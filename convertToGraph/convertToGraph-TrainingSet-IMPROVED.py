#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二步：训练集图结构转换（KNN + Delaunay 统一图构建版本）
将vector_data和vector_data_attacked下的训练集矢量数据转换为GAT可处理的图结构
使用 KNN + Delaunay 统一图构建方式

【核心改进】：
1. 引入20维最优几何不变特征（方案D：自适应版，替代原19维）
   - 维度0-2:  几何类型编码（one-hot）
   - 维度3:    Hu不变矩φ1（完全几何不变）⭐
   - 维度4:    边界复杂度（缩放不变）
   - 维度5-7:  当前地图相对位置（宏观空间信息）
   - 维度8-10: 局部相对位置（微观空间信息，抗裁剪）⭐核心
   - 维度11-12: 长宽比 + 矩形度（旋转不变）
   - 维度13:   Solidity（形状复杂度）
   - 维度14:   对数顶点数（复杂度指标）
   - 维度15-17: 拓扑邻域特征（图结构相关）
   - 维度18:   孔洞数量（拓扑特征）
   - 维度19:   节点数编码（图规模信息）⭐新增
2. 多尺度位置表达：全局+局部并存，GAT自动学习权重
3. 实施全局标准化（替代逐图标准化）
4. **KNN + Delaunay 统一图构建**：⭐⭐⭐
   - 自适应K值：根据节点数动态调整（K最大为8）
   - KNN保证局部密集连接：每个节点至多8个邻居
   - Delaunay保证全局连通：覆盖所有节点，填补稀疏区域
   - 适用于所有数据类型（点/线/面），无孤岛节点
5. 对各种攻击（特别是裁剪和删除对象）具有极强鲁棒性

【性能优化】：
6. KD-tree加速KNN构建：O(n log n) 复杂度
7. Delaunay三角剖分：O(n log n) 复杂度
8. 无需R-tree依赖（Delaunay自带空间索引）

【依赖安装】：
  pip install scipy scikit-learn
  
  注意：scipy用于Delaunay三角剖分，sklearn用于KNN
"""

import os
import geopandas as gpd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected  # ✅ 用于无向图转换
from tqdm import tqdm
import shutil
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from scipy.spatial import Delaunay
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import gc
import time
import hashlib
import json

class ImprovedTrainSetVectorToGraphConverter:
    """改进的训练集矢量数据转图结构转换器（KNN + Delaunay 统一图构建）"""
    
    @staticmethod
    def adaptive_k_for_graph(n_nodes):
        """
        ⭐ 根据节点数自适应确定K值（K最小为1，完全自适应）
        
        公式：K = min(round(2 * log10(n)) + 2, n-1)
        - K随节点数对数增长
        - 限制范围：[1, min(12, n-1)]
        
        示例：
        - n=1      → K=1   (n-1)
        - n=2      → K=1   (n-1)
        - n=3-7    → K=min(4, n-1)
        - n=8-999  → K=min(计算值, n-1)
        - n=1,000  → K=8   (2*3 + 2 = 8)
        - n=10,000 → K=10  (2*4 + 2 = 10)
        - n≥100,000→ K=12  (达到上限)
        
        优势：
        - 极小图（n<3）：K=n-1，能构建基本连接
        - 小图：K自适应，不会超过节点数
        - 中大图：K适中，保持局部连接
        - 最大K限制为12，避免过密
        
        Args:
            n_nodes: 图的节点总数
            
        Returns:
            int: 推荐的K值（范围1-12）
        """
        if n_nodes < 2:
            return 1  # 单节点图，K=1
        
        if n_nodes == 2:
            return 1  # 2个节点，K=1
        
        # K = 2 * log10(n) + 2，但不超过 n-1
        k = int(round(2 * np.log10(n_nodes) + 2))
        
        # 限制范围 [1, min(12, n-1)]
        k = max(1, min(min(12, n_nodes - 1), k))
        
        return k
    
    @staticmethod
    def hilbert_distance(x, y, order=16):
        """
        ⭐ 计算2D点在Hilbert曲线上的距离（用于空间排序）
        
        Hilbert曲线是一种空间填充曲线，能保持空间局部性：
        - 在曲线上相邻的点，在2D空间中也相邻
        - 用于对点排序，使得相邻的点在同一分块中
        
        Args:
            x, y: 归一化坐标 [0, 1]
            order: Hilbert曲线阶数（默认16，支持2^16=65536个点）
            
        Returns:
            int: Hilbert距离
        """
        # 将[0,1]坐标映射到[0, 2^order-1]整数空间
        max_coord = (1 << order) - 1
        xi = int(x * max_coord)
        yi = int(y * max_coord)
        
        # Hilbert曲线编码（迭代版本）
        d = 0
        s = 1 << (order - 1)
        
        while s > 0:
            rx = 1 if (xi & s) > 0 else 0
            ry = 1 if (yi & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            
            # 旋转坐标
            if ry == 0:
                if rx == 1:
                    xi = max_coord - xi
                    yi = max_coord - yi
                xi, yi = yi, xi
            
            s >>= 1
        
        return d
    
    @staticmethod
    def partition_by_hilbert(centroids, block_size=2000):
        """
        ⭐ 使用Hilbert曲线将节点分块（保持空间局部性）
        
        优势：
        - 每个块内的节点在空间上聚集
        - 块间连接数量少，减少跨块边
        - Delaunay复杂度：O(n_block * log n_block) << O(n_total^2)
        
        Args:
            centroids: (N, 2) numpy数组，节点坐标
            block_size: 每个块的最大节点数（默认2000）
            
        Returns:
            list: [block1_indices, block2_indices, ...]
        """
        n = len(centroids)
        
        if n <= block_size:
            return [list(range(n))]  # 单个块
        
        # 归一化坐标到[0, 1]
        min_x, min_y = centroids.min(axis=0)
        max_x, max_y = centroids.max(axis=0)
        
        # 防止除零
        range_x = max_x - min_x if max_x > min_x else 1.0
        range_y = max_y - min_y if max_y > min_y else 1.0
        
        norm_x = (centroids[:, 0] - min_x) / range_x
        norm_y = (centroids[:, 1] - min_y) / range_y
        
        # 计算每个点的Hilbert距离
        hilbert_distances = np.array([
            ImprovedTrainSetVectorToGraphConverter.hilbert_distance(x, y)
            for x, y in zip(norm_x, norm_y)
        ])
        
        # 按Hilbert距离排序
        sorted_indices = np.argsort(hilbert_distances)
        
        # 分块
        blocks = []
        for i in range(0, n, block_size):
            block_indices = sorted_indices[i:i+block_size].tolist()
            blocks.append(block_indices)
        
        print(f"  ⭐ Hilbert分块：{n}节点 → {len(blocks)}块（每块≤{block_size}）")
        
        return blocks
    
    def __init__(self, vector_dir="../convertToGeoJson/GeoJson/TrainingSet", 
                 attacked_dir="../convertToGeoJson-Attacked/GeoJson-Attacked/TrainingSet", 
                 graph_dir="Graph/TrainingSet",
                 batch_size=100,
                 max_workers=None,
                 use_cache=True):
        self.vector_dir = vector_dir
        self.attacked_dir = attacked_dir
        self.graph_dir = graph_dir
        self.batch_size = batch_size
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.use_cache = use_cache
        
        self.ensure_graph_dir()
        
        # 使用全局标准化器
        self.global_scaler = StandardScaler()
        self.scaler_fitted = False
        
        # 存储全局统计量（用于计算相对位置）
        self.global_bounds = None
        self.global_centroid = None
        
        # 缓存相关
        self.cache_dir = os.path.join(self.graph_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # ⭐ 方案A：仅用原始图生成scaler，缓存文件名区分
        self.features_cache_file = os.path.join(self.cache_dir, "features_cache_original_only.pkl")
        self.file_hashes_file = os.path.join(self.cache_dir, "file_hashes_original_only.json")
        
    def ensure_graph_dir(self):
        """确保图数据目录存在"""
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        os.makedirs(os.path.join(self.graph_dir, 'Original'), exist_ok=True)
        os.makedirs(os.path.join(self.graph_dir, 'Attacked'), exist_ok=True)

    def clean_output_dirs(self):
        """清空输出目录，确保每次运行可完全替换"""
        original_path = os.path.join(self.graph_dir, 'Original')
        attacked_path = os.path.join(self.graph_dir, 'Attacked')

        if os.path.exists(original_path):
            for name in os.listdir(original_path):
                # 跳过 macOS 的隐藏元数据文件
                if name.startswith('._'):
                    continue
                
                file_path = os.path.join(original_path, name)
                try:
                    # 检查路径是否存在
                    if not os.path.exists(file_path):
                        continue
                    
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"⚠️  删除文件失败 {file_path}: {e}")
                    continue
        else:
            os.makedirs(original_path, exist_ok=True)

        if os.path.exists(attacked_path):
            try:
                shutil.rmtree(attacked_path)
            except Exception as e:
                print(f"⚠️  删除目录失败 {attacked_path}: {e}")
        os.makedirs(attacked_path, exist_ok=True)
    
    def get_file_hash(self, file_path):
        """计算文件哈希值用于缓存验证"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None

    def load_file_hashes(self):
        """加载文件哈希缓存"""
        if os.path.exists(self.file_hashes_file):
            try:
                with open(self.file_hashes_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_file_hashes(self, file_hashes):
        """保存文件哈希缓存"""
        try:
            with open(self.file_hashes_file, 'w') as f:
                json.dump(file_hashes, f)
        except Exception as e:
            print(f"保存文件哈希失败: {e}")
    
    def check_original_files_changed(self, old_hashes):
        """检查原始图文件是否有变化"""
        if not os.path.exists(self.vector_dir):
            return False
        
        for filename in os.listdir(self.vector_dir):
            if filename.endswith('.geojson') and not filename.startswith('._'):
                file_path = os.path.join(self.vector_dir, filename)
                current_hash = self.get_file_hash(file_path)
                
                # 如果文件是新的或者哈希变化了
                if file_path not in old_hashes or old_hashes.get(file_path) != current_hash:
                    return True
        
        return False
    
    def get_graph_output_path(self, filename, attacked_subdir=None, data_type='Original'):
        """获取图文件的输出路径"""
        graph_name = filename.replace('.geojson', '')
        
        if data_type == 'Original':
            output_dir = os.path.join(self.graph_dir, 'Original')
            output_path = os.path.join(output_dir, f"{graph_name}_graph.pkl")
        else:  # Attacked
            output_dir = os.path.join(self.graph_dir, 'Attacked', attacked_subdir)
            output_path = os.path.join(output_dir, f"{graph_name}_graph.pkl")
        
        return output_path
    
    def should_update_file(self, file_path, old_hashes, output_path):
        """判断文件是否需要更新（基于哈希和输出文件存在性）"""
        # 如果输出文件不存在，必须更新
        if not os.path.exists(output_path):
            return True, "输出文件不存在"
        
        # 计算当前文件哈希
        current_hash = self.get_file_hash(file_path)
        if current_hash is None:
            return True, "无法计算哈希"
        
        # 如果文件是新的或哈希变化了
        if file_path not in old_hashes:
            return True, "新文件"
        
        if old_hashes.get(file_path) != current_hash:
            return True, "文件已修改"
        
        return False, "无变化"

    def monitor_system_resources(self):
        """监控系统资源使用情况"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            print(f"💻 系统资源: CPU {cpu:.1f}%, 内存 {memory.percent:.1f}% ({memory.used // 1024**3}GB/{memory.total // 1024**3}GB)")
            
            if memory.percent > 85:
                print("⚠️  内存使用率过高，建议减少batch_size或max_workers")
                
            return memory.percent < 90  # 返回是否可以继续处理
        except:
            return True

    def get_all_file_paths(self):
        """获取所有需要处理的文件路径"""
        file_paths = []
        
        # 原始数据
        if os.path.exists(self.vector_dir):
            for filename in os.listdir(self.vector_dir):
                if filename.endswith('.geojson') and not filename.startswith('._'):
                    file_paths.append(('original', os.path.join(self.vector_dir, filename)))
        
        # 攻击数据
        if os.path.exists(self.attacked_dir):
            for attacked_subdir in os.listdir(self.attacked_dir):
                attack_dir_path = os.path.join(self.attacked_dir, attacked_subdir)
                if os.path.isdir(attack_dir_path):
                    for filename in os.listdir(attack_dir_path):
                        if filename.endswith('.geojson') and not filename.startswith('._'):
                            file_paths.append(('attacked', os.path.join(attack_dir_path, filename)))
        
        return file_paths
    
    def calculate_global_statistics(self, all_gdfs):
        """计算所有几何要素的全局统计量"""
        print("计算全局统计量...")
        
        # 收集所有几何要素
        all_geometries = []
        for gdf in all_gdfs:
            all_geometries.extend(gdf.geometry.tolist())
        
        # 计算全局边界框
        all_bounds = [geom.bounds for geom in all_geometries]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        self.global_bounds = (min_x, min_y, max_x, max_y)
        
        # 计算全局质心
        all_centroids = [geom.centroid for geom in all_geometries]
        avg_x = np.mean([c.x for c in all_centroids])
        avg_y = np.mean([c.y for c in all_centroids])
        self.global_centroid = Point(avg_x, avg_y)
        
        print(f"全局边界框: {self.global_bounds}")
        print(f"全局质心: ({avg_x:.2f}, {avg_y:.2f})")
    
    def extract_improved_features(self, geometry, row, geometry_index=None, all_centroids=None, 
                                  local_bounds=None, local_centroid=None, k_neighbors_info=None, total_nodes=None):
        """
        提取改进的20维几何不变特征（方案D：自适应版，全局+局部多尺度+节点数编码）
        
        特征列表：
        0-2.   几何类型编码（3维）- one-hot
        3.     Hu不变矩φ1（1维）- 完全几何不变，最经典的形状描述符⭐
        4.     边界复杂度 Boundary Complexity（1维）- 缩放不变，对噪声鲁棒
        5-7.   当前地图相对位置（3维）- 当前地图的宏观空间信息（独立全局）
        8-10.  局部相对位置（3维）- 微观空间信息（抗裁剪）⭐核心
        11-12. 长宽比 + 矩形度（2维）- 旋转不变
        13.    Solidity（1维）- 形状复杂度
        14.    对数顶点数（1维）- 复杂度指标
        15-17. 拓扑邻域特征（3维）- 基于拓扑邻接，与图结构一致
        18.    孔洞数量 Holes（1维）- 拓扑特征，抗攻击
        19.    节点数编码（1维）- 图规模信息⭐新增
        
        【方案D核心思想】：
        - 当前地图位置：描述节点在当前地图中的宏观位置（独立归一化）
        - 局部位置：描述节点在邻域中的微观位置（裁剪后仍稳定）
        - 节点数编码：补偿自适应K值归一化后丢失的图规模信息
        - GAT注意力机制会自动学习在不同场景下使用不同特征
        
        Args:
            geometry: 当前几何要素
            row: GeoDataFrame的行数据
            geometry_index: 当前几何索引（用于计算局部位置）
            all_centroids: 所有几何要素的质心列表（用于计算局部位置）
            local_bounds: 当前地图的边界框（用于归一化位置）
            local_centroid: 当前地图的质心（用于计算相对距离）
            k_neighbors_info: 预计算的K近邻信息（用于加速）
            total_nodes: 当前地图的总节点数（用于节点数编码）
        """
        features = []
        
        # ===== 1-3. 几何类型编码（3维）=====
        geom_type = geometry.geom_type if hasattr(geometry, 'geom_type') else 'Unknown'
        if geom_type == 'Point':
            geom_features = [1, 0, 0]
        elif geom_type in ['LineString', 'MultiLineString']:
            geom_features = [0, 1, 0]
        elif geom_type in ['Polygon', 'MultiPolygon']:
            geom_features = [0, 0, 1]
        else:
            geom_features = [0, 0, 0]
        features.extend(geom_features)
        
        # 获取基本几何属性
        area = geometry.area if hasattr(geometry, 'area') else 0.0
        perimeter = geometry.length if hasattr(geometry, 'length') else 0.0
        
        # ===== 4. Hu不变矩φ1（完全几何不变）=====
        # Hu矩是最经典的形状不变量：对平移、缩放、旋转都完全不变
        # φ1 = η20 + η02（第一个Hu不变矩，最稳定）
        # 替代紧凑度，消除与边界复杂度的冗余
        
        if area > 1e-6 and geom_type in ['Polygon', 'MultiPolygon']:
            try:
                # 提取边界坐标
                if geom_type == 'Polygon':
                    coords = np.array(geometry.exterior.coords[:-1])  # 去掉重复的最后一点
                else:  # MultiPolygon，取最大的多边形
                    largest_poly = max(geometry.geoms, key=lambda p: p.area)
                    coords = np.array(largest_poly.exterior.coords[:-1])
                
                if len(coords) >= 3:
                    # 计算质心
                    cx = np.mean(coords[:, 0])
                    cy = np.mean(coords[:, 1])
                    
                    # 计算中心矩 μpq = Σ(x-cx)^p * (y-cy)^q
                    x_centered = coords[:, 0] - cx
                    y_centered = coords[:, 1] - cy
                    
                    mu20 = np.sum(x_centered**2) / len(coords)
                    mu02 = np.sum(y_centered**2) / len(coords)
                    mu11 = np.sum(x_centered * y_centered) / len(coords)
                    
                    # 归一化中心矩 ηpq = μpq / μ00^((p+q)/2+1)
                    # μ00 近似为 area
                    if area > 1e-6:
                        nu20 = mu20 / (area ** 1.0)  # (2+0)/2+1 = 2
                        nu02 = mu02 / (area ** 1.0)
                        
                        # 第一个Hu不变矩：φ1 = η20 + η02
                        hu1 = nu20 + nu02
                        
                        # 对数归一化（Hu矩值域可能很大）
                        hu1_normalized = np.log1p(abs(hu1)) / 10.0
                    else:
                        hu1_normalized = 0.0
                else:
                    hu1_normalized = 0.0
            except Exception:
                hu1_normalized = 0.0
        else:
            # Point和LineString使用简化值
            hu1_normalized = 0.0 if geom_type == 'Point' else 0.5
        
        features.append(hu1_normalized)
        
        # ===== 5. 边界复杂度 Boundary Complexity（缩放不变）=====
        # 公式: perimeter / sqrt(area)
        # 比形状指数更稳定，对噪声更鲁棒
        if area > 1e-6:
            boundary_complexity = perimeter / np.sqrt(area)
            # 对数归一化，避免值过大
            boundary_complexity = np.log1p(boundary_complexity) / 5.0  # 经验性归一化
        else:
            boundary_complexity = 0.0
        features.append(boundary_complexity)
        
        # ===== 6-8. 当前地图相对位置（独立全局，宏观空间信息）=====
        # 描述节点在当前地图中的位置
        # 使用独立归一化，裁剪攻击后仍保持相对稳定
        centroid = geometry.centroid
        
        if local_bounds is not None:
            # 维度6: 相对X位置（归一化到[0,1]）
            local_width = local_bounds[2] - local_bounds[0]
            if local_width > 1e-6:
                local_relative_x = (centroid.x - local_bounds[0]) / local_width
            else:
                local_relative_x = 0.5
            
            # 维度7: 相对Y位置（归一化到[0,1]）
            local_height = local_bounds[3] - local_bounds[1]
            if local_height > 1e-6:
                local_relative_y = (centroid.y - local_bounds[1]) / local_height
            else:
                local_relative_y = 0.5
            
            # 维度8: 相对于当前地图质心的距离（归一化）
            local_diagonal = np.sqrt(local_width**2 + local_height**2)
            if local_diagonal > 1e-6 and local_centroid is not None:
                distance_to_local_center = centroid.distance(local_centroid) / local_diagonal
            else:
                distance_to_local_center = 0.0
            
            features.extend([local_relative_x, local_relative_y, distance_to_local_center])
        else:
            # 如果没有提供边界框，使用默认值
            features.extend([0.5, 0.5, 0.0])
        
        # ===== 9-11. 局部相对位置（微观空间信息，抗裁剪）=====
        # 基于K近邻的局部参考系，即使裁剪后也稳定
        # 这是方案B的核心：全局+局部多尺度表达
        
        if k_neighbors_info is not None:
            # 使用预计算的 K近邻（KD-tree 加速，O(n log n)）
            neighbor_centroids = k_neighbors_info['centroids']
            neighbor_distances = k_neighbors_info['distances']
            
            if len(neighbor_centroids) > 0:
                # 计算局部质心（K近邻的平均位置）
                local_centroid_x = np.mean([c.x for c in neighbor_centroids])
                local_centroid_y = np.mean([c.y for c in neighbor_centroids])
                local_centroid = Point(local_centroid_x, local_centroid_y)
                
                # 计算局部半径（K近邻的平均距离）
                local_radius = np.mean(neighbor_distances)
                
                # 维度9: 相对于局部质心的X偏移（归一化）
                if local_radius > 1e-6:
                    local_relative_x = (centroid.x - local_centroid.x) / (local_radius * 2)
                    local_relative_x = np.clip(local_relative_x, -1, 1)  # 限制到[-1, 1]
                else:
                    local_relative_x = 0.0
                
                # 维度10: 相对于局部质心的Y偏移（归一化）
                if local_radius > 1e-6:
                    local_relative_y = (centroid.y - local_centroid.y) / (local_radius * 2)
                    local_relative_y = np.clip(local_relative_y, -1, 1)
                else:
                    local_relative_y = 0.0
                
                # 维度11: 到局部质心的距离（归一化）
                if local_radius > 1e-6:
                    distance_to_local_center = centroid.distance(local_centroid) / local_radius
                else:
                    distance_to_local_center = 0.0
                
                features.extend([local_relative_x, local_relative_y, distance_to_local_center])
            else:
                # 如果没有其他节点，使用默认值
                features.extend([0.0, 0.0, 0.0])
        else:
            # 如果没有提供all_centroids，使用默认值（第一次遍历时）
            features.extend([0.0, 0.0, 0.0])
        
        # ===== 12-13. 长宽比 + 矩形度（旋转不变）=====
        if geom_type in ['Polygon', 'MultiPolygon'] and area > 0:
            try:
                # 最小外接矩形
                min_rect = geometry.minimum_rotated_rectangle
                
                # 处理MultiPolygon的情况
                if min_rect.geom_type == 'MultiPolygon':
                    # 如果最小外接矩形是MultiPolygon，取最大的那个
                    largest_rect = max(min_rect.geoms, key=lambda p: p.area)
                    rect_coords = list(largest_rect.exterior.coords)
                else:
                    rect_coords = list(min_rect.exterior.coords)
                
                # 计算矩形的两条边长
                edge1 = np.linalg.norm(
                    np.array(rect_coords[0]) - np.array(rect_coords[1])
                )
                edge2 = np.linalg.norm(
                    np.array(rect_coords[1]) - np.array(rect_coords[2])
                )
                
                # 长宽比（归一化）
                if min(edge1, edge2) > 0:
                    aspect_ratio = max(edge1, edge2) / min(edge1, edge2)
                    # 对数变换，避免极端值
                    aspect_ratio = np.log1p(aspect_ratio) / 3.0  # 经验性归一化
                else:
                    aspect_ratio = 0.0
                
                # 矩形度：原图形面积 / 最小外接矩形面积
                rect_area = min_rect.area
                if rect_area > 0:
                    rectangularity = area / rect_area
                else:
                    rectangularity = 0.0
                
            except Exception as e:
                aspect_ratio, rectangularity = 0.0, 0.0
        else:
            aspect_ratio, rectangularity = 0.0, 1.0 if geom_type == 'Point' else 0.0
        
        features.extend([aspect_ratio, rectangularity])
        
        # ===== 14. Solidity 实心度（形状复杂度）=====
        # 公式: area / convex_hull.area
        # 衡量形状的凹凸程度：凸多边形=1.0，凹进去越多值越小
        if area > 0:
            try:
                convex_hull = geometry.convex_hull
                convex_area = convex_hull.area
                if convex_area > 0:
                    solidity = area / convex_area
                else:
                    solidity = 1.0
            except:
                solidity = 1.0
        else:
            solidity = 1.0 if geom_type == 'Point' else 0.0
        features.append(solidity)
        
        # ===== 15. 对数顶点数（复杂度指标）=====
        if geom_type == 'Point':
            num_vertices = 1
        elif geom_type in ['LineString', 'MultiLineString']:
            if geom_type == 'LineString':
                num_vertices = len(list(geometry.coords))
            else:  # MultiLineString
                num_vertices = sum(len(list(line.coords)) for line in geometry.geoms)
        elif geom_type in ['Polygon', 'MultiPolygon']:
            if geom_type == 'Polygon':
                num_vertices = len(list(geometry.exterior.coords))
            else:  # MultiPolygon
                num_vertices = sum(len(list(poly.exterior.coords)) for poly in geometry.geoms)
        else:
            num_vertices = 1
        
        # 对数归一化
        log_vertices = np.log1p(num_vertices) / 10.0  # 经验性归一化
        features.append(log_vertices)
        
        # ===== 16-18. 拓扑邻域特征（占位符，后续更新）=====
        # 这些特征需要在构建拓扑邻接图后计算
        # 暂时填充0，在update_topology_neighborhood_features中更新
        features.extend([0.0, 0.0, 0.0])
        
        # ===== 19. 孔洞数量 Holes（拓扑特征）=====
        # 多边形内部的孔洞数量，对删除/裁剪攻击鲁棒
        if geom_type == 'Polygon':
            try:
                num_holes = len(geometry.interiors)
            except:
                num_holes = 0
        elif geom_type == 'MultiPolygon':
            try:
                num_holes = sum(len(poly.interiors) for poly in geometry.geoms)
            except:
                num_holes = 0
        else:
            # Point和LineString没有孔洞
            num_holes = 0
        
        # 对数归一化（大多数多边形没有孔洞）
        log_holes = np.log1p(num_holes) / 5.0
        features.append(log_holes)
        
        # ===== 20. 节点数编码（图规模信息）⭐新增 =====
        # 补偿自适应K值归一化后丢失的图规模信息
        # 对数归一化：log10(n+1) / 4.0
        # 范围示例：
        #   n=10    -> 0.25
        #   n=100   -> 0.50
        #   n=1000  -> 0.75
        #   n=10000 -> 1.00
        if total_nodes is not None and total_nodes > 0:
            node_count_feature = np.log10(total_nodes + 1) / 4.0
        else:
            # 如果未提供，使用默认值（假设中等规模图）
            node_count_feature = 0.5
        features.append(node_count_feature)
        
        # ⭐ 方案A优化：添加 clip 防止极端攻击产生超出范围的特征值
        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -10.0, 10.0)  # 限制特征范围，防止极端值影响标准化
        
        return features
    
    def update_topology_neighborhood_features(self, node_features, geometries, topology_edges, local_bounds=None):
        """
        更新空间邻域特征（特征维度15-17）- 基于Delaunay三角剖分⭐优化版
        
        🎯 核心改进：用Delaunay邻居替代原始拓扑边，彻底解决NaN问题
        
        基于Delaunay三角剖分邻接关系计算：
        - 维度15: 与Delaunay邻居的平均距离（归一化，clip限制）
        - 维度16: Delaunay邻居数量（对数归一化，自动反映密度）⭐
        - 维度17: Delaunay邻域密度（邻居数/面积）
        
        优势：
        ✅ 完全几何不变：只依赖质心坐标
        ✅ 抗拓扑攻击：顶点打乱不影响Delaunay结构
        ✅ 邻居数自适应：密集区域多邻居，稀疏区域少邻居
        ✅ 数值稳定：添加clip防止溢出到万亿级别
        ✅ 无需重复计算：复用build_knn_delaunay_edges的结果
        
        Args:
            node_features: 节点特征矩阵
            geometries: 几何要素列表
            topology_edges: Delaunay边列表 [[i, j], ...]（从build_knn_delaunay_edges获取）
            local_bounds: 当前地图边界框（用于归一化距离）
        """
        n_samples = len(geometries)
        if n_samples < 2:
            return node_features
        
        # 构建邻接表
        adjacency = {i: set() for i in range(n_samples)}
        for edge in topology_edges:
            if len(edge) == 2:
                i, j = edge
                adjacency[i].add(j)
                # 边已经是双向的，不需要重复添加
        
        # 提取质心
        centroids = [g.centroid for g in geometries]
        
        # 计算当前地图对角线用于归一化
        if local_bounds is not None:
            local_width = local_bounds[2] - local_bounds[0]
            local_height = local_bounds[3] - local_bounds[1]
            local_diagonal = np.sqrt(local_width**2 + local_height**2)
        else:
            local_diagonal = 1.0
        
        # 防止除零
        if local_diagonal < 1e-6:
            local_diagonal = 1.0
        
        # 更新每个节点的Delaunay邻域特征
        for i in range(n_samples):
            neighbors = list(adjacency[i])
            
            if len(neighbors) > 0:
                # 计算到Delaunay邻居的距离
                distances = [centroids[i].distance(centroids[j]) for j in neighbors]
                avg_dist_raw = np.mean(distances)
                
                # 维度15: 平均距离（归一化）⭐添加clip防止溢出
                avg_distance = avg_dist_raw / local_diagonal
                avg_distance = np.clip(avg_distance, 0.0, 1.0)  # ⭐关键：限制到[0,1]
                
                # 维度16: Delaunay邻居数量（对数归一化）⭐自动反映密度差异
                num_neighbors = np.log1p(len(neighbors)) / 5.0
                num_neighbors = np.clip(num_neighbors, 0.0, 1.0)  # ⭐限制范围
                
                # 维度17: Delaunay邻域密度⭐添加clip防止溢出
                if avg_dist_raw > 1e-10:
                    # 邻域面积（圆形假设：π * r²）
                    neighborhood_area = np.pi * (avg_dist_raw ** 2)
                    density = len(neighbors) / neighborhood_area
                    
                    # 对数归一化
                    density = np.log1p(density * 1000) / 10.0
                    density = np.clip(density, 0.0, 1.0)  # ⭐限制范围
                else:
                    density = 0.0
            else:
                # 如果没有邻居（理论上Delaunay不会有孤岛，但保险起见）
                avg_distance = 0.0
                num_neighbors = 0.0
                density = 0.0
            
            node_features[i, 15] = avg_distance
            node_features[i, 16] = num_neighbors
            node_features[i, 17] = density
        
        return node_features
    
    def build_knn_delaunay_edges(self, geometries):
        """
        ⭐ 构建KNN + Delaunay统一图（支持大图Hilbert分块优化）
        
        策略：
        1. KNN保证局部密集连接（每个节点至多k个邻居）
        2. Delaunay三角剖分保证全局连通：
           - 节点≤5000：直接Delaunay
           - 节点>5000：⭐ Hilbert曲线分块 + 跨块KNN连接
        3. 合并去重，返回单向边列表
        
        Args:
            geometries: 几何要素列表
            
        Returns:
            list: 边列表 [[src, dst], ...]
            dict: 统计信息
        """
        from sklearn.neighbors import NearestNeighbors
        from scipy.spatial import Delaunay
        
        n = len(geometries)
        
        # 自适应K值
        k = self.adaptive_k_for_graph(n)
        
        # 限制K值不超过节点数-1
        k = min(k, n - 1)
        
        if n < 2:
            return [], {'total_nodes': n, 'knn_k': 0, 'knn_edges': 0, 
                       'delaunay_edges': 0, 'total_edges': 0, 
                       'isolated_nodes': n, 'avg_degree': 0, 
                       'delaunay_edges_list': []}
        
        # 提取质心
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in geometries])
        
        edges_set = set()
        
        # ==== 第1步：KNN边（全局） ====
        print(f"  [1/2] 构建KNN图（K={k}，共{n}个节点）...")
        knn_start = time.time()
        nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='kd_tree').fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)
        
        for i in range(n):
            for j in indices[i][1:]:  # 跳过自己（第一个是自己）
                edge = tuple(sorted([i, j]))  # 无向边：统一为 (min, max)
                edges_set.add(edge)
        
        knn_edges_count = len(edges_set)
        print(f"  ✓ KNN完成，边数: {knn_edges_count}，耗时: {time.time()-knn_start:.2f}秒")
        
        # ==== 第2步：Delaunay边（自适应策略，n<3时跳过） ====
        delaunay_edges_list = []  # 保存Delaunay边列表用于特征计算
        
        if n < 3:
            # 节点数<3，无法构建Delaunay三角剖分，只使用KNN
            print(f"  [2/2] 跳过Delaunay（节点数={n}<3，只使用KNN边）")
        elif n >= 3:
            if n <= 5000:
                # 🔹 小图：直接Delaunay
                print(f"  [2/2] 构建Delaunay三角剖分（{n}个节点）...")
                delaunay_start = time.time()
                try:
                    tri = Delaunay(centroids)
                    delaunay_edges_count = 0
                    
                    for simplex in tri.simplices:
                        # 三角形的三条边
                        for i in range(3):
                            v1 = simplex[i]
                            v2 = simplex[(i+1) % 3]
                            edge = tuple(sorted([v1, v2]))
                            delaunay_edges_list.append([v1, v2])
                            if edge not in edges_set:
                                delaunay_edges_count += 1
                            edges_set.add(edge)
                    
                    print(f"  ✓ Delaunay完成，新增边数: {delaunay_edges_count}，耗时: {time.time()-delaunay_start:.2f}秒")
                except Exception as e:
                    print(f"  ⚠️  Delaunay失败: {e}，仅使用KNN边")
            else:
                # ⭐ 大图：Hilbert分块Delaunay
                print(f"  [2/2] ⭐ 大图优化：Hilbert分块Delaunay（{n}个节点）...")
                delaunay_start = time.time()
                
                # 分块
                blocks = self.partition_by_hilbert(centroids, block_size=2000)
                
                total_delaunay_edges = 0
                
                # 对每个块做Delaunay
                for block_idx, block_indices in enumerate(blocks):
                    if len(block_indices) < 3:
                        continue
                    
                    block_centroids = centroids[block_indices]
                    
                    try:
                        tri = Delaunay(block_centroids)
                        
                        for simplex in tri.simplices:
                            for i in range(3):
                                local_v1 = simplex[i]
                                local_v2 = simplex[(i+1) % 3]
                                # 映射回全局索引
                                global_v1 = block_indices[local_v1]
                                global_v2 = block_indices[local_v2]
                                edge = tuple(sorted([global_v1, global_v2]))
                                delaunay_edges_list.append([global_v1, global_v2])
                                if edge not in edges_set:
                                    total_delaunay_edges += 1
                                edges_set.add(edge)
                    except Exception as e:
                        print(f"    ⚠️  块{block_idx}的Delaunay失败: {e}")
                        continue
                
                # 跨块连接：使用KNN连接相邻块的边界节点
                print(f"  ⭐ 跨块连接（KNN K={k//2}）...")
                cross_block_edges = 0
                
                for i in range(len(blocks) - 1):
                    # 取当前块的最后10%节点和下一块的前10%节点
                    block1_indices = blocks[i]
                    block2_indices = blocks[i+1]
                    
                    boundary1_size = max(1, len(block1_indices) // 10)
                    boundary2_size = max(1, len(block2_indices) // 10)
                    
                    boundary1 = block1_indices[-boundary1_size:]
                    boundary2 = block2_indices[:boundary2_size]
                    
                    # 在边界节点间做KNN连接
                    boundary_centroids = centroids[boundary1 + boundary2]
                    boundary_k = min(k//2, len(boundary_centroids)-1)
                    
                    if boundary_k >= 1:
                        nbrs_boundary = NearestNeighbors(
                            n_neighbors=boundary_k+1, algorithm='kd_tree'
                        ).fit(boundary_centroids)
                        _, indices_boundary = nbrs_boundary.kneighbors(boundary_centroids)
                        
                        for local_i, neighbors in enumerate(indices_boundary):
                            global_i = (boundary1 + boundary2)[local_i]
                            for local_j in neighbors[1:]:
                                global_j = (boundary1 + boundary2)[local_j]
                                edge = tuple(sorted([global_i, global_j]))
                                if edge not in edges_set:
                                    cross_block_edges += 1
                                edges_set.add(edge)
                
                print(f"  ✓ 分块Delaunay完成：")
                print(f"    - 块内边: {total_delaunay_edges}")
                print(f"    - 跨块边: {cross_block_edges}")
                print(f"    - 耗时: {time.time()-delaunay_start:.2f}秒")
        else:
            print(f"  [2/2] 节点数<3，跳过Delaunay")
        
        # ✅ 第3步：转换为边列表（单向表示）
        edges_list = [[e[0], e[1]] for e in edges_set]
        
        print(f"  ✅ 去重完成，无向边数: {len(edges_list)} 条（单向表示）")
        
        # 统计孤岛节点
        connected_nodes = set()
        for edge in edges_list:
            connected_nodes.add(edge[0])
            connected_nodes.add(edge[1])
        
        isolated_count = n - len(connected_nodes)
        
        stats = {
            'total_nodes': n,
            'knn_k': k,
            'knn_edges': knn_edges_count,
            'delaunay_edges': len(delaunay_edges_list),
            'total_edges': len(edges_list),
            'isolated_nodes': isolated_count,
            'avg_degree': (len(edges_list) * 2) / n if n > 0 else 0,
            'delaunay_edges_list': delaunay_edges_list  # ⭐保存Delaunay边列表
        }
        
        return edges_list, stats
    
    def build_rng_edges(self, geometries, node_indices=None, k=5):
        """
        构建RNG补充边（K近邻快速版本）
        
        原理：
        - 原始RNG: O(n³) - 对每对节点检查所有其他节点
        - 优化版本: O(n log n) - 对孤岛节点连接K个最近邻
        
        优化说明：
        - 严格的RNG对于孤岛节点补充来说过于严格且计算昂贵
        - 使用K近邻作为RNG的实用近似：保证连通性，计算快速
        - K=5通常足够保证连通性，同时保持图的稀疏性
        
        Args:
            geometries: 所有几何要素
            node_indices: 需要处理的节点索引列表（None表示全部节点）
            k: 每个孤岛节点连接的最近邻数量（默认5）
        """
        if node_indices is None:
            node_indices = list(range(len(geometries)))
        
        if len(node_indices) == 0:
            return []
        
        from sklearn.neighbors import NearestNeighbors
        
        # 提取所有节点的质心
        all_centroids = np.array([[g.centroid.x, g.centroid.y] for g in geometries])
        
        edges = []
        n_total = len(geometries)
        
        # 动态调整k：不能超过总节点数-1
        actual_k = min(k, n_total - 1)
        
        if actual_k < 1:
            return []
        
        print(f"  K近邻补充（K={actual_k}，处理 {len(node_indices)} 个孤岛节点）...")
        
        # 使用KD树加速最近邻搜索
        nbrs = NearestNeighbors(
            n_neighbors=actual_k + 1,  # +1因为包括自己
            algorithm='kd_tree'
        ).fit(all_centroids)
        
        # 为每个孤岛节点找K个最近邻
        for i in node_indices:
            distances, indices = nbrs.kneighbors([all_centroids[i]])
            
            # 排除自己（第一个是自己），取K个最近邻
            for j in indices[0][1:actual_k+1]:
                edges.append([i, int(j)])
        
        print(f"  补充边数量: {len(edges)}")
        print(f"  加速效果: O(n log n) vs 原RNG的 O(n³)")
        
        return edges
    
    def find_isolated_nodes(self, edges, n_nodes):
        """
        找出孤岛节点（没有任何边的节点）
        
        Args:
            edges: 边列表 [[src, dst], ...]
            n_nodes: 总节点数
        
        Returns:
            set: 孤岛节点的索引集合
        """
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge[0])
            connected_nodes.add(edge[1])
        
        all_nodes = set(range(n_nodes))
        isolated = all_nodes - connected_nodes
        
        return isolated
    
    def build_level0_edges(self, geometries):
        """
        构建第0层（节点层）的边
        策略：KNN + Delaunay 统一图构建
        
        Returns:
            list: 边列表
            dict: 统计信息
        """
        n = len(geometries)
        print(f"\n=== 构建图结构（KNN + Delaunay，{n}个节点）===")
        
        # 直接使用 KNN + Delaunay 统一图构建
        unique_edges, stats = self.build_knn_delaunay_edges(geometries)
        
        return unique_edges, stats
    
    
    def build_knn_delaunay_graph(self, geometries, node_features):
        """
        构建 KNN + Delaunay 统一图（简化版，移除聚类计算）
        
        策略：
        1. 自适应K值：K最大为20（根据节点数动态调整）
        2. KNN构建：每个节点连接K个最近邻（局部密集）
        3. Delaunay三角剖分：覆盖全图（全局连通）
        
        Returns:
            dict: 包含图结构信息的字典
        """
        n = len(geometries)
        
        print(f"\n{'='*60}")
        print(f"构建 KNN + Delaunay 统一图（{n} 个节点）")
        print(f"{'='*60}")
        
        # === 构建主图结构 ===
        level0_edges, level0_stats = self.build_level0_edges(geometries)
        
        print(f"\n{'='*60}")
        print(f"KNN + Delaunay 图构建完成")
        print(f"{'='*60}")
        print(f"节点数: {n}")
        print(f"边数: {level0_stats['total_edges']} 对")
        print(f"自适应K值: {level0_stats.get('knn_k', 'N/A')}")
        print(f"平均度数: {level0_stats.get('avg_degree', 0):.2f}")
        print(f"连通性: 无孤岛节点（Delaunay保证）")
        print(f"{'='*60}\n")
        
        return {
            'edges': level0_edges,
            'stats': level0_stats
        }
    
    def build_graph_from_gdf(self, gdf, graph_name, use_global_scaler=True):
        """
        从GeoDataFrame构建拓扑增强图（简化优化版）
        
        Returns:
            Data: PyTorch Geometric Data对象，包含聚类信息
        """
        print(f"\n{'='*60}")
        print(f"处理: {graph_name}")
        print(f"{'='*60}")
        
        # 提取几何要素
        geometries = gdf.geometry.tolist()
        
        # 第一步：计算当前地图的边界框和质心（独立全局）
        all_bounds = [geom.bounds for geom in geometries]
        local_bounds = (
            min(b[0] for b in all_bounds),  # min_x
            min(b[1] for b in all_bounds),  # min_y
            max(b[2] for b in all_bounds),  # max_x
            max(b[3] for b in all_bounds)   # max_y
        )
        
        # 计算当前地图的质心
        all_centroids = [Point(geom.centroid.x, geom.centroid.y) for geom in geometries]
        local_centroid = Point(
            np.mean([c.x for c in all_centroids]),
            np.mean([c.y for c in all_centroids])
        )
        
        print(f"当前地图边界框: {local_bounds}")
        print(f"当前地图质心: ({local_centroid.x:.2f}, {local_centroid.y:.2f})")
        
        # 使用 KD-tree 预计算 K近邻（大幅加速）
        k_neighbors_dict = self.precompute_k_neighbors(all_centroids)
        
        # 第二步：提取特征（传入当前地图的统计量）
        total_nodes = len(geometries)  # 总节点数
        node_features = []
        for idx, row in gdf.iterrows():
            features = self.extract_improved_features(
                row.geometry, 
                row, 
                geometry_index=idx,  # 传入索引
                all_centroids=all_centroids,  # 传入所有质心
                local_bounds=local_bounds,  # 传入当前地图边界框
                local_centroid=local_centroid,  # 传入当前地图质心
                k_neighbors_info=k_neighbors_dict.get(idx),  # 传入预计算的K近邻
                total_nodes=total_nodes  # 传入总节点数（第20维特征）⭐新增
            )
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # 第三步：构建KNN+Delaunay图（用于邻域特征计算）
        print("\n构建图用于邻域特征计算...")
        graph_edges_merged, graph_stats = self.build_knn_delaunay_edges(geometries)
        
        # ⭐提取纯Delaunay边用于特征计算（抗攻击，数值稳定）
        delaunay_edges_for_features = graph_stats.get('delaunay_edges_list', [])
        if not delaunay_edges_for_features:
            # 如果没有Delaunay边（节点<3），使用合并后的边
            delaunay_edges_for_features = graph_edges_merged
        
        # 第四步：更新Delaunay邻域特征（维度15-17）⭐使用纯Delaunay边
        node_features = self.update_topology_neighborhood_features(
            node_features, 
            geometries, 
            delaunay_edges_for_features,  # ⭐改用Delaunay边
            local_bounds=local_bounds
        )
        print(f"✅ 已更新邻域特征（基于 {len(delaunay_edges_for_features)} 条Delaunay边）")
        
        # 第五步：标准化特征
        if len(node_features) > 0:
            if use_global_scaler:
                # 必须使用全局标准化器
                if not self.scaler_fitted:
                    raise RuntimeError(
                        f"❌ 尝试使用全局标准化器但未拟合！\n"
                        f"   图名: {graph_name}\n"
                        f"   请先调用 first_pass_collect_features() 拟合全局标准化器"
                    )
                node_features = self.global_scaler.transform(node_features)
            else:
                # 临时标准化（仅用于第一遍收集统计量）
                scaler = StandardScaler()
                node_features = scaler.fit_transform(node_features)
        
        # 第六步：构建 KNN + Delaunay 统一图
        graph_info = self.build_knn_delaunay_graph(geometries, node_features)
        
        # 提取边信息
        edges = graph_info['edges']
        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).T
            # ✅ 转换为无向图（自动添加反向边）
            edge_index = to_undirected(edge_index)
        else:
            # 如果没有边（极端情况：只有一个节点），创建空边索引
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 创建PyTorch Geometric Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index
        )
        
        # 保存简化的统计信息（只保存数值，不保存字典引用）
        data.n_nodes = graph_info['stats']['total_nodes']
        data.n_edges = graph_info['stats']['total_edges']
        data.n_isolated_nodes = graph_info['stats']['isolated_nodes']
        
        return data
    
    def save_graph_data(self, data, filename, subdir=None, data_type='Original'):
        """保存图数据，如果文件存在则覆盖"""
        if data_type == 'Original':
            save_dir = os.path.join(self.graph_dir, 'Original')
        else:
            save_dir = os.path.join(self.graph_dir, 'Attacked', subdir if subdir else '')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f"{filename}_graph.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"已保存: {save_path}")
        
    def check_graph_exists(self, filename, subdir=None, data_type='Original'):
        """检查图文件是否已存在"""
        if data_type == 'Original':
            save_dir = os.path.join(self.graph_dir, 'Original')
        else:
            save_dir = os.path.join(self.graph_dir, 'Attacked', subdir if subdir else '')
        
        save_path = os.path.join(save_dir, f"{filename}_graph.pkl")
        return os.path.exists(save_path)
    
    def first_pass_collect_features(self):
        """
        优化版第一遍：流式处理收集特征（独立全局）
        ⭐ 仅用原始图生成 global_scaler（方案A）
        使用缓存加速，每个地图独立计算边界框
        """
        print("\n=== 🚀 优化版第一遍：收集原始图特征（仅原始图生成scaler） ===")
        print("【策略】仅用原始图生成 global_scaler.pkl，攻击图使用此scaler标准化")
        print("【优势】速度快、符合零水印逻辑、特征空间以原始图为基准\n")
        
        start_time = time.time()
        self.monitor_system_resources()
        
        # ⭐ 仅获取原始图路径
        file_paths = []
        if os.path.exists(self.vector_dir):
            for filename in os.listdir(self.vector_dir):
                if filename.endswith('.geojson') and not filename.startswith('._'):
                    file_paths.append(('original', os.path.join(self.vector_dir, filename)))
        
        print(f"发现 {len(file_paths)} 个原始图文件（仅用于生成scaler）")
        
        if not file_paths:
            print("❌ 未找到任何文件，请检查路径设置")
            return
        
        # 检查缓存
        all_features = None
        if self.use_cache:
            all_features = self.try_load_cached_features(file_paths)
        
        if all_features is None:
            print("💾 缓存无效或未启用，开始重新计算特征...")
            
            # 直接提取特征（使用独立全局，每个地图自己计算边界框）
            print("🔧 提取特征（使用独立全局归一化）...")
            all_features = self.process_files_with_local_bounds(file_paths)
            
            if self.use_cache and all_features is not None:
                self.save_features_cache(all_features, file_paths)
        
        if all_features is None or len(all_features) == 0:
            raise RuntimeError("❌ 特征提取失败：无法收集到有效特征，可能是内存不足或文件损坏")
        
        all_features = np.array(all_features, dtype=np.float32)
        print(f"✅ 共提取 {len(all_features)} 个节点的特征，特征维度: {all_features.shape[1]}")
        
        # 拟合全局标准化器
        print("🔧 拟合全局标准化器（仅基于原始图）...")
        self.global_scaler.fit(all_features)
        self.scaler_fitted = True
        
        # 保存标准化器
        scaler_path = os.path.join(self.graph_dir, 'global_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.global_scaler,
                'global_bounds': self.global_bounds,
                'global_centroid': self.global_centroid,
                'strategy': 'original_only'  # ⭐ 标记使用方案A
            }, f)
        print(f"✅ 已保存全局标准化器: {scaler_path}")
        
        # 打印统计信息
        elapsed_time = time.time() - start_time
        print(f"\n" + "="*70)
        print(f"📊 第一遍完成：仅原始图特征收集")
        print(f"="*70)
        print(f"⏱️  耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.1f}分钟)")
        print(f"📊 原始图数量: {len(file_paths)} 个")
        print(f"📊 节点总数: {len(all_features)} 个")
        print(f"📈 处理速度: {len(all_features)/elapsed_time:.1f} 特征/秒")
        print(f"🎯 特征维度: {all_features.shape[1]}")
        print(f"✅ 攻击图将使用此scaler标准化（无需重新扫描）")
        print(f"="*70)
        
        # 清理内存
        del all_features
        gc.collect()
        self.monitor_system_resources()
    
    def try_load_cached_features(self, file_paths):
        """尝试加载缓存的特征数据"""
        if not os.path.exists(self.features_cache_file):
            return None
        
        print("🔍 检查缓存有效性...")
        
        # 改进的缓存验证策略：
        # 1. 检查缓存文件的修改时间
        # 2. 采样检查文件哈希（智能采样）
        # 3. 检查文件总数是否一致
        
        cached_hashes = self.load_file_hashes()
        
        # 检查文件总数是否一致
        if len(cached_hashes) != len(file_paths):
            print(f"📝 文件数量变化: 缓存 {len(cached_hashes)} vs 当前 {len(file_paths)}")
            return None
        
        # 智能采样检查：检查最新修改的文件和随机采样
        import random
        
        # 按文件修改时间排序，检查最新的文件
        file_paths_with_mtime = []
        for file_type, file_path in file_paths:
            try:
                mtime = os.path.getmtime(file_path)
                file_paths_with_mtime.append((file_type, file_path, mtime))
            except:
                continue
        
        file_paths_with_mtime.sort(key=lambda x: x[2], reverse=True)  # 按修改时间降序
        
        # 检查最新的20个文件 + 随机采样30个文件
        recent_files = file_paths_with_mtime[:20]
        remaining_files = file_paths_with_mtime[20:]
        random_sample = random.sample(remaining_files, min(30, len(remaining_files)))
        
        files_to_check = recent_files + random_sample
        
        for file_type, file_path, mtime in files_to_check:
            current_hash = self.get_file_hash(file_path)
            if current_hash is None:
                continue
                
            if file_path not in cached_hashes or cached_hashes[file_path] != current_hash:
                print(f"📝 文件已变更: {os.path.basename(file_path)}")
                return None
        
        try:
            print("✅ 缓存验证通过，加载特征数据...")
            with open(self.features_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # 检查缓存版本
            cache_version = cached_data.get('version', '0.0')
            current_version = '1.3'  # 版本1.3：方案A（仅原始图生成scaler）+ clip优化
            if cache_version != current_version:
                print(f"📝 缓存版本不匹配: {cache_version} vs {current_version}")
                return None
                
            self.global_bounds = cached_data.get('global_bounds')
            self.global_centroid = cached_data.get('global_centroid')
            
            features = cached_data.get('features', [])
            print(f"📦 从缓存加载 {len(features)} 个特征")
            print(f"🔍 采样检查了 {len(files_to_check)} 个文件的完整性")
            return features
            
        except Exception as e:
            print(f"❌ 缓存加载失败: {e}")
            return None
    
    def save_features_cache(self, features, file_paths):
        """保存特征数据到缓存"""
        try:
            print("💾 保存特征缓存...")
            
            # 计算并保存文件哈希
            file_hashes = {}
            for file_type, file_path in file_paths:
                file_hash = self.get_file_hash(file_path)
                if file_hash:
                    file_hashes[file_path] = file_hash
            
            # 保存缓存数据
            cache_data = {
                'features': features,
                'global_bounds': self.global_bounds,
                'global_centroid': self.global_centroid,
                'version': '1.3',  # 版本1.3：方案A（仅原始图生成scaler）+ clip优化
                'strategy': 'original_only'  # ⭐ 标记使用方案A
            }
            
            with open(self.features_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.save_file_hashes(file_hashes)
            print("✅ 缓存保存完成")
            
        except Exception as e:
            print(f"❌ 缓存保存失败: {e}")
    
    def process_files_optimized(self, file_paths):
        """优化的文件处理方法"""
        all_features = []
        all_geometries = []
        
        # 分批处理文件
        batches = [file_paths[i:i + self.batch_size] for i in range(0, len(file_paths), self.batch_size)]
        print(f"🔄 分为 {len(batches)} 个批次处理，每批 {self.batch_size} 个文件")
        
        # 串行处理（避免多进程的复杂性，但保留批次处理的内存优化）
        for batch_idx, batch in enumerate(batches):
            if not self.monitor_system_resources():
                print("⚠️  内存不足，停止处理")
                break
                
            print(f"📂 处理批次 {batch_idx + 1}/{len(batches)}")
            
            batch_features, batch_geometries = self.process_file_batch_serial(batch)
            all_features.extend(batch_features)
            all_geometries.extend(batch_geometries)
            
            # 定期清理内存
            if batch_idx % 10 == 0:
                gc.collect()
        
        # 全局统计量已在第一阶段计算，这里不需要重复计算
        return all_features
    
    def process_file_batch_serial(self, file_batch):
        """串行处理一批文件"""
        batch_features = []
        batch_geometries = []
        
        for file_type, file_path in tqdm(file_batch, desc="处理文件", leave=False):
            try:
                gdf = gpd.read_file(file_path)
                
                for idx, row in gdf.iterrows():
                    try:
                        features = self.extract_improved_features(row.geometry, row)
                        batch_features.append(features)
                        
                        # 收集几何信息用于全局统计
                        batch_geometries.append({
                            'bounds': row.geometry.bounds,
                            'centroid': (row.geometry.centroid.x, row.geometry.centroid.y)
                        })
                    except Exception as e:
                        print(f"⚠️  特征提取失败: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ 文件读取失败 {os.path.basename(file_path)}: {e}")
                continue
        
        return batch_features, batch_geometries
    
    def precompute_k_neighbors(self, all_centroids, k=5):
        """
        使用 KD-tree 预计算所有节点的 K近邻（O(n log n) 而不是 O(n²)）
        
        Args:
            all_centroids: 所有节点的质心列表（Point对象）
            k: 近邻数量
            
        Returns:
            dict: {节点索引: {'indices': [...], 'distances': [...], 'centroids': [...]}}
        """
        from sklearn.neighbors import NearestNeighbors
        
        n = len(all_centroids)
        if n < 2:
            return {}
        
        # 转换为 numpy 数组
        centroids_array = np.array([[c.x, c.y] for c in all_centroids])
        
        # 构建 KD-tree 并查询
        actual_k = min(k, n - 1)
        nbrs = NearestNeighbors(
            n_neighbors=actual_k + 1,  # +1 因为包括自己
            algorithm='kd_tree'
        ).fit(centroids_array)
        
        distances, indices = nbrs.kneighbors(centroids_array)
        
        # 构建结果字典
        result = {}
        for i in range(n):
            # 排除自己（第一个是自己，距离为0）
            neighbor_indices = indices[i][1:actual_k+1]
            neighbor_distances = distances[i][1:actual_k+1]
            neighbor_centroids = [all_centroids[j] for j in neighbor_indices]
            
            result[i] = {
                'indices': neighbor_indices,
                'distances': neighbor_distances,
                'centroids': neighbor_centroids
            }
        
        return result
    
    def process_files_with_local_bounds(self, file_paths):
        """使用独立全局处理文件（每个地图自己计算边界框）"""
        all_features = []
        
        # 分批处理文件
        batches = [file_paths[i:i + self.batch_size] for i in range(0, len(file_paths), self.batch_size)]
        print(f"🔄 分为 {len(batches)} 个批次处理，每批 {self.batch_size} 个文件")
        
        for batch_idx, batch in enumerate(batches):
            if not self.monitor_system_resources():
                print("⚠️  内存不足，停止处理")
                break
                
            print(f"📂 处理批次 {batch_idx + 1}/{len(batches)}")
            
            for file_type, file_path in tqdm(batch, desc="处理文件", leave=False):
                try:
                    gdf = gpd.read_file(file_path)
                    geometries = gdf.geometry.tolist()
                    
                    # 计算当前地图的边界框和质心
                    all_bounds = [geom.bounds for geom in geometries]
                    local_bounds = (
                        min(b[0] for b in all_bounds),
                        min(b[1] for b in all_bounds),
                        max(b[2] for b in all_bounds),
                        max(b[3] for b in all_bounds)
                    )
                    
                    all_centroids = [Point(geom.centroid.x, geom.centroid.y) for geom in geometries]
                    local_centroid = Point(
                        np.mean([c.x for c in all_centroids]),
                        np.mean([c.y for c in all_centroids])
                    )
                    
                    # 使用 KD-tree 预计算 K近邻（大幅加速）
                    k_neighbors_dict = self.precompute_k_neighbors(all_centroids)
                    
                    # 提取特征
                    total_nodes = len(geometries)  # 总节点数
                    for idx, row in gdf.iterrows():
                        try:
                            features = self.extract_improved_features(
                                row.geometry, 
                                row,
                                geometry_index=idx,
                                all_centroids=all_centroids,
                                local_bounds=local_bounds,
                                local_centroid=local_centroid,
                                k_neighbors_info=k_neighbors_dict.get(idx),
                                total_nodes=total_nodes  # 传入总节点数（第20维特征）⭐新增
                            )
                            all_features.append(features)
                        except Exception as e:
                            print(f"⚠️  特征提取失败: {e}")
                            continue
                            
                except Exception as e:
                    print(f"❌ 文件读取失败 {os.path.basename(file_path)}: {e}")
                    continue
            
            # 定期清理内存
            if batch_idx % 10 == 0:
                gc.collect()
        
        return all_features
    
    def calculate_global_statistics_from_geometries(self, geometries):
        """从几何信息计算全局统计量"""
        print("📊 计算全局统计量...")
        
        if not geometries:
            self.global_bounds = (0, 0, 1, 1)
            self.global_centroid = (0.5, 0.5)
            return
        
        # 计算全局边界框
        min_x = min(geom['bounds'][0] for geom in geometries)
        min_y = min(geom['bounds'][1] for geom in geometries)
        max_x = max(geom['bounds'][2] for geom in geometries)
        max_y = max(geom['bounds'][3] for geom in geometries)
        
        self.global_bounds = (min_x, min_y, max_x, max_y)
        
        # 计算全局质心
        centroids_x = [geom['centroid'][0] for geom in geometries]
        centroids_y = [geom['centroid'][1] for geom in geometries]
        
        self.global_centroid = (
            np.mean(centroids_x),
            np.mean(centroids_y)
        )
        
        print(f"🌍 全局边界框: {self.global_bounds}")
        print(f"🎯 全局质心: {self.global_centroid}")

    def calculate_global_stats_from_files(self, file_paths):
        """
        快速扫描所有文件计算全局统计量
        只读取几何信息，不提取完整特征
        """
        print("📊 快速扫描计算全局统计量...")
        
        all_bounds = []
        all_centroids = []
        
        for file_type, file_path in tqdm(file_paths, desc="扫描文件"):
            try:
                gdf = gpd.read_file(file_path)
                
                for idx, row in gdf.iterrows():
                    try:
                        geom = row.geometry
                        all_bounds.append(geom.bounds)
                        all_centroids.append((geom.centroid.x, geom.centroid.y))
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"⚠️  扫描文件失败 {os.path.basename(file_path)}: {e}")
                continue
        
        if not all_bounds:
            print("⚠️  未找到有效几何数据，使用默认全局统计量")
            self.global_bounds = (0, 0, 1, 1)
            self.global_centroid = Point(0.5, 0.5)
            return
        
        # 计算全局边界框
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        self.global_bounds = (min_x, min_y, max_x, max_y)
        
        # 计算全局质心
        avg_x = np.mean([c[0] for c in all_centroids])
        avg_y = np.mean([c[1] for c in all_centroids])
        self.global_centroid = Point(avg_x, avg_y)
        
        print(f"🌍 全局边界框: {self.global_bounds}")
        print(f"🎯 全局质心: ({avg_x:.2f}, {avg_y:.2f})")
        print(f"📈 扫描了 {len(all_centroids)} 个几何要素")
    
    def second_pass_convert_and_save(self, incremental_mode=True):
        """
        第二遍：使用全局标准化器转换并保存图数据（原始图+攻击图）
        
        Args:
            incremental_mode: 是否使用增量更新模式
                - True: 只更新变化的文件（基于文件哈希）
                - False: 清空并重新生成所有文件
        """
        print("\n=== 第二遍：转换并保存图数据（原始图 + 攻击图） ===")
        print("【策略】所有图使用第一遍生成的 global_scaler.pkl 标准化")
        print(f"【模式】{'🔄 增量更新模式（只更新变化的文件）' if incremental_mode else '🔥 完全重新生成模式'}\n")
        
        if not self.scaler_fitted:
            raise ValueError("必须先调用 first_pass_collect_features()")
        
        # 总体时间统计
        total_start_time = time.time()
        
        # 加载旧的哈希记录（用于判断文件是否变化）
        old_hashes = self.load_file_hashes() if incremental_mode else {}
        new_hashes = {}
        
        # 检查原始图是否有变化
        if incremental_mode and old_hashes:
            original_changed = self.check_original_files_changed(old_hashes)
            if original_changed:
                print("⚠️  检测到原始图文件有变化！")
                print("⚠️  原始图变化会影响 global_scaler.pkl，需要重新生成所有图")
                print("⚠️  切换到完全重新生成模式...\n")
                incremental_mode = False
                old_hashes = {}
        
        # 根据模式决定是否清理输出目录
        if not incremental_mode:
            print("\n🧹 清理旧图数据...")
            self.clean_output_dirs()
            print("✅ 旧数据已清理\n")
        else:
            print("\n🔍 增量更新模式：只更新变化的文件，保留未变化的文件\n")
        
        # 处理原始数据
        print("\n" + "="*70)
        print("📂 处理原始数据（TrainingSet/Original）")
        print("="*70)
        original_start_time = time.time()
        skipped_count = 0
        processed_count = 0
        
        for filename in os.listdir(self.vector_dir):
            if filename.endswith('.geojson') and not filename.startswith('._'):
                graph_name = filename.replace('.geojson', '')
                file_path = os.path.join(self.vector_dir, filename)
                output_path = self.get_graph_output_path(filename, data_type='Original')
                
                try:
                    # 增量更新模式：检查是否需要更新
                    if incremental_mode:
                        should_update, reason = self.should_update_file(file_path, old_hashes, output_path)
                        
                        if not should_update:
                            print(f"⏭️  跳过 {filename} ({reason})")
                            skipped_count += 1
                            # 记录哈希（即使跳过也要记录）
                            new_hashes[file_path] = self.get_file_hash(file_path)
                            continue
                        else:
                            print(f"🔄 更新 {filename} ({reason})")
                    
                    # 读取并处理文件
                    gdf = gpd.read_file(file_path)
                    
                    # 构建图（使用全局标准化器）
                    data = self.build_graph_from_gdf(gdf, filename, use_global_scaler=True)
                    
                    self.save_graph_data(data, graph_name, data_type='Original')
                    processed_count += 1
                    
                    # 记录文件哈希
                    new_hashes[file_path] = self.get_file_hash(file_path)
                    
                except Exception as e:
                    print(f"❌ 处理文件 {filename} 时出错: {e}")
                    continue
        
        original_elapsed = time.time() - original_start_time
        print(f"\n✅ 原始数据处理完成")
        print(f"   - 处理数量: {processed_count} 个")
        if incremental_mode:
            print(f"   - 跳过数量: {skipped_count} 个")
        print(f"   - 耗时: {original_elapsed:.2f}秒 ({original_elapsed/60:.1f}分钟)")
        print(f"   - 平均速度: {original_elapsed/processed_count if processed_count > 0 else 0:.2f}秒/图")
        
        # 处理攻击数据
        print("\n" + "="*70)
        print("📂 处理攻击数据（TrainingSet/Attacked）")
        print("="*70)
        attacked_start_time = time.time()
        
        # 统计信息
        attack_type_stats = {}  # {attack_type: {'count': N, 'time': T, 'skipped': S}}
        
        for attacked_subdir in os.listdir(self.attacked_dir):
            attack_dir_path = os.path.join(self.attacked_dir, attacked_subdir)
            if os.path.isdir(attack_dir_path):
                # 统计处理的文件数
                subdir_start_time = time.time()
                subdir_processed = 0
                subdir_skipped = 0
                total_files = len([f for f in os.listdir(attack_dir_path) if f.endswith('.geojson') and not f.startswith('._')])
                
                print(f"\n📂 [{attacked_subdir}] 共 {total_files} 个文件")
                
                # 在增量模式下，显示更详细的信息
                use_tqdm = not incremental_mode  # 增量模式下不用进度条，改用逐行输出
                
                file_list = os.listdir(attack_dir_path)
                iterator = tqdm(file_list, desc=f"  处理中", leave=False) if use_tqdm else file_list
                
                for filename in iterator:
                    if filename.endswith('.geojson') and not filename.startswith('._'):
                        graph_name = filename.replace('.geojson', '')
                        file_path = os.path.join(attack_dir_path, filename)
                        output_path = self.get_graph_output_path(filename, attacked_subdir, 'Attacked')
                        
                        try:
                            # 增量更新模式：检查是否需要更新
                            if incremental_mode:
                                should_update, reason = self.should_update_file(file_path, old_hashes, output_path)
                                
                                if not should_update:
                                    if subdir_skipped < 3:  # 只显示前几个跳过的文件
                                        print(f"  ⏭️  跳过 {attacked_subdir}/{filename} ({reason})")
                                    subdir_skipped += 1
                                    # 记录哈希（即使跳过也要记录）
                                    new_hashes[file_path] = self.get_file_hash(file_path)
                                    continue
                                else:
                                    print(f"  🔄 更新 {attacked_subdir}/{filename} ({reason})")
                            
                            # 读取并处理文件
                            gdf = gpd.read_file(file_path)
                            
                            data = self.build_graph_from_gdf(gdf, filename, use_global_scaler=True)
                            self.save_graph_data(data, graph_name, attacked_subdir, 'Attacked')
                            subdir_processed += 1
                            
                            # 记录文件哈希
                            new_hashes[file_path] = self.get_file_hash(file_path)
                            
                        except Exception as e:
                            print(f"    ❌ 处理文件 {filename} 时出错: {e}")
                            continue
                
                subdir_elapsed = time.time() - subdir_start_time
                attack_type_stats[attacked_subdir] = {
                    'count': subdir_processed,
                    'time': subdir_elapsed,
                    'skipped': subdir_skipped
                }
                
                if incremental_mode and subdir_skipped > 3:
                    print(f"  ... 还有 {subdir_skipped - 3} 个文件被跳过")
                
                print(f"  ✅ {attacked_subdir}: {subdir_processed} 个处理" + 
                      (f", {subdir_skipped} 个跳过" if incremental_mode else "") +
                      f"，耗时 {subdir_elapsed:.2f}秒 " +
                      f"(平均 {subdir_elapsed/subdir_processed if subdir_processed > 0 else 0:.2f}秒/图)")
        
        # 计算总体统计
        attacked_elapsed = time.time() - attacked_start_time
        total_attacked_count = sum(stats['count'] for stats in attack_type_stats.values())
        total_attacked_skipped = sum(stats.get('skipped', 0) for stats in attack_type_stats.values())
        total_elapsed = time.time() - total_start_time
        
        # 保存哈希记录（增量模式）
        if incremental_mode:
            print("\n💾 保存文件哈希记录...")
            self.save_file_hashes(new_hashes)
            print(f"✅ 已保存 {len(new_hashes)} 个文件的哈希记录")
        
        print("\n" + "="*70)
        print("✅ 训练集转换完成！")
        print("="*70)
        
        # 打印详细统计
        print(f"\n📊 转换统计汇总:")
        print(f"   原始图: {processed_count} 个处理" + 
              (f", {skipped_count} 个跳过" if incremental_mode else "") +
              f"，耗时 {original_elapsed:.2f}秒")
        print(f"   攻击图: {total_attacked_count} 个处理" + 
              (f", {total_attacked_skipped} 个跳过" if incremental_mode else "") +
              f"，耗时 {attacked_elapsed:.2f}秒")
        
        total_processed = processed_count + total_attacked_count
        total_skipped = skipped_count + total_attacked_skipped if incremental_mode else 0
        
        print(f"   总计: {total_processed} 个图处理" + 
              (f", {total_skipped} 个跳过" if incremental_mode else "") +
              f"，耗时 {total_elapsed:.2f}秒 ({total_elapsed/60:.1f}分钟)")
        print(f"   平均速度: {total_elapsed/total_processed if total_processed > 0 else 0:.2f}秒/图")
        
        if incremental_mode:
            print(f"\n⚡ 增量更新效果:")
            print(f"   - 跳过率: {total_skipped / (total_processed + total_skipped) * 100 if (total_processed + total_skipped) > 0 else 0:.1f}%")
            print(f"   - 节省时间: 约 {total_skipped * (total_elapsed/total_processed if total_processed > 0 else 0) / 60:.1f}分钟")
        
        print(f"\n📂 攻击类型统计 (Top 10 最慢):")
        # 按时间排序
        sorted_attacks = sorted(attack_type_stats.items(), key=lambda x: x[1]['time'], reverse=True)
        for i, (attack_type, stats) in enumerate(sorted_attacks[:10], 1):
            avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0
            skip_info = f" (跳过{stats.get('skipped', 0)})" if incremental_mode and stats.get('skipped', 0) > 0 else ""
            print(f"   {i:2d}. {attack_type:40s} {stats['count']:4d}图{skip_info} {stats['time']:7.2f}秒 (平均{avg_time:.2f}秒/图)")
        
        print("="*70)
    
    def convert_train_set_to_graph(self, incremental_mode=True):
        """
        完整的两遍转换流程（KNN + Delaunay 统一图构建 + 方案A优化 + 增量更新）：
        1. 第一遍：⭐仅收集原始图特征，拟合全局标准化器（方案A）
        2. 第二遍：原始图+攻击图都使用此标准化器转换并保存
        
        Args:
            incremental_mode: 是否使用增量更新模式（默认True）
                - True: 只更新变化的GeoJSON文件（基于MD5哈希）
                - False: 清空并重新生成所有图文件
        
        【方案A核心优势】：
        - ⚡ 速度极快：第一遍只需5分钟（vs 方案B的6小时）
        - 🎯 符合零水印逻辑：原始图是基准，攻击图向原始图对齐
        - 🛡️ 特征空间纯净：不被极端攻击"污染"scaler
        - 📊 理论正确：攻击图是原始图的几何变换
        - ✅ 添加clip防护：特征值限制在[-10, 10]范围内
        
        【增量更新优势】⭐新增：
        - 🔄 智能检测：基于文件哈希判断GeoJSON是否变化
        - ⚡ 大幅提速：只处理变化的文件，跳过未变化的文件
        - 🛡️ 安全机制：原始图变化时自动切换到完全重新生成模式
        - 📊 详细统计：显示跳过率和节省时间
        
        【其他优化】：
        - 每个地图使用自己的边界框归一化位置特征
        - 自适应K值：K最大为8（节点数<50时K=5）
        - 节点数编码：补偿自适应归一化后的图规模信息
        """
        print("\n" + "="*70)
        print("KNN + Delaunay 统一图训练集转换（方案A：仅原始图生成scaler）")
        print("="*70)
        print("【核心特性】")
        print("  1. 特征：20维几何不变特征 + clip防护（-10~10）")
        print("  2. ⭐ 标准化策略：方案A（仅用原始图生成scaler）")
        print("     - 第一遍：仅原始图 → 生成 global_scaler.pkl")
        print("     - 第二遍：所有图使用此scaler标准化")
        print("     - 优势：速度快、符合零水印逻辑、特征空间纯净")
        print("  3. ⭐ 增量更新：" + ("启用 🔄" if incremental_mode else "禁用 🔥"))
        print("     - 智能检测：基于文件哈希判断是否需要更新")
        print("     - 原始图变化：自动完全重新生成")
        print("     - 攻击图变化：仅更新变化的文件")
        print("  4. 位置归一化：独立全局（每个地图独立归一化）")
        print("  5. 图构建：KNN + Delaunay 统一图")
        print("     - 自适应K值：K最大为8（节点数<50时K=5）")
        print("     - KNN保证局部密集：每个节点至多8个邻居")
        print("     - Delaunay保证全局连通：覆盖所有节点")
        print("     - 适用所有数据类型：点/线/面，无孤岛节点")
        print("  6. 鲁棒性：对裁剪、删除对象攻击极强鲁棒性")
        print("="*70 + "\n")
        
        # 第一遍：收集特征和统计量
        self.first_pass_collect_features()
        
        # ✅ 验证scaler是否已拟合
        if not self.scaler_fitted:
            raise RuntimeError("❌ 第一遍特征收集失败：全局标准化器未拟合，无法继续第二遍处理")
        
        # 第二遍：转换并保存（传递增量模式参数）
        self.second_pass_convert_and_save(incremental_mode=incremental_mode)
        
        print("\n" + "="*70)
        print("训练集图结构转换完成！")
        print("="*70)
        print(f"输出目录: {self.graph_dir}")
        print("\n生成文件:")
        print("  - Original/: 原始图数据（KNN + Delaunay统一图）")
        print("  - Attacked/: 攻击图数据（KNN + Delaunay统一图）")
        print("  - global_scaler.pkl: 全局标准化器（供测试集使用）")
        print("\n图结构特点:")
        print("  - 无孤岛节点（Delaunay保证全局连通）")
        print("  - 局部密集（KNN保证局部连接，K≤8）")
        print("  - 自适应K值（K最大为8，节点数<50时K=5）")
        print("  - 聚类信息保留（作为节点属性）")
        print("  - 适用所有数据类型（点/线/面统一方案）")
        print("  - 对几何变换和内容破坏攻击高度鲁棒")
        print("="*70 + "\n")

def main():
    """主函数"""
    print("\n" + "="*70)
    print("    KNN + Delaunay 统一图 - 训练集图结构转换（自适应版）")
    print("="*70 + "\n")
    
    # 检查必需依赖
    missing_deps = []
    try:
        from scipy.spatial import Delaunay
        print("✅ 已检测到 scipy.spatial.Delaunay")
    except ImportError:
        missing_deps.append("scipy")
    
    try:
        from sklearn.neighbors import NearestNeighbors
        print("✅ 已检测到 sklearn.neighbors.NearestNeighbors")
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        print("\n" + "="*70)
        print("错误：缺少必需依赖")
        print("="*70)
        print(f"\n缺少的库: {', '.join(missing_deps)}\n")
        print("请安装缺少的依赖：")
        print("  pip install scipy scikit-learn\n")
        print("或者使用conda：")
        print("  conda install scipy scikit-learn\n")
        print("安装后重新运行此脚本。")
        print("="*70 + "\n")
        return
    
    print()
    
    # 创建优化版转换器
    converter = ImprovedTrainSetVectorToGraphConverter(
        batch_size=25,      # ⭐降低为25（原50），减少内存占用
        max_workers=4,      # 可根据CPU核心数调整 (默认8)
        use_cache=True      # 启用缓存加速
    )
    
    # ⭐ 增量更新模式（默认启用）
    # 如需完全重新生成，设置 incremental_mode=False
    incremental_mode = True  # True: 增量更新 | False: 完全重新生成
    
    print(f"🔧 配置参数:")
    print(f"   - 批次大小: {converter.batch_size}")
    print(f"   - 最大工作进程: {converter.max_workers}")
    print(f"   - 缓存启用: {converter.use_cache}")
    print(f"   - 增量更新: {'启用 🔄' if incremental_mode else '禁用 🔥'}")
    print()
    
    # 转换训练集数据
    try:
        converter.convert_train_set_to_graph(incremental_mode=incremental_mode)
    except RuntimeError as e:
        print(f"\n{'='*70}")
        print("❌ 转换失败")
        print(f"{'='*70}")
        print(f"错误信息: {e}")
        print(f"\n可能原因:")
        print("  1. 内存不足（当前内存使用率>90%）")
        print("  2. GeoJSON文件损坏")
        print("  3. 磁盘空间不足")
        print(f"\n建议:")
        print("  1. 关闭其他程序释放内存")
        print("  2. 减小 batch_size（当前50 → 建议25）")
        print("  3. 检查 GeoJSON 文件完整性")
        print(f"{'='*70}\n")
        return
    
    print("="*70)
    print("【重要提示】")
    print("="*70)
    print("1. ✅ 已生成全局标准化器 global_scaler.pkl（方案A：仅基于原始图）")
    print("2. ✅ 图数据包含聚类信息（data.node_cluster_ids）")
    print("3. ⚠️  模型训练时需要修改 input_dim=20（原来是19）")
    print("4. 📊 图构建方式：KNN + Delaunay 统一图")
    print("5. 🔧 如需使用聚类信息，可访问 data.node_cluster_ids")
    print("6. 🎯 自适应K值：K最大为8（节点数<50时K=5）")
    print("7. ⭐ 标准化策略：方案A（速度快、符合零水印逻辑）")
    
    print("\n【方案A优势】⭐⭐⭐")
    print("="*70)
    print("✓ 速度极快: 第一遍只需5分钟（vs 方案B的6小时）")
    print("✓ 符合零水印逻辑: 原始图是anchor，攻击图是positive")
    print("✓ 特征空间纯净: 不被极端攻击污染scaler")
    print("✓ 数值稳定: 添加clip防护，限制特征范围")
    print("✓ 理论正确: 攻击图应该向原始图对齐")
    
    print("\n【性能优化】⭐新增增量更新")
    print("="*70)
    print("✓ KD-tree加速KNN: O(n log n) 复杂度")
    print("✓ Delaunay三角剖分: O(n log n) 复杂度")
    print("✓ 智能缓存机制: 避免重复计算，二次运行极速")
    print("✓ 分批处理: 内存占用优化，支持大规模数据")
    print("✓ 增量更新: 基于文件哈希，只更新变化的文件")
    print("  - 原始图变化 → 自动完全重新生成")
    print("  - 攻击图变化 → 仅更新变化的文件")
    print("  - 大幅节省时间，特别是修改少量攻击文件时")
    print("✓ 实时监控: 系统资源监控，防止内存溢出")
    print("✓ 自适应K值: K最大为8，保持图稀疏性")
    
    print("\n【KNN + Delaunay 统一图的优势】")
    print("="*70)
    print("  ✓ 适用所有数据类型（点/线/面统一方案）")
    print("  ✓ 无孤岛节点（Delaunay保证100%连通）")
    print("  ✓ 局部密集（KNN保证邻域充分，K≤8）")
    print("  ✓ 自适应K值（K最大为8，节点数<50时K=5）")
    print("  ✓ 图稀疏性好（避免过度连接，训练更快）")
    print("  ✓ 20维特征（新增节点数编码）")
    print("  ✓ 模型泛化能力提升（结构归一化）")
    print("  ✓ 计算高效（O(n log n)，无需R-tree）")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

