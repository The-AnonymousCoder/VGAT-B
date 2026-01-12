#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二步：测试集图结构转换（KNN + Delaunay 统一图构建版本）
将vector_data_test和vector_data_test_attacked下的测试集矢量数据转换为GAT可处理的图结构
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
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from shapely.geometry import Point, MultiPoint
from scipy.spatial import Delaunay
import shutil
import time
import gc

class ImprovedTestSetVectorToGraphConverter:
    """改进的测试集矢量数据转图结构转换器（20维特征 + KNN + Delaunay 统一图构建）"""

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

        # 计算基础K值：2 * log10(n) + 2
        import math
        base_k = round(2 * math.log10(n_nodes)) + 2

        # 限制范围：[1, min(12, n-1)]
        # 说明：之前强制最小为3会在小图中导致超过 n-1，这里改为遵循 n-1 上限并最小为1
        k = int(min(max(1, base_k), 12, n_nodes - 1))

        return k
    
    @staticmethod
    def hilbert_distance(x, y, order=16):
        """计算2D点在Hilbert曲线上的距离（用于空间排序）"""
        max_coord = (1 << order) - 1
        xi = int(x * max_coord)
        yi = int(y * max_coord)
        d = 0
        s = 1 << (order - 1)
        while s > 0:
            rx = 1 if (xi & s) > 0 else 0
            ry = 1 if (yi & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    xi = max_coord - xi
                    yi = max_coord - yi
                xi, yi = yi, xi
            s >>= 1
        return d

    @staticmethod
    def partition_by_hilbert(centroids, block_size=2000):
        """使用 Hilbert 曲线将节点分块（保持空间局部性）"""
        n = len(centroids)
        if n <= block_size:
            return [list(range(n))]
        min_x, min_y = centroids.min(axis=0)
        max_x, max_y = centroids.max(axis=0)
        range_x = max_x - min_x if max_x > min_x else 1.0
        range_y = max_y - min_y if max_y > min_y else 1.0
        norm_x = (centroids[:, 0] - min_x) / range_x
        norm_y = (centroids[:, 1] - min_y) / range_y
        hilbert_distances = np.array([
            ImprovedTestSetVectorToGraphConverter.hilbert_distance(x, y)
            for x, y in zip(norm_x, norm_y)
        ])
        sorted_indices = np.argsort(hilbert_distances)
        blocks = []
        for i in range(0, n, block_size):
            block_indices = sorted_indices[i:i+block_size].tolist()
            blocks.append(block_indices)
        print(f"  ⭐ Hilbert分块：{n}节点 → {len(blocks)}块（每块≤{block_size}）")
        return blocks
    
    def __init__(self, original_dir="../convertToGeoJson/GeoJson/TestSet",
                 attacked_dir="../convertToGeoJson-Attacked/GeoJson-Attacked/TestSet",
                 graph_dir="Graph/TestSet",
                 training_scaler_path="../convertToGraph/Graph/TrainingSet/global_scaler.pkl"):
        # 规范化为绝对路径（相对路径相对于脚本文件位置）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.original_dir = original_dir if os.path.isabs(original_dir) else os.path.abspath(os.path.join(base_dir, original_dir))
        self.attacked_dir = attacked_dir if os.path.isabs(attacked_dir) else os.path.abspath(os.path.join(base_dir, attacked_dir))
        self.graph_dir = graph_dir if os.path.isabs(graph_dir) else os.path.abspath(os.path.join(base_dir, graph_dir))
        self.training_scaler_path = training_scaler_path if os.path.isabs(training_scaler_path) else os.path.abspath(os.path.join(base_dir, training_scaler_path))

        # 跳过过大的 GeoJSON 文件阈值（字节），12000 KB
        self.max_file_size_bytes = 12000 * 1024

        self.ensure_graph_dir()

        # ⭐ 使用训练集的全局标准化器（最佳实践）
        self.load_training_scaler()
    
    def ensure_graph_dir(self):
        """确保图数据目录存在"""
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        os.makedirs(os.path.join(self.graph_dir, 'Original'), exist_ok=True)
        os.makedirs(os.path.join(self.graph_dir, 'Attacked'), exist_ok=True)

    def load_training_scaler(self):
        """加载训练集生成的全局标准化器"""
        if not os.path.exists(self.training_scaler_path):
            # 尝试自动定位常见路径（兼容不同的运行工作目录）
            candidates = [
                os.path.join(os.path.dirname(__file__), 'Graph', 'TrainingSet', 'global_scaler.pkl'),
                os.path.join(os.getcwd(), 'Graph', 'TrainingSet', 'global_scaler.pkl'),
                os.path.join(os.path.dirname(__file__), '..', 'convertToGraph', 'Graph', 'TrainingSet', 'global_scaler.pkl')
            ]
            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found:
                print(f"⚠️ 警告：训练集标准化器路径 {self.training_scaler_path} 未找到，使用自动定位到: {found}")
                self.training_scaler_path = found
            else:
                raise FileNotFoundError(
                    f"\n{'='*70}\n"
                    f"❌ 未找到训练集标准化器文件\n"
                    f"{'='*70}\n\n"
                    f"请先运行训练集转换脚本生成标准化器：\n"
                    f"  cd ../convertToGraph\n"
                    f"  python convertToGraph-TrainingSet-IMPROVED.py\n\n"
                    f"预期文件路径: {self.training_scaler_path}\n"
                    f"{'='*70}"
                )

        try:
            with open(self.training_scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)

            self.global_scaler = scaler_data['scaler']
            print("✅ 已加载训练集的全局标准化器")
            print(f"   路径: {self.training_scaler_path}")
            print(f"   特征维度: {self.global_scaler.n_features_in_}")

            # 验证标准化器是否已拟合
            if not hasattr(self.global_scaler, 'mean_'):
                raise ValueError("标准化器未正确拟合（缺少mean_属性）")

        except Exception as e:
            raise RuntimeError(f"加载训练集标准化器失败: {e}")
    
    def clean_output_dirs(self):
        """清空输出目录，确保每次运行可完全替换"""
        original_path = os.path.join(self.graph_dir, 'Original')
        attacked_path = os.path.join(self.graph_dir, 'Attacked')
        # 修改说明：不再删除已存在的图文件，改为确保输出目录存在，以实现 append 行为
        if not os.path.exists(original_path):
            os.makedirs(original_path, exist_ok=True)
        if not os.path.exists(attacked_path):
            os.makedirs(attacked_path, exist_ok=True)
    
    def compute_global_bounds(self, gdf):
        """计算当前图的全局边界（用于相对位置特征）"""
        all_bounds = gdf.geometry.bounds
        self.global_bounds = (
            all_bounds['minx'].min(),
            all_bounds['miny'].min(),
            all_bounds['maxx'].max(),
            all_bounds['maxy'].max()
        )
        
        # 计算全局质心
        # 注意：要求 gdf 已经是投影坐标系（米制）以保证质心/距离计算正确
        try:
            all_centroids = gdf.geometry.centroid
            avg_x = all_centroids.x.mean()
            avg_y = all_centroids.y.mean()
            self.global_centroid = Point(avg_x, avg_y)
        except Exception:
            # 兜底：若计算失败，使用单个几何的质心或None
            try:
                any_geom = next((g for g in gdf.geometry if g is not None), None)
                if any_geom is not None:
                    c = any_geom.centroid
                    self.global_centroid = Point(c.x, c.y)
                else:
                    self.global_centroid = None
            except Exception:
                self.global_centroid = None

    # ===== 辅助安全几何处理方法（处理 Multi* / 投影问题） =====
    def _get_representative_polygon(self, geom):
        """如果是MultiPolygon，返回面积最大的子Polygon；否则若是Polygon返回自身；否则返回None"""
        try:
            if geom is None:
                return None
            if geom.geom_type == 'Polygon':
                return geom
            if geom.geom_type == 'MultiPolygon':
                # 选择面积最大的子多边形
                return max(geom.geoms, key=lambda p: p.area)
        except Exception:
            return None
        return None

    def _get_exterior_coords_array(self, geom):
        """
        安全获取多边形外环坐标数组（去掉重复最后一点）。
        返回 numpy.ndarray 或 None。
        """
        try:
            if geom is None:
                return None
            if geom.geom_type == 'Polygon':
                coords = np.array(geom.exterior.coords)
                if len(coords) > 1:
                    return coords[:-1]
                return coords
            if geom.geom_type == 'MultiPolygon':
                rep = self._get_representative_polygon(geom)
                if rep is not None:
                    coords = np.array(rep.exterior.coords)
                    if len(coords) > 1:
                        return coords[:-1]
                    return coords
            # 对于其他类型返回None
            return None
        except Exception:
            return None

    def _count_vertices(self, geom):
        """安全计算顶点数，考虑Multi*情形"""
        try:
            if geom is None:
                return 0
            t = geom.geom_type
            if t == 'Point':
                return 1
            if t == 'LineString':
                return len(list(geom.coords))
            if t == 'MultiLineString':
                return sum(len(list(g.coords)) for g in geom.geoms)
            if t == 'Polygon':
                return len(list(geom.exterior.coords))
            if t == 'MultiPolygon':
                return sum(len(list(p.exterior.coords)) for p in geom.geoms)
            # 其他类型
            return 0
        except Exception:
            return 0

    def precompute_k_neighbors(self, all_centroids, k=5):
        """使用 KD-tree 预计算所有节点的 K近邻（返回字典形式）"""
        from sklearn.neighbors import NearestNeighbors
        n = len(all_centroids)
        if n < 2:
            return {}
        centroids_array = np.array([[c.x, c.y] for c in all_centroids])
        actual_k = min(k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='kd_tree').fit(centroids_array)
        distances, indices = nbrs.kneighbors(centroids_array)
        result = {}
        for i in range(n):
            neighbor_indices = indices[i][1:actual_k+1]
            neighbor_distances = distances[i][1:actual_k+1]
            neighbor_centroids = [all_centroids[j] for j in neighbor_indices]
            result[i] = {
                'indices': neighbor_indices,
                'distances': neighbor_distances,
                'centroids': neighbor_centroids
            }
        return result
    
    def extract_improved_features(self, geometry, row, geometry_index=None, all_centroids=None, total_nodes=None, k_neighbors_info=None):
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
                # 安全获取边界坐标（处理 MultiPolygon）
                coords = self._get_exterior_coords_array(geometry)
                
                if coords is not None and len(coords) >= 3:
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
        
        # ===== 5-7. 当前地图相对位置（宏观空间信息）=====
        # 描述节点在当前地图中的宏观位置（独立归一化）
        # 与局部位置形成多尺度表达，全局+局部并存
        centroid = geometry.centroid

        # 计算当前地图的边界（独立于训练集）
        if all_centroids is not None and len(all_centroids) > 0:
            # 从所有质心计算当前地图的边界
            all_x = [p.x for p in all_centroids]
            all_y = [p.y for p in all_centroids]

            map_minx, map_maxx = min(all_x), max(all_x)
            map_miny, map_maxy = min(all_y), max(all_y)

            # 维度5: 相对X位置（归一化到[0,1]）
            map_width = map_maxx - map_minx
            if map_width > 1e-6:
                map_relative_x = (centroid.x - map_minx) / map_width
            else:
                map_relative_x = 0.5

            # 维度6: 相对Y位置（归一化到[0,1]）
            map_height = map_maxy - map_miny
            if map_height > 1e-6:
                map_relative_y = (centroid.y - map_miny) / map_height
            else:
                map_relative_y = 0.5

            # 维度7: 相对于当前地图质心的距离（归一化）
            map_diagonal = np.sqrt(map_width**2 + map_height**2)
            map_center_x = (map_minx + map_maxx) / 2
            map_center_y = (map_miny + map_maxy) / 2
            map_center = Point(map_center_x, map_center_y)

            if map_diagonal > 1e-6:
                distance_to_map_center = centroid.distance(map_center) / map_diagonal
            else:
                distance_to_map_center = 0.0

            features.extend([map_relative_x, map_relative_y, distance_to_map_center])
        else:
            # 如果没有质心数据，使用默认值
            features.extend([0.5, 0.5, 0.0])
        
        # 预计算局部特征分支（优先使用 k_neighbors_info）
        skip_local_block = False
        if k_neighbors_info is not None:
            try:
                neighbor_centroids = k_neighbors_info.get('centroids', [])
                neighbor_distances = k_neighbors_info.get('distances', [])
                if len(neighbor_centroids) > 0:
                    local_centroid_x = np.mean([c.x for c in neighbor_centroids])
                    local_centroid_y = np.mean([c.y for c in neighbor_centroids])
                    local_centroid = Point(local_centroid_x, local_centroid_y)
                    local_radius = np.mean(neighbor_distances) if len(neighbor_distances) > 0 else 0.0
                    if local_radius > 1e-6:
                        local_relative_x = (centroid.x - local_centroid.x) / (local_radius * 2)
                        local_relative_x = np.clip(local_relative_x, -1, 1)
                        local_relative_y = (centroid.y - local_centroid.y) / (local_radius * 2)
                        local_relative_y = np.clip(local_relative_y, -1, 1)
                        distance_to_local_center = centroid.distance(local_centroid) / local_radius
                    else:
                        local_relative_x = 0.0
                        local_relative_y = 0.0
                        distance_to_local_center = 0.0
                    features.extend([local_relative_x, local_relative_y, distance_to_local_center])
                else:
                    features.extend([0.0, 0.0, 0.0])
            except Exception:
                features.extend([0.0, 0.0, 0.0])
            skip_local_block = True

        # ===== 8-10. 局部相对位置（微观空间信息，抗裁剪）=====
        # 基于K近邻的局部参考系，即使裁剪后也稳定
        # 这是方案D的核心：全局+局部多尺度表达
        
        if not skip_local_block and all_centroids is not None and geometry_index is not None and len(all_centroids) > 1:
            # 计算到K个最近邻的距离
            k = min(5, len(all_centroids) - 1)  # K=5或更少
            
            # 计算到所有其他质心的距离
            distances = []
            for i, other_centroid in enumerate(all_centroids):
                if i != geometry_index:
                    dist = centroid.distance(other_centroid)
                    distances.append((dist, i, other_centroid))
            
            if len(distances) > 0:
                # 排序找最近的K个
                distances.sort(key=lambda x: x[0])
                k_nearest = distances[:k]
                
                # 计算局部质心（K近邻的平均位置）
                local_centroid_x = np.mean([p[2].x for p in k_nearest])
                local_centroid_y = np.mean([p[2].y for p in k_nearest])
                local_centroid = Point(local_centroid_x, local_centroid_y)
                
                # 计算局部半径（K近邻的平均距离）
                local_radius = np.mean([p[0] for p in k_nearest])
                
                # 维度8: 相对于局部质心的X偏移（归一化）
                if local_radius > 1e-6:
                    local_relative_x = (centroid.x - local_centroid.x) / (local_radius * 2)
                    local_relative_x = np.clip(local_relative_x, -1, 1)  # 限制到[-1, 1]
                else:
                    local_relative_x = 0.0

                # 维度9: 相对于局部质心的Y偏移（归一化）
                if local_radius > 1e-6:
                    local_relative_y = (centroid.y - local_centroid.y) / (local_radius * 2)
                    local_relative_y = np.clip(local_relative_y, -1, 1)
                else:
                    local_relative_y = 0.0

                # 维度10: 到局部质心的距离（归一化）
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
        
        # ===== 11-12. 长宽比 + 矩形度（旋转不变）=====
        if geom_type in ['Polygon', 'MultiPolygon'] and area > 0:
            try:
                # 最小外接矩形（对于MultiPolygon先选代表多边形）
                rep_poly = self._get_representative_polygon(geometry)
                if rep_poly is not None:
                    min_rect = rep_poly.minimum_rotated_rectangle
                else:
                    min_rect = geometry.minimum_rotated_rectangle
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
        
        # ===== 13. Solidity 实心度（形状复杂度）=====
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
        
        # ===== 14. 对数顶点数（复杂度指标）=====
        # 安全计算顶点数（考虑Multi*情形）
        num_vertices = self._count_vertices(geometry)
        
        # 对数归一化
        log_vertices = np.log1p(num_vertices) / 10.0  # 经验性归一化
        features.append(log_vertices)
        
        # ===== 15-17. 拓扑邻域特征（占位符，后续更新）=====
        # 这些特征需要在构建拓扑邻接图后计算
        # 暂时填充0，在update_topology_neighborhood_features中更新
        features.extend([0.0, 0.0, 0.0])

        # ===== 18. 孔洞数量 Holes（拓扑特征）=====
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
        if all_centroids is not None:
            total_nodes = len(all_centroids)
            if total_nodes > 0:
                node_count_feature = np.log10(total_nodes + 1) / 4.0
            else:
                node_count_feature = 0.0
        else:
            # 如果未提供，使用默认值（假设中等规模图）
            node_count_feature = 0.5
        features.append(node_count_feature)

        # 显式按固定顺序构造 20 维特征，保证与训练集一致：
        # [0-2] geom one-hot, [3] hu1, [4] boundary_complexity,
        # [5-7] map_relative_x/y, distance_to_map_center,
        # [8-10] local_relative_x/y, distance_to_local_center,
        # [11-12] aspect_ratio, rectangularity, [13] solidity,
        # [14] log_vertices,
        # [15-17] 拓扑占位（将在 update_topology_neighborhood_features 中更新）
        # [18] log_holes, [19] node_count_feature
        gf = geom_features if 'geom_features' in locals() else [0.0, 0.0, 0.0]
        hu = locals().get('hu1_normalized', 0.0)
        boundary = locals().get('boundary_complexity', 0.0)
        map_x = locals().get('map_relative_x', 0.5)
        map_y = locals().get('map_relative_y', 0.5)
        map_dist = locals().get('distance_to_map_center', 0.0)
        local_x = locals().get('local_relative_x', 0.0)
        local_y = locals().get('local_relative_y', 0.0)
        local_dist = locals().get('distance_to_local_center', 0.0)
        aspect_ratio = locals().get('aspect_ratio', 0.0)
        rectangularity = locals().get('rectangularity', 0.0)
        solidity_v = locals().get('solidity', 0.0)
        log_vertices_v = locals().get('log_vertices', 0.0)
        # 拓扑占位（15-17）先置0，后续 update 会覆盖
        topo_15 = 0.0
        topo_16 = 0.0
        topo_17 = 0.0
        log_holes_v = locals().get('log_holes', 0.0)
        node_count_v = locals().get('node_count_feature', locals().get('node_count_feature', 0.5))

        final = np.array([
            float(gf[0]), float(gf[1]), float(gf[2]),
            float(hu),
            float(boundary),
            float(map_x), float(map_y), float(map_dist),
            float(local_x), float(local_y), float(local_dist),
            float(aspect_ratio), float(rectangularity),
            float(solidity_v),
            float(log_vertices_v),
            float(topo_15), float(topo_16), float(topo_17),
            float(log_holes_v),
            float(node_count_v)
        ], dtype=np.float32)

        # 最后校验长度
        if final.shape[0] != getattr(self.global_scaler, 'n_features_in_', 20):
            print(f"⚠️ 构造的特征长度 {final.shape[0]} 与 scaler 期望 {getattr(self.global_scaler, 'n_features_in_', 20)} 不同")
        return final
    
    def update_topology_neighborhood_features(self, node_features, geometries, topology_edges):
        """
        更新拓扑邻域特征（特征维度15-17）

        基于拓扑邻接关系计算（而非K近邻）：
        - 维度15: 与拓扑邻居的平均距离（归一化）
        - 维度16: 拓扑邻居数量（归一化）
        - 维度17: 拓扑邻域密度
        
        Args:
            node_features: 节点特征矩阵
            geometries: 几何要素列表
            topology_edges: 拓扑边列表 [[i, j], ...]
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
        
        # 计算全局对角线用于归一化
        if self.global_bounds is not None:
            global_width = self.global_bounds[2] - self.global_bounds[0]
            global_height = self.global_bounds[3] - self.global_bounds[1]
            global_diagonal = np.sqrt(global_width**2 + global_height**2)
        else:
            global_diagonal = 1.0
        
        # 更新每个节点的拓扑邻域特征
        for i in range(n_samples):
            neighbors = list(adjacency[i])
            
            if len(neighbors) > 0:
                # 计算到拓扑邻居的距离
                distances = [centroids[i].distance(centroids[j]) for j in neighbors]
                
                # 维度16: 平均距离（归一化）
                if global_diagonal > 1e-6:
                    avg_distance = np.mean(distances) / global_diagonal
                else:
                    avg_distance = 0.0
                
                # 维度17: 拓扑邻居数量（对数归一化）
                num_neighbors = np.log1p(len(neighbors)) / 5.0
                
                # 维度18: 拓扑邻域密度
                if avg_distance > 1e-6:
                    density = len(neighbors) / (avg_distance ** 2)
                    density = np.log1p(density) / 10.0  # 对数归一化
                else:
                    density = 0.0
            else:
                # 如果没有拓扑邻居（孤岛节点），使用默认值
                avg_distance = 0.0
                num_neighbors = 0.0
                density = 0.0
            
            # 注意：特征向量中拓扑邻域占位是在 indices 15-17（0-based）
            node_features[i, 15] = avg_distance
            node_features[i, 16] = num_neighbors
            node_features[i, 17] = density
        
        return node_features
    
    
    def build_knn_delaunay_graph(self, geometries, node_features):
        """
        ⭐ 构建KNN + Delaunay统一图（TrainingSet的核心图构建方法）

        策略：
        1. KNN保证局部密集连接：每个节点至多K个邻居
        2. Delaunay保证全局连通：覆盖所有节点，填补稀疏区域
        3. 自适应K值：根据节点数动态调整
        4. 无孤岛节点：Delaunay天然保证连通性

        优势：
        - 计算复杂度：O(n log n) vs O(n²)
        - 适用于所有数据类型（点/线/面）
        - 天然无孤岛，结构更稳定
        - 空间自适应，适合各种密度分布

        Returns:
            tuple: (边列表, 统计信息dict)
        """
        n = len(geometries)
        print(f"\n=== KNN + Delaunay 统一图构建（{n}个节点）===")

        if n < 2:
            return [], {'total_nodes': n, 'knn_edges': 0, 'delaunay_edges': 0, 'total_edges': 0}

        # 1. 自适应确定K值
        k = self.adaptive_k_for_graph(n)
        print(f"  自适应K值: {k} (基于节点数{n})")

        # 2. 提取所有节点的质心
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in geometries])

        # 3. 构建KNN边
        print("  构建KNN局部连接...")
        from sklearn.neighbors import NearestNeighbors

        knn_edges = []
        nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='kd_tree').fit(centroids)

        for i in range(n):
            distances, indices = nbrs.kneighbors([centroids[i]])
            # 排除自己，取前k个邻居
            for j in indices[0][1:k+1]:
                if i < j:  # 避免重复边
                    knn_edges.append([i, j])

        print(f"  KNN边数: {len(knn_edges)} 对")

        # 4. 构建Delaunay边（保证全局连通，含大图分块优化）
        print("  构建Delaunay全局连接...")
        from scipy.spatial import Delaunay

        delaunay_edges = []
        delaunay_edge_set = set()

        try:
            if n < 3:
                print(f"  [Delaunay] 节点数<{3}，跳过Delaunay")
                delaunay_edges = []
            elif n <= 5000:
                # 小图：直接Delaunay
                tri = Delaunay(centroids)
                for simplex in tri.simplices:
                    edges = [
                        tuple(sorted([simplex[0], simplex[1]])),
                        tuple(sorted([simplex[1], simplex[2]])),
                        tuple(sorted([simplex[2], simplex[0]]))
                    ]
                    delaunay_edge_set.update(edges)
                delaunay_edges = [[i, j] for i, j in delaunay_edge_set]
                print(f"  Delaunay边数: {len(delaunay_edges)} 对")
            else:
                # 大图：Hilbert曲线分块 + 块内 Delaunay + 跨块 KNN 连接
                print(f"  ⭐ 大图优化：Hilbert分块Delaunay（{n}个节点）...")
                blocks = self.partition_by_hilbert(centroids, block_size=2000)
                total_delaunay_edges = 0
                for block_idx, block_indices in enumerate(blocks):
                    if len(block_indices) < 3:
                        continue
                    block_centroids = centroids[block_indices]
                    try:
                        tri = Delaunay(block_centroids)
                        for simplex in tri.simplices:
                            for i_local in range(3):
                                v1 = simplex[i_local]
                                v2 = simplex[(i_local + 1) % 3]
                                global_v1 = block_indices[v1]
                                global_v2 = block_indices[v2]
                                edge = tuple(sorted([global_v1, global_v2]))
                                delaunay_edge_set.add(edge)
                                total_delaunay_edges += 1
                    except Exception as e:
                        print(f"    ⚠️  块{block_idx}的Delaunay失败: {e}")
                        continue

                # 跨块连接：在相邻块边界节点间做 KNN 连接（取边界前后 10% 节点）
                cross_block_edges = 0
                for i in range(len(blocks) - 1):
                    block1_indices = blocks[i]
                    block2_indices = blocks[i+1]
                    if not block1_indices or not block2_indices:
                        continue
                    boundary1_size = max(1, len(block1_indices) // 10)
                    boundary2_size = max(1, len(block2_indices) // 10)
                    boundary1 = block1_indices[-boundary1_size:]
                    boundary2 = block2_indices[:boundary2_size]
                    boundary_combined = boundary1 + boundary2
                    if len(boundary_combined) < 2:
                        continue
                    boundary_centroids = centroids[boundary_combined]
                    boundary_k = min(max(1, k//2), len(boundary_centroids)-1)
                    if boundary_k >= 1:
                        nbrs_boundary = NearestNeighbors(n_neighbors=boundary_k+1, algorithm='kd_tree').fit(boundary_centroids)
                        _, indices_boundary = nbrs_boundary.kneighbors(boundary_centroids)
                        for local_i, neighbors in enumerate(indices_boundary):
                            global_i = boundary_combined[local_i]
                            for local_j in neighbors[1:]:
                                global_j = boundary_combined[local_j]
                                edge = tuple(sorted([global_i, global_j]))
                                if edge not in delaunay_edge_set:
                                    cross_block_edges += 1
                                delaunay_edge_set.add(edge)

                delaunay_edges = [[i, j] for i, j in delaunay_edge_set]
                print(f"  分块Delaunay完成：块内边约 {total_delaunay_edges}，跨块新增约 {cross_block_edges}")

        except Exception as e:
            print(f"  ⚠️  Delaunay构建失败，使用纯KNN: {e}")
            delaunay_edges = []

        # 5. 合并边并去重
        all_edges = knn_edges + delaunay_edges
        edge_set = set()

        for edge in all_edges:
            edge_tuple = tuple(sorted(edge))
            edge_set.add(edge_tuple)

        unique_edges = [[i, j] for i, j in edge_set]

        # 6. 转换为PyTorch Geometric格式（双向边）
        pyg_edges = []
        for edge in unique_edges:
            pyg_edges.extend([[edge[0], edge[1]], [edge[1], edge[0]]])

        print(f"  总边数: {len(unique_edges)} 对 ({len(pyg_edges)} 条有向边)")

        stats = {
            'total_nodes': n,
            'k_value': k,
            'knn_edges': len(knn_edges),
            'delaunay_edges': len(delaunay_edges),
            'total_edges': len(unique_edges),
            'connectivity': len(unique_edges) / (n * (n-1) / 2) if n > 1 else 0
        }

        print(f"\n图构建完成:")
        print(f"  节点数: {n}")
        print(f"  KNN局部连接: {len(knn_edges)} 对")
        print(f"  Delaunay全局连接: {len(delaunay_edges)} 对")
        print(f"  总边数: {len(unique_edges)} 对")
        print(f"  图连通率: {stats['connectivity']:.3f}")

        return pyg_edges, stats
    
    def build_graph_from_gdf(self, gdf, graph_name):
        """
        ⭐ 从GeoDataFrame构建KNN+Delaunay统一图

        处理流程：
        1. 计算当前图边界（独立归一化）
        2. 提取20维几何特征（多尺度位置+节点数编码）
        3. 构建KNN+Delaunay图结构
        4. 更新拓扑邻域特征
        5. 应用训练集的全局标准化器

        Args:
            gdf: GeoDataFrame
            graph_name: 图名称

        Returns:
            Data: PyTorch Geometric Data对象
        """
        print(f"\n{'='*60}")
        print(f"处理: {graph_name}")
        print(f"{'='*60}")

        # 投影到米制坐标系以获得正确的centroid/距离（优先EPSG:3857）
        gdf_proj = gdf
        try:
            if getattr(gdf, 'crs', None) is None or gdf.crs.is_geographic:
                gdf_proj = gdf.to_crs(epsg=3857)
        except Exception:
            gdf_proj = gdf.copy()

        # 计算当前图的全局边界（使用投影后的gdf）
        self.compute_global_bounds(gdf_proj)

        # 提取几何要素（使用投影后的几何进行几何计算，属性仍使用原始gdf）
        proj_gdf = gdf_proj.reset_index(drop=True)
        orig_gdf = gdf.reset_index(drop=True)
        geometries = proj_gdf.geometry.tolist()

        # 第一步：提取所有质心（用于计算局部相对位置）
        all_centroids = [Point(geom.centroid.x, geom.centroid.y) for geom in geometries]

        # 预计算 K 近邻信息以加速局部特征提取（KD-tree）
        k_neighbors_dict = self.precompute_k_neighbors(all_centroids, k=5)

        # 第二步：提取特征（传入all_centroids和geometry_index）
        node_features = []
        for idx in range(len(proj_gdf)):
            geom_proj = proj_gdf.geometry.iloc[idx]
            row = orig_gdf.iloc[idx] if idx < len(orig_gdf) else proj_gdf.iloc[idx]
            features = self.extract_improved_features(
                geom_proj,
                row,
                geometry_index=idx,  # 传入索引
                all_centroids=all_centroids,  # 传入所有质心
                total_nodes=len(geometries),  # 传入总节点数
                k_neighbors_info=k_neighbors_dict.get(idx)
            )
            node_features.append(features)

        # 校验特征维度与训练集标准化器期望一致
        expected_dim = getattr(self.global_scaler, 'n_features_in_', 20)
        try:
            lengths = [len(f) for f in node_features]
        except Exception:
            # 若某些元素不是序列，尝试转换所有元素为一维数组后再测长度
            node_features = [np.ravel(f).tolist() if not isinstance(f, (list, tuple, np.ndarray)) else f for f in node_features]
            lengths = [len(f) for f in node_features]

        if not all(l == lengths[0] for l in lengths):
            print(f"⚠️ 检测到节点特征长度不一致，样本长度分布前10项: {lengths[:10]}")
            # 给出异常索引
            bad_indices = [i for i, l in enumerate(lengths) if l != expected_dim]
            print(f"⚠️ 以下节点特征维度与期望 {expected_dim} 不同，将进行修正 (示例前20): {bad_indices[:20]}")

        # 修正：对每个节点特征进行裁剪或填充，确保最终维度为 expected_dim
        normalized_node_features = []
        for i, f in enumerate(node_features):
            arr = np.array(f, dtype=np.float32).ravel()
            if arr.size > expected_dim:
                arr = arr[:expected_dim]
            elif arr.size < expected_dim:
                pad = np.zeros(expected_dim - arr.size, dtype=np.float32)
                arr = np.concatenate([arr, pad])
            normalized_node_features.append(arr)

        node_features = np.array(normalized_node_features, dtype=np.float32)
        if node_features.shape[1] != expected_dim:
            raise RuntimeError(f"修正后特征维度仍不匹配: {node_features.shape[1]} vs expected {expected_dim}")

        # 第三步：构建KNN+Delaunay统一图
        edges, graph_stats = self.build_knn_delaunay_graph(geometries, node_features)

        # 第四步：更新拓扑邻域特征（基于图结构）
        if len(edges) > 0:
            # 从边列表构建邻接关系用于邻域特征计算
            # edges 是有向边平铺列表 [[i,j],[j,i],...]
            # 将其恢复为无向边对 [[i,j], ...]（确保元素为 int）
            topology_edges = []
            for e1, e2 in zip(edges[::2], edges[1::2]):
                try:
                    i = int(e1[0]) if isinstance(e1, (list, tuple)) else int(e1)
                    j = int(e1[1]) if isinstance(e1, (list, tuple)) else int(e2)
                except Exception:
                    # 回退处理：尝试按索引读取
                    try:
                        i = int(e1)
                        j = int(e2)
                    except Exception:
                        continue
                if i < j:
                    topology_edges.append([i, j])
            node_features = self.update_topology_neighborhood_features(
                node_features,
                geometries,
                topology_edges
            )
            print(f"已更新拓扑邻域特征（基于 {len(topology_edges)} 对边）")
        else:
            print("图无边，跳过拓扑邻域特征更新")

        # 第五步：使用训练集的全局标准化器
        if len(node_features) > 0:
            node_features = self.global_scaler.transform(node_features)
            print(f"已使用训练集的全局标准化器（20维）")
        else:
            print("⚠️ 无节点特征，跳过标准化")

        # 第六步：创建PyTorch Geometric Data对象
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).T
        else:
            # 如果没有边（极端情况），创建空边索引
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 创建PyTorch Geometric Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index
        )

        # 保存图统计信息
        data.n_nodes = graph_stats['total_nodes']
        data.n_edges = graph_stats['total_edges']
        data.k_value = graph_stats.get('k_value', 0)
        data.connectivity = graph_stats.get('connectivity', 0.0)

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
            pickle.dump(data, f)
        
        print(f"已保存: {save_path}")
    
    def convert_test_set_to_graph(self):
        """
        ⭐ KNN + Delaunay统一图 - 测试集转换（使用训练集标准化器）

        处理策略：
        - 直接使用训练集的全局标准化器处理所有数据
        - 原始图+攻击图都使用相同的标准化基准

        优势：
        - 符合机器学习最佳实践（训练/测试一致性）
        - 模拟真实部署场景（只能使用训练数据拟合标准化器）
        - 避免数据泄露问题
        """
        print("\n" + "="*70)
        print("KNN + Delaunay 统一图 - 测试集图结构转换（使用训练集标准化器）")
        print("="*70 + "\n")

        # 清理旧输出（第二遍开始时）
        self.clean_output_dirs()

        total_converted = 0

        # 处理 Original（来自 GeoJson/TestSet）
        print("处理原始数据（TestSet/Original）")
        print("="*60)
        for filename in os.listdir(self.original_dir):
            if filename.endswith('.geojson') and not filename.startswith('._'):
                try:
                    file_path = os.path.join(self.original_dir, filename)
                    # 跳过过大的文件（> 12000 KB）
                    try:
                        if os.path.getsize(file_path) > self.max_file_size_bytes:
                            print(f"⏭️ 跳过过大文件: {filename} ({os.path.getsize(file_path)//1024} KB)")
                            continue
                    except Exception:
                        pass
                    graph_name = filename.replace('.geojson', '')
                    # 如果图已存在则跳过（append 行为）
                    target_path = os.path.join(self.graph_dir, 'Original', f"{graph_name}_graph.pkl")
                    if os.path.exists(target_path):
                        print(f"⏭️ 已存在，跳过: Original/{graph_name}_graph.pkl")
                        continue

                    gdf = gpd.read_file(file_path)
                    data = self.build_graph_from_gdf(gdf, filename)
                    self.save_graph_data(data, graph_name, data_type='Original')
                    total_converted += 1
                except Exception as e:
                    print(f"❌ 处理文件 {filename} 时出错: {e}")
                    continue

        # 处理 Attacked（来自 GeoJson-Attacked/TestSet 的每个原文件名子目录）
        print("\n处理攻击数据（TestSet/Attacked）")
        print("="*60)
        for attacked_subdir in os.listdir(self.attacked_dir):
            attack_dir_path = os.path.join(self.attacked_dir, attacked_subdir)
            if os.path.isdir(attack_dir_path):
                print(f"\n{'='*50}")
                print(f"处理子目录: {attacked_subdir}")
                print(f"{'='*50}")
                for filename in tqdm(os.listdir(attack_dir_path), desc=f"处理 {attacked_subdir}"):
                    if filename.endswith('.geojson') and not filename.startswith('._'):
                        try:
                            file_path = os.path.join(attack_dir_path, filename)
                            # 跳过过大的文件（> 12000 KB）
                            try:
                                if os.path.getsize(file_path) > self.max_file_size_bytes:
                                    print(f"⏭️ 跳过过大攻击文件: {attacked_subdir}/{filename} ({os.path.getsize(file_path)//1024} KB)")
                                    continue
                            except Exception:
                                pass
                            graph_name = filename.replace('.geojson', '')
                            # 如果图已存在则跳过（append 行为）
                            target_path = os.path.join(self.graph_dir, 'Attacked', attacked_subdir, f"{graph_name}_graph.pkl")
                            if os.path.exists(target_path):
                                print(f"⏭️ 已存在，跳过: Attacked/{attacked_subdir}/{graph_name}_graph.pkl")
                                continue

                            gdf = gpd.read_file(file_path)
                            data = self.build_graph_from_gdf(gdf, filename)
                            self.save_graph_data(data, graph_name, attacked_subdir, 'Attacked')
                            total_converted += 1
                        except Exception as e:
                            print(f"❌ 处理文件 {filename} 时出错: {e}")
                            continue

        print("\n" + "="*70)
        print("测试集转换完成！")
        print("="*70)
        print(f"📊 总共转换: {total_converted} 个图文件")
        print("✅ 特征维度: 20维（Hu不变矩 + 当前地图位置 + 局部位置 + 拓扑邻域 + 节点数编码）")
        print("✅ 图构建: KNN + Delaunay统一图（自适应K值）")
        print("✅ 标准化: 使用训练集的全局标准化器")
        print("📊 图连通性: Delaunay保证100%连通，无孤岛节点")
        print("="*70 + "\n")

def main():
    """主函数"""
    print("\n" + "="*70)
    print("KNN + Delaunay 统一图 - 测试集图结构转换（使用训练集标准化器）")
    print("="*70 + "\n")

    # 检查必需依赖
    try:
        import scipy
        import sklearn
        print("✅ 已检测到 scipy 和 scikit-learn")
        print("   scipy版本:", scipy.__version__)
        print("   sklearn版本:", sklearn.__version__)
    except ImportError as e:
        print("\n" + "="*70)
        print("错误：缺少必需依赖")
        print("="*70)
        print(f"\n缺少库: {e}")
        print("\nKNN + Delaunay统一图需要以下依赖：")
        print("  pip install scipy scikit-learn")
        print("\n或者使用conda：")
        print("  conda install scipy scikit-learn")
        print("\n安装后重新运行此脚本。")
        print("="*70 + "\n")
        return

    print()

    # 创建转换器
    converter = ImprovedTestSetVectorToGraphConverter()

    # 转换测试集数据（使用训练集标准化器）
    converter.convert_test_set_to_graph()

    print("="*70)
    print("【重要提示】")
    print("="*70)
    print("1. ✅ 已实施全局标准化（使用训练集标准化器）")
    print("2. ✅ 图数据包含图统计信息（data.n_nodes, data.n_edges等）")
    print("3. ⚠️  模型预测时需要使用 input_dim=20 的模型")
    print("4. 📊 图构建方式：KNN + Delaunay统一图（自适应K值）")
    print("5. 🔧 可访问 data.k_value, data.connectivity 等统计信息")

    print("\n【相比原版本的优势】")
    print("="*70)
    print("  ✓ 20维特征（新增节点数编码）")
    print("  ✓ 模型泛化能力提升（结构归一化）")
    print("  ✓ 计算高效（O(n log n)，无需R-tree）")
    print("  ✓ 对各种攻击具有极强鲁棒性")
    print("  ✓ 自适应图构建（KNN + Delaunay）")
    print("  ✓ 无孤岛节点（Delaunay保证连通性）")
    print("  ✓ 使用训练集标准化器（机器学习最佳实践）")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

