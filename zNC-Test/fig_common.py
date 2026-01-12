#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig1-Fig12 共享逻辑模块
提取公共函数，避免代码冗余

包含内容：
1. 路径配置
2. 20维特征提取函数
3. 图构建函数（KNN + Delaunay）
4. 模型加载函数
5. 零水印工具函数（load_cat32, features_to_matrix, calc_nc）
6. 结果保存函数
"""

from pathlib import Path
import sys
from typing import List, Tuple, Optional
import pickle

import numpy as np
from shapely.geometry import Point  # type: ignore

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:
    StandardScaler = None
    NearestNeighbors = None

try:
    from scipy.spatial import Delaunay  # type: ignore
except Exception:
    Delaunay = None

try:
    import torch  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except Exception:
    Data = None
    torch = None

# ====================
# 路径配置
# ====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

# 模型和资源路径
MODEL_PATH = PROJECT_ROOT / 'VGAT' / 'models' / 'gat_model_IMPROVED_best.pth'
CAT32_PATH = PROJECT_ROOT / 'ZeroWatermark' / 'Cat32.png'
GLOBAL_SCALER_PATH = PROJECT_ROOT / 'convertToGraph' / 'Graph' / 'TrainingSet' / 'global_scaler.pkl'

# 配置参数
K_FOR_KNN = 8  # KNN邻居数（已弃用，改用自适应K值）

# 全局标准化器（延迟加载）
_global_scaler = None
_global_scaler_loaded = False


def adaptive_k_for_graph(n_nodes):
    """
    ⭐ 根据节点数自适应确定K值（与训练集完全一致）
    
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


def load_global_scaler():
    """
    加载训练集的全局标准化器（强制要求）
    
    Returns:
        StandardScaler: 全局标准化器
        
    Raises:
        FileNotFoundError: 如果未找到全局标准化器文件
        RuntimeError: 如果加载失败或标准化器无效
    """
    global _global_scaler, _global_scaler_loaded
    
    # 如果已经尝试过加载，直接返回结果（或抛出之前的错误）
    if _global_scaler_loaded:
        if _global_scaler is None:
            raise RuntimeError("全局标准化器加载失败，无法继续")
        return _global_scaler
    
    _global_scaler_loaded = True  # 标记已尝试加载
    
    # 检查文件是否存在
    if not GLOBAL_SCALER_PATH.exists():
        error_msg = (
            f"❌ 未找到全局标准化器文件！\n"
            f"   路径: {GLOBAL_SCALER_PATH}\n"
            f"   \n"
            f"   请先运行以下步骤生成全局标准化器：\n"
            f"   1. cd convertToGraph\n"
            f"   2. python convertToGraph-TrainingSet-IMPROVED.py\n"
            f"   \n"
            f"   全局标准化器是保证训练集和测试集特征一致性的关键！"
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(GLOBAL_SCALER_PATH, 'rb') as f:
            scaler_data = pickle.load(f)
        
        if isinstance(scaler_data, dict):
            _global_scaler = scaler_data.get('scaler')
        else:
            _global_scaler = scaler_data
        
        # 验证标准化器是否有效
        if _global_scaler is None:
            raise RuntimeError("全局标准化器文件已损坏（scaler为None）")
        
        print(f"✓ 已加载训练集的全局标准化器: {GLOBAL_SCALER_PATH}")
        return _global_scaler
        
    except Exception as e:
        error_msg = (
            f"❌ 加载全局标准化器失败！\n"
            f"   错误: {e}\n"
            f"   路径: {GLOBAL_SCALER_PATH}\n"
            f"   \n"
            f"   请重新生成全局标准化器：\n"
            f"   1. cd convertToGraph\n"
            f"   2. python convertToGraph-TrainingSet-IMPROVED.py\n"
        )
        print(error_msg)
        raise RuntimeError(error_msg) from e


# ====================
# 特征提取函数（20维）
# ====================

def extract_features_20d(geometry, all_geometries=None, idx=None, bounds_stats=None) -> np.ndarray:
    """
    提取20维几何不变特征（适配IMPROVED模型）
    
    特征列表：
    0-2:   几何类型编码（one-hot）
    3:     Hu不变矩φ1
    4:     边界复杂度
    5-7:   当前地图相对位置（宏观）
    8-10:  局部相对位置（微观，K近邻）
    11-12: 长宽比 + 矩形度
    13:    Solidity
    14:    对数顶点数
    15-17: 拓扑邻域特征（默认值）
    18:    孔洞数量
    19:    节点数编码
    """
    feats: List[float] = []
    
    # 维度0-2: 几何类型编码（one-hot）
    geom_type = getattr(geometry, 'geom_type', 'Unknown')
    if geom_type == 'Point':
        feats.extend([1, 0, 0])
    elif geom_type in ['LineString', 'MultiLineString']:
        feats.extend([0, 1, 0])
    elif geom_type in ['Polygon', 'MultiPolygon']:
        feats.extend([0, 0, 1])
    else:
        feats.extend([0, 0, 0])
    
    # 基本属性
    area = getattr(geometry, 'area', 0.0) or 0.0
    perimeter = getattr(geometry, 'length', 0.0) or 0.0
    
    # 维度3: Hu不变矩φ1（简化版）
    hu1 = 0.0
    if area > 1e-6 and geom_type in ['Polygon', 'MultiPolygon']:
        try:
            coords = np.array(geometry.exterior.coords[:-1]) if geom_type == 'Polygon' else np.array(max(geometry.geoms, key=lambda p: p.area).exterior.coords[:-1])
            if len(coords) >= 3:
                cx, cy = np.mean(coords[:, 0]), np.mean(coords[:, 1])
                mu20 = np.sum((coords[:, 0] - cx)**2) / len(coords)
                mu02 = np.sum((coords[:, 1] - cy)**2) / len(coords)
                nu20, nu02 = mu20 / area, mu02 / area
                hu1 = np.log1p(abs(nu20 + nu02)) / 10.0
        except:
            pass
    feats.append(hu1)
    
    # 维度4: 边界复杂度
    boundary_complexity = np.log1p(perimeter / np.sqrt(area)) / 5.0 if area > 1e-6 else 0.0
    feats.append(boundary_complexity)
    
    # 维度5-7: 当前地图相对位置（宏观空间）
    centroid = geometry.centroid
    if bounds_stats:
        minx, miny, maxx, maxy = bounds_stats['bounds']
        local_width, local_height = maxx - minx, maxy - miny
        local_cx, local_cy = bounds_stats['centroid']
        local_diagonal = np.sqrt(local_width**2 + local_height**2)
        rel_x = (centroid.x - minx) / local_width if local_width > 1e-6 else 0.5
        rel_y = (centroid.y - miny) / local_height if local_height > 1e-6 else 0.5
        dist_to_center = centroid.distance(Point(local_cx, local_cy)) / local_diagonal if local_diagonal > 1e-6 else 0.0
        feats.extend([rel_x, rel_y, dist_to_center])
    else:
        feats.extend([0.5, 0.5, 0.0])
    
    # 维度8-10: 局部相对位置（微观空间，基于K近邻）
    if all_geometries and idx is not None and len(all_geometries) > 1:
        try:
            # ⚡ 性能优化：使用预计算的质心数组
            if bounds_stats and 'precomputed_centroids' in bounds_stats:
                centroids = bounds_stats['precomputed_centroids']
            else:
                # 降级方案：重新计算（不应该执行到这里）
                centroids = np.array([[g.centroid.x, g.centroid.y] for g in all_geometries])
            
            k = min(K_FOR_KNN, len(all_geometries) - 1)
            dists = np.linalg.norm(centroids - centroids[idx], axis=1)
            neighbor_idxs = np.argsort(dists)[1:k+1]
            neighbor_centroids = centroids[neighbor_idxs]
            local_cx, local_cy = np.mean(neighbor_centroids, axis=0)
            local_radius = np.mean(dists[neighbor_idxs])
            local_rel_x = np.clip((centroid.x - local_cx) / (local_radius * 2), -1, 1) if local_radius > 1e-6 else 0.0
            local_rel_y = np.clip((centroid.y - local_cy) / (local_radius * 2), -1, 1) if local_radius > 1e-6 else 0.0
            local_dist = np.sqrt((centroid.x - local_cx)**2 + (centroid.y - local_cy)**2) / local_radius if local_radius > 1e-6 else 0.0
            feats.extend([local_rel_x, local_rel_y, local_dist])
        except:
            feats.extend([0.0, 0.0, 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0])
    
    # 维度11-12: 长宽比 + 矩形度
    if geom_type in ['Polygon', 'MultiPolygon'] and area > 0:
        try:
            min_rect = geometry.minimum_rotated_rectangle
            rect_area = min_rect.area if min_rect.area > 0 else area
            coords = list(min_rect.exterior.coords)
            d1 = Point(coords[0]).distance(Point(coords[1]))
            d2 = Point(coords[1]).distance(Point(coords[2]))
            aspect_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 1e-6 else 1.0
            rectangularity = area / rect_area if rect_area > 0 else 1.0
            feats.extend([aspect_ratio, rectangularity])
        except:
            feats.extend([0.5, 0.8])
    else:
        feats.extend([0.5, 0.8])
    
    # 维度13: Solidity
    solidity = 0.8
    if geom_type in ['Polygon', 'MultiPolygon'] and area > 0:
        try:
            convex_hull_area = geometry.convex_hull.area
            solidity = area / convex_hull_area if convex_hull_area > 0 else 0.8
        except:
            pass
    feats.append(solidity)
    
    # 维度14: 对数顶点数
    n_vertices = 0
    if geom_type == 'Polygon':
        n_vertices = len(geometry.exterior.coords) - 1
    elif geom_type == 'MultiPolygon':
        n_vertices = sum(len(p.exterior.coords) - 1 for p in geometry.geoms)
    elif geom_type == 'LineString':
        n_vertices = len(geometry.coords)
    elif geom_type == 'MultiLineString':
        n_vertices = sum(len(line.coords) for line in geometry.geoms)
    log_vertices = np.log1p(n_vertices) / 10.0
    feats.append(log_vertices)
    
    # 维度15-17: 拓扑邻域特征（简化：使用默认值，在图构建后会更新）
    feats.extend([0.5, 0.5, 0.5])
    
    # 维度18: 孔洞数量
    n_holes = 0
    if geom_type == 'Polygon':
        n_holes = len(geometry.interiors)
    elif geom_type == 'MultiPolygon':
        n_holes = sum(len(p.interiors) for p in geometry.geoms)
    holes_normalized = np.log1p(n_holes) / 5.0
    feats.append(holes_normalized)
    
    # 维度19: 节点数编码
    total_nodes = len(all_geometries) if all_geometries else 100
    node_count_encoding = np.log1p(total_nodes) / 10.0
    feats.append(node_count_encoding)
    
    return np.array(feats, dtype=np.float32)


# ====================
# 图构建函数
# ====================

def _hilbert_curve_sort(centroids):
    """
    使用Hilbert曲线对坐标点排序（保持空间局部性）
    
    Args:
        centroids: nx2的坐标数组
        
    Returns:
        排序后的索引数组
    """
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except ImportError:
        # 降级到简单的x+y排序
        print(f"      ⚠️ hilbertcurve未安装，使用简化排序")
        print(f"      提示：pip install hilbertcurve 可获得更好性能")
        return np.argsort(centroids[:, 0] + centroids[:, 1])
    
    n = len(centroids)
    
    # 标准化坐标到[0, 2^p-1]范围
    x_min, x_max = centroids[:, 0].min(), centroids[:, 0].max()
    y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
    
    # 计算合适的Hilbert曲线阶数（p值）
    # 2^p应该足够大以保证精度，但不能太大导致溢出
    p = min(15, max(8, int(np.log2(np.sqrt(n))) + 3))
    max_coord = (1 << p) - 1  # 2^p - 1
    
    # 标准化坐标
    if x_max > x_min:
        x_norm = ((centroids[:, 0] - x_min) / (x_max - x_min) * max_coord).astype(int)
    else:
        x_norm = np.zeros(n, dtype=int)
    
    if y_max > y_min:
        y_norm = ((centroids[:, 1] - y_min) / (y_max - y_min) * max_coord).astype(int)
    else:
        y_norm = np.zeros(n, dtype=int)
    
    # 计算Hilbert距离
    hc = HilbertCurve(p, 2)  # p阶，2维
    hilbert_distances = np.array([
        hc.distance_from_point([int(x), int(y)]) 
        for x, y in zip(x_norm, y_norm)
    ])
    
    # 按Hilbert距离排序
    return np.argsort(hilbert_distances)


def build_knn_delaunay_edges(geometries, k: int = None):
    """
    构建 KNN + Delaunay 统一图（与训练集逻辑完全一致）
    
    核心策略：
    1. 使用原始几何质心坐标（不是标准化后的特征）
    2. KNN构建：每个节点连接K个最近邻（⭐自适应K值）
    3. Delaunay三角剖分：保证全局连通
    4. 合并去重：取并集
    
    Args:
        geometries: 几何要素列表
        k: KNN邻居数（None表示使用自适应K值，推荐）
        
    Returns:
        edges: 边列表 [[src, dst], ...]
    """
    from sklearn.neighbors import NearestNeighbors
    try:
        from scipy.spatial import Delaunay as DelaunayTri
    except ImportError:
        DelaunayTri = None
    
    n = len(geometries)
    if n < 2:
        return []
    
    # ✅ 使用原始几何质心坐标
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in geometries])
    
    # ⭐ 自适应K值（与训练集完全一致）
    if k is None:
        k = adaptive_k_for_graph(n)
    
    # === KNN 构建（所有规模都执行） ===
    actual_k = min(k, n - 1)
    knn_edges = []
    
    if actual_k >= 1:
        print(f"    [1/2] 构建KNN图（K={actual_k}，共{n}个节点）...")
        nbrs = NearestNeighbors(
            n_neighbors=actual_k + 1,  # +1因为包括自己
            algorithm='kd_tree'
        ).fit(centroids)
        
        distances, indices = nbrs.kneighbors(centroids)
        
        for i in range(n):
            # 排除自己（第一个），取K个最近邻
            for j in indices[i][1:actual_k+1]:
                knn_edges.append([i, int(j)])
        
        print(f"    ✓ KNN完成，边数: {len(knn_edges)}")
    
    # === Delaunay 三角剖分（所有节点统一处理） ===
    delaunay_edges = []
    
    if DelaunayTri is None:
        print(f"    ⚠️ scipy.spatial.Delaunay不可用，跳过Delaunay")
    elif n < 3:
        if n == 2:
            delaunay_edges = [[0, 1]]
            print(f"    [2/2] Delaunay: 2节点直接连接")
    else:
        # 所有规模数据都直接做Delaunay（与训练集完全一致）
        print(f"    [2/2] 构建Delaunay三角剖分（{n}个节点）...")
        try:
            tri = DelaunayTri(centroids)
            
            edge_set = set()
            for simplex in tri.simplices:
                edge_set.add(tuple(sorted([simplex[0], simplex[1]])))
                edge_set.add(tuple(sorted([simplex[1], simplex[2]])))
                edge_set.add(tuple(sorted([simplex[2], simplex[0]])))
            
            for edge in edge_set:
                delaunay_edges.append([edge[0], edge[1]])
            
            print(f"    ✓ Delaunay完成，边数: {len(delaunay_edges)}")
        except Exception as e:
            print(f"    ⚠️ Delaunay失败: {e}")
    
    # === 合并去重（与训练集完全一致：先存储无向边，后续用to_undirected转换） ===
    all_edges = knn_edges + delaunay_edges
    
    edge_set = set()
    
    for edge in all_edges:
        edge_tuple = tuple(sorted(edge))  # 无向边：统一为 (min, max)
        edge_set.add(edge_tuple)
    
    # ✅ 转换为边列表（单向表示，与训练集一致）
    unique_edges = [[e[0], e[1]] for e in edge_set]
    
    print(f"    ✓ 合并去重完成，无向边数: {len(unique_edges)}对")
    
    return unique_edges


def gdf_to_graph(gdf, max_nodes=None) -> Optional[Data]:
    """
    从GeoDataFrame构建图结构（20维特征 + KNN+Delaunay图）
    
    Args:
        gdf: GeoDataFrame对象
        max_nodes: 最大节点数阈值，超过则返回None（None表示不限制）
    
    Returns:
        Data对象或None（如果节点数超过阈值）
    """
    if Data is None or StandardScaler is None:
        print("缺少依赖：torch-geometric 或 scikit-learn")
        return None
    
    # ⭐检查节点数，超过阈值则跳过（仅当max_nodes不为None时）
    if max_nodes is not None:
        num_nodes = len(gdf)
        if num_nodes > max_nodes:
            return None
    
    geometries = gdf.geometry.tolist()
    
    # ⚡ 性能优化：预计算所有质心（避免在extract_features_20d中重复计算）
    print(f"    预计算质心数组...")
    all_centroids = np.array([[g.centroid.x, g.centroid.y] for g in geometries])
    
    # 计算边界统计信息
    all_bounds = [g.bounds for g in geometries]
    bounds_stats = {
        'bounds': (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds)
        ),
        'centroid': (
            np.mean(all_centroids[:, 0]),
            np.mean(all_centroids[:, 1])
        ),
        'precomputed_centroids': all_centroids  # ⚡ 传递预计算的质心
    }
    
    # 提取特征
    print(f"    提取特征...")
    feats = [extract_features_20d(geometries[i], geometries, i, bounds_stats) 
             for i in range(len(geometries))]
    feats = np.array(feats, dtype=np.float32)
    
    # 特征归一化（必须使用全局标准化器）
    if len(feats) > 0:
        # 加载训练集的全局标准化器（如果不存在会抛出异常）
        global_scaler = load_global_scaler()
        # ✅ 使用训练集的全局标准化器（符合机器学习标准实践）
        feats = global_scaler.transform(feats)
    
    # ✅ 构建边：使用KNN+Delaunay（基于原始几何质心，自适应K值）
    edges = build_knn_delaunay_edges(geometries, k=None)  # None表示使用自适应K值
    
    # 转换为edge_index格式
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).T
        # ✅ 转换为无向图（自动添加反向边，与训练集完全一致）
        from torch_geometric.utils import to_undirected
        edge_index = to_undirected(edge_index)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 创建Data对象
    data = Data(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=edge_index
    )
    
    return data


# ====================
# 模型加载函数
# ====================

def load_improved_gat_model(device='cpu', model_path=None):
    """
    加载ImprovedGATModel模型
    
    Returns:
        model: 加载好的模型（已设置为eval模式）
        device: 使用的设备
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    # 允许自动回退：默认best损坏/截断时，尝试备用模型或checkpoint
    model_path = Path(model_path)
    models_dir = PROJECT_ROOT / 'VGAT' / 'models'
    ckpt_dir = PROJECT_ROOT / 'VGAT' / 'checkpoints'
    candidates = [
        model_path,
        models_dir / 'gat_model_IMPROVED_best_V2.pth',
        ckpt_dir / 'gat_checkpoint_latest.pth',
        ckpt_dir / 'gat_checkpoint_emergency_epoch39.pth',
        ckpt_dir / 'gat_checkpoint_emergency_epoch21.pth',
    ]
    
    # 添加VGAT路径到sys.path
    vgat_path = str(PROJECT_ROOT / 'VGAT')
    if vgat_path not in sys.path:
        sys.path.insert(0, vgat_path)
    
    try:
        from VGAT import ImprovedGATModel  # type: ignore
    except Exception as exc:
        print(f'导入ImprovedGATModel失败: {exc}')
        print('尝试直接加载模块...')
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "vgat_improved", 
                str(PROJECT_ROOT / 'VGAT' / 'VGAT-IMPROVED.py')
            )
            vgat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vgat_module)
            ImprovedGATModel = vgat_module.ImprovedGATModel
        except Exception as e2:
            raise ImportError(f"无法加载模型类: {e2}")
    
    # 创建模型（20维输入，256隐藏层，1024输出，8个注意力头）
    model = ImprovedGATModel(
        input_dim=20,
        hidden_dim=256,
        output_dim=1024,
        num_heads=8,
        dropout=0.3
    )
    
    # 加载权重（容错：支持plain state_dict或带'model_state_dict'的dict；尝试多个候选路径）
    last_error = None
    for cand in candidates:
        try:
            if not cand.exists():
                continue
            # 优先过滤明显异常的小文件（<1MB）
            try:
                if cand.stat().st_size < (1 << 20):  # 1MB
                    continue
            except Exception:
                pass
            try:
                ckpt = torch.load(str(cand), map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(str(cand), map_location=device)
            state = None
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif isinstance(ckpt, dict):
                # 可能直接是state_dict
                state = ckpt
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
                model.to(device)
                model.eval()
                print(f'✓ 成功加载模型: {cand.name}')
                print(f'  使用候选路径: {cand}')
                break
        except Exception as e:
            last_error = e
            continue
    else:
        raise RuntimeError(f"无法加载模型。已尝试: {[str(c) for c in candidates if c.exists()]}\n最后错误: {last_error}")
    
    print(f'  设备: {device}')
    print(f'  输入维度: 20')
    print(f'  输出维度: 1024')
    
    return model, device


# ====================
# 零水印工具函数
# ====================

def load_cat32(img_path=None):
    """加载并预处理版权图像为32x32二值图"""
    if img_path is None:
        img_path = CAT32_PATH
    
    try:
        from PIL import Image  # type: ignore
    except Exception:
        raise ImportError("需要安装 pillow: pip install pillow")
    
    try:
        img = Image.open(img_path)
        img = img.convert('L').resize((32, 32))
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        print(f'加载版权图像失败: {exc}')
        # 返回随机图像作为备选
        return (np.random.rand(32, 32) > 0.5).astype(np.uint8)


def features_to_matrix(features: np.ndarray, shape=(32, 32)) -> np.ndarray:
    """将特征向量转换为二值矩阵（基于中位数阈值）"""
    total = shape[0] * shape[1]
    
    # 展平特征
    if features.ndim > 1:
        features_1d = features.flatten()
    else:
        features_1d = features
    
    # 如果特征不足，重复填充
    if len(features_1d) < total:
        rep = (total + len(features_1d) - 1) // len(features_1d)
        features_1d = np.tile(features_1d, rep)
    
    # 截取到目标大小并reshape
    mat = features_1d[:total].reshape(shape)
    
    # 中位数阈值二值化
    thr = np.median(mat)
    return (mat > thr).astype(np.uint8)


def calc_nc(a: np.ndarray, b: np.ndarray) -> float:
    """计算归一化相关系数（NC）"""
    va = a.flatten().astype(float)
    vb = b.flatten().astype(float)
    
    dot = float(np.sum(va * vb))
    na = float(np.sqrt(np.sum(va ** 2)))
    nb = float(np.sqrt(np.sum(vb ** 2)))
    
    if na == 0 or nb == 0:
        return 0.0
    
    return dot / (na * nb)


def extract_features_from_graph(graph_data, model, device, copyright_shape=(32, 32)):
    """
    从图数据中提取1024维特征并转为二值矩阵
    
    Args:
        graph_data: PyTorch Geometric Data对象
        model: VGAT模型
        device: 设备
        copyright_shape: 版权图像形状
    
    Returns:
        feat_matrix: 二值特征矩阵
    """
    with torch.no_grad():
        feat = model(
            graph_data.x.to(device),
            graph_data.edge_index.to(device)
        ).detach().cpu().numpy()
    
    # 确保是1024维
    if feat.ndim > 1:
        feat = feat.flatten()
    
    if len(feat) != 1024:
        if len(feat) < 1024:
            rep = (1024 + len(feat) - 1) // len(feat)
            feat = np.tile(feat, rep)
        feat = feat[:1024]
    
    # 转为二值矩阵
    feat_matrix = features_to_matrix(feat, copyright_shape)
    
    return feat_matrix


# ====================
# GeoJSON转换函数
# ====================

def convert_to_geojson(input_paths: List[Path], output_dir: Path) -> List[Path]:
    """
    批量转换矢量数据为GeoJSON（与 convertToGeoJson.py 对齐）

    行为：
      - 对 .shp 按多编码顺序尝试读取（避免编码问题）
      - 若读取失败且缺少 .shx，可通过外部设置环境变量 SHAPE_RESTORE_SHX=YES
      - 将坐标系统一转换为 EPSG:4326
      - 以 UTF-8 编码输出 GeoJSON
      - Append 模式：不会删除已有文件
    """
    if gpd is None:
        raise ImportError("需要安装 geopandas")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    # 与 convertToGeoJson.py 对齐的编码尝试顺序
    shp_encodings = ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin1', 'iso-8859-1']

    for src in input_paths:
        # 跳过 macOS 元数据文件
        if src.name.startswith('._'):
            continue

        try:
            base = src.stem
            out_path = output_dir / f'{base}.geojson'

            # 如果目标已存在，保持 append 模式，跳过
            if out_path.exists():
                print(f'  ⚠️ 目标已存在，跳过: {out_path.name}')
                outputs.append(out_path)
                continue

            # 针对 SHP 使用多编码尝试
            if src.suffix.lower() == ".shp":
                gdf = None
                last_err: Optional[Exception] = None
                for encoding in shp_encodings:
                    try:
                        gdf = gpd.read_file(src, encoding=encoding)
                        print(f'  ✓ {src.name} 使用编码 {encoding} 读取成功 ({len(gdf)} 要素)')
                        break
                    except UnicodeDecodeError as e:
                        last_err = e
                        continue
                    except Exception as e:
                        last_err = e
                        continue

                if gdf is None:
                    print(f'  ✗ {src.name}: 所有编码 {shp_encodings} 均无法读取 ({last_err})')
                    continue
            else:
                # 其他格式（GeoJSON 等）直接读取
                gdf = gpd.read_file(src)
                print(f'  ✓ {src.name} 读取成功 ({len(gdf)} 要素)')

            # 转换为 WGS84（如果有 CRS 且不是 EPSG:4326）
            if getattr(gdf, 'crs', None) and str(gdf.crs) != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')

            # 读取后标准化（仅针对部分图层以减少不必要修改）
            try:
                standardize_layers = {'BRGA', 'HYDP', 'LRDL'}
                if base in standardize_layers:
                    # 在本作用域内导入所需 shapely 类，避免全局改动
                    from shapely.geometry import MultiPolygon, MultiLineString, MultiPoint  # type: ignore

                    def _to_multipart(geom):
                        if geom is None:
                            return geom
                        try:
                            gt = geom.geom_type
                        except Exception:
                            return geom
                        if gt == 'Polygon':
                            return MultiPolygon([geom])
                        if gt == 'LineString':
                            return MultiLineString([geom])
                        if gt == 'Point':
                            return MultiPoint([geom])
                        return geom

                    if 'geometry' in gdf:
                        try:
                            gdf['geometry'] = gdf['geometry'].apply(lambda x: _to_multipart(x) if x is not None else x)
                        except Exception:
                            new_geoms = []
                            for _, row in gdf.iterrows():
                                geom = row.get('geometry', None) if isinstance(row, dict) else row.geometry if hasattr(row, 'geometry') else None
                                new_geoms.append(_to_multipart(geom) if geom is not None else geom)
                            gdf['geometry'] = new_geoms

                    # 统一可能的字段名变体（避免后续处理因字段截断产生差异）
                    rename_map = {}
                    for c in list(gdf.columns):
                        lc = c.lower()
                        if lc.startswith('shape_leng'):
                            rename_map[c] = 'SHAPE_Length'
                        if lc.startswith('shape_area'):
                            rename_map[c] = 'SHAPE_Area'
                    if rename_map:
                        gdf = gdf.rename(columns=rename_map)
            except Exception:
                # 若标准化失败，继续使用原始读取结果（不可阻塞流程）
                pass

            # 保存为 GeoJSON，UTF-8 编码
            gdf.to_file(out_path, driver='GeoJSON', encoding='utf-8')
            print(f'  ✓ 导出 {out_path.name} ({len(gdf)} 要素)')
            outputs.append(out_path)

        except Exception as exc:
            print(f'  ✗ {src.name}: {exc}')
            continue

    return outputs


# ====================
# 通用图数据转换函数
# ====================

def convert_geojsons_to_graphs(
    original_geojsons: List[Path],
    attacked_geojson_map: dict,
    output_dir_original: Path,
    output_dir_attacked: Path,
    max_nodes=None
):
    """
    批量转换GeoJSON为图结构（可选节点数过滤）
    
    Args:
        original_geojsons: 原始GeoJSON文件列表
        attacked_geojson_map: 攻击后的GeoJSON文件映射 {base_name: {attack_param: path}}
        output_dir_original: 原始图输出目录
        output_dir_attacked: 攻击图输出目录
        max_nodes: 最大节点数阈值（None表示不限制，默认None）
    """
    import shutil
    
    # 确保输出目录存在（不删除已有文件，支持断点续传）
    # ⚠️ 不删除已有的图文件，允许增量转换和断点续传
    output_dir_original.mkdir(parents=True, exist_ok=True)
    output_dir_attacked.mkdir(parents=True, exist_ok=True)
    
    # 统计
    skipped_original = []
    processed_original = []
    skipped_attacked = 0
    processed_attacked = 0
    
    # 转换原始图
    threshold_info = f"不限制" if max_nodes is None else f"{max_nodes}"
    print(f'\n[转换原始图] (节点数阈值: {threshold_info})')
    for src in original_geojsons:
        # ⭐跳过macOS元数据文件（以._开头的文件）
        if src.name.startswith('._'):
            print(f'  ⚠️  跳过macOS元数据文件: {src.name}')
            continue
        
        # ⭐检查原始图是否已存在，若存在则跳过
        out_path = output_dir_original / f"{src.stem}_graph.pkl"
        if out_path.exists():
            print(f'  ⏭️  跳过 {src.name}：原始图已存在')
            processed_original.append(src.stem)
            continue
        
        try:
            gdf = gpd.read_file(src)
            num_nodes = len(gdf)
            
            # ⭐检查节点数（仅当max_nodes不为None时）
            if max_nodes is not None and num_nodes > max_nodes:
                print(f'  ⚠️  跳过 {src.name}: 节点数={num_nodes} > {max_nodes}')
                skipped_original.append(src.stem)
                continue
            
            data = gdf_to_graph(gdf, max_nodes)
            
            if data is not None:
                with open(out_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f'  ✓ {out_path.name} (节点数: {num_nodes})')
                processed_original.append(src.stem)
        except Exception as exc:
            print(f'  ✗ {src.name}: {exc}')
            skipped_original.append(src.stem)
    
    # 转换攻击图（跳过被过滤的原始图对应的攻击图）
    print('\n[转换攻击图]')
    for base, attack_map in attacked_geojson_map.items():
        # ⭐如果原始图被跳过，则跳过所有对应的攻击图
        if base in skipped_original:
            print(f'  ⚠️  跳过 {base} 的所有攻击图（原始图节点数超限）')
            skipped_attacked += len(attack_map)
            continue
        
        subdir = output_dir_attacked / base
        # respect KEEP_EXISTING env: if set, don't rmtree; only generate missing graphs
        KEEP_EXISTING = os.environ.get('KEEP_EXISTING', '0') in ['1', 'true', 'True']
        # ⭐ 如果子目录已存在且已有完整的攻击图，则跳过
        if subdir.exists() and len(list(subdir.glob('*_graph.pkl'))) == len(attack_map):
            print(f'  ⏭️  跳过 {base}：攻击图已完整 ({len(attack_map)} 个)')
            processed_attacked += len(attack_map)
            continue
        # 清理并重建该数据集的子目录（除非 KEEP_EXISTING 为真，此时只生成缺失项）
        if subdir.exists():
            if KEEP_EXISTING:
                print(f'  ⚠️  子目录存在，KEEP_EXISTING=True，保留已有文件，仅生成缺失项: {subdir}')
            else:
                shutil.rmtree(subdir)
                subdir.mkdir(parents=True, exist_ok=True)
        else:
            subdir.mkdir(parents=True, exist_ok=True)
        
        for param, geojson_path in sorted(attack_map.items()):
            try:
                out_path = subdir / f"{geojson_path.stem}_graph.pkl"
                if out_path.exists() and KEEP_EXISTING:
                    print(f'  ⏭️  跳过已存在攻击图: {base}/{out_path.name}')
                    processed_attacked += 1
                    continue
                gdf = gpd.read_file(geojson_path)
                data = gdf_to_graph(gdf, max_nodes)
                
                if data is not None:
                    with open(out_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f'  ✓ {base}/{out_path.name}')
                    processed_attacked += 1
                else:
                    skipped_attacked += 1
            except Exception as exc:
                print(f'  ✗ {base}/{param}: {exc}')
                skipped_attacked += 1
    
    # ⭐输出转换统计
    print('\n' + '='*60)
    threshold_info = "不限制" if max_nodes is None else str(max_nodes)
    print(f'📊 转换统计（节点数阈值: {threshold_info}）')
    print('='*60)
    print(f'原始图: 处理 {len(processed_original)} 个, 跳过 {len(skipped_original)} 个')
    if skipped_original:
        print(f'  跳过的文件: {", ".join(skipped_original)}')
    print(f'攻击图: 处理 {processed_attacked} 个, 跳过 {skipped_attacked} 个')
    print('='*60)


# ====================
# 导出接口
# ====================

__all__ = [
    # 路径配置
    'PROJECT_ROOT',
    'SCRIPT_DIR',
    'MODEL_PATH',
    'CAT32_PATH',
    'GLOBAL_SCALER_PATH',
    'K_FOR_KNN',
    
    # 自适应K值函数
    'adaptive_k_for_graph',
    
    # 标准化器
    'load_global_scaler',
    
    # 特征提取
    'extract_features_20d',
    
    # 图构建
    'build_knn_delaunay_edges',
    'gdf_to_graph',
    'convert_geojsons_to_graphs',
    
    # 模型加载
    'load_improved_gat_model',
    
    # 零水印工具
    'load_cat32',
    'features_to_matrix',
    'calc_nc',
    'extract_features_from_graph',
    
    # GeoJSON转换
    'convert_to_geojson',
]

