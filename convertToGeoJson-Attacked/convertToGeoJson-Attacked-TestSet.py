#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一步：生成测试集被攻击的矢量数据
按照test100.py的逻辑：前50个指定攻击方式，后50个随机组合攻击
为每个图生成100个被攻击的矢量地图类型放入vector_data_test的各个图的子文件夹中

⭐ 代码版本：v2.0_global_center
   - 几何变换（缩放、旋转、翻转）使用全局中心，与TrainingSet保持一致
   - XY翻转使用rotate(180°)实现，更稳定
"""

import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import rotate, scale, translate
from shapely.ops import split as shp_split
from shapely.geometry import LineString as ShpLineString
import random
from tqdm import tqdm
import shutil
import math

class TestVectorAttackGenerator:
    """测试集矢量数据攻击生成器"""
    
    def __init__(self, input_dir="../convertToGeoJson/GeoJson/TestSet", output_dir="GeoJson-Attacked/TestSet", noise_only: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.noise_only = noise_only

        # 规范相对路径为基于脚本文件的绝对路径，避免工作目录导致的路径解析错误
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(self.input_dir):
            self.input_dir = os.path.abspath(os.path.join(base_dir, self.input_dir))
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.abspath(os.path.join(base_dir, self.output_dir))

        self.ensure_output_dir()
        
        # 定义单体攻击方式，在zNC-Test/Fig1.py-Fig10.py的区间里进行随机偏移
        self.single_attacks = []
        
        # Fig1: 删除顶点 0%-90% 随机 (生成10个随机样本)
        for _ in range(10):
            pct = random.randint(0, 90)
            self.single_attacks.append((f"test_del_vertices_{pct}pct.geojson", f"随机删除{pct}%顶点"))
        
        # Fig2: 添加顶点 强度随机0-2 × 比例0%-90% 随机 (生成10个随机样本)
        for _ in range(10):
            strength = random.randint(0, 2)
            pct = random.randint(0, 90)
            self.single_attacks.append((f"test_add_strength{strength}_{pct}pct_vertices.geojson", f"添加{pct}%顶点_强度{strength}"))
        
        # Fig3: 删除对象 0%-90% 随机 (生成5个随机样本)
        for _ in range(5):
            pct = random.randint(0, 90)
            self.single_attacks.append((f"test_del_objects_{pct}pct.geojson", f"删除{pct}%图形对象"))
        
        # Fig4: 噪声扰动 强度0.4-0.8随机 × 比例10-90随机 (生成5个随机样本)
        for _ in range(5):
            strength = round(random.uniform(0.4, 0.8), 2)
            pct = random.randint(10, 90)
            self.single_attacks.append((f"test_noise_{pct}pct_strength_{strength}.geojson", f"噪声扰动{pct}%顶点_强度{strength}"))
        
        # Fig5-Fig10: 保持单一攻击方式，不需要随机化 (共17个)
        # Fig5: 裁剪 (5种)
        self.single_attacks.extend([
            ("test_crop_x_center_50pct.geojson", "沿X轴中心裁剪50%"),
            ("test_crop_y_center_50pct.geojson", "沿Y轴中心裁剪50%"),
            ("test_crop_top_left.geojson", "裁剪左上角区域"),
            ("test_crop_bottom_right.geojson", "裁剪右下角区域"),
            ("test_crop_random_40pct.geojson", "随机裁剪40%"),
        ])
        
        # Fig6: 平移 (随机生成5个)
        for _ in range(5):
            dx = random.randint(-30, 30)
            dy = random.randint(-30, 30)
            self.single_attacks.append((f"test_translate_{dx}_{dy}.geojson", f"平移({dx}_{dy})"))
        
        # Fig7: 缩放 (在0.1-2.1范围随机生成3个)
        for _ in range(3):
            factor = round(random.uniform(0.1, 2.1), 2)
            pct = int(round(factor * 100))
            self.single_attacks.append((f"test_scale_{pct}pct.geojson", f"缩放{pct}%"))
        
        # Fig8: 旋转 (在45-360范围随机生成3个)
        for _ in range(3):
            deg = random.choice([45, 90, 135, 180, 225, 270, 315, 360])
            self.single_attacks.append((f"test_rotate_{deg}deg.geojson", f"旋转{deg}度"))
        
        # Fig9: 翻转 (3种)
        self.single_attacks.extend([
            ("test_flip_x.geojson", "X轴镜像翻转"),
            ("test_flip_y.geojson", "Y轴镜像翻转"),
            ("test_flip_xy.geojson", "同时X_Y轴镜像翻转"),
        ])
        
        # Fig10: 打乱顺序 (4种)
        self.single_attacks.extend([
            ("test_reverse_vertices.geojson", "反转顶点顺序"),
            ("test_shuffle_vertices.geojson", "打乱顶点顺序"),
            ("test_reverse_objects.geojson", "反转对象顺序"),
            ("test_shuffle_objects.geojson", "打乱对象顺序"),
        ])
        
        # 创建50个组合攻击（从单体攻击中选择组合）
        self.combo_count = 50
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def count_total_vertices(self, gdf):
        """计算GeoDataFrame中所有几何对象的总顶点数（支持Multi类型）"""
        if gdf is None or len(gdf) == 0:
            return 0
        
        total_vertices = 0
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            
            # 递归处理单个几何对象
            total_vertices += self._count_geom_vertices(geom)
        
        return total_vertices
    
    def _count_geom_vertices(self, geom):
        """计算单个几何对象的顶点数（递归处理Multi类型）"""
        from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
        
        if geom is None or geom.is_empty:
            return 0
        
        if isinstance(geom, Point):
            return 1
        elif isinstance(geom, LineString):
            return len(geom.coords)
        elif isinstance(geom, Polygon):
            count = len(geom.exterior.coords)
            # 包括孔洞的顶点
            for interior in geom.interiors:
                count += len(interior.coords)
            return count
        elif isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
            # 递归处理Multi类型和GeometryCollection
            count = 0
            for sub_geom in geom.geoms:
                count += self._count_geom_vertices(sub_geom)
            return count
        else:
            # 未知类型，返回0
            return 0
    
    def is_valid_gdf(self, gdf):
        """验证GeoDataFrame是否有效（只检查是否为空，不限制对象数）
        
        Args:
            gdf: GeoDataFrame
            
        Returns:
            bool: 是否为有效的GeoDataFrame
        """
        if gdf is None or len(gdf) == 0:
            return False
        return True
    
    def load_vector_data(self, filename):
        """加载矢量数据"""
        filepath = os.path.join(self.input_dir, filename)
        try:
            gdf = gpd.read_file(filepath)
            print(f"成功加载矢量数据: {filename}")
            print(f"数据包含 {len(gdf)} 个要素")
            return gdf
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def apply_delete_vertices_attack(self, gdf, percentage):
        """删除指定百分比的顶点"""
        def delete_vertices_from_geom(geom, pct):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) <= 2:
                    return geom
                n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
                if n_to_delete >= len(coords) - 2:
                    return geom
                indices = list(range(1, len(coords) - 1))
                to_delete = set(random.sample(indices, n_to_delete))
                new_coords = [coords[0]] + [coords[i] for i in range(1, len(coords) - 1) if i not in to_delete] + [coords[-1]]
                return LineString(new_coords)
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                if len(ext_coords) <= 4:
                    return geom
                n_to_delete = max(1, int((len(ext_coords) - 4) * pct / 100))
                if n_to_delete >= len(ext_coords) - 4:
                    return geom
                indices = list(range(1, len(ext_coords) - 2))
                to_delete = set(random.sample(indices, n_to_delete))
                new_ext_coords = [ext_coords[0]] + [ext_coords[i] for i in range(1, len(ext_coords) - 2) if i not in to_delete] + [ext_coords[-2], ext_coords[-1]]
                holes = []
                for ring in geom.interiors:
                    hole_coords = list(ring.coords)
                    if len(hole_coords) > 4:
                        n_to_delete_hole = max(1, int((len(hole_coords) - 4) * pct / 100))
                        if n_to_delete_hole < len(hole_coords) - 4:
                            indices_hole = list(range(1, len(hole_coords) - 2))
                            to_delete_hole = set(random.sample(indices_hole, n_to_delete_hole))
                            new_hole_coords = [hole_coords[0]] + [hole_coords[i] for i in range(1, len(hole_coords) - 2) if i not in to_delete_hole] + [hole_coords[-2], hole_coords[-1]]
                            holes.append(new_hole_coords)
                        else:
                            holes.append(hole_coords)
                    else:
                        holes.append(hole_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: delete_vertices_from_geom(geom, percentage)
        )
        return gdf_attacked
    
    def apply_delete_objects_attack(self, gdf, percentage):
        """删除指定百分比的对象"""
        gdf_attacked = gdf.copy()
        num_objects = len(gdf_attacked)
        num_to_delete = int(num_objects * percentage / 100)
        if num_to_delete > 0:
            indices_to_delete = random.sample(range(num_objects), num_to_delete)
            gdf_attacked = gdf_attacked.drop(indices_to_delete).reset_index(drop=True)
        return gdf_attacked
    
    def apply_add_vertices_attack(self, gdf, percentage, strength=0):
        """添加指定百分比的顶点，strength控制噪声强度（0=无噪声，1=小噪声，2=大噪声）"""
        def add_vertices_to_geom(geom, pct, strength):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) < 2:
                    return geom
                # 限制添加的顶点数量，避免过度复杂化
                n_to_add = min(3, max(1, int((len(coords) - 1) * pct / 100)))
                new_coords = [coords[0]]
                for i in range(len(coords) - 1):
                    p1, p2 = coords[i], coords[i + 1]
                    new_coords.append(p1)
                    for j in range(n_to_add):
                        t = (j + 1) / (n_to_add + 1)
                        mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                        # 根据强度添加噪声
                        if strength == 1:
                            noise = np.random.normal(0, 0.01, 2)
                            mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                        elif strength == 2:
                            noise = np.random.normal(0, 0.05, 2)
                            mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                        new_coords.append(mid_point)
                new_coords.append(coords[-1])
                return LineString(new_coords)
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                if len(ext_coords) < 4:
                    return geom
                # 限制添加的顶点数量，避免过度复杂化
                n_to_add = min(3, max(1, int((len(ext_coords) - 1) * pct / 100)))
                new_ext_coords = [ext_coords[0]]
                for i in range(len(ext_coords) - 1):
                    p1, p2 = ext_coords[i], ext_coords[i + 1]
                    new_ext_coords.append(p1)
                    for j in range(n_to_add):
                        t = (j + 1) / (n_to_add + 1)
                        mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                        # 根据强度添加噪声
                        if strength == 1:
                            noise = np.random.normal(0, 0.01, 2)
                            mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                        elif strength == 2:
                            noise = np.random.normal(0, 0.05, 2)
                            mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                        new_ext_coords.append(mid_point)
                new_ext_coords.append(ext_coords[-1])
                holes = []
                for ring in geom.interiors:
                    ring_coords = list(ring.coords)
                    if len(ring_coords) >= 4:
                        new_ring_coords = [ring_coords[0]]
                        for i in range(len(ring_coords) - 1):
                            p1, p2 = ring_coords[i], ring_coords[i + 1]
                            new_ring_coords.append(p1)
                            for j in range(n_to_add):
                                t = (j + 1) / (n_to_add + 1)
                                mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                                # 根据强度添加噪声
                                if strength == 1:
                                    noise = np.random.normal(0, 0.01, 2)
                                    mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                                elif strength == 2:
                                    noise = np.random.normal(0, 0.05, 2)
                                    mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                                new_ring_coords.append(mid_point)
                        new_ring_coords.append(ring_coords[-1])
                        holes.append(new_ring_coords)
                    else:
                        holes.append(ring_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: add_vertices_to_geom(geom, percentage, strength)
        )
        return gdf_attacked
    
    def apply_noise_attack(self, gdf, percentage, strength):
        """噪声扰动攻击 - 顶点级扰动"""
        def jitter_vertices(geom, pct, strength):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                n = len(coords)
                k = max(1, int(n * pct / 100))
                indices = list(range(n))
                chosen = set(random.sample(indices, min(k, len(indices))))
                new_coords = []
                for i, coord in enumerate(coords):
                    if i in chosen:
                        new_coords.append((
                            coord[0] + random.uniform(-strength, strength),
                            coord[1] + random.uniform(-strength, strength)
                        ))
                    else:
                        new_coords.append(coord)
                return LineString(new_coords)
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                n = len(ext_coords)
                k = max(1, int(n * pct / 100))
                indices = list(range(n))
                chosen = set(random.sample(indices, min(k, len(indices))))
                new_ext_coords = []
                for i, coord in enumerate(ext_coords):
                    if i in chosen:
                        new_ext_coords.append((
                            coord[0] + random.uniform(-strength, strength),
                            coord[1] + random.uniform(-strength, strength)
                        ))
                    else:
                        new_ext_coords.append(coord)
                holes = []
                for ring in geom.interiors:
                    ring_coords = list(ring.coords)
                    n_ring = len(ring_coords)
                    k_ring = max(1, int(n_ring * pct / 100))
                    indices_ring = list(range(n_ring))
                    chosen_ring = set(random.sample(indices_ring, min(k_ring, len(indices_ring))))
                    new_ring_coords = []
                    for i, coord in enumerate(ring_coords):
                        if i in chosen_ring:
                            new_ring_coords.append((
                                coord[0] + random.uniform(-strength, strength),
                                coord[1] + random.uniform(-strength, strength)
                            ))
                        else:
                            new_ring_coords.append(coord)
                    holes.append(new_ring_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: jitter_vertices(geom, percentage, strength)
        )
        return gdf_attacked
    
    def apply_crop_attack(self, gdf, crop_type):
        """裁剪攻击（✅ 修复：所有分支都重置索引，避免组合攻击中索引问题）"""
        gdf_attacked = gdf.copy()
        bounds = gdf_attacked.total_bounds
        bdf = gdf_attacked.geometry.bounds  # DataFrame: minx, miny, maxx, maxy
        
        if crop_type == "x_40pct":
            # 沿X轴裁剪40%
            mid_x = bounds[0] + (bounds[2] - bounds[0]) * 0.4
            gdf_attacked = gdf_attacked[bdf['minx'] < mid_x].reset_index(drop=True)
        elif crop_type == "y_35pct":
            # 沿Y轴裁剪35%
            mid_y = bounds[1] + (bounds[3] - bounds[1]) * 0.35
            gdf_attacked = gdf_attacked[bdf['miny'] < mid_y].reset_index(drop=True)
        elif crop_type == "top_25pct":
            # 裁剪上部25%区域
            top_y = bounds[3] - (bounds[3] - bounds[1]) * 0.25
            gdf_attacked = gdf_attacked[bdf['miny'] > top_y].reset_index(drop=True)
        elif crop_type == "bottom_20pct":
            # 裁剪下部20%区域
            bottom_y = bounds[1] + (bounds[3] - bounds[1]) * 0.2
            gdf_attacked = gdf_attacked[bdf['miny'] < bottom_y].reset_index(drop=True)
        elif crop_type == "random_30pct":
            # 随机裁剪30%
            num_objects = len(gdf_attacked)
            num_to_keep = int(num_objects * 0.7)
            if num_to_keep > 0:
                indices_to_keep = random.sample(range(num_objects), num_to_keep)
                gdf_attacked = gdf_attacked.iloc[indices_to_keep].reset_index(drop=True)
        
        return gdf_attacked
    
    def apply_translate_attack(self, gdf, dx, dy):
        """平移攻击"""
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: translate(geom, dx, dy)
        )
        return gdf_attacked
    
    def apply_scale_attack(self, gdf, scale_x, scale_y=None):
        """缩放攻击（使用全局中心，与TrainingSet一致）"""
        gdf_attacked = gdf.copy()
        if scale_y is None:
            scale_y = scale_x
        # ✅ 计算全局中心作为缩放原点
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center_x = (bounds[0] + bounds[2]) / 2
        global_center_y = (bounds[1] + bounds[3]) / 2
        global_center = (global_center_x, global_center_y)

        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: scale(geom, scale_x, scale_y, origin=global_center)
        )
        return gdf_attacked
    
    def apply_rotate_attack(self, gdf, angle):
        """旋转攻击（使用全局中心，与TrainingSet一致）"""
        gdf_attacked = gdf.copy()
        # ✅ 计算全局中心作为旋转原点
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center_x = (bounds[0] + bounds[2]) / 2
        global_center_y = (bounds[1] + bounds[3]) / 2
        global_center = (global_center_x, global_center_y)

        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: rotate(geom, angle, origin=global_center)
        )
        return gdf_attacked
    
    def apply_flip_attack(self, gdf, flip_type):
        """翻转攻击（使用全局中心，与TrainingSet一致）"""
        gdf_attacked = gdf.copy()
        # ✅ 计算全局中心作为变换原点
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center_x = (bounds[0] + bounds[2]) / 2
        global_center_y = (bounds[1] + bounds[3]) / 2
        global_center = (global_center_x, global_center_y)

        if flip_type == "x":
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
                lambda geom: scale(geom, -1, 1, origin=global_center)
            )
        elif flip_type == "y":
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
                lambda geom: scale(geom, 1, -1, origin=global_center)
            )
        elif flip_type == "xy":
            # ✅ 双轴翻转使用rotate(180°)实现，与TrainingSet一致
            # 数学等价且更稳定，避免scale(-1,-1)可能导致的顶点顺序问题
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
                lambda geom: rotate(geom, 180, origin=global_center)
            )
        return gdf_attacked
    
    def apply_shuffle_attack(self, gdf, shuffle_type):
        """打乱攻击"""
        gdf_attacked = gdf.copy()
        if shuffle_type == "objects":
            gdf_attacked = gdf_attacked.sample(frac=1).reset_index(drop=True)
        elif shuffle_type == "vertices":
            def shuffle_vertices(geom):
                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                    if len(coords) <= 2:
                        return geom
                    core = coords[1:-1]
                    random.shuffle(core)
                    return LineString([coords[0]] + core + [coords[-1]])
                elif isinstance(geom, Polygon):
                    ext_coords = list(geom.exterior.coords)
                    if len(ext_coords) <= 4:
                        return geom
                    core = ext_coords[1:-2]
                    random.shuffle(core)
                    new_ext_coords = [ext_coords[0]] + core + [ext_coords[-2], ext_coords[-1]]
                    holes = []
                    for ring in geom.interiors:
                        ring_coords = list(ring.coords)
                        if len(ring_coords) > 4:
                            core_ring = ring_coords[1:-2]
                            random.shuffle(core_ring)
                            new_ring_coords = [ring_coords[0]] + core_ring + [ring_coords[-2], ring_coords[-1]]
                            holes.append(new_ring_coords)
                        else:
                            holes.append(ring_coords)
                    return Polygon(new_ext_coords, holes=holes if holes else None)
                return geom
            
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(shuffle_vertices)
        return gdf_attacked
    
    def apply_merge_objects_attack(self, gdf):
        """合并对象攻击"""
        if len(gdf) < 2:
            return gdf.copy()
        
        gdf_attacked = gdf.copy()
        indices = list(range(len(gdf_attacked)))
        random.shuffle(indices)
        
        merged_geoms = []
        used = set()
        
        for i in range(0, len(indices) - 1, 2):
            idx1, idx2 = indices[i], indices[i + 1]
            try:
                geom1 = gdf_attacked.geometry.iloc[idx1]
                geom2 = gdf_attacked.geometry.iloc[idx2]
                merged = geom1.union(geom2)
                merged_geoms.append(merged)
                used.add(idx1)
                used.add(idx2)
            except Exception:
                pass
        
        # 保留未合并的对象
        remaining_geoms = [gdf_attacked.geometry.iloc[i] for i in range(len(gdf_attacked)) if i not in used]
        
        # 创建新的GeoDataFrame
        new_gdf = gdf_attacked.iloc[:0].copy()
        new_gdf['geometry'] = None
        new_gdf = new_gdf.reindex(range(len(remaining_geoms) + len(merged_geoms)))
        new_gdf['geometry'] = remaining_geoms + merged_geoms
        new_gdf = new_gdf.reset_index(drop=True)
        
        return new_gdf
    
    def apply_split_objects_attack(self, gdf):
        """拆分对象攻击"""
        def split_polygon(geom):
            if isinstance(geom, Polygon):
                bounds = geom.bounds
                cx = (bounds[0] + bounds[2]) / 2
                cy = (bounds[1] + bounds[3]) / 2
                length = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2
                
                # 随机选择切割角度
                angle = random.uniform(0, math.pi)
                dx = math.cos(angle) * length
                dy = math.sin(angle) * length
                
                cutter = ShpLineString([(cx - dx, cy - dy), (cx + dx, cy + dy)])
                
                try:
                    parts = shp_split(geom, cutter)
                    return list(parts.geoms)
                except Exception:
                    return [geom]
            return [geom]
        
        gdf_attacked = gdf.copy()
        new_geoms = []
        
        for geom in gdf_attacked.geometry:
            if random.random() < 0.5:  # 50%概率进行拆分
                split_parts = split_polygon(geom)
                new_geoms.extend(split_parts)
            else:
                new_geoms.append(geom)
        
        # 创建新的GeoDataFrame
        new_gdf = gdf_attacked.iloc[:0].copy()
        new_gdf['geometry'] = None
        new_gdf = new_gdf.reindex(range(len(new_geoms)))
        new_gdf['geometry'] = new_geoms
        new_gdf = new_gdf.reset_index(drop=True)
        
        return new_gdf
    
    def apply_single_attack(self, gdf, attack_name):
        """应用单体攻击"""
        # 提取删除顶点百分比
        import re
        match = re.search(r'test_del_vertices_(\d+)pct', attack_name)
        if match:
            pct = int(match.group(1))
            return self.apply_delete_vertices_attack(gdf, pct)
        
        # 提取添加顶点参数
        match = re.search(r'test_add_strength(\d+)_(\d+)pct_vertices', attack_name)
        if match:
            strength = int(match.group(1))
            pct = int(match.group(2))
            return self.apply_add_vertices_attack(gdf, pct, strength)
        
        # 提取删除对象百分比
        match = re.search(r'test_del_objects_(\d+)pct', attack_name)
        if match:
            pct = int(match.group(1))
            return self.apply_delete_objects_attack(gdf, pct)
        
        # 提取噪声攻击参数（避免匹配到文件扩展名前的点）
        match = re.search(r'test_noise_(\d+)pct_strength_([0-9]+(?:\.[0-9]+)?)(?:\.geojson)?', attack_name)
        if match:
            pct = int(match.group(1))
            strength = float(match.group(2))
            return self.apply_noise_attack(gdf, pct, strength)
        
        # 裁剪攻击
        if "test_crop_x_center_50pct" in attack_name:
            return self.apply_crop_attack(gdf, "x_center_50pct")
        elif "test_crop_y_center_50pct" in attack_name:
            return self.apply_crop_attack(gdf, "y_center_50pct")
        elif "test_crop_top_left" in attack_name:
            return self.apply_crop_attack(gdf, "top_left")
        elif "test_crop_bottom_right" in attack_name:
            return self.apply_crop_attack(gdf, "bottom_right")
        elif "test_crop_random_40pct" in attack_name:
            return self.apply_crop_attack(gdf, "random_40pct")
        
        # 提取平移参数
        match = re.search(r'test_translate_(-?\d+)_(-?\d+)', attack_name)
        if match:
            dx = int(match.group(1))
            dy = int(match.group(2))
            return self.apply_translate_attack(gdf, dx, dy)
        
        # 提取缩放参数
        match = re.search(r'test_scale_(\d+)pct', attack_name)
        if match:
            pct = int(match.group(1))
            factor = pct / 100.0
            return self.apply_scale_attack(gdf, factor)
        
        # 提取旋转角度
        match = re.search(r'test_rotate_(\d+)deg', attack_name)
        if match:
            deg = int(match.group(1))
            return self.apply_rotate_attack(gdf, deg)
        
        # 翻转攻击
        if "test_flip_xy" in attack_name:
            return self.apply_flip_attack(gdf, "xy")
        elif "test_flip_x" in attack_name:
            return self.apply_flip_attack(gdf, "x")
        elif "test_flip_y" in attack_name:
            return self.apply_flip_attack(gdf, "y")
        
        # 打乱顺序攻击
        elif "test_reverse_vertices" in attack_name:
            def reverse_vertices(geom):
                if isinstance(geom, LineString):
                    return LineString(list(geom.coords)[::-1])
                elif isinstance(geom, Polygon):
                    ext_coords = list(geom.exterior.coords)
                    ext_coords = ext_coords[:-1][::-1] + [ext_coords[0]]
                    holes = []
                    for ring in geom.interiors:
                        ring_coords = list(ring.coords)
                        ring_coords = ring_coords[:-1][::-1] + [ring_coords[0]]
                        holes.append(ring_coords)
                    return Polygon(ext_coords, holes=holes if holes else None)
                return geom
            
            gdf_attacked = gdf.copy()
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(reverse_vertices)
            return gdf_attacked
        elif "test_shuffle_vertices" in attack_name:
            return self.apply_shuffle_attack(gdf, "vertices")
        elif "test_reverse_objects" in attack_name:
            return gdf.iloc[::-1].reset_index(drop=True)
        elif "test_shuffle_objects" in attack_name:
            return self.apply_shuffle_attack(gdf, "objects")
        
        else:
            # 未知的攻击类型
            print(f"警告：未知的攻击类型 {attack_name}，返回原始数据")
            return gdf.copy()
    
    def apply_combo_attack(self, gdf):
        """应用组合攻击，从单体攻击中随机选择2-3种组合（带有中间验证）"""
        gdf_attacked = gdf.copy()
        
        # 随机选择2-3个单体放击进行组合
        num_attacks = random.randint(2, 3)
        selected_attacks = random.sample(self.single_attacks, num_attacks)
        
        # 不再限制最小对象数，所有攻击数据都保存
        
        attack_descriptions = []
        for attack_name, attack_desc in selected_attacks:
            try:
                # 备份当前状态
                gdf_backup = gdf_attacked.copy()
                
                # 应用攻击
                gdf_attacked = self.apply_single_attack(gdf_attacked, attack_name)
                
                # ✅ 验证攻击后的数据是否有效
                if not self.is_valid_gdf(gdf_attacked):
                    # 数据无效（为空），恢复到备份状态，跳过此攻击
                    gdf_attacked = gdf_backup
                else:
                    # 数据有效，记录攻击
                    attack_descriptions.append(attack_desc)
                    
            except Exception as e:
                # 攻击失败，恢复备份防止脏数据
                gdf_attacked = gdf_backup
                print(f"警告：攻击{attack_name}失败: {e}，已恢复备份，跳过此攻击")
        
        # 返回攻击后的数据和政击描述
        return gdf_attacked, attack_descriptions
    
    def apply_random_attack(self, gdf):
        """应用随机攻击"""
        attack_types = ['translate', 'rotate', 'scale', 'noise', 'crop', 'flip']
        attack_type = random.choice(attack_types)
        
        if attack_type == 'translate':
            dx = random.uniform(-20, 20)
            dy = random.uniform(-20, 20)
            return self.apply_translate_attack(gdf, dx, dy)
        elif attack_type == 'rotate':
            angle = random.uniform(-45, 45)
            return self.apply_rotate_attack(gdf, angle)
        elif attack_type == 'scale':
            scale_factor = random.uniform(0.8, 1.2)
            return self.apply_scale_attack(gdf, scale_factor)
        elif attack_type == 'noise':
            strength = random.uniform(0.05, 0.3)
            return self.apply_noise_attack(gdf, 100, strength)
        elif attack_type == 'crop':
            crop_type = random.choice(['x_40pct', 'y_35pct', 'random_30pct'])
            return self.apply_crop_attack(gdf, crop_type)
        elif attack_type == 'flip':
            flip_type = random.choice(['x', 'y', 'xy'])
            return self.apply_flip_attack(gdf, flip_type)
    
    def clean_output_subdir(self, output_subdir):
        """清理输出子目录的旧文件"""
        subdir_path = os.path.join(self.output_dir, output_subdir)
        # 改为 append 模式：不删除已存在的被攻击文件，仅确保目录存在
        if not os.path.exists(subdir_path):
            print(f"📁 创建新目录: {subdir_path}")
            os.makedirs(subdir_path, exist_ok=True)
        else:
            print(f"🔍 追加模式：保留已存在文件 {subdir_path}")

    def save_attacked_data(self, gdf, filename, attack_name, output_subdir):
        """
        保存被攻击的数据（保存前验证顶点数）
        
        Returns:
            bool: True=保存成功, False=保存失败或数据无效
        """
        base_name = os.path.splitext(filename)[0]
        attack_base_name = os.path.splitext(attack_name)[0]
        output_filename = f"{attack_base_name}.geojson"
        output_path = os.path.join(self.output_dir, output_subdir, output_filename)
        
        # ✅ 保存前验证数据有效性（只检查是否为空，不限制对象数）
        if not self.is_valid_gdf(gdf):
            print(f"❌ 跳过保存 {output_filename}: 数据为空")
            return False
        
        # Append 模式：如果目标文件已存在则跳过，不覆盖
        if os.path.exists(output_path):
            print(f"⏭️ 已存在，跳过保存: {output_filename}")
            return False

        try:
            gdf.to_file(output_path, driver='GeoJSON')
            return True
        except Exception as e:
            print(f"❌ 保存失败 {output_filename}: {e}")
            return False
    
    def generate_attacks(self):
        """生成100个攻击版本（前50个指定方式，后50个随机组合）"""
        # 获取所有geojson文件
        geojson_files = [f for f in os.listdir(self.input_dir) if f.endswith('.geojson')]
        
        if not geojson_files:
            print("未找到geojson文件")
            return
        
        print(f"找到 {len(geojson_files)} 个矢量文件\n")
        
        # ✅ 全局统计
        total_generated = 0
        total_invalid = 0
        
        for geojson_file in geojson_files:
            print(f"\n{'='*70}")
            print(f"处理文件: {geojson_file}")
            print(f"{'='*70}")
            
            gdf = self.load_vector_data(geojson_file)
            
            if gdf is None:
                continue
            
            # 获取文件名称（去掉.geojson后缀）作为子文件夹名
            file_base_name = os.path.splitext(geojson_file)[0]
            
            # 清理并创建输出子目录
            self.clean_output_subdir(file_base_name)
            
            # ✅ 文件级统计
            file_generated = 0
            file_invalid = 0
            
            # 生成前50个指定攻击方式
            print(f"\n生成前{len(self.single_attacks)}个指定攻击方式...")
            # 为每个数据集增加两个不同强度的噪声攻击（append 形式）
            noise_strengths = [0.4, 0.7]
            noise_pcts = [30, 60]  # 两个不同的扰动比例，可根据需要调整
            per_file_attacks = list(self.single_attacks)  # 复制基础攻击清单
            for s, p in zip(noise_strengths, noise_pcts):
                per_file_attacks.append((f"test_noise_{p}pct_strength_{s}.geojson", f"噪声扰动{p}%顶点_强度{s}"))

            for i, (attack_name, attack_desc) in enumerate(tqdm(per_file_attacks, desc="指定攻击")):
                try:
                    gdf_attacked = self.apply_single_attack(gdf, attack_name)
                    saved = self.save_attacked_data(gdf_attacked, geojson_file, attack_name, file_base_name)
                    if saved:
                        file_generated += 1
                    else:
                        file_invalid += 1
                except Exception as e:
                    print(f"应用攻击 {attack_name} 时出错: {e}")
                    continue
            
            # 生成后50个随机组合攻击
            print(f"\n生成后{self.combo_count}个随机组合攻击...")
            for i in tqdm(range(self.combo_count), desc="组合攻击"):
                try:
                    gdf_attacked, attack_descriptions = self.apply_combo_attack(gdf)
                    # 生成描述性的文件名
                    combo_name = "_".join([desc.replace("%", "pct").replace("，", "_").replace(" ", "").replace("、", "_").replace("(", "").replace(")", "") for desc in attack_descriptions])
                    # 限制文件名长度，避免过长
                    if len(combo_name) > 150:
                        combo_name = f"{combo_name[:150]}_test_combo{i+1}.geojson"
                    else:
                        combo_name = f"test_combo_{combo_name}.geojson"
                    saved = self.save_attacked_data(gdf_attacked, geojson_file, combo_name, file_base_name)
                    if saved:
                        file_generated += 1
                    else:
                        file_invalid += 1
                except Exception as e:
                    print(f"应用组合攻击 {i+1} 时出错: {e}")
                    continue
            
            # ✅ 文件级统计输出
            total_generated += file_generated
            total_invalid += file_invalid
            
            print(f"\n✅ {file_base_name} 完成:")
            print(f"   - 生成成功: {file_generated} 个")
            if file_invalid > 0:
                print(f"   - 数据无效跳过（顶点数<8）: {file_invalid} 个")
        
        # ✅ 全局统计输出
        print(f"\n{'='*70}")
        print(f"✅ 所有测试集文件处理完成！")
        print(f"{'='*70}")
        print(f"📊 总体统计:")
        print(f"   - 总共生成成功: {total_generated} 个攻击样本")
        if total_invalid > 0:
            print(f"   - 数据无效跳过（顶点数<8）: {total_invalid} 个")
            total_attempts = total_generated + total_invalid
            print(f"   - 成功率: {total_generated/total_attempts*100:.1f}%")
        print(f"\n✅ 数据验证: 所有保存的攻击样本总顶点数均 ≥ 8")
        print(f"{'='*70}\n")

def main():
    """主函数"""
    print("=== 第一步：生成测试集被攻击的矢量数据 ===")
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 强制只生成噪声攻击（append 形式），避免误传参导致生成其它攻击
    noise_only = True

    # 创建攻击生成器
    generator = TestVectorAttackGenerator(noise_only=noise_only)

    # 生成攻击数据（如果需要仅生成噪声，传入 --noise-only）
    generator.generate_attacks()
    
    print("\n测试集攻击数据生成完成！")
    print(f"攻击数据保存在: {generator.output_dir}")
    print("为每个图生成了100个被攻击的矢量地图数据")

if __name__ == "__main__":
    main()
