#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一步：生成被攻击的矢量数据
按照attack200.py的逻辑：前100个指定攻击方式，后100个随机组合攻击
为每个图生成200个被攻击的矢量地图类型放入vector_data_attacked的各个图的子文件夹中
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

class VectorAttackGenerator:
    """矢量数据攻击生成器"""
    
    def __init__(self, input_dir="../convertToGeoJson/GeoJson/TrainingSet", output_dir="GeoJson-Attacked/TrainingSet", incremental_mode=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.incremental_mode = incremental_mode
        self.ensure_output_dir()
        
        # ⭐ 代码版本号：标记几何攻击逻辑的变化
        self.code_version = "v2.0_global_center"  # v2.0: 缩放/旋转/翻转改用全局中心
        
        # ⭐ 受影响的攻击类型（本次修改影响的攻击）
        # 这些攻击需要强制重新生成，即使文件已存在
        self.affected_attack_keywords = {
            'scale',      # 缩放攻击：改为全局中心
            'rotate',     # 旋转攻击：改为全局中心
            'flip',       # 翻转攻击：改为全局中心
            'combo',      # 组合攻击：可能包含上述攻击
            'full_attack' # 全攻击链：包含所有攻击
        }
        
        # ✅ 优化：定义单体攻击方式，增加低NC攻击的样本数量
        self.single_attacks = []
        
        # Fig1: 删除顶点 10%-90% (9个) + ✅ 额外增加高比例删除样本 (5个)
        for pct in range(10, 100, 10):
            self.single_attacks.append((f"delete_{pct}pct_vertices.geojson", f"删除{pct}%顶点"))
        # ✅ 针对90%删除（NC=0.944），增加多个变体
        for i in range(1, 6):
            self.single_attacks.append((f"delete_85pct_vertices_v{i}.geojson", f"删除85%顶点_变体{i}"))
        
        # Fig2: 添加顶点 - ✅ 大幅增加样本（从27个增加到48个）
        # 强度0,1,2 × 比例10%-90% (基础27个)
        for strength in [0, 1, 2]:
            for pct in range(10, 100, 10):
                self.single_attacks.append((f"add_strength{strength}_{pct}pct_vertices.geojson", f"添加{pct}%顶点，强度{strength}"))
        # ✅ 针对强度1和2（NC<0.85），增加更多比例 (16个)
        for strength in [1, 2]:
            for pct in [15, 25, 35, 45, 55, 65, 75, 85]:  # 增加中间比例
                self.single_attacks.append((f"add_strength{strength}_{pct}pct_vertices_extra.geojson", f"添加{pct}%顶点，强度{strength}_增强"))
        # ✅ 再增加5个强度1的高比例样本
        for i in range(1, 6):
            self.single_attacks.append((f"add_strength1_50pct_vertices_v{i}.geojson", f"添加50%顶点，强度1_变体{i}"))
        
        # Fig3: 删除对象 10%-90% (9个) + ✅ 额外增加高比例删除样本 (5个)
        for pct in range(10, 100, 10):
            self.single_attacks.append((f"delete_{pct}pct_objects.geojson", f"删除{pct}%图形对象"))
        # ✅ 针对90%删除（NC=0.853），增加多个变体
        for i in range(1, 6):
            self.single_attacks.append((f"delete_85pct_objects_v{i}.geojson", f"删除85%对象_变体{i}"))
        
        # Fig4: 噪声扰动 - ✅ 大幅增加样本（从15个增加到45个）
        # 基础：强度[0.4, 0.6, 0.8] × 比例[10, 30, 50, 70, 90] (15个)
        for strength in [0.4, 0.6, 0.8]:
            for pct in [10, 30, 50, 70, 90]:
                self.single_attacks.append((f"noise_{pct}pct_strength_{strength}.geojson", f"噪声扰动{pct}%顶点，强度{strength}"))
        # ✅ 增加更多强度和比例组合 (30个)
        for strength in [0.5, 0.7]:  # 新增中间强度
            for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                self.single_attacks.append((f"noise_{pct}pct_strength_{strength}_extra.geojson", f"噪声扰动{pct}%顶点，强度{strength}_增强"))
        # 再增加12个高强度样本
        for strength in [0.6, 0.8]:
            for pct in [40, 60, 80]:
                for i in range(1, 3):
                    self.single_attacks.append((f"noise_{pct}pct_strength_{strength}_v{i}.geojson", f"噪声扰动{pct}%顶点，强度{strength}_变体{i}"))
        
        # Fig5: 裁剪 - ✅ 增加左上角和右下角的变体（从5个增加到15个）
        self.single_attacks.extend([
            ("crop_x_center_50pct.geojson", "沿X轴中心裁剪50%"),
            ("crop_y_center_50pct.geojson", "沿Y轴中心裁剪50%"),
            ("crop_top_left.geojson", "裁剪左上角区域"),
            ("crop_bottom_right.geojson", "裁剪右下角区域"),
            ("crop_random_40pct.geojson", "随机裁剪40%"),
        ])
        # ✅ 针对左上角（NC=0.842）和右下角（NC=0.892），增加多个变体和不同裁剪比例 (10个)
        for i in range(1, 6):
            self.single_attacks.append((f"crop_top_left_v{i}.geojson", f"裁剪左上角_变体{i}"))
        for i in range(1, 6):
            self.single_attacks.append((f"crop_bottom_right_v{i}.geojson", f"裁剪右下角_变体{i}"))
        
        # Fig6: 平移 5种方式 (5个)
        self.single_attacks.extend([
            ("translate_x_20.geojson", "沿X轴平移20"),
            ("translate_y_20.geojson", "沿Y轴平移20"),
            ("translate_20_20.geojson", "沿X、Y轴分别平移20"),
            ("translate_20_40.geojson", "沿X轴平移20，沿Y轴平移40"),
            ("translate_30_10.geojson", "沿X轴平移30，沿Y轴平移10"),
        ])
        
        # Fig7: 缩放 6个因子 (6个)
        for factor in [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]:
            pct = int(round(factor * 100))
            self.single_attacks.append((f"scale_{pct}pct.geojson", f"缩放{pct}%"))
        
        # Fig8: 旋转 8个角度 (8个)
        for deg in [45, 90, 135, 180, 225, 270, 315, 360]:
            self.single_attacks.append((f"rotate_{deg}deg.geojson", f"旋转{deg}度"))
        
        # Fig9: 翻转 3种方式 (3个)
        self.single_attacks.extend([
            ("flip_x.geojson", "X轴镜像翻转"),
            ("flip_y.geojson", "Y轴镜像翻转"),
            ("flip_xy.geojson", "同时X、Y轴镜像翻转"),
        ])
        
        # Fig10: 打乱顺序 - ✅ 增加打乱顶点的样本（从4个增加到18个）
        self.single_attacks.extend([
            ("reverse_vertices.geojson", "反转顶点顺序"),
            ("shuffle_vertices.geojson", "打乱顶点顺序"),
            ("reverse_objects.geojson", "反转对象顺序"),
            ("shuffle_objects.geojson", "打乱对象顺序"),
        ])
        # ✅ 针对打乱顶点顺序（NC=0.873），增加多个变体（不同随机种子） (14个)
        for i in range(1, 15):
            self.single_attacks.append((f"shuffle_vertices_v{i}.geojson", f"打乱顶点顺序_变体{i}"))
        
        # ✅ 优化：增加组合攻击样本数量和多样性
        # 1个全攻击链（Fig12风格）+ 149个多样化组合 = 150个组合攻击
        self.combo_attacks = []
        
        # 1. 全攻击链组合（Fig12风格）- 最强攻击
        self.combo_attacks.append(("combo_full_attack_chain.geojson", "全攻击链组合(Fig1→Fig10)"))
        
        # 2. 重度组合攻击（6-8种攻击）- 30个
        for i in range(1, 31):
            self.combo_attacks.append((f"combo_heavy_{i:03d}.geojson", f"重度组合攻击{i}(6-8种)"))
        
        # 3. 中度组合攻击（4-5种攻击）- 50个
        for i in range(1, 51):
            self.combo_attacks.append((f"combo_medium_{i:03d}.geojson", f"中度组合攻击{i}(4-5种)"))
        
        # 4. 轻度组合攻击（2-3种攻击）- 69个（保持向后兼容）
        for i in range(1, 70):
            self.combo_attacks.append((f"combo_light_{i:03d}.geojson", f"轻度组合攻击{i}(2-3种)"))
    
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
        
        if crop_type == "x_center_50pct":
            # 沿X轴中心裁剪50%
            mid_x = (bounds[0] + bounds[2]) / 2
            gdf_attacked = gdf_attacked[bdf['minx'] < mid_x].reset_index(drop=True)
        elif crop_type == "y_center_50pct":
            # 沿Y轴中心裁剪50%
            mid_y = (bounds[1] + bounds[3]) / 2
            gdf_attacked = gdf_attacked[bdf['miny'] < mid_y].reset_index(drop=True)
        elif crop_type == "top_left":
            # 裁剪左上角区域
            gdf_attacked = gdf_attacked[
                (bdf['minx'] < (bounds[0] + bounds[2]) / 2) &
                (bdf['miny'] > (bounds[1] + bounds[3]) / 2)
            ].reset_index(drop=True)
        elif crop_type == "bottom_right":
            # 裁剪右下角区域
            gdf_attacked = gdf_attacked[
                (bdf['minx'] > (bounds[0] + bounds[2]) / 2) &
                (bdf['miny'] < (bounds[1] + bounds[3]) / 2)
            ].reset_index(drop=True)
        elif crop_type == "random_40pct":
            # 随机裁剪40%
            num_objects = len(gdf_attacked)
            num_to_keep = int(num_objects * 0.6)
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
        """缩放攻击（使用全局中心，与Fig7一致）"""
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
        """旋转攻击（使用全局中心，与Fig8一致）"""
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
        """翻转攻击（使用全局中心，与Fig9一致）"""
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
            # ✅ 双轴翻转使用rotate(180°)实现，与Fig9一致
            # 数学等价且更稳定，避免scale(-1,-1)可能导致的顶点顺序问题
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
                lambda geom: rotate(geom, 180, origin=global_center)
            )
        return gdf_attacked
    
    def apply_reverse_vertices_attack(self, gdf):
        """反转顶点顺序攻击"""
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
    
    def apply_shuffle_vertices_attack(self, gdf):
        """打乱顶点顺序攻击"""
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
        
        gdf_attacked = gdf.copy()
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
        # ✅ 修复：移除后缀变体标识，使所有变体都能正确匹配基础攻击逻辑
        import re
        # 移除 _extra、_v1、_v2 等后缀（但保留前面的参数）
        normalized_name = re.sub(r'_(extra|v\d+)(?=\.geojson)', '', attack_name)
        
        # Fig1: 删除顶点 10%-90% (包括85%变体)
        for pct in range(10, 100, 10):
            if f"delete_{pct}pct_vertices" in normalized_name:
                return self.apply_delete_vertices_attack(gdf, pct)
        # 额外支持85%删除变体
        if "delete_85pct_vertices" in normalized_name:
            return self.apply_delete_vertices_attack(gdf, 85)
        
        # Fig2: 添加顶点 强度0,1,2 × 比例10%-90% (包括额外比例)
        for strength in [0, 1, 2]:
            for pct in range(10, 100, 10):
                if f"add_strength{strength}_{pct}pct_vertices" in normalized_name:
                    return self.apply_add_vertices_attack(gdf, pct, strength)
            # 额外支持 15, 25, 35, 45, 55, 65, 75, 85 等比例
            for pct in [15, 25, 35, 45, 55, 65, 75, 85]:
                if f"add_strength{strength}_{pct}pct_vertices" in normalized_name:
                    return self.apply_add_vertices_attack(gdf, pct, strength)
        
        # Fig3: 删除对象 10%-90% (包括85%变体)
        for pct in range(10, 100, 10):
            if f"delete_{pct}pct_objects" in normalized_name:
                return self.apply_delete_objects_attack(gdf, pct)
        # 额外支持85%删除变体
        if "delete_85pct_objects" in normalized_name:
            return self.apply_delete_objects_attack(gdf, 85)
        
        # Fig4: 噪声扰动 强度[0.4, 0.5, 0.6, 0.7, 0.8] × 比例[10-90]
        for strength in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                if f"noise_{pct}pct_strength_{strength}" in normalized_name:
                    return self.apply_noise_attack(gdf, pct, strength)
        
        # Fig5: 裁剪 (使用标准化名称，支持变体)
        if "crop_x_center_50pct" in normalized_name:
            return self.apply_crop_attack(gdf, "x_center_50pct")
        elif "crop_y_center_50pct" in normalized_name:
            return self.apply_crop_attack(gdf, "y_center_50pct")
        elif "crop_top_left" in normalized_name:
            return self.apply_crop_attack(gdf, "top_left")
        elif "crop_bottom_right" in normalized_name:
            return self.apply_crop_attack(gdf, "bottom_right")
        elif "crop_random_40pct" in normalized_name:
            return self.apply_crop_attack(gdf, "random_40pct")
        
        # Fig6: 平移 (使用标准化名称)
        elif "translate_x_20" in normalized_name and "translate_20_20" not in normalized_name and "translate_20_40" not in normalized_name:
            return self.apply_translate_attack(gdf, 20, 0)
        elif "translate_y_20" in normalized_name and "translate_20_20" not in normalized_name:
            return self.apply_translate_attack(gdf, 0, 20)
        elif "translate_20_20" in normalized_name:
            return self.apply_translate_attack(gdf, 20, 20)
        elif "translate_20_40" in normalized_name:
            return self.apply_translate_attack(gdf, 20, 40)
        elif "translate_30_10" in normalized_name:
            return self.apply_translate_attack(gdf, 30, 10)
        
        # Fig7: 缩放 (使用标准化名称)
        for factor in [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]:
            pct = int(round(factor * 100))
            if f"scale_{pct}pct" in normalized_name:
                return self.apply_scale_attack(gdf, factor)
        
        # Fig8: 旋转 (使用标准化名称)
        for deg in [45, 90, 135, 180, 225, 270, 315, 360]:
            if f"rotate_{deg}deg" in normalized_name:
                return self.apply_rotate_attack(gdf, deg)
        
        # Fig9: 翻转 (使用标准化名称)
        if "flip_xy" in normalized_name:  # 必须在flip_x和flip_y之前检查
            return self.apply_flip_attack(gdf, "xy")
        elif "flip_x" in normalized_name:
            return self.apply_flip_attack(gdf, "x")
        elif "flip_y" in normalized_name:
            return self.apply_flip_attack(gdf, "y")
        
        # Fig10: 打乱顺序 (使用标准化名称，支持变体)
        elif "reverse_vertices" in normalized_name:
            return self.apply_reverse_vertices_attack(gdf)
        elif "shuffle_vertices" in normalized_name:
            return self.apply_shuffle_vertices_attack(gdf)
        elif "reverse_objects" in normalized_name:
            return gdf.iloc[::-1].reset_index(drop=True)
        elif "shuffle_objects" in normalized_name:
            return gdf.sample(frac=1).reset_index(drop=True)
        
        else:
            # 未知的攻击类型
            print(f"警告：未知的攻击类型 {attack_name}，返回原始数据")
            return gdf.copy()
    
    def apply_combo_attack(self, gdf, combo_type='light'):
        """
        应用组合攻击（带有中间验证，确保每步攻击后数据仍然有效）
        
        Args:
            gdf: GeoDataFrame
            combo_type: 组合类型
                - 'full': 全攻击链（Fig12风格，所有攻击顺序执行）
                - 'heavy': 重度组合（6-8种攻击）
                - 'medium': 中度组合（4-5种攻击）
                - 'light': 轻度组合（2-3种攻击）
        """
        gdf_attacked = gdf.copy()
        attack_descriptions = []
        
        # 不再限制最小对象数，所有攻击数据都保存
        
        if combo_type == 'full':
            # ✅ Fig12风格：按顺序执行所有攻击类型（模拟最严酷的攻击）
            attack_sequence = [
                ("delete_10pct_vertices.geojson", "删除10%顶点"),
                ("add_strength1_50pct_vertices.geojson", "添加50%顶点(强度1)"),
                ("delete_50pct_objects.geojson", "删除50%对象"),
                ("noise_50pct_strength_0.8.geojson", "噪声50%(强度0.8)"),
                ("crop_y_center_50pct.geojson", "沿Y轴裁剪50%"),
                ("translate_20_40.geojson", "平移(20,40)"),
                ("scale_90pct.geojson", "缩放90%"),
                ("rotate_180deg.geojson", "旋转180度"),
                ("flip_y.geojson", "Y轴翻转"),
                ("reverse_vertices.geojson", "反转顶点顺序"),
            ]
            
            for attack_name, attack_desc in attack_sequence:
                try:
                    # 备份当前状态
                    gdf_backup = gdf_attacked.copy()
                    
                    # 应用攻击
                    gdf_attacked = self.apply_single_attack(gdf_attacked, attack_name)
                    
                    # ✅ 验证攻击后的数据是否有效
                    if not self.is_valid_gdf(gdf_attacked):
                        # 数据无效（为空），恢复到备份状态，跳过此攻击
                        gdf_attacked = gdf_backup
                        # print(f"  ⚠️ 跳过攻击 {attack_name}：攻击后数据为空")
                    else:
                        # 数据有效，记录攻击
                        attack_descriptions.append(attack_desc)
                        
                except Exception as e:
                    # 攻击失败，恢复备份防止脏数据
                    gdf_attacked = gdf_backup
                    print(f"警告：攻击{attack_name}失败: {e}，已恢复备份，继续执行")
                    
        elif combo_type == 'heavy':
            # 重度组合：随机选择6-8种攻击
            num_attacks = random.randint(6, 8)
            selected_attacks = random.sample(self.single_attacks, min(num_attacks, len(self.single_attacks)))
            
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
                        # print(f"  ⚠️ 跳过攻击 {attack_name}：攻击后数据为空")
                    else:
                        # 数据有效，记录攻击
                        attack_descriptions.append(attack_desc)
                        
                except Exception as e:
                    print(f"警告：攻击{attack_name}失败: {e}，跳过此攻击")
                
        elif combo_type == 'medium':
            # 中度组合：随机选择4-5种攻击
            num_attacks = random.randint(4, 5)
            selected_attacks = random.sample(self.single_attacks, min(num_attacks, len(self.single_attacks)))
            
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
                        # print(f"  ⚠️ 跳过攻击 {attack_name}：攻击后数据为空")
                    else:
                        # 数据有效，记录攻击
                        attack_descriptions.append(attack_desc)
                        
                except Exception as e:
                    print(f"警告：攻击{attack_name}失败: {e}，跳过此攻击")
                
        else:  # light
            # 轻度组合：随机选择2-3种攻击（原有逻辑）
            num_attacks = random.randint(2, 3)
            selected_attacks = random.sample(self.single_attacks, min(num_attacks, len(self.single_attacks)))
            
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
                        # print(f"  ⚠️ 跳过攻击 {attack_name}：攻击后数据为空")
                    else:
                        # 数据有效，记录攻击
                        attack_descriptions.append(attack_desc)
                        
                except Exception as e:
                    print(f"警告：攻击{attack_name}失败: {e}，跳过此攻击")
        
        return gdf_attacked, attack_descriptions
    
    def apply_random_attack(self, gdf):
        """应用随机攻击"""
        attack_types = ['translate', 'rotate', 'scale', 'noise', 'crop', 'flip']
        attack_type = random.choice(attack_types)
        
        if attack_type == 'translate':
            dx = random.uniform(-30, 30)
            dy = random.uniform(-30, 30)
            return self.apply_translate_attack(gdf, dx, dy)
        elif attack_type == 'rotate':
            angle = random.uniform(-90, 90)
            return self.apply_rotate_attack(gdf, angle)
        elif attack_type == 'scale':
            scale_factor = random.uniform(0.7, 1.3)
            return self.apply_scale_attack(gdf, scale_factor)
        elif attack_type == 'noise':
            strength = random.uniform(0.1, 0.5)
            return self.apply_noise_attack(gdf, 100, strength)
        elif attack_type == 'crop':
            crop_type = random.choice(['x_center_50pct', 'y_center_50pct', 'random_40pct'])
            return self.apply_crop_attack(gdf, crop_type)
        elif attack_type == 'flip':
            flip_type = random.choice(['x', 'y', 'xy'])
            return self.apply_flip_attack(gdf, flip_type)
    
    def should_regenerate_attack(self, attack_name, output_path):
        """
        判断是否需要重新生成攻击文件
        
        Args:
            attack_name: 攻击文件名（如 'scale_90pct.geojson'）
            output_path: 输出文件的完整路径
            
        Returns:
            (bool, str): (是否需要生成, 原因说明)
        """
        # 1. 输出文件不存在，必须生成
        if not os.path.exists(output_path):
            return True, "输出文件不存在"
        
        # 2. 如果不是增量模式，全部重新生成
        if not self.incremental_mode:
            return True, "完全重新生成模式"
        
        # 3. 检查攻击类型是否受本次代码修改影响
        attack_name_lower = attack_name.lower()
        
        for keyword in self.affected_attack_keywords:
            if keyword in attack_name_lower:
                return True, f"攻击逻辑已更新({keyword}→全局中心)"
        
        # 4. 其他情况：文件存在且攻击逻辑未变化，跳过
        return False, "攻击逻辑未变化"
    
    def clean_output_subdir(self, output_subdir):
        """确保输出子目录存在（增量模式下不清空）"""
        subdir_path = os.path.join(self.output_dir, output_subdir)
        
        if not self.incremental_mode:
            # 完全重新生成模式：清空目录
            if os.path.exists(subdir_path):
                print(f"🔥 清理旧文件: {subdir_path}")
                shutil.rmtree(subdir_path)
        else:
            # 增量模式：保留目录，只更新变化的文件
            if os.path.exists(subdir_path):
                print(f"🔄 增量更新模式：保留未变化的文件")
            else:
                print(f"📁 创建新目录: {subdir_path}")
        
        os.makedirs(subdir_path, exist_ok=True)

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
        
        try:
            gdf.to_file(output_path, driver='GeoJSON')
            # print(f"保存攻击数据: {output_filename}")
            return True
        except Exception as e:
            print(f"❌ 保存失败 {output_filename}: {e}")
            return False
    
    def generate_attacks(self):
        """✅ 优化：生成更多训练样本（前91个指定方式 + 后150个组合攻击） + ⭐ 增量更新"""
        # 获取所有geojson文件
        geojson_files = [f for f in os.listdir(self.input_dir) if f.endswith('.geojson')]
        
        if not geojson_files:
            print("未找到geojson文件")
            return
        
        print(f"\n{'='*70}")
        print(f"找到 {len(geojson_files)} 个矢量文件")
        print(f"总攻击样本数: {len(self.single_attacks)} 单一攻击 + {len(self.combo_attacks)} 组合攻击 = {len(self.single_attacks) + len(self.combo_attacks)} 样本")
        print(f"代码版本: {self.code_version}")
        print(f"模式: {'🔄 增量更新（只更新受影响的攻击）' if self.incremental_mode else '🔥 完全重新生成'}")
        print(f"{'='*70}\n")
        
        # 全局统计
        total_generated = 0
        total_skipped = 0
        total_invalid = 0  # ✅ 新增：因数据无效（顶点数<8）被跳过的数量
        
        for geojson_file in geojson_files:
            print(f"\n{'='*70}")
            print(f"📂 处理文件: {geojson_file}")
            print(f"{'='*70}")
            
            gdf = self.load_vector_data(geojson_file)
            
            if gdf is None:
                continue
            
            # 获取文件名称（去掉.geojson后缀）作为子文件夹名
            file_base_name = os.path.splitext(geojson_file)[0]
            
            # 清理并创建输出子目录（增量模式下不清空）
            self.clean_output_subdir(file_base_name)
            
            # 文件级统计
            file_generated = 0
            file_skipped = 0
            file_invalid = 0  # ✅ 新增：因数据无效被跳过的数量
            
            # 生成单一攻击方式（91个）
            print(f"\n📌 处理 {len(self.single_attacks)} 个单一攻击...")
            for i, (attack_name, attack_desc) in enumerate(self.single_attacks):
                try:
                    # 获取输出路径
                    attack_base_name = os.path.splitext(attack_name)[0]
                    output_filename = f"{attack_base_name}.geojson"
                    output_path = os.path.join(self.output_dir, file_base_name, output_filename)
                    
                    # ⭐ 判断是否需要重新生成
                    should_update, reason = self.should_regenerate_attack(attack_name, output_path)
                    
                    if should_update:
                        if file_generated < 5 or '全局中心' in reason:  # 只显示前几个或受影响的
                            print(f"  🔄 更新: {attack_name:50s} ({reason})")
                        gdf_attacked = self.apply_single_attack(gdf, attack_name)
                        # ✅ 保存时会自动验证，返回False表示数据无效
                        saved = self.save_attacked_data(gdf_attacked, geojson_file, attack_name, file_base_name)
                        if saved:
                            file_generated += 1
                        else:
                            file_invalid += 1
                    else:
                        if file_skipped < 3:  # 只显示前几个跳过的
                            print(f"  ⏭️  跳过: {attack_name:50s} ({reason})")
                        file_skipped += 1
                        
                except Exception as e:
                    print(f"  ❌ 错误: {attack_name} - {e}")
                    continue
            
            if file_skipped > 3:
                print(f"  ... 还有 {file_skipped - 3} 个单一攻击被跳过")
            
            # ✅ 优化：生成150个组合攻击（包括Fig12全攻击链）
            print(f"\n📌 处理 {len(self.combo_attacks)} 个组合攻击...")
            combo_generated = 0
            combo_skipped = 0
            combo_invalid = 0  # ✅ 新增：组合攻击中因数据无效被跳过的数量
            
            for i, (combo_filename, combo_desc) in enumerate(self.combo_attacks):
                try:
                    # 获取输出路径
                    combo_base_name = os.path.splitext(combo_filename)[0]
                    output_filename = f"{combo_base_name}.geojson"
                    output_path = os.path.join(self.output_dir, file_base_name, output_filename)
                    
                    # ⭐ 判断是否需要重新生成
                    should_update, reason = self.should_regenerate_attack(combo_filename, output_path)
                    
                    if should_update:
                        # 根据文件名判断组合类型
                        if 'full_attack_chain' in combo_filename:
                            combo_type = 'full'
                            if combo_generated == 0:
                                print(f"  🔥 生成全攻击链 (Fig12风格)...")
                        elif 'heavy' in combo_filename:
                            combo_type = 'heavy'
                        elif 'medium' in combo_filename:
                            combo_type = 'medium'
                        else:
                            combo_type = 'light'
                        
                        if combo_generated < 3 or 'full_attack' in combo_filename:
                            print(f"  🔄 更新: {combo_filename:50s} ({reason})")
                        
                        gdf_attacked, attack_descriptions = self.apply_combo_attack(gdf, combo_type)
                        # ✅ 保存时会自动验证，返回False表示数据无效
                        saved = self.save_attacked_data(gdf_attacked, geojson_file, combo_filename, file_base_name)
                        if saved:
                            combo_generated += 1
                        else:
                            combo_invalid += 1
                    else:
                        if combo_skipped < 3:
                            print(f"  ⏭️  跳过: {combo_filename:50s} ({reason})")
                        combo_skipped += 1
                        
                except Exception as e:
                    print(f"  ❌ 错误: {combo_filename} - {e}")
                    continue
            
            file_generated += combo_generated
            file_skipped += combo_skipped
            file_invalid += combo_invalid
            
            if combo_skipped > 3:
                print(f"  ... 还有 {combo_skipped - 3} 个组合攻击被跳过")
            
            # 文件级统计
            total_generated += file_generated
            total_skipped += file_skipped
            total_invalid += file_invalid
            
            print(f"\n✅ {file_base_name} 完成:")
            print(f"   - 生成成功: {file_generated} 个")
            if file_invalid > 0:
                print(f"   - 数据无效（顶点数<8）: {file_invalid} 个")
            if self.incremental_mode:
                print(f"   - 跳过（已存在且逻辑未变）: {file_skipped} 个")
                total_attempts = file_generated + file_skipped + file_invalid
                if total_attempts > 0:
                    print(f"   - 跳过率: {file_skipped/total_attempts*100:.1f}%")
        
        # 全局统计
        print(f"\n{'='*70}")
        print(f"✅ 所有文件处理完成！")
        print(f"{'='*70}")
        print(f"📊 总体统计:")
        print(f"   - 总共生成成功: {total_generated} 个攻击样本")
        if total_invalid > 0:
            print(f"   - 数据无效跳过（顶点数<8）: {total_invalid} 个")
        if self.incremental_mode:
            print(f"   - 文件已存在跳过: {total_skipped} 个攻击样本")
            total = total_generated + total_skipped + total_invalid
            print(f"   - 总尝试数: {total} 个")
            print(f"   - 成功率: {total_generated/total*100 if total > 0 else 0:.1f}%")
            print(f"   - 跳过率（已存在）: {total_skipped/total*100 if total > 0 else 0:.1f}%")
            if total_invalid > 0:
                print(f"   - 无效率（顶点<8）: {total_invalid/total*100 if total > 0 else 0:.1f}%")
            print(f"\n⚡ 增量更新效果:")
            print(f"   - 仅更新受影响的攻击类型: {', '.join(self.affected_attack_keywords)}")
            print(f"   - 节省文件生成数: {total_skipped} 个")
        else:
            if total_invalid > 0:
                print(f"   - 无效数据比例: {total_invalid/(total_generated+total_invalid)*100:.1f}%")
        print(f"\n✅ 数据验证: 所有保存的攻击样本总顶点数均 ≥ 8")
        print(f"{'='*70}\n")

def main():
    """主函数"""
    print("="*70)
    print("第一步：生成被攻击的矢量数据 (⭐ 支持增量更新)")
    print("="*70)
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # ⭐ 增量更新模式（默认启用）
    # True: 只更新受影响的攻击类型（scale, rotate, flip, combo）
    # False: 完全重新生成所有文件
    incremental_mode = True
    
    print(f"\n⚙️  配置:")
    print(f"   - 模式: {'🔄 增量更新模式' if incremental_mode else '🔥 完全重新生成模式'}")
    print(f"   - 受影响的攻击: scale（缩放）, rotate（旋转）, flip（翻转）, combo（组合）")
    print(f"   - 版本: v2.0_global_center（几何变换改用全局中心）")
    print()
    
    # 创建攻击生成器
    generator = VectorAttackGenerator(incremental_mode=incremental_mode)
    
    # 生成攻击数据
    generator.generate_attacks()
    
    print("\n" + "="*70)
    print("✅ 攻击数据生成完成！")
    print("="*70)
    print(f"📂 攻击数据保存在: {generator.output_dir}")
    print(f"📊 为每个图生成/更新了攻击样本")
    if incremental_mode:
        print(f"⚡ 增量更新: 已精确覆盖受影响的攻击文件")
        print(f"   - scale（缩放）: 已更新为全局中心")
        print(f"   - rotate（旋转）: 已更新为全局中心")
        print(f"   - flip（翻转）: 已更新为全局中心")
        print(f"   - combo（组合）: 已更新包含上述攻击的组合")
    print("="*70)

if __name__ == "__main__":
    main() 