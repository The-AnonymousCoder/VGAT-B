#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只生成测试集的噪声攻击（Noise-only）
为每个测试集 GeoJSON 生成两个不同强度的噪声攻击并追加到对应 attacked 子目录（append，不覆盖）

用法:
  python convertToGeoJson-Attacked-TestSet-NoiseOnly.py

脚本保证：
 - 不覆盖已有文件（若已存在则跳过）
 - 路径解析基于脚本文件位置，避免工作目录差异
 - 生成文件示例名： noise_30pct_strength_0.40.geojson
"""

import os
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon, Point
import sys


class NoiseOnlyGenerator:
    """为 TestSet 只生成噪声攻击的生成器"""

    def __init__(self,
                 input_dir="../convertToGeoJson/GeoJson/TestSet",
                 output_dir="GeoJson-Attacked/TestSet",
                 noise_specs=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_dir = input_dir if os.path.isabs(input_dir) else os.path.abspath(os.path.join(base_dir, input_dir))
        self.output_dir = output_dir if os.path.isabs(output_dir) else os.path.abspath(os.path.join(base_dir, output_dir))
        # 默认两个噪声：30% @0.4 and 60% @0.7
        self.noise_specs = noise_specs or [(30, 0.4), (60, 0.7)]

        self.ensure_output_dir()

    def ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def load_vector_data(self, filename):
        fp = os.path.join(self.input_dir, filename)
        try:
            gdf = gpd.read_file(fp)
            print(f"加载: {filename} ({len(gdf)} 要素)")
            return gdf
        except Exception as e:
            print(f"加载失败 {filename}: {e}")
            return None

    def apply_noise_attack(self, gdf, percentage, strength):
        """将噪声应用到 gdf 中选定比例的顶点（顶点级扰动）"""
        def jitter_vertices(geom, pct, strength):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                n = len(coords)
                if n == 0:
                    return geom
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
                if n == 0:
                    return geom
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
                    if n_ring == 0:
                        holes.append(ring_coords)
                        continue
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
            else:
                return geom

        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: jitter_vertices(geom, percentage, strength)
        )
        return gdf_attacked

    def save_attacked(self, gdf, src_filename, attack_name, subdir):
        base = os.path.splitext(src_filename)[0]
        out_dir = os.path.join(self.output_dir, base)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_filename = os.path.splitext(attack_name)[0] + ".geojson"
        out_path = os.path.join(out_dir, out_filename)
        # append behavior: do not overwrite
        if os.path.exists(out_path):
            print(f"⏭️ 已存在，跳过: {os.path.join(base, out_filename)}")
            return False
        try:
            gdf.to_file(out_path, driver="GeoJSON")
            print(f"✅ 已保存: {os.path.join(base, out_filename)}")
            return True
        except Exception as e:
            print(f"❌ 保存失败 {out_path}: {e}")
            return False

    def generate_noise_only(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.geojson')]
        if not files:
            print("未找到任何测试集 geojson 文件:", self.input_dir)
            return
        total = 0
        for fn in files:
            gdf = self.load_vector_data(fn)
            if gdf is None:
                continue
            # ensure subdir exists (append)
            base = os.path.splitext(fn)[0]
            subdir = os.path.join(self.output_dir, base)
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
            for pct, strength in self.noise_specs:
                attack_name = f"noise_{pct}pct_strength_{strength:.2f}.geojson"
                try:
                    gdf_att = self.apply_noise_attack(gdf, pct, strength)
                    saved = self.save_attacked(gdf_att, fn, attack_name, subdir)
                    if saved:
                        total += 1
                except Exception as e:
                    print(f"应用攻击 {attack_name} 时出错: {e}")
                    continue
        print(f"完成：共生成噪声攻击 {total} 个（append 模式）")


def main():
    random.seed(42)
    np.random.seed(42)
    gen = NoiseOnlyGenerator()
    gen.generate_noise_only()


if __name__ == "__main__":
    main()






