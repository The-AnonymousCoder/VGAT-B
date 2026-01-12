#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只转换 TestSet 下已生成的噪声攻击 GeoJSON 为图（Noise-only）
行为：
 - 遍历 convertToGeoJson-Attacked/GeoJson-Attacked/TestSet/<original> 下的 noise_*.geojson
 - 跳过过大文件（基于 converter.max_file_size_bytes）
 - 若目标 graph 已存在则跳过（append）
 - 使用训练集的 global_scaler.pkl（由 TestSet 转换器加载）
 - 转换逻辑直接复用 convertToGraph-TestSet-IMPROVED.py 中的 ImprovedTestSetVectorToGraphConverter
"""

import os
import re
import importlib.util
import geopandas as gpd
from pathlib import Path


def load_converter_module(script_path):
    spec = importlib.util.spec_from_file_location("testset_converter_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # path to the existing TestSet converter script
    conv_script = os.path.join(base_dir, "convertToGraph-TestSet-IMPROVED.py")
    if not os.path.exists(conv_script):
        raise FileNotFoundError(f"无法找到转换器脚本: {conv_script}")

    module = load_converter_module(conv_script)
    ConverterClass = getattr(module, "ImprovedTestSetVectorToGraphConverter")

    converter = ConverterClass()

    # attacked geojson root (where noise files were generated)
    attacked_root = os.path.abspath(os.path.join(base_dir, "..", "convertToGeoJson-Attacked", "GeoJson-Attacked", "TestSet"))
    if not os.path.exists(attacked_root):
        print("未找到 attacked 根目录:", attacked_root)
        return

    noise_pattern = re.compile(r'noise_\d+pct_strength_[0-9]+(?:\.[0-9]+)?\.geojson$', re.IGNORECASE)

    total_converted = 0
    total_skipped = 0

    for subdir in sorted(os.listdir(attacked_root)):
        subdir_path = os.path.join(attacked_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        print(f"\n处理子目录: {subdir}")
        for fname in sorted(os.listdir(subdir_path)):
            if noise_pattern.search(fname):
                file_path = os.path.join(subdir_path, fname)
                try:
                    # skip large files
                    try:
                        if os.path.getsize(file_path) > converter.max_file_size_bytes:
                            print(f"⏭️ 跳过过大文件: {fname} ({os.path.getsize(file_path)//1024} KB)")
                            total_skipped += 1
                            continue
                    except Exception:
                        pass

                    graph_name = os.path.splitext(fname)[0]
                    target_save = os.path.join(converter.graph_dir, "Attacked", subdir, f"{graph_name}_graph.pkl")
                    if os.path.exists(target_save):
                        print(f"⏭️ 图已存在，跳过: {subdir}/{graph_name}_graph.pkl")
                        total_skipped += 1
                        continue

                    # load geojson
                    gdf = gpd.read_file(file_path)
                    if gdf is None or len(gdf) == 0:
                        print(f"❌ 空数据，跳过: {fname}")
                        total_skipped += 1
                        continue

                    data = converter.build_graph_from_gdf(gdf, fname)

                    # ensure attacked save dir exists
                    out_dir = os.path.join(converter.graph_dir, "Attacked", subdir)
                    os.makedirs(out_dir, exist_ok=True)

                    # save (append behavior)
                    if os.path.exists(target_save):
                        print(f"⏭️ 目标已存在，跳过保存: {target_save}")
                        total_skipped += 1
                        continue
                    with open(target_save, "wb") as f:
                        import pickle
                        pickle.dump(data, f)
                    print(f"✅ 已转换并保存: {subdir}/{graph_name}_graph.pkl")
                    total_converted += 1

                except Exception as e:
                    print(f"❌ 转换失败 {subdir}/{fname}: {e}")
                    total_skipped += 1
                    continue

    print(f"\n完成：已转换 {total_converted} 个噪声图，跳过 {total_skipped} 个（存在/过大/空数据）")


if __name__ == "__main__":
    main()






