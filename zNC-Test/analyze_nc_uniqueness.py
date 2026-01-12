#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析测试集零水印的NC唯一性
只分析NC值（归一化相关系数）
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re
import pandas as pd

def generate_label_from_filename(filename):
    """从文件名生成简洁的标签"""
    # 移除扩展名和_watermark后缀
    base = filename.replace('_watermark.png', '').replace('_watermark.npy', '')
    
    # 提取关键部分
    # 处理 H50-XXX 格式
    if base.startswith('H50-'):
        return base.replace('H50-', '')
    
    # 处理 shanghai-latest-free.shp-gis_osm_xxx 格式
    if 'gis_osm_' in base:
        match = re.search(r'gis_osm_(\w+)_', base)
        if match:
            category = match.group(1)
            # 转换为更友好的名称
            category_map = {
                'landuse': 'Landuse',
                'natural': 'Natural',
                'railways': 'Railways',
                'waterways': 'Waterways',
                'places': 'Places',
                'transport': 'Transport'
            }
            return category_map.get(category, category.capitalize())
    
    # 如果文件名太长，截取关键部分
    if len(base) > 15:
        # 尝试提取最后一个有意义的部分
        parts = base.split('-')
        if len(parts) > 1:
            return parts[-1]
        return base[:15]
    
    return base

def analyze_nc_uniqueness():
    """分析零水印NC矩阵"""
    
    script_dir = Path(__file__).resolve().parent
    watermark_dir = script_dir / 'vector-data-zerowatermark'
    
    # 自适应扫描所有水印文件（优先使用.png）
    watermark_files = {}
    for ext in ['.png', '.npy']:
        for file_path in watermark_dir.glob(f'*_watermark{ext}'):
            base_name = file_path.stem.replace('_watermark', '')
            if base_name not in watermark_files:
                watermark_files[base_name] = file_path
    
    if not watermark_files:
        print("❌ 错误: 未找到任何水印文件")
        return
    
    # 加载零水印向量
    vectors = []
    found_labels = []
    found_files = []
    
    print(f"\n📊 找到 {len(watermark_files)} 个水印文件，开始加载...\n")
    
    for base_name, file_path in sorted(watermark_files.items()):
        label = generate_label_from_filename(file_path.name)
        
        try:
            if file_path.suffix == '.png':
                img = cv2.imread(str(file_path), 0)
                if img is None:
                    print(f"⚠️  警告: 无法读取图片 {file_path.name}")
                    continue
                vec = (img.flatten() / 255).astype(np.uint8)
            else:  # .npy
                vec = np.load(file_path).astype(np.uint8)
                # 如果是2D数组，展平
                if vec.ndim > 1:
                    vec = vec.flatten()
            
            vectors.append(vec)
            found_labels.append(label)
            found_files.append(base_name)
            print(f"✓ 加载 {label:20s}: {vec.shape} -> {len(vec)} bits")
        except Exception as e:
            print(f"⚠️  警告: 加载 {file_path.name} 失败: {e}")
            continue
    
    if not vectors:
        print("❌ 错误: 没有成功加载任何水印文件")
        return
    
    n = len(vectors)
    print(f"\n📐 计算 {n}x{n} NC矩阵...\n")
    
    # 计算NC矩阵
    nc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v1 = vectors[i].astype(float)
            v2 = vectors[j].astype(float)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                nc = np.dot(v1, v2) / (norm1 * norm2)
            else:
                nc = 0.0
            nc_matrix[i, j] = nc
    
    # 提取非对角线元素
    mask = ~np.eye(n, dtype=bool)
    off_diag = nc_matrix[mask]
    
    # 统计信息
    max_off_diag_nc = float(np.max(off_diag))
    min_off_diag_nc = float(np.min(off_diag))
    mean_off_diag_nc = float(np.mean(off_diag))
    std_off_diag_nc = float(np.std(off_diag))
    median_off_diag_nc = float(np.median(off_diag))
    
    # 统计各区间的配对数
    ranges = [
        (0.0, 0.5, "极低相似"),
        (0.5, 0.75, "低相似"),
        (0.75, 0.85, "中等相似"),
        (0.85, 0.9, "高相似"),
        (0.9, 1.0, "极高相似")
    ]
    
    total_pairs = len(off_diag)
    
    # 统计不同阈值的配对数
    threshold_uniqueness = 0.82
    pairs_ge_080 = int(np.sum(off_diag >= 0.80))
    pairs_ge_082 = int(np.sum(off_diag >= 0.82))
    pairs_ge_085 = int(np.sum(off_diag >= 0.85))
    pairs_ge_090 = int(np.sum(off_diag >= 0.90))
    
    # 找出最高的NC值配对
    indices = np.triu_indices(n, k=1)
    nc_pairs = [(found_labels[i], found_labels[j], nc_matrix[i, j]) 
                for i, j in zip(indices[0], indices[1])]
    nc_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 详细分析输出
    print("\n" + "=" * 80)
    print("📊 NC唯一性分析报告")
    print("=" * 80)
    print(f"\n📈 基本统计信息:")
    print(f"   总配对数: {total_pairs}")
    print(f"   最大非对角线NC值: {max_off_diag_nc:.6f}")
    print(f"   最小非对角线NC值: {min_off_diag_nc:.6f}")
    print(f"   平均非对角线NC值: {mean_off_diag_nc:.6f}")
    print(f"   标准差: {std_off_diag_nc:.6f}")
    print(f"   中位数: {median_off_diag_nc:.6f}")
    
    print(f"\n🎯 阈值统计:")
    print(f"   NC ≥ 0.80 的配对数: {pairs_ge_080} ({pairs_ge_080/total_pairs*100:.2f}%)")
    print(f"   NC ≥ 0.82 的配对数: {pairs_ge_082} ({pairs_ge_082/total_pairs*100:.2f}%)")
    print(f"   NC ≥ 0.85 的配对数: {pairs_ge_085} ({pairs_ge_085/total_pairs*100:.2f}%)")
    print(f"   NC ≥ 0.90 的配对数: {pairs_ge_090} ({pairs_ge_090/total_pairs*100:.2f}%)")
    
    print(f"\n📊 相似度分布:")
    for low, high, desc in ranges:
        count = int(np.sum((off_diag >= low) & (off_diag < high)))
        if high == 1.0:
            count = int(np.sum(off_diag >= low))
        pct = count / total_pairs * 100 if total_pairs > 0 else 0
        print(f"   {desc:8s} [{low:.2f}-{high:.2f}): {count:3d} 对 ({pct:5.2f}%)")
    
    # 唯一性评估
    uniqueness_ok = max_off_diag_nc < threshold_uniqueness
    print(f"\n✅ 唯一性评估 (阈值={threshold_uniqueness}):")
    if uniqueness_ok:
        print(f"   ✓ 通过: 最大非对角线NC值 {max_off_diag_nc:.6f} < {threshold_uniqueness}")
    else:
        print(f"   ✗ 未通过: 最大非对角线NC值 {max_off_diag_nc:.6f} ≥ {threshold_uniqueness}")
    
    # 显示高相似度配对
    high_sim_pairs = [p for p in nc_pairs if p[2] >= 0.75]
    if high_sim_pairs:
        print(f"\n⚠️  高相似度配对 (NC ≥ 0.75):")
        for label1, label2, nc_val in high_sim_pairs[:10]:  # 只显示前10个
            print(f"   {label1:20s} <-> {label2:20s}: {nc_val:.6f}")
        if len(high_sim_pairs) > 10:
            print(f"   ... 还有 {len(high_sim_pairs) - 10} 对未显示")
    
    print("\n" + "=" * 80 + "\n")
    
    # 保存统计信息到JSON
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / 'zzManuscript' / 'Figure'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "max_off_diag_nc": max_off_diag_nc,
        "min_off_diag_nc": min_off_diag_nc,
        "mean_off_diag_nc": mean_off_diag_nc,
        "std_off_diag_nc": std_off_diag_nc,
        "median_off_diag_nc": median_off_diag_nc,
        "pairs_ge_0.80": pairs_ge_080,
        "pairs_ge_0.82": pairs_ge_082,
        "pairs_ge_0.85": pairs_ge_085,
        "pairs_ge_0.90": pairs_ge_090,
        "total_pairs": total_pairs,
        "threshold_ok": uniqueness_ok,
        "threshold_uniqueness": threshold_uniqueness,
        "high_similarity_pairs": [
            {"label1": label1, "label2": label2, "nc": float(nc_val)}
            for label1, label2, nc_val in high_sim_pairs[:20]
        ]
    }
    
    # 保存统计信息到两个位置
    stats_path = out_dir / 'nc_uniqueness_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ 统计信息已保存: {stats_path}")
    
    # 同时保存到 zNC-Test 文件夹
    script_dir = Path(__file__).resolve().parent
    local_stats_path = script_dir / 'nc_uniqueness_stats.json'
    with open(local_stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ 统计信息已保存: {local_stats_path}")
    
    # 保存NC矩阵为CSV文件（短标签）与全称CSV
    nc_df = pd.DataFrame(nc_matrix, index=found_labels, columns=found_labels)
    csv_path = script_dir / 'NC_Matrix.csv'
    nc_df.to_csv(csv_path, float_format='%.6f')
    print(f"✓ NC矩阵CSV已保存: {csv_path}")

    # 全称版本（使用原始文件基名，便于完整显示）
    nc_df_full = pd.DataFrame(nc_matrix, index=found_files, columns=found_files)
    csv_full_path = script_dir / 'NC_Matrix_full.csv'
    nc_df_full.to_csv(csv_full_path, float_format='%.6f')
    print(f"✓ NC矩阵全称CSV已保存: {csv_full_path}")

    # 保存标签映射（短标签 -> 全称）
    mapping_path = script_dir / 'label_mapping.json'
    mapping = [{"short": s, "full": f} for s, f in zip(found_labels, found_files)]
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"✓ 标签映射已保存: {mapping_path}")
    
    # 生成热力图（统一风格：无标题，标注文字为黑色）
    # 使用“全称”标签（不再缩写），直接取原始文件基名以便在热力图上完整展示
    display_labels = found_files
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(nc_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NC Value', rotation=270, labelpad=20)
    
    # 设置刻度（使用全称标签以便在热力图上完整展示矢量图名）
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels)
    ax.set_yticklabels(display_labels)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标注（统一为黑色字体）
    for i in range(len(found_labels)):
        for j in range(len(found_labels)):
            text = ax.text(j, i, f'{nc_matrix[i, j]:.3f}',
                          ha="center", va="center", 
                          color="black",
                          fontsize=11)
    
    plt.tight_layout()
    
    # 保存热力图到两个位置
    output_path = out_dir / 'NC_Matrix_Heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存: {output_path}")
    
    local_heatmap_path = script_dir / 'NC_Matrix_Heatmap.png'
    plt.savefig(local_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存: {local_heatmap_path}")
    plt.close()

if __name__ == '__main__':
    analyze_nc_uniqueness()
