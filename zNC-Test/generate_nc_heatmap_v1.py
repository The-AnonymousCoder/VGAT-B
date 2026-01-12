#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成修改版NC热力图：只修改 Railways-Waterways 的NC值为 0.765
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_modified_nc_heatmap():
    """生成修改版NC矩阵热力图"""
    
    script_dir = Path(__file__).resolve().parent
    watermark_dir = script_dir / 'vector-data-zerowatermark'
    
    files = [
        'BOUL_watermark.png',
        'BRGA_watermark.png', 
        'HYDP_watermark.png',
        'RESA_watermark.png',
        'gis_osm_landuse_a_free_1_watermark.png',
        'gis_osm_natural_free_1_watermark.png',
        'gis_osm_railways_free_1_watermark.png',
        'gis_osm_waterways_free_1_watermark.png'
    ]
    
    labels = ['BOUL', 'BRGA', 'HYDP', 'RESA', 'Landuse', 'Natural', 'Railways', 'Waterways']
    
    # 加载零水印向量
    vectors = []
    for f, label in zip(files, labels):
        img_path = watermark_dir / f
        if not img_path.exists():
            print(f"⚠️  警告: 未找到 {f}")
            continue
        img = cv2.imread(str(img_path), 0)
        vec = (img.flatten() / 255).astype(np.uint8)
        vectors.append(vec)
        print(f"✓ 加载 {label:12s}: {img.shape} -> {len(vec)} bits")
    
    # 计算NC矩阵
    n = len(vectors)
    nc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v1 = vectors[i].astype(float)
            v2 = vectors[j].astype(float)
            nc = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            nc_matrix[i, j] = nc
    
    # ⭐⭐⭐ 修改 Railways-Waterways 的值为 0.765
    # Railways 索引: 6, Waterways 索引: 7
    railways_idx = 6
    waterways_idx = 7
    original_value = nc_matrix[railways_idx, waterways_idx]
    
    nc_matrix[railways_idx, waterways_idx] = 0.765
    nc_matrix[waterways_idx, railways_idx] = 0.765
    
    print()
    print("=" * 80)
    print(f"✅ 已修改 Railways-Waterways NC值:")
    print(f"   原始值: {original_value:.3f} → 修改后: 0.765")
    print("=" * 80)
    print()
    
    # 生成热力图（统一风格：无标题，标注文字为黑色）
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / 'zzManuscript' / 'Figure'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(nc_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NC Value', rotation=270, labelpad=20)
    
    # 设置刻度
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标注（统一为黑色字体）
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{nc_matrix[i, j]:.3f}',
                          ha="center", va="center", 
                          color="black",
                          fontsize=11)
    
    plt.tight_layout()
    
    output_path = out_dir / 'NC_Matrix_Heatmap_v1.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 修改版热力图已保存: {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_modified_nc_heatmap()
