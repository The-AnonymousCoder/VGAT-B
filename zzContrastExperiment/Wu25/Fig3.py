# -*- coding: utf-8 -*-
"""
Wu25/Fig3.py - 对象删除攻击的NC评估（可逆水印，复用 Wu25 的提取与攻击管线）
- 攻击：删除比例 0.0, 0.1, 0.3, 0.5, 0.7, 0.9
- 数据：使用 `embed/` 下 6 个含水印矢量文件
- 结果：输出到 `NC-Results/Fig3/` 下的 CSV、XLSX、PNG
- 结构与 Tan24/zNC-Test 的 Fig3 保持一致的层次化展示
"""

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import geopandas as gpd

from extract import extract

try:
    preferred_fonts = [
        'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC',
        'Heiti SC', 'Hiragino Sans GB', 'Source Han Sans CN', 'STHeiti',
        'Arial Unicode MS'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for fname in preferred_fonts:
        if fname in available:
            chosen = fname
            break
    if chosen:
        rcParams['font.sans-serif'] = [chosen]
    rcParams['axes.unicode_minus'] = False
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig3'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'

# 自动发现 embed 目录 Cat32_*.shp（适配 8 数据集）
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

# 与 Tan24 对齐：删除对象比例 10%~90%
DELETE_RATIOS = [i / 100 for i in range(10, 100, 10)]


def run_evaluation():
    print('=== Wu25 Fig3：对象删除攻击 NC 评估 ===')
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig3_delete_objects_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig3: {csv_check}')
            return
    if not (SCRIPT_DIR / WATERMARK).exists():
        print('缺少必要文件: Cat32.png，请先生成水印')
        return
    if len(VECTOR_FILES) == 0:
        print('未发现嵌入矢量：请先运行 embed.py（期望在 embed/ 下生成 Cat32_*.shp）')
        return

    rows = []
    nc_matrix = np.zeros((len(FILE_NAMES), len(DELETE_RATIOS)))

    def generate_delete_objects_variant(src_path: Path, pct: int) -> Path:
        base = src_path.stem
        subdir = (SCRIPT_DIR / 'attacked' / 'delete_objects' / 'Fig3_delete_objects' / base)
        subdir.mkdir(parents=True, exist_ok=True)
        gdf = gpd.read_file(str(src_path))
        attacked = gdf.copy()
        n_total = len(gdf)
        num_to_delete = int(n_total * pct / 100)
        if num_to_delete > 0 and n_total > 0:
            import random
            random.seed(42)
            indices = list(range(n_total))
            to_del = random.sample(indices, min(num_to_delete, len(indices)))
            attacked = attacked.drop(to_del).reset_index(drop=True)
        if getattr(gdf, 'crs', None) is not None:
            try:
                attacked.set_crs(gdf.crs, allow_override=True, inplace=True)  # type: ignore
            except Exception:
                pass
        out_path = subdir / f'delete_{pct}pct_objects.shp'
        if KEEP_EXISTING and out_path.exists():
            print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
            return out_path
        attacked.to_file(str(out_path), driver='ESRI Shapefile')
        return out_path

    for file_idx, (vector_file, file_name) in enumerate(zip(VECTOR_FILES, FILE_NAMES)):
        abs_vector = str(vector_file)
        print(f'处理: {file_name}')
        for ratio_idx, delete_ratio in enumerate(DELETE_RATIOS):
            try:
                pct = int(round(delete_ratio * 100))
                attacked_path = generate_delete_objects_variant(Path(abs_vector), pct)
                _, eva = extract(str(attacked_path), str(SCRIPT_DIR / WATERMARK))
                nc_value = float(eva.get('NC', 0.0))
                ber_value = float(eva.get('BER', 1.0))
                rows.append({
                    'file': file_name,
                    'delete_ratio': delete_ratio,
                    'nc': nc_value,
                    'ber': ber_value
                })
                nc_matrix[file_idx, ratio_idx] = nc_value
                print(f'  ratio={delete_ratio:.1f} -> NC={nc_value:.4f}, BER={ber_value:.4f}')
            except Exception as exc:
                print('  失败:', file_name, delete_ratio, exc)
                nc_matrix[file_idx, ratio_idx] = 0.0

    df = pd.DataFrame(rows)

    # 防御性检查：避免 rows 全部失败导致后续通过 df['file'] 抛出 KeyError
    if df.empty or 'file' not in df.columns:
        print('没有有效的评估结果，跳过 Fig3 的汇总与绘图（rows 长度:', len(rows), ')')
        print('=== 完成（无有效结果） ===')
        return

    # 层次结构输出
    hierarchical_rows: List[dict] = []
    for ratio in DELETE_RATIOS:
        pct = int(round(ratio * 100))
        hierarchical_rows.append({'对象删除': f'比例: {pct}%', 'VGAT': '', '类型': 'subheader'})
        for file_name in FILE_NAMES:
            sub = df[(df['file'] == file_name) & (df['delete_ratio'] == ratio)]
            nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
            hierarchical_rows.append({'对象删除': f'  {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['delete_ratio'] == ratio]['nc'].mean()) if not df.empty else 0.0
        hierarchical_rows.append({'对象删除': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV/XLSX
    csv_path = DIR_RESULTS / 'fig3_delete_objects_nc.csv'
    xlsx_path = DIR_RESULTS / 'fig3_delete_objects_nc.xlsx'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception as e:
        print('Excel保存警告:', e)

    # 绘图
    plt.figure(figsize=(10, 6))
    for file_idx, file_name in enumerate(FILE_NAMES):
        plt.plot(DELETE_RATIOS, nc_matrix[file_idx, :], '-o', label=file_name)
    avg_curve = np.mean(nc_matrix, axis=0)
    plt.plot(DELETE_RATIOS, avg_curve, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xlabel('删除比例')
    plt.ylabel('NC')
    plt.title('对象删除攻击的NC鲁棒性（Wu25/Fig3）')
    plt.ylim(0, 1.05)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig3_delete_objects_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存:', csv_path)
    print('Excel保存:', xlsx_path)
    print('曲线图保存:', fig_path)
    print('=== 完成 ===')


if __name__ == '__main__':
    run_evaluation()