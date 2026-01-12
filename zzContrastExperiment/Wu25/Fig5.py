# -*- coding: utf-8 -*-
"""
Wu25/Fig5.py - 裁剪攻击（5种方式）NC评估（复用 Wu25 提取与数据管线）
- 攻击：
  - x_center_50pct: 沿X轴中心裁剪50%
  - y_center_50pct: 沿Y轴中心裁剪50%
  - top_left: 裁剪左上角
  - bottom_right: 裁剪右下角
  - random_40pct: 随机保留60%对象
- 数据：`embed/` 下 8 个 `Cat32_*.shp`
- 结果：输出到 `NC-Results/Fig5/` 下的 CSV、XLSX、PNG
"""

from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from extract import extract

# 中文显示（优先选择可用的中文字体，避免乱码）
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig5'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'crop' / 'Fig5_crop'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

# 裁剪方式定义
CROP_TYPES: List[Tuple[str, str]] = [
    ("x_center_50pct", "沿X轴中心裁剪50%"),
    ("y_center_50pct", "沿Y轴中心裁剪50%"),
    ("top_left", "裁剪左上角"),
    ("bottom_right", "裁剪右下角"),
    ("random_40pct", "随机裁剪40%对象"),
]


def generate_attacked_variants(src_path: Path) -> Dict[str, Path]:
    """针对单个源矢量生成五种裁剪攻击，返回 {key: shapefile_path}."""
    base = src_path.stem
    subdir = DIR_ATTACKED / base
    if subdir.exists():
        import shutil
        shutil.rmtree(subdir, ignore_errors=True)
    subdir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(str(src_path))
    # 裁剪辅助信息
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    mid_x = (bounds[0] + bounds[2]) / 2
    mid_y = (bounds[1] + bounds[3]) / 2
    bdf = gdf.geometry.bounds

    outputs: Dict[str, Path] = {}
    base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'

    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    for key, _label in CROP_TYPES:
        attacked = gdf.copy()
        if key == 'x_center_50pct':
            attacked = attacked[bdf['minx'] < mid_x]
        elif key == 'y_center_50pct':
            attacked = attacked[bdf['miny'] < mid_y]
        elif key == 'top_left':
            attacked = attacked[(bdf['minx'] < mid_x) & (bdf['miny'] > mid_y)]
        elif key == 'bottom_right':
            attacked = attacked[(bdf['minx'] > mid_x) & (bdf['miny'] < mid_y)]
        elif key == 'random_40pct':
            num_objects = len(attacked)
            num_to_keep = int(num_objects * 0.6)
            if num_to_keep > 0 and num_objects > 0:
                idx_keep = random.sample(range(num_objects), min(num_to_keep, num_objects))
                attacked = attacked.iloc[idx_keep].reset_index(drop=True)

        try:
            attacked.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
        except Exception:
            pass

        out_path = subdir / f'crop_{key}.shp'
        attacked.to_file(str(out_path), driver='ESRI Shapefile')
        outputs[key] = out_path
    return outputs


def run_evaluation():
    print('=== Wu25 Fig5：裁剪攻击 NC 评估 ===')
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig5_crop_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig5: {csv_check}')
            return
    if not (SCRIPT_DIR / WATERMARK).exists():
        print('缺少水印图像: ', WATERMARK)
        return
    if len(VECTOR_FILES) == 0:
        print('未发现嵌入矢量：请先在 embed/ 生成 Cat32_*.shp')
        return

    rows: List[Dict] = []

    for vector_file, file_name in zip(VECTOR_FILES, FILE_NAMES):
        print(f'处理: {file_name}')
        attacked_map = generate_attacked_variants(Path(vector_file))
        for key, shp_path in sorted(attacked_map.items()):
            try:
                _, eva = extract(str(shp_path), str(SCRIPT_DIR / WATERMARK))
                nc_value = float(eva.get('NC', 0.0))
                rows.append({'file': file_name, 'crop_key': key, 'nc': nc_value})
                print(f'  {key} -> NC={nc_value:.4f}')
            except Exception as exc:
                print('  失败:', file_name, key, exc)
                rows.append({'file': file_name, 'crop_key': key, 'nc': 0.0})

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 防御性检查：避免 rows 全部失败导致后续通过 df['file'] 抛出 KeyError
    if df.empty or 'file' not in df.columns:
        print('没有有效的评估结果，跳过 Fig5 的汇总与绘图（rows 长度:', len(rows), ')')
        print('=== 完成（无有效结果） ===')
        return

    # 层次结构输出：裁剪方式 -> 各图 -> 平均
    key_to_label = {k: lbl for k, lbl in CROP_TYPES}
    order_keys = [k for k, _ in CROP_TYPES]
    hierarchical_rows: List[dict] = []
    base_names = sorted(df['file'].unique())
    for k in order_keys:
        hierarchical_rows.append({'裁剪方式': key_to_label.get(k, k), 'VGAT': '', '类型': 'header'})
        for name in base_names:
            sub = df[(df['file'] == name) & (df['crop_key'] == k)]
            nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
            hierarchical_rows.append({'裁剪方式': f'  {name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['crop_key'] == k]['nc'].mean())
        hierarchical_rows.append({'裁剪方式': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV/XLSX
    csv_path = DIR_RESULTS / 'fig5_crop_nc.csv'
    xlsx_path = DIR_RESULTS / 'fig5_crop_nc.xlsx'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception as e:
        print('Excel保存警告:', e)

    # 绘图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(order_keys))
    labels = [key_to_label.get(k, k) for k in order_keys]
    for name, sub in df.groupby('file'):
        y = []
        for k in order_keys:
            val = sub[sub['crop_key'] == k]['nc']
            y.append(val.iloc[0] if len(val) > 0 else np.nan)
        plt.plot(x, y, '-o', alpha=0.7, label=name)
    avg_vals = [df[df['crop_key'] == k]['nc'].mean() for k in order_keys]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, labels, rotation=20)
    plt.xlabel('裁剪方式')
    plt.ylabel('NC')
    plt.title('裁剪攻击的NC鲁棒性（Wu25/Fig5）')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig5_crop_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存:', csv_path)
    print('Excel保存:', xlsx_path)
    print('曲线图保存:', fig_path)
    print('=== 完成 ===')


if __name__ == '__main__':
    run_evaluation()