# -*- coding: utf-8 -*-
"""
Wu25/Fig7.py - 缩放攻击（0.1, 0.5, 0.9, 1.3, 1.7, 2.1）NC评估（复用 Wu25 提取与数据管线）
- 数据：`embed/` 下 8 个 `Cat32_*.shp`
- 结果：输出到 `NC-Results/Fig7/` 下的 CSV、XLSX、PNG
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.affinity import scale as shp_scale  # type: ignore
from extract import extract

# 中文显示
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig7'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'scale' / 'Fig7_scale'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

SCALE_FACTORS = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]


def generate_attacked_variants(src_path: Path) -> Dict[float, Path]:
    base = src_path.stem
    subdir = DIR_ATTACKED / base
    if subdir.exists():
        import shutil
        shutil.rmtree(subdir, ignore_errors=True)
    subdir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(str(src_path))
    outputs: Dict[float, Path] = {}
    base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'

    # ✅ 使用全局中心作为缩放原点，与zNC-Test/Fig7策略一致
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    global_center_x = (bounds[0] + bounds[2]) / 2
    global_center_y = (bounds[1] + bounds[3]) / 2
    global_center = (global_center_x, global_center_y)

    def _safe_scale(geom, s):
        try:
            if geom is None:
                return None
            return shp_scale(geom, s, s, origin=global_center)
        except Exception:
            return geom

    for s in SCALE_FACTORS:
        attacked = gdf.copy()
        attacked['geometry'] = attacked['geometry'].apply(lambda geom: _safe_scale(geom, s))
        try:
            attacked.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
        except Exception:
            pass
        out_path = subdir / f'scale_{int(round(s*100))}pct.shp'
        attacked.to_file(str(out_path), driver='ESRI Shapefile')
        outputs[s] = out_path
    return outputs


def run_evaluation():
    print('=== Wu25 Fig7：缩放攻击 NC 评估 ===')
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig7_scale_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig7: {csv_check}')
            return
    if not (SCRIPT_DIR / WATERMARK).exists():
        print('缺少水印图像: ', WATERMARK)
        return
    if len(VECTOR_FILES) == 0:
        print('未发现嵌入矢量：请先在 embed/ 生成 Cat32_*.shp')
        return

    rows: List[Dict] = []  # type: ignore
    for vector_file, file_name in zip(VECTOR_FILES, FILE_NAMES):
        print(f'处理: {file_name}')
        attacked_map = generate_attacked_variants(Path(vector_file))
        for s, shp_path in sorted(attacked_map.items()):
            try:
                _, eva = extract(str(shp_path), str(SCRIPT_DIR / WATERMARK))
                nc_value = float(eva.get('NC', 0.0))
                rows.append({'file': file_name, 'scale': s, 'nc': nc_value})
                print(f'  scale={s:.2f} -> NC={nc_value:.4f}')
            except Exception as exc:
                print('  失败:', file_name, s, exc)
                rows.append({'file': file_name, 'scale': s, 'nc': 0.0})

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 防御性检查：避免 rows 全部失败导致后续通过 df['file'] 抛出 KeyError
    if df.empty or 'file' not in df.columns:
        print('没有有效的评估结果，跳过 Fig7 的汇总与绘图（rows 长度:', len(rows), ')')
        print('=== 完成（无有效结果） ===')
        return
    order_scales = SCALE_FACTORS

    hierarchical_rows: List[dict] = []
    base_names = sorted(df['file'].unique())
    for s in order_scales:
        label = f'{int(round(s*100))}%'
        hierarchical_rows.append({'缩放比例': label, 'VGAT': '', '类型': 'header'})
        for name in base_names:
            sub = df[(df['file'] == name) & (df['scale'] == s)]
            nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
            hierarchical_rows.append({'缩放比例': f'  {name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['scale'] == s]['nc'].mean())
        hierarchical_rows.append({'缩放比例': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    csv_path = DIR_RESULTS / 'fig7_scale_nc.csv'
    xlsx_path = DIR_RESULTS / 'fig7_scale_nc.xlsx'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception as e:
        print('Excel保存警告:', e)

    # 绘图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(order_scales))
    labels = [f'{int(round(s*100))}%' for s in order_scales]
    for name, sub in df.groupby('file'):
        y = []
        for s in order_scales:
            val = sub[sub['scale'] == s]['nc']
            y.append(val.iloc[0] if len(val) > 0 else np.nan)
        plt.plot(x, y, '-o', alpha=0.7, label=name)
    avg_vals = [df[df['scale'] == s]['nc'].mean() for s in order_scales]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, labels)
    plt.xlabel('缩放比例')
    plt.ylabel('NC')
    plt.title('缩放攻击的NC鲁棒性（Wu25/Fig7）')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig7_scale_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存:', csv_path)
    print('Excel保存:', xlsx_path)
    print('曲线图保存:', fig_path)
    print('=== 完成 ===')


if __name__ == '__main__':
    run_evaluation()