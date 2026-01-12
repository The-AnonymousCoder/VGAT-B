#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig7：Lin18 对比试验（缩放攻击：0.1, 0.5, 0.9, 1.3, 1.7, 2.1）
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

try:
    import geopandas as gpd
    from shapely import affinity
except Exception:
    gpd = None

from extract import extract

try:
    preferred_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fname in preferred_fonts:
        if fname in available:
            rcParams['font.sans-serif'] = [fname]
            break
    rcParams['axes.unicode_minus'] = False
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
import os
KEEP_EXISTING = True
PROJECT_ROOT = SCRIPT_DIR.resolve().parents[1]
DIR_PSO = Path(os.environ.get('PSO_DATA_DIR', str(PROJECT_ROOT / 'zNC-Test' / 'vector-data')))
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'scale' / 'Fig7_scale'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig7'
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

SCALE_FACTORS = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]


def discover_inputs() -> List[Path]:
    # Prefer embedded watermarked files (embed/) if present; otherwise fall back to original data DIR_PSO
    embed_dir = SCRIPT_DIR / 'embed'
    search_dir = embed_dir if embed_dir.exists() and any(embed_dir.glob('*.shp')) else DIR_PSO
    if not Path(search_dir).exists():
        return []
    shp_files = [p for p in sorted(Path(search_dir).glob('*.shp')) if not p.name.startswith('._')]
    valid = []
    for p in shp_files:
        if not (p.with_suffix('.dbf').exists() and p.with_suffix('.shx').exists()):
            continue
        # 过滤点矢量数据（Lin18 不支持 Point/MultiPoint）
        try:
            gdf = gpd.read_file(str(p))
            geom_types = gdf.geometry.geom_type.unique()
            if any(gt in ['Point', 'MultiPoint'] for gt in geom_types):
                print(f'跳过点矢量: {p.stem}')
                continue
            valid.append(p)
        except Exception:
            continue
    return valid[:6]


def scale_gdf(gdf, factor: float):
    bounds = gdf.total_bounds
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    scaled = gdf.copy()
    scaled['geometry'] = scaled['geometry'].apply(lambda g: affinity.scale(g, xfact=factor, yfact=factor, origin=(cx, cy)))
    return scaled


def generate_attacks(shp_files: List[Path], force_regenerate: bool = False) -> Dict[str, Dict[float, Path]]:
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        return {}
    outputs: Dict[str, Dict[float, Path]] = {}
    
    for src in shp_files:
        base = src.stem
        subdir = DIR_ATTACKED / base
        subdir.mkdir(parents=True, exist_ok=True)
        try:
            gdf = gpd.read_file(src)
        except Exception:
            continue
        outputs[base] = {}
        for factor in SCALE_FACTORS:
            try:
                out_path = subdir / f'scale_{int(factor*10)}.shp'
                if not force_regenerate and KEEP_EXISTING and out_path.exists():
                    print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                    outputs[base][factor] = out_path
                    continue
                attacked = scale_gdf(gdf, factor)
                attacked.to_file(out_path, driver='ESRI Shapefile')
                outputs[base][factor] = out_path
            except Exception:
                continue
    return outputs


def evaluate_nc(attacked_map: Dict[str, Dict[str, Path]], force_regenerate: bool = False):
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if not force_regenerate and KEEP_EXISTING:
        csv_path = DIR_RESULTS / 'fig7_scale_nc.csv'
        if csv_path.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 NC 评估: {csv_path}')
            return
    rows = []
    for base, f_map in attacked_map.items():
        for factor, shp_path in sorted(f_map.items()):
            try:
                _, error, nc = extract(str(shp_path), str(CAT32_PATH))
                rows.append({'base': base, 'factor': factor, 'nc': float(nc)})
            except Exception:
                continue
    
    if not rows:
        return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    for factor in SCALE_FACTORS:
        hierarchical_rows.append({'缩放因子': f'{factor}', 'Lin18': '', '类型': 'header'})
        for base in sorted(df['base'].unique()):
            match = df[(df['base'] == base) & (df['factor'] == factor)]
            nc_value = float(match['nc'].iloc[0]) if len(match) > 0 else 0.0
            hierarchical_rows.append({'缩放因子': f'  {base}', 'Lin18': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['factor'] == factor]['nc'].mean())
        hierarchical_rows.append({'缩放因子': '  Average', 'Lin18': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig7_scale_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    try:
        with pd.ExcelWriter(DIR_RESULTS / 'fig7_scale_nc.xlsx', engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception:
        pass
    
    plt.figure(figsize=(10, 6))
    for base, sub in df.groupby('base'):
        sub2 = sub.sort_values('factor')
        plt.plot(sub2['factor'], sub2['nc'], '-o', alpha=0.7, label=base)
    avg = df.groupby('factor')['nc'].mean().reset_index()
    plt.plot(avg['factor'], avg['nc'], 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xlabel('缩放因子')
    plt.ylabel('NC')
    plt.title('Fig7：Lin18 缩放攻击 NC鲁棒性')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / 'fig7_scale_nc.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print('=== Fig7：Lin18 缩放攻击 ===')
    inputs = discover_inputs()
    if not inputs:
        return
    attacked_map = generate_attacks(inputs)
    evaluate_nc(attacked_map)
    print('=== 完成 ===')


if __name__ == '__main__':
    import sys
    force = '--force' in sys.argv or '-f' in sys.argv
    main(force_regenerate=force)



