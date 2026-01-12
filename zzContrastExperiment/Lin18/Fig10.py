#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig10：Lin18 对比试验（打乱顺序：反转顶点、打乱顶点、反转对象、打乱对象）
"""

from pathlib import Path
from typing import Dict, List
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon
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
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'reorder' / 'Fig10_reorder'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig10'
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

REORDER_METHODS = ['反转顶点', '打乱顶点', '反转对象', '打乱对象']


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


def reorder_vertices(geom):
    if geom is None:
        return geom
    try:
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            return LineString(coords[::-1])
        elif isinstance(geom, Polygon):
            ext = list(geom.exterior.coords)
            # 处理内环
            holes = []
            for ring in geom.interiors:
                ring_coords = list(ring.coords)
                holes.append(ring_coords[::-1])
            return Polygon(ext[::-1], holes=holes if holes else None)
    except Exception:
        pass
    return geom


def shuffle_vertices(geom):
    if geom is None:
        return geom
    try:
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) <= 2:
                return geom
            middle = coords[1:-1]
            random.shuffle(middle)
            return LineString([coords[0]] + middle + [coords[-1]])
        elif isinstance(geom, Polygon):
            ext = list(geom.exterior.coords)
            if len(ext) <= 4:
                return geom
            middle = ext[1:-2]
            random.shuffle(middle)
            # 处理内环
            holes = []
            for ring in geom.interiors:
                ring_coords = list(ring.coords)
                if len(ring_coords) > 4:
                    ring_middle = ring_coords[1:-2]
                    random.shuffle(ring_middle)
                    holes.append([ring_coords[0]] + ring_middle + [ring_coords[-2], ring_coords[-1]])
                else:
                    holes.append(ring_coords)
            return Polygon([ext[0]] + middle + [ext[-2], ext[-1]], holes=holes if holes else None)
    except Exception:
        pass
    return geom


def generate_attacks(shp_files: List[Path], force_regenerate: bool = False) -> Dict[str, Dict[str, Path]]:
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        return {}
    outputs: Dict[str, Dict[str, Path]] = {}
    random.seed(42)
    np.random.seed(42)
    
    for src in shp_files:
        base = src.stem
        subdir = DIR_ATTACKED / base
        subdir.mkdir(parents=True, exist_ok=True)
        try:
            gdf = gpd.read_file(src)
        except Exception:
            continue
        outputs[base] = {}
        
        for method in REORDER_METHODS:
            try:
                if method == '反转顶点':
                    attacked = gdf.copy()
                    attacked['geometry'] = attacked['geometry'].apply(reorder_vertices)
                elif method == '打乱顶点':
                    attacked = gdf.copy()
                    attacked['geometry'] = attacked['geometry'].apply(shuffle_vertices)
                elif method == '反转对象':
                    attacked = gdf.iloc[::-1].reset_index(drop=True)
                elif method == '打乱对象':
                    indices = list(range(len(gdf)))
                    random.shuffle(indices)
                    attacked = gdf.iloc[indices].reset_index(drop=True)
                else:
                    continue
                
                out_path = subdir / f'reorder_{method}.shp'
                if not force_regenerate and KEEP_EXISTING and out_path.exists():
                    print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                    outputs[base][method] = out_path
                    continue
                attacked.to_file(out_path, driver='ESRI Shapefile')
                outputs[base][method] = out_path
            except Exception:
                continue
    return outputs


def evaluate_nc(attacked_map: Dict[str, Dict[str, Path]], force_regenerate: bool = False):
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if not force_regenerate and KEEP_EXISTING:
        csv_path = DIR_RESULTS / 'fig10_reorder_nc.csv'
        if csv_path.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 NC 评估: {csv_path}')
            return
    rows = []
    for base, m_map in attacked_map.items():
        for method, shp_path in m_map.items():
            try:
                _, error, nc = extract(str(shp_path), str(CAT32_PATH))
                rows.append({'base': base, 'method': method, 'nc': float(nc)})
            except Exception:
                continue
    
    if not rows:
        return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    for method in REORDER_METHODS:
        hierarchical_rows.append({'打乱方式': method, 'Lin18': '', '类型': 'header'})
        for base in sorted(df['base'].unique()):
            match = df[(df['base'] == base) & (df['method'] == method)]
            nc_value = float(match['nc'].iloc[0]) if len(match) > 0 else 0.0
            hierarchical_rows.append({'打乱方式': f'  {base}', 'Lin18': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['method'] == method]['nc'].mean())
        hierarchical_rows.append({'打乱方式': '  Average', 'Lin18': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig10_reorder_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    try:
        with pd.ExcelWriter(DIR_RESULTS / 'fig10_reorder_nc.xlsx', engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception:
        pass
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(REORDER_METHODS))
    for base, sub in df.groupby('base'):
        nc_vals = [float(sub[sub['method'] == m]['nc'].iloc[0]) if len(sub[sub['method'] == m]) > 0 else 0.0 for m in REORDER_METHODS]
        plt.plot(x_pos, nc_vals, '-o', alpha=0.7, label=base)
    avg_vals = [float(df[df['method'] == m]['nc'].mean()) for m in REORDER_METHODS]
    plt.plot(x_pos, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.xticks(x_pos, REORDER_METHODS, rotation=15)
    plt.grid(True, alpha=0.3)
    plt.ylabel('NC')
    plt.title('Fig10：Lin18 打乱顺序攻击 NC鲁棒性')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / 'fig10_reorder_nc.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print('=== Fig10：Lin18 打乱顺序攻击 ===')
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



