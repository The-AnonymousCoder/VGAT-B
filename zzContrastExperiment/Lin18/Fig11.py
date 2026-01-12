#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig11：Lin18 对比试验（组合攻击：10种固定组合）
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
    from shapely.geometry import LineString, Polygon, box
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
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'combined' / 'Fig11_combined'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig11'
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

COMBINATIONS = [
    '删点10%',
    '增点强度1比例50%',
    '删对象50%',
    '噪声强度0.8比例50%',
    '裁剪左10%',
    '平移右1%',
    '缩放0.9',
    '旋转45度',
    'X镜像',
    '顺序: 反转顶点→反转对象'
]


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


def apply_combination(gdf, combo: str):
    random.seed(42)
    np.random.seed(42)
    result = gdf.copy()
    
    try:
        if combo == '删点10%':
            def del_vertices(geom):
                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                    if len(coords) > 2:
                        n_del = max(1, int((len(coords) - 2) * 0.1))
                        idx = list(range(1, len(coords) - 1))
                        to_del = set(random.sample(idx, min(n_del, len(idx))))
                        return LineString([coords[0]] + [c for i, c in enumerate(coords[1:-1], 1) if i not in to_del] + [coords[-1]])
                elif isinstance(geom, Polygon):
                    ext = list(geom.exterior.coords)
                    if len(ext) > 4:
                        n_del = max(1, int((len(ext) - 4) * 0.1))
                        idx = list(range(1, len(ext) - 2))
                        to_del = set(random.sample(idx, min(n_del, len(idx))))
                        new_ext = [ext[0]] + [c for i, c in enumerate(ext[1:-2], 1) if i not in to_del] + [ext[-2], ext[-1]]
                        # 处理内环
                        holes = []
                        for ring in geom.interiors:
                            ring_coords = list(ring.coords)
                            if len(ring_coords) > 4:
                                n_h = max(1, int((len(ring_coords) - 4) * 0.1))
                                idx_h = list(range(1, len(ring_coords) - 2))
                                to_del_h = set(random.sample(idx_h, min(n_h, len(idx_h))))
                                holes.append([ring_coords[0]] + [ring_coords[i] for i in range(1, len(ring_coords) - 2) if i not in to_del_h] + [ring_coords[-2], ring_coords[-1]])
                            else:
                                holes.append(ring_coords)
                        return Polygon(new_ext, holes=holes if holes else None)
                return geom
            result['geometry'] = result['geometry'].apply(del_vertices)
        
        elif combo == '增点强度1比例50%':
            def add_vertices(geom):
                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                    new_coords = []
                    for i in range(len(coords) - 1):
                        new_coords.append(coords[i])
                        mid = ((coords[i][0] + coords[i+1][0]) / 2, (coords[i][1] + coords[i+1][1]) / 2)
                        noise = np.random.normal(0, 0.01, 2)
                        new_coords.append((mid[0] + noise[0], mid[1] + noise[1]))
                    new_coords.append(coords[-1])
                    return LineString(new_coords)
                return geom
            result['geometry'] = result['geometry'].apply(add_vertices)
        
        elif combo == '删对象50%':
            n_del = max(1, int(len(result) * 0.5))
            indices = random.sample(range(len(result)), n_del)
            result = result.drop(indices).reset_index(drop=True)
        
        elif combo == '噪声强度0.8比例50%':
            def add_noise(geom):
                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                    n_noise = max(1, int(len(coords) * 0.5))
                    idx = random.sample(range(len(coords)), n_noise)
                    return LineString([((c[0] + np.random.normal(0, 0.8)) if i in idx else c[0], 
                                       (c[1] + np.random.normal(0, 0.8)) if i in idx else c[1]) for i, c in enumerate(coords)])
                return geom
            result['geometry'] = result['geometry'].apply(add_noise)
        
        elif combo == '裁剪左10%':
            bounds = result.total_bounds
            crop_box = box(bounds[0] + (bounds[2] - bounds[0]) * 0.1, bounds[1], bounds[2], bounds[3])
            result = result[result.intersects(crop_box)].copy()
        
        elif combo == '平移右1%':
            bounds = result.total_bounds
            xoff = (bounds[2] - bounds[0]) * 0.01
            result['geometry'] = result['geometry'].apply(lambda g: affinity.translate(g, xoff=xoff, yoff=0))
        
        elif combo == '缩放0.9':
            bounds = result.total_bounds
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            result['geometry'] = result['geometry'].apply(lambda g: affinity.scale(g, xfact=0.9, yfact=0.9, origin=(cx, cy)))
        
        elif combo == '旋转45度':
            bounds = result.total_bounds
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            result['geometry'] = result['geometry'].apply(lambda g: affinity.rotate(g, 45, origin=(cx, cy)))
        
        elif combo == 'X镜像':
            bounds = result.total_bounds
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            result['geometry'] = result['geometry'].apply(lambda g: affinity.scale(g, xfact=1, yfact=-1, origin=(cx, cy)))
        
        elif combo == '顺序: 反转顶点→反转对象':
            # Step 1: 反转顶点顺序
            def reverse_vertices(geom):
                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                    return LineString(coords[::-1])
                elif isinstance(geom, Polygon):
                    ext = list(geom.exterior.coords)
                    holes = [list(ring.coords[::-1]) for ring in geom.interiors]
                    return Polygon(ext[::-1], holes=holes if holes else None)
                return geom
            result['geometry'] = result['geometry'].apply(reverse_vertices)
            
            # Step 2: 反转对象顺序
            result = result.iloc[::-1].reset_index(drop=True)
    
    except Exception:
        pass
    
    return result


def generate_attacks(shp_files: List[Path], force_regenerate: bool = False) -> Dict[str, Dict[str, Path]]:
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        return {}
    outputs: Dict[str, Dict[str, Path]] = {}
    
    for src in shp_files:
        base = src.stem
        subdir = DIR_ATTACKED / base
        subdir.mkdir(parents=True, exist_ok=True)
        try:
            gdf = gpd.read_file(src)
        except Exception:
            continue
        outputs[base] = {}
        
        for combo in COMBINATIONS:
            try:
                safe_name = combo.replace('%', 'pct').replace('度', 'deg').replace(' ', '_').replace(':', '')
                out_path = subdir / f'combo_{safe_name}.shp'
                if not force_regenerate and KEEP_EXISTING and out_path.exists():
                    print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                    outputs[base][combo] = out_path
                    continue
                attacked = apply_combination(gdf, combo)
                attacked.to_file(out_path, driver='ESRI Shapefile')
                outputs[base][combo] = out_path
            except Exception:
                continue
    return outputs


def evaluate_nc(attacked_map: Dict[str, Dict[str, Path]], force_regenerate: bool = False):
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if not force_regenerate and KEEP_EXISTING:
        csv_path = DIR_RESULTS / 'fig11_combined_nc.csv'
        if csv_path.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 NC 评估: {csv_path}')
            return
    rows = []
    for base, c_map in attacked_map.items():
        for combo, shp_path in c_map.items():
            try:
                _, error, nc = extract(str(shp_path), str(CAT32_PATH))
                rows.append({'base': base, 'combo': combo, 'nc': float(nc)})
            except Exception:
                continue
    
    if not rows:
        return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    for combo in COMBINATIONS:
        hierarchical_rows.append({'组合攻击': combo, 'Lin18': '', '类型': 'header'})
        for base in sorted(df['base'].unique()):
            match = df[(df['base'] == base) & (df['combo'] == combo)]
            nc_value = float(match['nc'].iloc[0]) if len(match) > 0 else 0.0
            hierarchical_rows.append({'组合攻击': f'  {base}', 'Lin18': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['combo'] == combo]['nc'].mean())
        hierarchical_rows.append({'组合攻击': '  Average', 'Lin18': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig11_combined_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    try:
        with pd.ExcelWriter(DIR_RESULTS / 'fig11_combined_nc.xlsx', engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception:
        pass
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(COMBINATIONS))
    for base, sub in df.groupby('base'):
        nc_vals = [float(sub[sub['combo'] == c]['nc'].iloc[0]) if len(sub[sub['combo'] == c]) > 0 else 0.0 for c in COMBINATIONS]
        plt.plot(x_pos, nc_vals, '-o', alpha=0.7, label=base, markersize=4)
    avg_vals = [float(df[df['combo'] == c]['nc'].mean()) for c in COMBINATIONS]
    plt.plot(x_pos, avg_vals, 'k-o', linewidth=2.5, label='平均', markersize=6)
    plt.xticks(x_pos, COMBINATIONS, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.ylabel('NC')
    plt.title('Fig11：Lin18 组合攻击 NC鲁棒性')
    plt.legend(loc='best', fontsize=7)
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / 'fig11_combined_nc.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print('=== Fig11：Lin18 组合攻击 ===')
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



