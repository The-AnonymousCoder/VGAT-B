#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig12：Lin18 对比试验（复合攻击：顺序执行多个攻击）
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
DIR_PSO = SCRIPT_DIR / 'pso_data'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'compound' / 'Fig12_compound'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig12'
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'


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


def apply_compound(gdf):
    random.seed(42)
    np.random.seed(42)
    result = gdf.copy()
    
    try:
        # 1. 删点10%
        def del_v(g):
            if isinstance(g, LineString):
                coords = list(g.coords)
                if len(coords) > 2:
                    n = max(1, int((len(coords)-2)*0.1))
                    idx = list(range(1, len(coords)-1))
                    to_del = set(random.sample(idx, min(n, len(idx))))
                    return LineString([coords[0]] + [c for i,c in enumerate(coords[1:-1],1) if i not in to_del] + [coords[-1]])
            return g
        result['geometry'] = result['geometry'].apply(del_v)
        
        # 2. 增点
        def add_v(g):
            if isinstance(g, LineString):
                coords = list(g.coords)
                new_c = []
                for i in range(len(coords)-1):
                    new_c.append(coords[i])
                    mid = ((coords[i][0]+coords[i+1][0])/2, (coords[i][1]+coords[i+1][1])/2)
                    new_c.append(mid)
                new_c.append(coords[-1])
                return LineString(new_c)
            return g
        result['geometry'] = result['geometry'].apply(add_v)
        
        # 3. 删对象
        n_del = max(1, int(len(result)*0.1))
        indices = random.sample(range(len(result)), min(n_del, len(result)))
        result = result.drop(indices).reset_index(drop=True)
        
        # 4. 噪声
        def noise(g):
            if isinstance(g, LineString):
                coords = list(g.coords)
                return LineString([(c[0]+np.random.normal(0,0.1), c[1]+np.random.normal(0,0.1)) for c in coords])
            return g
        result['geometry'] = result['geometry'].apply(noise)
        
        # 5. 平移
        bounds = result.total_bounds
        xoff = (bounds[2]-bounds[0])*0.01
        result['geometry'] = result['geometry'].apply(lambda g: affinity.translate(g, xoff=xoff, yoff=0))
        
        # 6. 缩放
        cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        result['geometry'] = result['geometry'].apply(lambda g: affinity.scale(g, xfact=0.95, yfact=0.95, origin=(cx,cy)))
        
        # 7. 旋转
        result['geometry'] = result['geometry'].apply(lambda g: affinity.rotate(g, 15, origin=(cx,cy)))
    except Exception:
        pass
    
    return result


def generate_attacks(shp_files: List[Path], force_regenerate: bool = False) -> Dict[str, Path]:
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        return {}
    outputs: Dict[str, Path] = {}
    
    for src in shp_files:
        base = src.stem
        try:
            out_path = DIR_ATTACKED / f'{base}_compound.shp'
            if not force_regenerate and KEEP_EXISTING and out_path.exists():
                print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                outputs[base] = out_path
                continue
            gdf = gpd.read_file(src)
            attacked = apply_compound(gdf)
            attacked.to_file(out_path, driver='ESRI Shapefile')
            outputs[base] = out_path
        except Exception:
            continue
    return outputs


def evaluate_nc(attacked_map: Dict[str, Dict[str, Path]], force_regenerate: bool = False):
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if not force_regenerate and KEEP_EXISTING:
        csv_path = DIR_RESULTS / 'fig12_compound_nc.csv'
        if csv_path.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 NC 评估: {csv_path}')
            return
    rows = []
    for base, shp_path in attacked_map.items():
        try:
            _, error, nc = extract(str(shp_path), str(CAT32_PATH))
            rows.append({'base': base, 'nc': float(nc)})
        except Exception:
            continue
    
    if not rows:
        return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    hierarchical_rows.append({'复合攻击': '全部攻击顺序执行', 'Lin18': '', '类型': 'header'})
    for base in sorted(df['base'].unique()):
        match = df[df['base'] == base]
        nc_value = float(match['nc'].iloc[0]) if len(match) > 0 else 0.0
        hierarchical_rows.append({'复合攻击': f'  {base}', 'Lin18': f'{nc_value:.6f}', '类型': 'data'})
    avg_nc = float(df['nc'].mean())
    hierarchical_rows.append({'复合攻击': '  Average', 'Lin18': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig12_compound_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    try:
        with pd.ExcelWriter(DIR_RESULTS / 'fig12_compound_nc.xlsx', engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception:
        pass
    
    plt.figure(figsize=(8, 6))
    bases = sorted(df['base'].unique())
    nc_vals = [float(df[df['base']==b]['nc'].iloc[0]) for b in bases]
    x_pos = np.arange(len(bases))
    plt.bar(x_pos, nc_vals, alpha=0.7)
    plt.axhline(y=avg_nc, color='r', linestyle='--', linewidth=2, label=f'平均={avg_nc:.4f}')
    plt.xticks(x_pos, bases, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylabel('NC')
    plt.title('Fig12：Lin18 复合攻击 NC鲁棒性')
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / 'fig12_compound_nc.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print('=== Fig12：Lin18 复合攻击 ===')
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



