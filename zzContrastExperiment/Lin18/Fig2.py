#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig2：Lin18 对比试验（增加顶点 强度0/1/2；比例10%→90%）
"""

from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
except Exception:
    pass

from extract import extract

try:
    preferred_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Microsoft YaHei']
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
# 默认改为使用本目录下的 pso_data（每个对比实验的源数据目录）
# 优先使用环境变量 PSO_DATA_DIR，其次使用脚本目录下的 pso_data
DIR_PSO = Path(os.environ.get('PSO_DATA_DIR', str(SCRIPT_DIR / 'pso_data')))
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'add' / 'Fig2_add_vertices'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig2'
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

STRENGTHS: List[int] = [0, 1, 2]
ADD_PCTS: List[int] = list(range(10, 100, 10))


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


def add_vertices_to_geom(geom, strength: int, pct: int):
    if geom is None:
        return geom
    try:
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) < 2:
                return geom
            n_to_add = min(3, max(1, int((len(coords) - 1) * pct / 100)))
            new_coords = []
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                new_coords.append(p1)
                for j in range(n_to_add):
                    t = (j + 1) / (n_to_add + 1)
                    mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                    if strength == 1:
                        noise = np.random.normal(0, 0.01, 2)
                        mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
                    elif strength == 2:
                        noise = np.random.normal(0, 0.05, 2)
                        mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
                    new_coords.append(mid)
            new_coords.append(coords[-1])
            return LineString(new_coords)
        elif isinstance(geom, Polygon):
            ext = list(geom.exterior.coords)
            if len(ext) < 4:
                return geom
            n_to_add = min(3, max(1, int((len(ext) - 1) * pct / 100)))
            new_ext = []
            for i in range(len(ext) - 1):
                p1, p2 = ext[i], ext[i + 1]
                new_ext.append(p1)
                for j in range(n_to_add):
                    t = (j + 1) / (n_to_add + 1)
                    mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                    if strength == 1:
                        noise = np.random.normal(0, 0.01, 2)
                        mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
                    elif strength == 2:
                        noise = np.random.normal(0, 0.05, 2)
                        mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
                    new_ext.append(mid)
            new_ext.append(ext[-1])
            # 处理内环
            holes = []
            for ring in geom.interiors:
                ring_coords = list(ring.coords)
                if len(ring_coords) >= 4:
                    new_ring = []
                    for i in range(len(ring_coords) - 1):
                        p1, p2 = ring_coords[i], ring_coords[i + 1]
                        new_ring.append(p1)
                        for j in range(n_to_add):
                            t = (j + 1) / (n_to_add + 1)
                            mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                            if strength == 1:
                                noise = np.random.normal(0, 0.01, 2)
                                mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
                            elif strength == 2:
                                noise = np.random.normal(0, 0.05, 2)
                                mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
                            new_ring.append(mid)
                    new_ring.append(ring_coords[-1])
                    holes.append(new_ring)
                else:
                    holes.append(ring_coords)
            return Polygon(new_ext, holes=holes if holes else None)
        elif isinstance(geom, (MultiLineString, MultiPolygon)):
            from shapely.geometry import MultiLineString, MultiPolygon
            geoms = []
            for g in geom.geoms:
                geoms.append(add_vertices_to_geom(g, strength, pct))
            if isinstance(geom, MultiLineString):
                return MultiLineString(geoms)
            else:
                return MultiPolygon(geoms)
    except Exception:
        pass
    return geom


def generate_attacks(shp_files: List[Path], force_regenerate: bool = False) -> Dict[str, Dict[Tuple[int, int], Path]]:
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        return {}
    outputs: Dict[str, Dict[Tuple[int, int], Path]] = {}
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
        for strength in STRENGTHS:
            for pct in ADD_PCTS:
                try:
                    out_path = subdir / f'add_strength{strength}_{pct}pct_vertices.shp'
                    if not force_regenerate and KEEP_EXISTING and out_path.exists():
                        print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                        outputs[base][(strength, pct)] = out_path
                        continue
                    attacked = gdf.copy()
                    attacked['geometry'] = attacked['geometry'].apply(lambda g: add_vertices_to_geom(g, strength, pct))
                    attacked.to_file(out_path, driver='ESRI Shapefile')
                    outputs[base][(strength, pct)] = out_path
                except Exception:
                    continue
    return outputs


def evaluate_nc(attacked_map: Dict[str, Dict[Tuple[int, int], Path]], force_regenerate: bool = False):
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if not force_regenerate and KEEP_EXISTING:
        csv_path = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
        if csv_path.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 NC 评估: {csv_path}')
            return
    rows = []
    for base, sp_map in attacked_map.items():
        for (strength, pct), shp_path in sorted(sp_map.items()):
            try:
                _, error, nc = extract(str(shp_path), str(CAT32_PATH))
                rows.append({'base': base, 'strength': strength, 'add_pct': pct, 'nc': float(nc)})
            except Exception:
                continue
    
    if not rows:
        return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    
    for strength in STRENGTHS:
        hierarchical_rows.append({'增加顶点': f'强度: {strength}', 'Lin18': '', '类型': 'header'})
        for pct in ADD_PCTS:
            hierarchical_rows.append({'增加顶点': f'  {pct}%', 'Lin18': '', '类型': 'subheader'})
            for base in sorted(df['base'].unique()):
                sub = df[(df['base'] == base) & (df['strength'] == strength) & (df['add_pct'] == pct)]
                nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
                hierarchical_rows.append({'增加顶点': f'    {base}', 'Lin18': f'{nc_value:.6f}', '类型': 'data'})
            avg_nc = float(df[(df['strength'] == strength) & (df['add_pct'] == pct)]['nc'].mean())
            hierarchical_rows.append({'增加顶点': '    Average', 'Lin18': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    try:
        xlsx_path = DIR_RESULTS / 'fig2_add_vertices_nc.xlsx'
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception:
        pass
    
    plt.figure(figsize=(12, 8))
    for i, strength in enumerate(STRENGTHS):
        plt.subplot(2, 2, i + 1)
        for base, sub in df[df['strength'] == strength].groupby('base'):
            sub2 = sub.sort_values('add_pct')
            plt.plot(sub2['add_pct'], sub2['nc'], '-o', alpha=0.7, label=base, markersize=4)
        avg = df[df['strength'] == strength].groupby('add_pct')['nc'].mean().reset_index()
        plt.plot(avg['add_pct'], avg['nc'], 'k-o', linewidth=2.5, label='平均', markersize=6)
        plt.grid(True, alpha=0.3)
        plt.xlabel('增加顶点比例（%）')
        plt.ylabel('NC')
        plt.title(f'强度 {strength}：Lin18 增加顶点攻击')
        plt.legend(loc='best', fontsize=6, ncol=2)
        plt.ylim(0, 1.1)
    
    plt.subplot(2, 2, 4)
    for strength in STRENGTHS:
        avg = df[df['strength'] == strength].groupby('add_pct')['nc'].mean().reset_index()
        plt.plot(avg['add_pct'], avg['nc'], '-o', linewidth=2, label=f'强度 {strength}', markersize=6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('增加顶点比例（%）')
    plt.ylabel('NC')
    plt.title('综合对比')
    plt.legend(loc='best')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / 'fig2_add_vertices_nc.png', dpi=300, bbox_inches='tight')
    plt.close()


def main(force_regenerate: bool = False):
    print('=== Fig2：Lin18 增加顶点攻击 ===')
    inputs = discover_inputs()
    if not inputs:
        return
    attacked_map = generate_attacks(inputs, force_regenerate=force_regenerate)
    evaluate_nc(attacked_map, force_regenerate=force_regenerate)
    print('=== 完成 ===')


if __name__ == '__main__':
    import sys
    force = '--force' in sys.argv or '-f' in sys.argv
    main(force_regenerate=force)
