#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig4：Lin18 对比试验（噪声攻击：强度0.4/0.6/0.8；扰动比例10/30/50/70/90）
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
    from shapely.geometry import LineString, Polygon, Point
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
DIR_PSO = Path(os.environ.get('PSO_DATA_DIR', str(PROJECT_ROOT / 'zNC-Test' / 'vector-data')))
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'noise' / 'Fig4_noise'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig4'
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

NOISE_STRENGTHS: List[float] = [0.4, 0.6, 0.8]
NOISE_PCTS: List[int] = [10, 30, 50, 70, 90]


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


def add_noise_to_geom(geom, strength: float, pct: int):
    if geom is None:
        return geom
    try:
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) <= 2:
                return geom
            n_noise = max(1, int((len(coords) - 2) * pct / 100))
            core_idx = list(range(1, len(coords) - 1))
            random.seed(42)
            noise_idx = random.sample(core_idx, min(n_noise, len(core_idx)))
            new_coords = []
            for i, c in enumerate(coords):
                if i in noise_idx:
                    noise = np.random.normal(0, strength, 2)
                    new_coords.append((c[0] + noise[0], c[1] + noise[1]))
                else:
                    new_coords.append(c)
            return LineString(new_coords)
        elif isinstance(geom, Polygon):
            ext = list(geom.exterior.coords)
            if len(ext) <= 4:
                return geom
            n_noise = max(1, int((len(ext) - 4) * pct / 100))
            core_idx = list(range(1, len(ext) - 2))
            random.seed(42)
            noise_idx = random.sample(core_idx, min(n_noise, len(core_idx)))
            new_ext = []
            for i, c in enumerate(ext):
                if i in noise_idx:
                    noise = np.random.normal(0, strength, 2)
                    new_ext.append((c[0] + noise[0], c[1] + noise[1]))
                else:
                    new_ext.append(c)
            # 处理内环
            holes = []
            for ring in geom.interiors:
                ring_coords = list(ring.coords)
                if len(ring_coords) > 4:
                    n_h = max(1, int((len(ring_coords) - 4) * pct / 100))
                    core_idx_h = list(range(1, len(ring_coords) - 2))
                    random.seed(42)
                    noise_idx_h = random.sample(core_idx_h, min(n_h, len(core_idx_h)))
                    new_ring = []
                    for i, c in enumerate(ring_coords):
                        if i in noise_idx_h:
                            noise = np.random.normal(0, strength, 2)
                            new_ring.append((c[0] + noise[0], c[1] + noise[1]))
                        else:
                            new_ring.append(c)
                    holes.append(new_ring)
                else:
                    holes.append(ring_coords)
            return Polygon(new_ext, holes=holes if holes else None)
    except Exception:
        pass
    return geom


def generate_attacks(shp_files: List[Path], force_regenerate: bool = False) -> Dict[str, Dict[Tuple[float, int], Path]]:
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        return {}
    outputs: Dict[str, Dict[Tuple[float, int], Path]] = {}
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
        for strength in NOISE_STRENGTHS:
            for pct in NOISE_PCTS:
                try:
                    out_path = subdir / f'noise_str{int(strength*10)}_{pct}pct.shp'
                    if not force_regenerate and KEEP_EXISTING and out_path.exists():
                        print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                        outputs[base][(strength, pct)] = out_path
                        continue
                    attacked = gdf.copy()
                    attacked['geometry'] = attacked['geometry'].apply(lambda g: add_noise_to_geom(g, strength, pct))
                    attacked.to_file(out_path, driver='ESRI Shapefile')
                    outputs[base][(strength, pct)] = out_path
                except Exception:
                    continue
    return outputs


def evaluate_nc(attacked_map: Dict[str, Dict[Tuple[float, int], Path]], force_regenerate: bool = False):
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if not force_regenerate and KEEP_EXISTING:
        csv_path = DIR_RESULTS / 'fig4_noise_nc.csv'
        if csv_path.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 NC 评估: {csv_path}')
            return
    rows = []
    for base, sp_map in attacked_map.items():
        for (strength, pct), shp_path in sorted(sp_map.items()):
            try:
                _, error, nc = extract(str(shp_path), str(CAT32_PATH))
                rows.append({'base': base, 'strength': strength, 'pct': pct, 'nc': float(nc)})
            except Exception:
                continue
    
    if not rows:
        return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    
    for strength in NOISE_STRENGTHS:
        hierarchical_rows.append({'噪声攻击': f'强度: {strength}', 'Lin18': '', '类型': 'header'})
        for pct in NOISE_PCTS:
            hierarchical_rows.append({'噪声攻击': f'  {pct}%', 'Lin18': '', '类型': 'subheader'})
            for base in sorted(df['base'].unique()):
                sub = df[(df['base'] == base) & (df['strength'] == strength) & (df['pct'] == pct)]
                nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
                hierarchical_rows.append({'噪声攻击': f'    {base}', 'Lin18': f'{nc_value:.6f}', '类型': 'data'})
            avg_nc = float(df[(df['strength'] == strength) & (df['pct'] == pct)]['nc'].mean())
            hierarchical_rows.append({'噪声攻击': '    Average', 'Lin18': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig4_noise_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    try:
        xlsx_path = DIR_RESULTS / 'fig4_noise_nc.xlsx'
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception:
        pass
    
    plt.figure(figsize=(12, 8))
    for i, strength in enumerate(NOISE_STRENGTHS):
        plt.subplot(2, 2, i + 1)
        for base, sub in df[df['strength'] == strength].groupby('base'):
            sub2 = sub.sort_values('pct')
            plt.plot(sub2['pct'], sub2['nc'], '-o', alpha=0.7, label=base, markersize=4)
        avg = df[df['strength'] == strength].groupby('pct')['nc'].mean().reset_index()
        plt.plot(avg['pct'], avg['nc'], 'k-o', linewidth=2.5, label='平均', markersize=6)
        plt.grid(True, alpha=0.3)
        plt.xlabel('噪声比例（%）')
        plt.ylabel('NC')
        plt.title(f'强度 {strength}：Lin18 噪声攻击')
        plt.legend(loc='best', fontsize=6, ncol=2)
        plt.ylim(0, 1.1)
    
    plt.subplot(2, 2, 4)
    for strength in NOISE_STRENGTHS:
        avg = df[df['strength'] == strength].groupby('pct')['nc'].mean().reset_index()
        plt.plot(avg['pct'], avg['nc'], '-o', linewidth=2, label=f'强度 {strength}', markersize=6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('噪声比例（%）')
    plt.ylabel('NC')
    plt.title('综合对比')
    plt.legend(loc='best')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / 'fig4_noise_nc.png', dpi=300, bbox_inches='tight')
    plt.close()


def main(force_regenerate: bool = False):
    print('=== Fig4：Lin18 噪声攻击 ===')
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
