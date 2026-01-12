#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig1：Lin18 对比试验（删点攻击 10%→90%）
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
except Exception:
    gpd = None

try:
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.geometry.polygon import orient
except Exception:
    pass

from extract import extract
from PIL import Image
import numpy as _np_inline


def _compute_nc_from_pngs(extracted_img_path, watermark_path) -> float:
    try:
        ex = _np_inline.array(Image.open(str(extracted_img_path)).convert('L')) > 127
        wm = _np_inline.array(Image.open(str(watermark_path)).convert('L')) > 127
        va = ex.flatten().astype(float)
        vb = wm.flatten().astype(float)
        na = _np_inline.linalg.norm(va)
        nb = _np_inline.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(_np_inline.sum(va * vb) / (na * nb))
    except Exception:
        return 0.0

# 中文字体配置
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
import os
PROJECT_ROOT = SCRIPT_DIR.resolve().parents[1]
DIR_PSO = Path(os.environ.get('PSO_DATA_DIR', str(PROJECT_ROOT / 'zNC-Test' / 'vector-data')))
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'delete' / 'Fig1_delete_vertices'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig1'
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'
DELETE_PCTS: List[int] = list(range(10, 100, 10))


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


def delete_vertices_from_geom(geom, pct: int):
    if geom is None:
        return geom
    try:
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) <= 2:
                return geom
            n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
            if n_to_delete >= len(coords) - 2:
                return geom
            core_idx = list(range(1, len(coords) - 1))
            random.seed(42)
            to_del = set(random.sample(core_idx, n_to_delete))
            new_coords = [coords[0]] + [c for i, c in enumerate(coords[1:-1], 1) if i not in to_del] + [coords[-1]]
            return LineString(new_coords)
        elif isinstance(geom, Polygon):
            ext = list(geom.exterior.coords)
            if len(ext) <= 4:
                return geom
            n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
            if n_to_delete >= len(ext) - 4:
                return geom
            core_idx = list(range(1, len(ext) - 2))
            random.seed(42)
            to_del = set(random.sample(core_idx, n_to_delete))
            new_ext = [ext[0]] + [c for i, c in enumerate(ext[1:-2], 1) if i not in to_del] + [ext[-2], ext[-1]]
            # 处理内环
            holes = []
            for ring in geom.interiors:
                ring_coords = list(ring.coords)
                if len(ring_coords) > 4:
                    n_h = max(1, int((len(ring_coords) - 4) * pct / 100))
                    if n_h < len(ring_coords) - 4:
                        core_idx_h = list(range(1, len(ring_coords) - 2))
                        random.seed(42)
                        to_del_h = set(random.sample(core_idx_h, n_h))
                        new_ring = [ring_coords[0]] + [ring_coords[i] for i in range(1, len(ring_coords) - 2) if i not in to_del_h] + [ring_coords[-2], ring_coords[-1]]
                        holes.append(new_ring)
                    else:
                        holes.append(ring_coords)
                else:
                    holes.append(ring_coords)
            return Polygon(new_ext, holes=holes if holes else None)
    except Exception:
        pass
    return geom


def generate_attacks(shp_files: List[Path], force_regenerate: bool = False) -> Dict[str, Dict[int, Path]]:
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
    if gpd is None:
        return {}
    outputs: Dict[str, Dict[int, Path]] = {}
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
        for pct in DELETE_PCTS:
            try:
                out_path = subdir / f'delete_{pct}pct_vertices.shp'
                if not force_regenerate and KEEP_EXISTING and out_path.exists():
                    print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                    outputs[base][pct] = out_path
                    continue
                attacked = gdf.copy()
                attacked['geometry'] = attacked['geometry'].apply(lambda g: delete_vertices_from_geom(g, pct))
                attacked.to_file(out_path, driver='ESRI Shapefile')
                outputs[base][pct] = out_path
            except Exception:
                continue
    return outputs


def evaluate_nc(attacked_map: Dict[str, Dict[int, Path]], force_regenerate: bool = False):
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    # 如果设置了 KEEP_EXISTING 并且结果 CSV 已存在，且不强制重新生成，则跳过评估
    import os
    if not force_regenerate and os.environ.get('KEEP_EXISTING', '0') in ['1', 'true', 'True']:
        csv_path = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
        if csv_path.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 NC 评估: {csv_path}')
            return
    rows = []
    for base, pct_map in attacked_map.items():
        for pct, shp_path in sorted(pct_map.items()):
            try:
                # If extraction result already exists and KEEP_EXISTING and not force_regenerate, compute NC from saved image
                extract_img = SCRIPT_DIR / 'extract' / 'watermark' / f"{Path(shp_path).stem}.png"
                shp_out = SCRIPT_DIR / 'extract' / 'shpfile' / Path(shp_path).name
                if not force_regenerate and KEEP_EXISTING and extract_img.exists() and shp_out.exists():
                    nc_val = _compute_nc_from_pngs(extract_img, CAT32_PATH)
                    rows.append({'base': base, 'attack_pct': pct, 'nc': float(nc_val)})
                else:
                    _, error, nc = extract(str(shp_path), str(CAT32_PATH))
                    rows.append({'base': base, 'attack_pct': pct, 'nc': float(nc)})
            except Exception:
                continue
    
    if not rows:
        return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    for pct in sorted(df['attack_pct'].unique()):
        hierarchical_rows.append({'删除比例': f'{pct}%', 'Lin18': '', '类型': 'header'})
        for base in sorted(df['base'].unique()):
            match = df[(df['base'] == base) & (df['attack_pct'] == pct)]
            nc_value = float(match['nc'].iloc[0]) if len(match) > 0 else 0.0
            hierarchical_rows.append({'删除比例': f'  {base}', 'Lin18': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['attack_pct'] == pct]['nc'].mean())
        hierarchical_rows.append({'删除比例': '  Average', 'Lin18': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    try:
        xlsx_path = DIR_RESULTS / 'fig1_delete_vertices_nc.xlsx'
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception:
        pass
    
    plt.figure(figsize=(10, 6))
    for base, sub in df.groupby('base'):
        sub2 = sub.sort_values('attack_pct')
        plt.plot(sub2['attack_pct'], sub2['nc'], '-o', alpha=0.7, label=base)
    avg = df.groupby('attack_pct')['nc'].mean().reset_index()
    plt.plot(avg['attack_pct'], avg['nc'], 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xlabel('删点比例（%）')
    plt.ylabel('NC')
    plt.title('Fig1：Lin18 删点攻击 NC鲁棒性')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / 'fig1_delete_vertices_nc.png', dpi=300, bbox_inches='tight')
    plt.close()


def main(force_regenerate: bool = False):
    print('=== Fig1：Lin18 删点攻击 ===')
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
