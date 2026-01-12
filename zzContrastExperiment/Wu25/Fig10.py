# -*- coding: utf-8 -*-
"""
Wu25/Fig10.py - 顺序扰动攻击（反转/打乱 顶点/对象）NC评估（复用 Wu25 提取与数据管线）
- 数据：`embed/` 下 8 个 `Cat32_*.shp`
- 结果：输出到 `NC-Results/Fig10/` 下的 CSV、XLSX、PNG
"""

from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.geometry import LineString, Polygon  # type: ignore
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig10'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'shuffle' / 'Fig10_shuffle'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

ORDER_ATTACKS: List[Tuple[str, str]] = [
    ('reverse_vertices', '反转顶点顺序'),
    ('shuffle_vertices', '打乱顶点顺序'),
    ('reverse_objects', '反转对象顺序'),
    ('shuffle_objects', '打乱对象顺序'),
]


def generate_attacked_variants(src_path: Path) -> Dict[str, Path]:
    base = src_path.stem
    subdir = DIR_ATTACKED / base
    # 如果存在已生成的 attacked 子目录且设置了 KEEP_EXISTING，则直接重用现有文件
    if subdir.exists():
        if KEEP_EXISTING and any(subdir.glob('*.shp')):
            outputs: Dict[str, Path] = {}
            for key, _label in ORDER_ATTACKS:
                p = subdir / f'{key}.shp'
                if p.exists():
                    outputs[key] = p
            # 若至少有一个现有文件，直接返回复用它们（避免重复生成）
            if outputs:
                return outputs
        else:
            import shutil
            shutil.rmtree(subdir, ignore_errors=True)
    subdir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(str(src_path))
    outputs: Dict[str, Path] = {}
    base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'

    def reverse_vertices_geom(geom):
        try:
            if geom is None:
                return None
            if geom.geom_type == 'LineString':
                return LineString(list(geom.coords)[::-1])
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                ext = ext[:-1][::-1] + [ext[0]]
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    rc = rc[:-1][::-1] + [rc[0]]
                    holes.append(rc)
                return Polygon(ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    def shuffle_vertices_geom(geom):
        try:
            if geom is None:
                return None
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) <= 2:
                    return geom
                core = coords[1:-1]
                random.shuffle(core)
                return LineString([coords[0]] + core + [coords[-1]])
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                if len(ext) <= 4:
                    return geom
                core = ext[1:-2]
                random.shuffle(core)
                new_ext = [ext[0]] + core + [ext[-2], ext[-1]]
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    if len(rc) > 4:
                        core_r = rc[1:-2]
                        random.shuffle(core_r)
                        holes.append([rc[0]] + core_r + [rc[-2], rc[-1]])
                    else:
                        holes.append(rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    for key, _label in ORDER_ATTACKS:
        attacked = gdf.copy()
        if key == 'reverse_vertices':
            attacked['geometry'] = attacked['geometry'].apply(reverse_vertices_geom)
        elif key == 'shuffle_vertices':
            attacked['geometry'] = attacked['geometry'].apply(shuffle_vertices_geom)
        elif key == 'reverse_objects':
            attacked = attacked.iloc[::-1].reset_index(drop=True)
        elif key == 'shuffle_objects':
            attacked = attacked.sample(frac=1).reset_index(drop=True)

        try:
            attacked.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
        except Exception:
            pass

        out_path = subdir / f'{key}.shp'
        attacked.to_file(str(out_path), driver='ESRI Shapefile')
        outputs[key] = out_path
    return outputs


def run_evaluation():
    print('=== Wu25 Fig10：顺序扰动攻击 NC 评估 ===')
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig10_shuffle_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig10: {csv_check}')
            return
    if not (SCRIPT_DIR / WATERMARK).exists():
        print('缺少水印图像: ', WATERMARK)
        return
    if len(VECTOR_FILES) == 0:
        print('未发现嵌入矢量：请先在 embed/ 生成 Cat32_*.shp')
        return

    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    rows: List[Dict] = []  # type: ignore
    for vector_file, file_name in zip(VECTOR_FILES, FILE_NAMES):
        print(f'处理: {file_name}')
        attacked_map = generate_attacked_variants(Path(vector_file))
        for key, shp_path in sorted(attacked_map.items()):
            try:
                _, eva = extract(str(shp_path), str(SCRIPT_DIR / WATERMARK))
                nc_value = float(eva.get('NC', 0.0))
                rows.append({'file': file_name, 'key': key, 'nc': nc_value})
                print(f'  {key} -> NC={nc_value:.4f}')
            except Exception as exc:
                print('  失败:', file_name, key, exc)
                rows.append({'file': file_name, 'key': key, 'nc': 0.0})

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 防御性检查：避免 rows 全部失败导致后续通过 df['file'] 抛出 KeyError
    if df.empty or 'file' not in df.columns:
        print('没有有效的评估结果，跳过 Fig10 的汇总与绘图（rows 长度:', len(rows), ')')
        print('=== 完成（无有效结果） ===')
        return
    key_to_label = dict(ORDER_ATTACKS)
    order_keys = [k for k, _ in ORDER_ATTACKS]

    hierarchical_rows: List[dict] = []
    base_names = sorted(df['file'].unique())
    for k in order_keys:
        hierarchical_rows.append({'打乱顺序方式': key_to_label.get(k, k), 'VGAT': '', '类型': 'header'})
        for name in base_names:
            sub = df[(df['file'] == name) & (df['key'] == k)]
            nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
            hierarchical_rows.append({'打乱顺序方式': f'  {name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['key'] == k]['nc'].mean())
        hierarchical_rows.append({'打乱顺序方式': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    csv_path = DIR_RESULTS / 'fig10_shuffle_nc.csv'
    xlsx_path = DIR_RESULTS / 'fig10_shuffle_nc.xlsx'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception as e:
        print('Excel保存警告:', e)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(order_keys))
    labels = [key_to_label.get(k, k) for k in order_keys]
    for name, sub in df.groupby('file'):
        y = []
        for k in order_keys:
            val = sub[sub['key'] == k]['nc']
            y.append(val.iloc[0] if len(val) > 0 else np.nan)
        plt.plot(x, y, '-o', alpha=0.7, label=name)
    avg_vals = [df[df['key'] == k]['nc'].mean() for k in order_keys]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, labels)
    plt.xlabel('打乱顺序方式')
    plt.ylabel('NC')
    plt.title('打乱顺序攻击的NC鲁棒性（Wu25/Fig10）')
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig10_shuffle_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存:', csv_path)
    print('Excel保存:', xlsx_path)
    print('曲线图保存:', fig_path)
    print('=== 完成 ===')


if __name__ == '__main__':
    run_evaluation()

