# -*- coding: utf-8 -*-
"""
Wu25/Fig1.py - 顶点删除攻击的NC评估（可逆水印，复用 Wu25 的提取与攻击管线）
- 攻击：删点比例 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
- 数据：使用 `embed/` 下 6 个含水印矢量文件
- 结果：输出到 `NC-Results/Fig1/` 下的 CSV、XLSX、PNG
- 结构与 Tan24/zNC-Test 的 Fig1 保持一致的层次化展示
"""

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.geometry.polygon import orient

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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig1'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'delete' / 'Fig1_delete_vertices'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'  # 必须与 embed 输出命名一致（生成 Cat32_*.shp）

# 自动发现 embed 目录下的 Cat32_*.shp（适配当前 8 个数据集）
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

# 与 Tan24 一致：删点比例 10%~90%
DELETE_FACTORS = [i / 100 for i in range(10, 100, 10)]


def delete_vertices_from_geom(geom, pct: int):
    """与 Tan24/Fig1 一致：逐要素/逐环删点，保留端点和闭合。"""
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
            import random
            random.seed(42)
            to_del = set(random.sample(core_idx, n_to_delete))
            new_coords = [coords[0]] + [coords[i] for i in range(1, len(coords) - 1) if i not in to_del] + [coords[-1]]
            return LineString(new_coords)
        elif isinstance(geom, Polygon):
            ext = list(geom.exterior.coords)
            if len(ext) <= 4:
                return geom
            n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
            if n_to_delete >= len(ext) - 4:
                return geom
            core_idx = list(range(1, len(ext) - 2))
            import random
            random.seed(42)
            to_del = set(random.sample(core_idx, n_to_delete))
            new_ext = [ext[0]] + [ext[i] for i in range(1, len(ext) - 2) if i not in to_del] + [ext[-2], ext[-1]]
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
            poly = Polygon(new_ext, holes=holes if holes else None)
            try:
                poly = orient(poly, sign=1.0)
            except Exception:
                pass
            return poly
        elif isinstance(geom, MultiPolygon):
            new_polys = []
            for pg in geom.geoms:
                new_polys.append(delete_vertices_from_geom(pg, pct))
            return MultiPolygon(new_polys)
    except Exception:
        return geom
    return geom


def generate_attacked_variant(src_path: Path, pct: int) -> Path:
    """生成单一删点比例的 attacked shapefile 并返回路径。"""
    base = src_path.stem
    subdir = DIR_ATTACKED / base
    subdir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(str(src_path))
    attacked = gdf.copy()
    attacked['geometry'] = attacked['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, pct))
    # 继承 CRS
    if getattr(gdf, 'crs', None) is not None:
        try:
            attacked.set_crs(gdf.crs, allow_override=True, inplace=True)  # type: ignore
        except Exception:
            pass
    out_path = subdir / f'delete_{pct}pct_vertices.shp'
    if KEEP_EXISTING and out_path.exists():
        print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
        return out_path
    attacked.to_file(str(out_path), driver='ESRI Shapefile')
    return out_path


def run_evaluation():
    print('=== Wu25 Fig1：顶点删除攻击 NC 评估 ===')
    if KEEP_EXISTING and any(DIR_RESULTS.glob('*.csv')):
        print(f'⚠️ KEEP_EXISTING=True，且 Fig1 结果已存在于 {DIR_RESULTS}，跳过')
        return
    missing = []
    if not (SCRIPT_DIR / WATERMARK).exists():
        missing.append(WATERMARK)
    if len(VECTOR_FILES) == 0:
        print('未发现嵌入矢量：请先运行 embed.py（期望在 embed/ 下生成 Cat32_*.shp）')
        return
    if missing:
        print(f'缺少必要文件: {missing}')
        return

    rows = []
    nc_matrix = np.zeros((len(FILE_NAMES), len(DELETE_FACTORS)))

    for file_idx, (vector_file, file_name) in enumerate(zip(VECTOR_FILES, FILE_NAMES)):
        abs_vector = str(vector_file)
        print(f'处理: {file_name}')
        for factor_idx, delete_factor in enumerate(DELETE_FACTORS):
            try:
                pct = int(round(delete_factor * 100))
                attacked_path = generate_attacked_variant(Path(abs_vector), pct)
                _, eva = extract(str(attacked_path), str(SCRIPT_DIR / WATERMARK))
                nc_value = float(eva.get('NC', 0.0))
                ber_value = float(eva.get('BER', 1.0))
                rows.append({
                    'file': file_name,
                    'delete_factor': delete_factor,
                    'nc': nc_value,
                    'ber': ber_value
                })
                nc_matrix[file_idx, factor_idx] = nc_value
                print(f'  ratio={delete_factor:.1f} -> NC={nc_value:.4f}, BER={ber_value:.4f}')
            except Exception as exc:
                print('  失败:', file_name, delete_factor, exc)
                nc_matrix[file_idx, factor_idx] = 0.0

    df = pd.DataFrame(rows)

    # 防御性检查：若没有列 'file' 或 'delete_factor'（可能全部评估失败或 rows 格式异常），则输出并返回
    if df.empty or 'file' not in df.columns or 'delete_factor' not in df.columns:
        print('没有有效的评估结果，跳过后续汇总与绘图（rows 长度:', len(rows), ')')
        # 仍然保存一个空的结果文件以便后续流程不出错
        csv_path = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
        try:
            pd.DataFrame([], columns=['删点比例', 'VGAT', '类型']).to_csv(csv_path, index=False, encoding='utf-8-sig')
        except Exception:
            pass
        print('=== 完成（无有效结果） ===')
        return

    # 层次结构输出（与 Tan24/zNC-Test 风格类似）
    hierarchical_rows: List[dict] = []
    for factor in DELETE_FACTORS:
        pct = int(round(factor * 100))
        hierarchical_rows.append({'删点比例': f'{pct}%', 'VGAT': '', '类型': 'subheader'})
        for file_name in FILE_NAMES:
            sub = df[(df['file'] == file_name) & (df['delete_factor'] == factor)]
            nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
            hierarchical_rows.append({'删点比例': f'  {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['delete_factor'] == factor]['nc'].mean()) if not df.empty else 0.0
        hierarchical_rows.append({'删点比例': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV/XLSX
    csv_path = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
    xlsx_path = DIR_RESULTS / 'fig1_delete_vertices_nc.xlsx'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception as e:
        print('Excel保存警告:', e)

    # 绘图
    plt.figure(figsize=(10, 6))
    for file_idx, file_name in enumerate(FILE_NAMES):
        plt.plot(DELETE_FACTORS, nc_matrix[file_idx, :], '-o', label=file_name)
    avg_curve = np.mean(nc_matrix, axis=0)
    plt.plot(DELETE_FACTORS, avg_curve, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xlabel('删点比例')
    plt.ylabel('NC')
    plt.title('顶点删除攻击的NC鲁棒性（Wu25/Fig1）')
    plt.ylim(0, 1.05)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig1_delete_vertices_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存:', csv_path)
    print('Excel保存:', xlsx_path)
    print('曲线图保存:', fig_path)
    print('=== 完成 ===')


if __name__ == '__main__':
    run_evaluation()