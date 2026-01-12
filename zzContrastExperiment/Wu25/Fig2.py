# -*- coding: utf-8 -*-
"""
Wu25/Fig2.py - 增加顶点攻击的NC评估（可逆水印，复用 Wu25 的提取与攻击管线）
- 攻击：addRatio 0.1, 0.3, 0.5 × strength 0, 0.5, 1.0
- 数据：使用 `embed/` 下 6 个含水印矢量文件
- 结果：输出到 `NC-Results/Fig2/` 下的 CSV、XLSX、PNG
- 结构与 Tan24/zNC-Test 的 Fig2 保持一致的层次化展示
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString

from extract import extract

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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig2'
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'

# 自动发现 embed 目录下 Cat32_*.shp（适配 8 数据集）
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

# 与 Tan24 对齐：强度 0/1/2；比例 10%~90%
STRENGTHS: List[int] = [0, 1, 2]
ADD_RATIOS: List[float] = [i / 100 for i in range(10, 100, 10)]
TOLERANCE = 0.01


def run_evaluation():
    print('=== Wu25 Fig2：增加顶点攻击 NC 评估 ===')
    # 确保结果目录存在
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig2: {csv_check}')
            return
    if not (SCRIPT_DIR / WATERMARK).exists():
        print('缺少必要文件: Cat32.png，请先生成水印')
        return
    if len(VECTOR_FILES) == 0:
        print('未发现嵌入矢量：请先运行 embed.py（期望在 embed/ 下生成 Cat32_*.shp）')
        return

    rows = []
    nc_tensor = np.zeros((len(FILE_NAMES), len(STRENGTHS), len(ADD_RATIOS)))

    def add_vertices_to_geom(geom, strength: int, pct: int):
        if geom is None:
            return geom
        try:
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) < 2:
                    return geom
                n_to_add = min(3, max(1, int((len(coords) - 1) * pct / 100)))
                new_coords = [coords[0]]
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
                new_ext = [ext[0]]
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
                holes = []
                for ring in geom.interiors:
                    ring_coords = list(ring.coords)
                    if len(ring_coords) >= 4:
                        new_ring = [ring_coords[0]]
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
                geoms = []
                for g in geom.geoms:
                    geoms.append(add_vertices_to_geom(g, strength, pct))
                if isinstance(geom, MultiLineString):
                    return MultiLineString(geoms)
                else:
                    return MultiPolygon(geoms)
        except Exception:
            return geom
        return geom

    def generate_add_variant(src_path: Path, strength: int, pct: int) -> Path:
        base = src_path.stem
        subdir = (SCRIPT_DIR / 'attacked' / 'add' / 'Fig2_add_vertices' / base)
        subdir.mkdir(parents=True, exist_ok=True)
        gdf = gpd.read_file(str(src_path))
        attacked = gdf.copy()
        attacked['geometry'] = attacked['geometry'].apply(lambda geom: add_vertices_to_geom(geom, strength, pct))
        if getattr(gdf, 'crs', None) is not None:
            try:
                attacked.set_crs(gdf.crs, allow_override=True, inplace=True)  # type: ignore
            except Exception:
                pass
        out_path = subdir / f'add_strength{strength}_{pct}pct_vertices.shp'
        if KEEP_EXISTING and out_path.exists():
            print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
            return out_path
        attacked.to_file(str(out_path), driver='ESRI Shapefile')
        return out_path

    for file_idx, (vector_file, file_name) in enumerate(zip(VECTOR_FILES, FILE_NAMES)):
        abs_vector = str(vector_file)
        print(f'处理: {file_name}')
        for s_idx, strength in enumerate(STRENGTHS):
            for r_idx, add_ratio in enumerate(ADD_RATIOS):
                try:
                    pct = int(round(add_ratio * 100))
                    attacked_path = generate_add_variant(Path(abs_vector), strength, pct)
                    _, eva = extract(str(attacked_path), str(SCRIPT_DIR / WATERMARK))
                    nc_value = float(eva.get('NC', 0.0))
                    ber_value = float(eva.get('BER', 1.0))
                    rows.append({
                        'file': file_name,
                        'strength': strength,
                        'add_ratio': add_ratio,
                        'nc': nc_value,
                        'ber': ber_value
                    })
                    nc_tensor[file_idx, s_idx, r_idx] = nc_value
                    print(f'  s={strength:.1f}, r={add_ratio:.1f} -> NC={nc_value:.4f}, BER={ber_value:.4f}')
                except Exception as exc:
                    print('  失败:', file_name, strength, add_ratio, exc)
                    nc_tensor[file_idx, s_idx, r_idx] = 0.0

    df = pd.DataFrame(rows)

    # 防御性检查：避免 rows 全部失败导致后续通过 df['file'] 抛出 KeyError
    if df.empty or 'file' not in df.columns:
        print('没有有效的评估结果，跳过 Fig2 的汇总与绘图（rows 长度:', len(rows), ')')
        print('=== 完成（无有效结果） ===')
        return

    # 层次结构输出：强度 -> 比例 -> 文件 -> 平均
    hierarchical_rows: List[dict] = []
    for strength in STRENGTHS:
        hierarchical_rows.append({'增加顶点': f'强度: {strength}', 'VGAT': '', '类型': 'header'})
        for add_ratio in ADD_RATIOS:
            hierarchical_rows.append({'增加顶点': f'  比例 {add_ratio:.1f}', 'VGAT': '', '类型': 'subheader'})
            for file_name in FILE_NAMES:
                sub = df[(df['file'] == file_name) & (df['strength'] == strength) & (df['add_ratio'] == add_ratio)]
                nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
                hierarchical_rows.append({'增加顶点': f'    {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
            avg_nc = float(df[(df['strength'] == strength) & (df['add_ratio'] == add_ratio)]['nc'].mean()) if not df.empty else 0.0
            hierarchical_rows.append({'增加顶点': '    Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV/XLSX
    csv_path = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
    xlsx_path = DIR_RESULTS / 'fig2_add_vertices_nc.xlsx'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception as e:
        print('Excel保存警告:', e)

    # 绘图：按强度分面+综合
    plt.figure(figsize=(12, 8))
    for i, strength in enumerate(STRENGTHS):
        plt.subplot(2, 2, i + 1)
        for file_idx, file_name in enumerate(FILE_NAMES):
            plt.plot(ADD_RATIOS, nc_tensor[file_idx, i, :], '-o', label=file_name, markersize=4)
        avg_curve = np.mean(nc_tensor[:, i, :], axis=0)
        plt.plot(ADD_RATIOS, avg_curve, 'k-o', linewidth=2.5, label='平均', markersize=6)
        plt.grid(True, alpha=0.3)
        plt.xlabel('增加比例')
        plt.ylabel('NC')
        plt.title(f'强度 {strength}：增加顶点攻击的NC鲁棒性')
        plt.legend(loc='best', fontsize=6, ncol=2)
        plt.ylim(0, 1.1)

    plt.subplot(2, 2, 4)
    for i, strength in enumerate(STRENGTHS):
        avg_curve = np.mean(nc_tensor[:, i, :], axis=0)
        plt.plot(ADD_RATIOS, avg_curve, '-o', linewidth=2, label=f'强度 {strength}', markersize=6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('增加比例')
    plt.ylabel('NC')
    plt.title('综合对比：不同强度的增加顶点攻击')
    plt.legend(loc='best')
    plt.ylim(0, 1.1)

    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig2_add_vertices_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存:', csv_path)
    print('Excel保存:', xlsx_path)
    print('曲线图保存:', fig_path)
    print('=== 完成 ===')


if __name__ == '__main__':
    run_evaluation()