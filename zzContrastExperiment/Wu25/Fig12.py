# -*- coding: utf-8 -*-
"""
Wu25/Fig12.py - 复合攻击（一次性顺序 Fig1→Fig10）NC评估（复用 Wu25 提取与数据管线）
顺序：
1) 顶点删除 10%
2) 顶点增加 强度1 比例50%
3) 对象删除 50%
4) 噪声扰动 强度0.8 比例50%
5) 沿Y轴中心裁剪50%
6) 平移 (20, 40)
7) 缩放 90%
8) 旋转 180°
9) Y轴镜像翻转
10) 顺序：反转顶点 → 反转对象
"""

from pathlib import Path
from typing import Dict, List
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.geometry import LineString, Polygon  # type: ignore
from shapely.affinity import translate as shp_translate, scale as shp_scale, rotate as shp_rotate  # type: ignore
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig12'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'compound_seq' / 'Fig12_compound_seq'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]


def generate_attacked_variants(src_path: Path) -> Dict[str, Path]:
    base = src_path.stem
    subdir = DIR_ATTACKED / base
    # 如果已有生成并且设置了 KEEP_EXISTING，重用已有文件
    if subdir.exists():
        if KEEP_EXISTING and any(subdir.glob('*.shp')):
            return {'compound_seq_all': subdir / 'compound_seq_all.shp'}
        else:
            import shutil
            shutil.rmtree(subdir, ignore_errors=True)
    subdir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(str(src_path))
    base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'

    # 裁剪辅助
    bounds = gdf.total_bounds
    mid_y = (bounds[1] + bounds[3]) / 2

    # 1 顶点删除10%
    def delete_vertices_from_geom(geom, pct: int):
        try:
            if geom is None:
                return None
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) <= 2:
                    return geom
                n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
                if n_to_delete >= len(coords) - 2:
                    return geom
                idx = list(range(1, len(coords) - 1))
                to_del = set(random.sample(idx, min(n_to_delete, len(idx))))
                new_coords = [coords[0]] + [coords[i] for i in idx if i not in to_del] + [coords[-1]]
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                if len(ext) <= 4:
                    return geom
                n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
                if n_to_delete >= len(ext) - 4:
                    return geom
                idx = list(range(1, len(ext) - 2))
                to_del = set(random.sample(idx, min(n_to_delete, len(idx))))
                new_ext = [ext[0]] + [ext[i] for i in idx if i not in to_del] + [ext[-2], ext[-1]]
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    if len(rc) > 4:
                        n_h = max(1, int((len(rc) - 4) * pct / 100))
                        if n_h < len(rc) - 4:
                            idx_h = list(range(1, len(rc) - 2))
                            del_h = set(random.sample(idx_h, min(n_h, len(idx_h))))
                            holes.append([rc[0]] + [rc[i] for i in range(1, len(rc) - 2) if i not in del_h] + [rc[-2], rc[-1]])
                        else:
                            holes.append(rc)
                    else:
                        holes.append(rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    gdf['geometry'] = gdf['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, 10))
    # 2 顶点增加 强度1 比例50%
    def add_vertices_to_geom(geom, pct: int, strength_level: int = 1):
        noise_sigma = 0.01 if strength_level == 1 else 0.0
        try:
            if geom is None:
                return None
            if geom.geom_type == 'LineString':
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
                        if noise_sigma > 0:
                            mid = (mid[0] + float(np.random.normal(0, noise_sigma)), mid[1] + float(np.random.normal(0, noise_sigma)))
                        new_coords.append(mid)
                new_coords.append(coords[-1])
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
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
                        if noise_sigma > 0:
                            mid = (mid[0] + float(np.random.normal(0, noise_sigma)), mid[1] + float(np.random.normal(0, noise_sigma)))
                        new_ext.append(mid)
                new_ext.append(ext[-1])
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    if len(rc) >= 4:
                        new_rc = [rc[0]]
                        for i in range(len(rc) - 1):
                            p1, p2 = rc[i], rc[i + 1]
                            new_rc.append(p1)
                            for j in range(n_to_add):
                                t = (j + 1) / (n_to_add + 1)
                                mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                                if noise_sigma > 0:
                                    mid = (mid[0] + float(np.random.normal(0, noise_sigma)), mid[1] + float(np.random.normal(0, noise_sigma)))
                                new_rc.append(mid)
                        new_rc.append(rc[-1])
                        holes.append(new_rc)
                    else:
                        holes.append(rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    gdf['geometry'] = gdf['geometry'].apply(lambda geom: add_vertices_to_geom(geom, 50, strength_level=1))
    # 3 对象删除50%
    n_total = len(gdf)
    n_del = int(n_total * 0.5)
    if n_del > 0 and n_total > 0:
        idx = random.sample(range(n_total), min(n_del, n_total))
        gdf = gdf.drop(idx).reset_index(drop=True)
    # 4 噪声扰动 强度0.8 比例50%
    def jitter_vertices_geom(geom, pct: int, strength: float):
        try:
            if geom is None:
                return None
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                n = len(coords)
                k = max(1, int(n * pct / 100))
                idx = list(range(n))
                chosen = set(random.sample(idx, min(k, len(idx))))
                new_coords = []
                for i, (x, y) in enumerate(coords):
                    if i in chosen:
                        new_coords.append((x + random.uniform(-strength, strength), y + random.uniform(-strength, strength)))
                    else:
                        new_coords.append((x, y))
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                n = len(ext)
                k = max(1, int(n * pct / 100))
                idx = list(range(n))
                chosen = set(random.sample(idx, min(k, len(idx))))
                new_ext = []
                for i, (x, y) in enumerate(ext):
                    if i in chosen:
                        new_ext.append((x + random.uniform(-strength, strength), y + random.uniform(-strength, strength)))
                    else:
                        new_ext.append((x, y))
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    n2 = len(rc)
                    k2 = max(1, int(n2 * pct / 100))
                    idx2 = list(range(n2))
                    chosen2 = set(random.sample(idx2, min(k2, len(idx2))))
                    new_rc = []
                    for i, (x, y) in enumerate(rc):
                        if i in chosen2:
                            new_rc.append((x + random.uniform(-strength, strength), y + random.uniform(-strength, strength)))
                        else:
                            new_rc.append((x, y))
                    holes.append(new_rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    gdf['geometry'] = gdf['geometry'].apply(lambda geom: jitter_vertices_geom(geom, 50, 0.8))
    # 5 沿Y轴中心裁剪50%
    bdf = gdf.geometry.bounds
    gdf = gdf[bdf['miny'] < mid_y].reset_index(drop=True)
    # 6 平移 (20, 40)
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_translate(geom, 20, 40) if geom is not None else None)
    
    # ✅ 使用全局中心作为变换原点，与zNC-Test/Fig12策略一致
    bounds = gdf.total_bounds
    global_center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    
    # 7 缩放 90%
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 0.9, 0.9, origin=global_center) if geom is not None else None)
    # 8 旋转 180°
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_rotate(geom, 180, origin=global_center) if geom is not None else None)
    # 9 Y轴镜像翻转
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 1.0, -1.0, origin=global_center) if geom is not None else None)
    # 10 反转顶点 → 反转对象
    def _reverse_vertices_geom(geom):
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
    gdf['geometry'] = gdf['geometry'].apply(_reverse_vertices_geom)
    gdf = gdf.iloc[::-1].reset_index(drop=True)

    try:
        gdf.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
    except Exception:
        pass

    out_path = subdir / 'compound_seq_all.shp'
    try:
        gdf.to_file(out_path, driver='ESRI Shapefile')
    except Exception as exc:
        print('保存失败:', exc)
    return {'compound_seq_all': out_path}


def run_evaluation():
    print('=== Wu25 Fig12：复合(顺序) 攻击 NC 评估 ===')
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig12_compound_seq_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig12: {csv_check}')
            return
    if not (SCRIPT_DIR / WATERMARK).exists():
        print('缺少水印图像: ', WATERMARK)
        return
    if len(VECTOR_FILES) == 0:
        print('未发现嵌入矢量：请先在 embed/ 生成 Cat32_*.shp')
        return

    # 创建输出目录
    out_dir = SCRIPT_DIR / 'extract' / 'watermark' / 'Fig12'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []  # type: ignore
    for vector_file, file_name in zip(VECTOR_FILES, FILE_NAMES):
        print(f'处理: {file_name}')
        attacked_map = generate_attacked_variants(Path(vector_file))
        shp_path = attacked_map.get('compound_seq_all')
        if not shp_path:
            continue
        try:
            extracted_img_path, eva = extract(str(shp_path), str(SCRIPT_DIR / WATERMARK))
            nc_value = float(eva.get('NC', 0.0))
            rows.append({'file': file_name, 'nc': nc_value})
            print(f'  NC={nc_value:.4f}')
            
            # 复制提取的水印图片到 Fig12 目录
            from PIL import Image
            import shutil
            temp_img_path = SCRIPT_DIR / 'extract' / 'watermark' / 'compound_seq_all.png'
            if temp_img_path.exists():
                out_img = out_dir / f'Cat32_{file_name}_extracted.png'
                shutil.copy(str(temp_img_path), str(out_img))
        except Exception as exc:
            print('  失败:', file_name, exc)
            rows.append({'file': file_name, 'nc': 0.0})

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 防御性检查：避免 rows 全部失败导致后续通过 df['file'] 抛出 KeyError
    if df.empty or 'file' not in df.columns:
        print('没有有效的评估结果，跳过 Fig12 的汇总与绘图（rows 长度:', len(rows), ')')
        print('=== 完成（无有效结果） ===')
        return

    df = df.sort_values('file')

    hierarchical_rows: List[Dict] = []  # type: ignore
    hierarchical_rows.append({'复合攻击(顺序)': '复合(Fig1→Fig10)', 'VGAT': '', '类型': 'header'})
    for _, row in df.iterrows():
        hierarchical_rows.append({'复合攻击(顺序)': f"  {row['file']}", 'VGAT': f"{row['nc']:.6f}", '类型': 'data'})
    hierarchical_rows.append({'复合攻击(顺序)': '  Average', 'VGAT': f"{df['nc'].mean():.6f}", '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    csv_path = DIR_RESULTS / 'fig12_compound_seq_nc.csv'
    xlsx_path = DIR_RESULTS / 'fig12_compound_seq_nc.xlsx'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
    except Exception as e:
        print('Excel保存警告:', e)

    plt.figure(figsize=(10, 6))
    bases = list(df['file'])
    vals = list(df['nc'])
    x = np.arange(len(bases))
    plt.bar(x, vals, alpha=0.85)
    plt.axhline(df['nc'].mean(), color='k', linestyle='--', label='平均')
    plt.xticks(x, bases, rotation=20)
    plt.ylabel('NC')
    plt.title('复合攻击(一次性顺序 Fig1→Fig10) 的NC鲁棒性（Wu25/Fig12）')
    plt.legend()
    fig_path = DIR_RESULTS / 'fig12_compound_seq_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()

    print('结果表保存:', csv_path)
    print('Excel保存:', xlsx_path)
    print('柱状图保存:', fig_path)
    print('=== 完成 ===')


if __name__ == '__main__':
    run_evaluation()

