# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig12：Tan24 对比试验（复合攻击：一次性按 Fig1→Fig10 顺序依次执行）

顺序：
1) 顶点删除 10%
2) 顶点增加 强度1 比例50%
3) 对象删除 50%
4) 噪声扰动 强度0.8 比例50%
5) 沿Y轴中心裁剪50%
6) 平移 X=20, Y=40
7) 缩放 90%
8) 顺时针旋转 180°
9) Y轴镜像翻转
10) 顺序操作：反转顶点顺序 → 反转对象顺序

目标：在保持 Tan24 零水印方法不变的前提下，采用与 zNC-Test/Fig12.py 一致的“顺序复合”总体流程，
以 Tan24 数据为输入，输出到 `Tan24/NC-Results/Fig12`（CSV/XLSX/PNG）。
"""

from pathlib import Path
from typing import Dict, List
import shutil
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# 依赖（地理数据 & 几何）
try:
    import geopandas as gpd  # type: ignore
except Exception as exc:
    print("需要安装 geopandas: pip install geopandas fiona pyproj shapely")
    print("geopandas_import_error", exc)
    gpd = None  # type: ignore

try:
    from shapely.geometry import LineString, Polygon  # type: ignore
    from shapely.affinity import translate as shp_translate, scale as shp_scale, rotate as shp_rotate  # type: ignore
except Exception as exc:
    print("需要安装 shapely: pip install shapely")

# Tan24 提取依赖
import cv2
from Zero_watermarking import *  # noqa: F401,F403 - 使用 Tan24 现有方法
from Extract_zero_watermarking import XOR2, Arnold_Decrypt  # noqa: F401
from NC import NC  # 使用 Tan24 的 NC 计算


# Matplotlib 中文字体配置
try:
    preferred_fonts = [
        'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Songti SC',
        'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for fname in preferred_fonts:
        if fname in available:
            chosen = fname
            break
    if chosen:
        rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        rcParams['font.sans-serif'] = preferred_fonts + ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# 路径（均相对 Tan24 目录）
SCRIPT_DIR = Path(__file__).resolve().parent
DIR_PSO = SCRIPT_DIR / 'pso_data'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'compound_seq' / 'Fig12_compound_seq'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig12'
KEEP_EXISTING = True

# 资源
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'


def discover_inputs() -> List[Path]:
    """扫描 Tan24 下 `pso_data` 中的8个 .shp 输入。"""
    print('[Step1] 扫描输入数据: ', DIR_PSO)
    if not DIR_PSO.exists():
        print('未找到目录: ', DIR_PSO)
        return []
    shp_files: List[Path] = [p for p in sorted(DIR_PSO.glob('*.shp')) if not p.name.startswith('._')]
    valid: List[Path] = []
    for p in shp_files:
        try:
            if p.with_suffix('.dbf').exists() and p.with_suffix('.shx').exists():
                valid.append(p)
        except Exception:
            continue
    selected = valid[:6]
    print('发现文件: ', [p.name for p in selected])
    return selected


# 基础操作用于顺序执行

def delete_vertices_from_geom(geom, pct: int):
    try:
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


def add_vertices_to_geom(geom, pct: int, strength_level: int = 1):
    noise_sigma = 0.01 if strength_level == 1 else 0.0
    try:
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


def jitter_vertices_geom(geom, pct: int, strength: float):
    try:
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            n = len(coords)
            k = max(1, int(n * pct / 100))
            idx = list(range(n))
            chosen = set(random.sample(idx, min(k, len(idx))))
            new_coords = []
            for i, c in enumerate(coords):
                if i in chosen:
                    new_coords.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
                else:
                    new_coords.append((c[0], c[1]))
            return LineString(new_coords)
        elif geom.geom_type == 'Polygon':
            ext = list(geom.exterior.coords)
            n = len(ext)
            k = max(1, int(n * pct / 100))
            idx = list(range(n))
            chosen = set(random.sample(idx, min(k, len(idx))))
            new_ext = []
            for i, c in enumerate(ext):
                if i in chosen:
                    new_ext.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
                else:
                    new_ext.append((c[0], c[1]))
            holes = []
            for ring in geom.interiors:
                rc = list(ring.coords)
                n2 = len(rc)
                k2 = max(1, int(n2 * pct / 100))
                idx2 = list(range(n2))
                chosen2 = set(random.sample(idx2, min(k2, len(idx2))))
                new_rc = []
                for i, c in enumerate(rc):
                    if i in chosen2:
                        new_rc.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
                    else:
                        new_rc.append((c[0], c[1]))
                holes.append(new_rc)
            return Polygon(new_ext, holes=holes if holes else None)
    except Exception:
        pass
    return geom


def generate_compound_seq_attacks(shp_files: List[Path]) -> Dict[str, Dict[str, Path]]:
    """对每个 base 一次性按 Fig1→Fig10 顺序依次执行，仅输出一个最终版本（Shapefile）。

    返回: {base: {"compound_seq_all": out_path}}
    """
    print('[Step2] 生成复合(顺序)攻击 ->', DIR_ATTACKED)
    outputs: Dict[str, Dict[str, Path]] = {}
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)

    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return outputs

    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    for src in shp_files:
        base = src.stem
        subdir = DIR_ATTACKED / base
        if subdir.exists():
            if KEEP_EXISTING and any(subdir.glob('*.shp')):
                pass
            else:
                shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_shp_error', src.name, exc)
            continue

        base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'

        # 裁剪辅助
        bounds = gdf.total_bounds
        mid_y = (bounds[1] + bounds[3]) / 2

        # 1 顶点删除10%
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, 10))
        # 2 顶点增加 强度1 比例50%
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: add_vertices_to_geom(geom, 50, strength_level=1))
        # 3 对象删除50%
        n_total = len(gdf)
        n_del = int(n_total * 0.5)
        if n_del > 0 and n_total > 0:
            idx = random.sample(range(n_total), min(n_del, n_total))
            gdf = gdf.drop(idx).reset_index(drop=True)
        # 4 噪声扰动 强度0.8 比例50%
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: jitter_vertices_geom(geom, 50, 0.8))
        # 5 沿Y轴中心裁剪50%
        bdf = gdf.geometry.bounds
        gdf = gdf[bdf['miny'] < mid_y].reset_index(drop=True)
        # 6 平移 X=20, Y=40
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_translate(geom, 20, 40))
        
        # ✅ 使用全局中心作为变换原点，与zNC-Test/Fig12策略一致
        bounds = gdf.total_bounds
        global_center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        
        # 7 缩放90%
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 0.9, 0.9, origin=global_center))
        # 8 旋转180°
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_rotate(geom, 180, origin=global_center))
        # 9 Y轴镜像翻转
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 1.0, -1.0, origin=global_center))
        # 10 反转顶点顺序 → 反转对象顺序
        def _reverse_vertices_geom(geom):
            try:
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
            if KEEP_EXISTING and out_path.exists():
                print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                outputs.setdefault(base, {})['compound_seq_all'] = out_path
            else:
                gdf.to_file(out_path, driver='ESRI Shapefile')
                print(f'输出: {base}/{out_path.name}')
                outputs.setdefault(base, {})['compound_seq_all'] = out_path
        except Exception as exc:
            print('attack_save_error', base, exc)
            continue

    return outputs


def construction_fast(Xlist: List[List[float]], feature_num: int, Lst_WaterMark: List[int]):
    """与 Fig1 相同的 O(W^2) 组合计数加速构造。"""
    W = len(Lst_WaterMark)
    counts_even = [0] * W
    counts_odd = [0] * W
    for coords in Xlist:
        Ni = len(coords)
        r = Ni % W
        if (Ni % 2) == 0:
            counts_even[r] += 1
        else:
            counts_odd[r] += 1
    categories = []
    for r in range(W):
        if counts_even[r]:
            categories.append((r, 0, counts_even[r]))
        if counts_odd[r]:
            categories.append((r, 1, counts_odd[r]))
    acc = [0] * W
    for i in range(len(categories)):
        r1, p1, c1 = categories[i]
        if c1 >= 2:
            k = (r1 * r1) % W
            acc[k] += (c1 * (c1 - 1) // 2)
        for j in range(i + 1, len(categories)):
            r2, p2, c2 = categories[j]
            k = (r1 * r2) % W
            acc[k] += (c1 * c2) if (p1 == p2) else -(c1 * c2)
    return [255 if v > 0 else 0 for v in acc]


def extract_watermark_from_attacked_vector(attacked_shp_path: str, original_watermark_path: str):
    """从受攻击矢量中按 Tan24 方法提取零水印并返回 NC/BER。"""
    try:
        img_original = cv2.imread(original_watermark_path, 0)
        img_original_deal = Watermark_deal(img_original)

        p = Path(attacked_shp_path)
        filename = p.stem
        parent_name = p.parent.name
        map_names = ['BOUL', 'gis_osm_landuse_a_free_1', 'gis_osm_railways_free_1', 'gis_osm_natural_free_1', 'gis_osm_waterways_free_1', 'BRGA', 'HYDP', 'RESA']
        base_name = None
        normalized_path = attacked_shp_path.replace('\\', '/')
        print(f"[debug] resolve base_name for {attacked_shp_path}; parent={parent_name}", flush=True)

        if parent_name in map_names:
            base_name = parent_name
        if base_name is None:
            for anc in p.parents:
                if anc.name in map_names:
                    base_name = anc.name
                    break
        if base_name is None:
            for map_name in map_names:
                if (map_name in filename) or (map_name in normalized_path):
                    base_name = map_name
                    break
        if base_name is None:
            try:
                pso_dir = SCRIPT_DIR / 'pso_data'
                if pso_dir.exists():
                    for f in pso_dir.iterdir():
                        if f.is_file():
                            cand = f.stem
                            if cand in filename or cand in normalized_path:
                                base_name = cand
                                break
            except Exception:
                pass
        if base_name is None:
            err_msg = f"无法识别的地图文件: {filename}; full_path: {attacked_shp_path}; parts: {p.parts}"
            try:
                with open(SCRIPT_DIR / 'extract_errors.log', 'a', encoding='utf-8') as ef:
                    ef.write(err_msg + '\n')
            except Exception:
                pass
            raise ValueError(err_msg)

        zero_watermark_file = SCRIPT_DIR / 'zero_watermark' / f'{base_name}_zero.png'
        if not zero_watermark_file.exists():
            raise FileNotFoundError(f"零水印文件不存在: {zero_watermark_file}")

        img_zero = cv2.imread(str(zero_watermark_file), 0)
        img_zero_deal = Watermark_deal(img_zero)
        List_Zero = img_zero_deal.flatten()
        print(f"[debug] zero file: {zero_watermark_file}, List_Zero length: {len(List_Zero)}", flush=True)

        XLst_attacked, YLst_attacked, feature_num_attacked = Read_Shapfile(attacked_shp_path)
        print(f"[debug] Read_Shapfile -> features: {feature_num_attacked} for {attacked_shp_path}", flush=True)
        List_Fea_attacked = construction_fast(XLst_attacked, feature_num_attacked, List_Zero)
        try:
            print(f"[debug] List_Fea_attacked length: {len(List_Fea_attacked)}", flush=True)
        except Exception:
            pass

        Lst_WaterMark_extract = XOR2(List_Fea_attacked, List_Zero)
        Re_mark = np.array(Lst_WaterMark_extract).reshape(32, 32)
        Decode_image = Arnold_Decrypt(Re_mark)

        nc_value = NC(img_original, Decode_image)
        different_bits = np.sum(img_original_deal != Decode_image)
        total_bits = img_original_deal.size
        ber_value = different_bits / total_bits

        return {
            'NC': nc_value,
            'BER': ber_value,
            'extracted_watermark': Decode_image,
            'feature_num': feature_num_attacked
        }
    except Exception as e:
        print(f"      ❌ 提取失败: {str(e)}")
        return {'NC': 0.0, 'BER': 1.0, 'extracted_watermark': None, 'feature_num': 0}


def evaluate_nc(attacked_map: Dict[str, Dict[str, Path]]):
    """评估复合顺序攻击版本的 NC，并保存 CSV/XLSX/PNG。"""
    print('[Step3] 评估NC ->', DIR_RESULTS)
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig12_compound_seq_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig12: {csv_check}')
            return

    # 创建输出目录
    out_dir = SCRIPT_DIR / 'extract' / 'watermark' / 'Fig12'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []  # type: ignore
    for base, file_map in attacked_map.items():
        shp_path = file_map.get('compound_seq_all')
        if not shp_path:
            continue
        try:
            eval_result = extract_watermark_from_attacked_vector(str(shp_path), str(CAT32_PATH))
            nc_value = float(eval_result.get('NC', 0.0))
            rows.append({'base': base, 'nc': nc_value})
            
            # 保存提取的水印图片
            extracted_watermark = eval_result.get('extracted_watermark')
            if extracted_watermark is not None:
                out_img = out_dir / f'Cat32_{base}_extracted.png'
                # extracted_watermark 已经是 0-255 范围，直接转换为 uint8
                to_save = extracted_watermark.astype(np.uint8)
                cv2.imwrite(str(out_img), to_save)
        except Exception as exc:
            print('nc_eval_error', base, exc)
            continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows).sort_values('base')

    # 层次表：仅一个“复合(Fig1→Fig10)”分组
    hierarchical_rows: List[Dict] = []  # type: ignore
    hierarchical_rows.append({'复合攻击(顺序)': '复合(Fig1→Fig10)', 'VGAT': '', '类型': 'header'})
    for _, row in df.iterrows():
        hierarchical_rows.append({'复合攻击(顺序)': f"  {row['base']}", 'VGAT': f"{row['nc']:.6f}", '类型': 'data'})
    hierarchical_rows.append({'复合攻击(顺序)': '  Average', 'VGAT': f"{df['nc'].mean():.6f}", '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV/Excel
    csv_path = DIR_RESULTS / 'fig12_compound_seq_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    xlsx_path = DIR_RESULTS / 'fig12_compound_seq_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            ws = writer.sheets['NC结果']
            try:
                from copy import copy as _copy
            except Exception:
                _copy = None
            try:
                ws.column_dimensions['A'].width = 30
                ws.column_dimensions['B'].width = 15
                for idx, row in hierarchical_df.iterrows():
                    cell_a = ws[f'A{idx+2}']
                    if row['类型'] == 'header' and _copy is not None:
                        nf = _copy(cell_a.font); nf.bold = True; cell_a.font = nf
                    elif row['类型'] == 'average' and _copy is not None:
                        nf = _copy(cell_a.font); nf.bold = True; nf.italic = True; cell_a.font = nf
                        cell_b = ws[f'B{idx+2}']
                        nfb = _copy(cell_b.font); nfb.bold = True; nfb.italic = True; cell_b.font = nfb
                    elif row['类型'] == 'data' and _copy is not None:
                        na = _copy(cell_a.alignment); na.horizontal = 'left'
                        try:
                            na.indent = 1
                        except Exception:
                            pass
                        cell_a.alignment = na
            except Exception:
                pass
    except Exception as e:
        print('Excel保存失败:', e)
        try:
            hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
        except Exception:
            pass

    # 柱状图：各基底的NC + 平均
    plt.figure(figsize=(10, 6))
    bases = list(df['base'])
    vals = list(df['nc'])
    x = np.arange(len(bases))
    plt.bar(x, vals, alpha=0.85)
    plt.axhline(df['nc'].mean(), color='k', linestyle='--', label='平均')
    plt.xticks(x, bases, rotation=20)
    plt.ylabel('NC')
    plt.title('Fig12：Tan24 复合攻击(一次性顺序 Fig1→Fig10) 的NC鲁棒性')
    plt.legend()
    fig_path = DIR_RESULTS / 'fig12_compound_seq_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()

    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)
    print('柱状图保存: ', fig_path.name)


def main():
    print('=== Fig12：Tan24 复合攻击(一次性顺序) 鲁棒性测试（与 Fig12 逻辑一致）===')

    inputs = discover_inputs()
    if not inputs:
        print('无输入数据，终止。')
        return

    attacked_map = generate_compound_seq_attacks(inputs)
    evaluate_nc(attacked_map)
    print('=== 完成 ===')


if __name__ == '__main__':
    main()