# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig2：Tan24 对比试验（增加顶点 强度0/1/2；比例10%→90%）

目标：在保持 Tan24 零水印方法不变的前提下，采用与 zNC-Test/Fig2.py 一致的“增加顶点”攻击逻辑、总体流程与结果输出结构，
以 Tan24 数据为输入，生成 `Tan24/NC-Results/Fig2` 下的 NC 结果（CSV/XLSX/PNG）。
"""

from pathlib import Path
from typing import Dict, List, Tuple
import random
import pickle

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
    from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, MultiPoint  # type: ignore
    from shapely.geometry.polygon import orient  # type: ignore
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
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'add' / 'Fig2_add_vertices'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig2'
KEEP_EXISTING = True

# 资源
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

# 配置
STRENGTHS: List[int] = [0, 1, 2]
ADD_PCTS: List[int] = list(range(10, 100, 10))


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


def add_vertices_to_geom(geom, strength: int, pct: int):
    """按 zNC-Test/Fig2 逻辑实现的增加顶点攻击。
    - 强度0：在线段/外环/内环中等分段插入新顶点
    - 强度1：等分插入并添加小幅噪声（~N(0, 0.01)）
    - 强度2：等分插入并添加较大噪声（~N(0, 0.05)）
    """
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
            # 外环
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
            # 内环
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
    categories: List[Tuple[int, int, int]] = []
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


def generate_add_attacks(shp_files: List[Path]) -> Dict[str, Dict[Tuple[int, int], Path]]:
    """生成 强度×比例 的增加顶点攻击，输出 Shapefile。
    返回: {base: {(strength, pct): path}}
    """
    print('[Step2] 生成增加顶点攻击 ->', DIR_ATTACKED)
    outputs: Dict[str, Dict[Tuple[int, int], Path]] = {}
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
            # 若要求保留已生成攻击文件且存在，则复用
            if KEEP_EXISTING and any(subdir.glob('*.shp')):
                pass
            else:
                import shutil
                shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_shp_error', src.name, exc)
            continue

        outputs[base] = {}
        for strength in STRENGTHS:
            for pct in ADD_PCTS:
                try:
                    attacked = gdf.copy()
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: add_vertices_to_geom(geom, strength, pct))

                    # 继承/设置 CRS，源缺失则默认 WGS84
                    try:
                        base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'
                        attacked.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
                    except Exception:
                        pass

                    out_path = subdir / f'add_strength{strength}_{pct}pct_vertices.shp'
                    if KEEP_EXISTING and out_path.exists():
                        print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                        outputs[base][(strength, pct)] = out_path
                    else:
                        attacked.to_file(out_path, driver='ESRI Shapefile')
                        print(f'输出: {base}/{out_path.name}')
                        outputs[base][(strength, pct)] = out_path
                except Exception as exc:
                    print('attack_error', base, strength, pct, exc)
                    continue

    return outputs


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
                    ef.write(err_msg + '\\n')
            except Exception:
                pass
            raise ValueError(err_msg)

        zero_watermark_file = SCRIPT_DIR / 'zero_watermark' / f'{base_name}_zero.png'
        if not zero_watermark_file.exists():
            try:
                from .ensure_zero import ensure_zero_for
            except Exception:
                from ensure_zero import ensure_zero_for
            ensure_zero_for(base_name)
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


def evaluate_nc(attacked_map: Dict[str, Dict[Tuple[int, int], Path]]):
    """评估增加顶点攻击版本的 NC，并保存 CSV/XLSX/PNG。"""
    print('[Step3] 评估NC ->', DIR_RESULTS)
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig2: {csv_check}')
            return

    rows: List[Dict] = []
    for base, sp_map in attacked_map.items():
        for (strength, pct), shp_path in sorted(sp_map.items()):
            try:
                eval_result = extract_watermark_from_attacked_vector(str(shp_path), str(CAT32_PATH))
                nc_value = float(eval_result.get('NC', 0.0))
                rows.append({'base': base, 'strength': strength, 'add_pct': pct, 'nc': nc_value})
            except Exception as exc:
                print('nc_eval_error', base, strength, pct, exc)
                continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 组织层次结构（与 zNC-Test/Fig2 一致）：强度 -> 比例 -> 每图 -> 平均
    hierarchical_rows: List[Dict] = []
    base_names = sorted(df['base'].unique())

    for strength in STRENGTHS:
        hierarchical_rows.append({'增加顶点': f'强度: {strength}', 'VGAT': '', '类型': 'header'})
        for pct in ADD_PCTS:
            hierarchical_rows.append({'增加顶点': f'  {pct}%', 'VGAT': '', '类型': 'subheader'})
            for base in base_names:
                sub = df[(df['base'] == base) & (df['strength'] == strength) & (df['add_pct'] == pct)]
                nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
                hierarchical_rows.append({'增加顶点': f'    {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
            avg_nc = float(df[(df['strength'] == strength) & (df['add_pct'] == pct)]['nc'].mean())
            hierarchical_rows.append({'增加顶点': '    Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV
    csv_path = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 保存 Excel
    xlsx_path = DIR_RESULTS / 'fig2_add_vertices_nc.xlsx'
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
                    elif row['类型'] == 'subheader' and _copy is not None:
                        nf = _copy(cell_a.font); nf.bold = True; cell_a.font = nf
                    elif row['类型'] == 'average' and _copy is not None:
                        nf = _copy(cell_a.font); nf.bold = True; nf.italic = True; cell_a.font = nf
                        cell_b = ws[f'B{idx+2}']
                        nfb = _copy(cell_b.font); nfb.bold = True; nfb.italic = True; cell_b.font = nfb
                    elif row['类型'] == 'data' and _copy is not None:
                        na = _copy(cell_a.alignment); na.horizontal = 'left'
                        try:
                            na.indent = 2
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

    # 绘制曲线：每强度一张 + 综合
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
        plt.title(f'强度 {strength}：增加顶点攻击的NC鲁棒性')
        plt.legend(loc='best', fontsize=6, ncol=2)
        plt.ylim(0, 1.1)

    plt.subplot(2, 2, 4)
    for strength in STRENGTHS:
        avg = df[df['strength'] == strength].groupby('add_pct')['nc'].mean().reset_index()
        plt.plot(avg['add_pct'], avg['nc'], '-o', linewidth=2, label=f'强度 {strength}', markersize=6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('增加顶点比例（%）')
    plt.ylabel('NC')
    plt.title('综合对比：不同强度的NC鲁棒性')
    plt.legend(loc='best')
    plt.ylim(0, 1.1)

    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig2_add_vertices_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)
    print('曲线图保存: ', fig_path.name)


def main():
    print('=== Fig2：Tan24 增加顶点鲁棒性测试（与 Fig2 逻辑一致）===')

    inputs = discover_inputs()
    if not inputs:
        print('无输入数据，终止。')
        return

    attacked_map = generate_add_attacks(inputs)
    evaluate_nc(attacked_map)
    print('=== 完成 ===')


if __name__ == '__main__':
    main()