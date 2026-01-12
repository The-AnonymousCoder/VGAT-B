# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig1：Tan24 对比试验（删点攻击 10%→90%）

目标：在保持 Tan24 零水印方法不变的前提下，采用与 zNC-Test/Fig1.py 一致的删点逻辑、总体流程与结果输出结构，
以 Tan24 数据为输入，生成 `Tan24/NC-Results/Fig1` 下的 NC 结果（CSV/XLSX/PNG）。
"""

from pathlib import Path
from typing import Dict, List
import os
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
    from shapely.geometry import LineString, Polygon, MultiPolygon  # type: ignore
    from shapely.geometry.polygon import orient  # type: ignore
except Exception as exc:
    print("需要安装 shapely: pip install shapely")

# Tan24 提取依赖
import cv2
from Zero_watermarking import *  # noqa: F401,F403 - 使用 Tan24 现有方法
from Extract_zero_watermarking import XOR2, Arnold_Decrypt  # noqa: F401
from NC import NC  # 使用 Tan24 的 NC 计算


# 路径（均相对 Tan24 目录）
SCRIPT_DIR = Path(__file__).resolve().parent

# Matplotlib 中文字体配置，避免中文字符显示为方块或缺字形
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

DIR_PSO = SCRIPT_DIR / 'pso_data'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'delete' / 'Fig1_delete_vertices'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig1'
KEEP_EXISTING = True

# 资源
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

# 配置（与 Fig1 一致：10..90，步长10）
DELETE_PCTS: List[int] = list(range(10, 100, 10))


def discover_inputs() -> List[Path]:
    """扫描 Tan24 下 `pso_data` 中的8个 .shp 输入。"""
    print('[Step1] 扫描输入数据: ', DIR_PSO)
    if not DIR_PSO.exists():
        print('未找到目录: ', DIR_PSO)
        return []
    # 过滤异常侧车文件（如 macOS 产生的 ._*.shp），并确保 .dbf/.shx 存在
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


def delete_vertices_from_geom(geom, pct: int):
    """与 zNC-Test/Fig1 相同的删点逻辑（Shapely）。"""
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
            return Polygon(new_ext, holes=holes if holes else None)
    except Exception:
        return geom
    return geom


def construction_fast(Xlist: List[List[float]], feature_num: int, Lst_WaterMark: List[int]):
    """精确但向量化的快速构造函数，等价于原始 Construction 的成对投票。

    将所有要素长度 Ni 做到模 W(=水印长度) 的直方图，并按奇偶拆分，
    再对类别对进行组合计数，复杂度 O(W^2) 而非 O(n^2)。
    """
    W = len(Lst_WaterMark)

    # 统计 Ni 的模值与奇偶
    counts_even = [0] * W
    counts_odd = [0] * W
    for coords in Xlist:
        Ni = len(coords)
        r = Ni % W
        if (Ni % 2) == 0:
            counts_even[r] += 1
        else:
            counts_odd[r] += 1

    # 聚合非空类别 (r, parity, count)
    categories: List[tuple] = []
    for r in range(W):
        c0 = counts_even[r]
        c1 = counts_odd[r]
        if c0:
            categories.append((r, 0, c0))
        if c1:
            categories.append((r, 1, c1))

    # 计算对所有无序对的贡献
    acc = [0] * W
    for i in range(len(categories)):
        r1, p1, c1 = categories[i]
        # 同一类别内部对
        if c1 >= 2:
            k = (r1 * r1) % W
            acc[k] += (c1 * (c1 - 1) // 2)  # 同奇偶 -> +1
        # 跨类别对
        for j in range(i + 1, len(categories)):
            r2, p2, c2 = categories[j]
            k = (r1 * r2) % W
            sign = 1 if p1 == p2 else -1
            acc[k] += sign * (c1 * c2)

    # 阈值化到 0/255
    List_Fea = [255 if v > 0 else 0 for v in acc]
    return List_Fea


def generate_delete_attacks(shp_files: List[Path]) -> Dict[str, Dict[int, Path]]:
    """生成 10%~90% 删点攻击（Shapely逻辑），输出为 Shapefile。

    返回: {base_name: {pct: out_path}}
    """
    print('[Step2] 生成删点攻击 ->', DIR_ATTACKED)
    outputs: Dict[str, Dict[int, Path]] = {}
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)

    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return outputs

    # 固定随机种子
    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    for src in shp_files:
        base = src.stem
        subdir = DIR_ATTACKED / base
        if subdir.exists():
            # 如果设置了 KEEP_EXISTING 并且已有 attacked 文件，则保留并重用
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
        for pct in DELETE_PCTS:
            try:
                attacked = gdf.copy()
                # 删点
                attacked['geometry'] = attacked['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, pct))

                # 修正多边形环方向，尽量降低 OGR 的 winding order 告警
                def _fix_geom(g):
                    if g is None:
                        return g
                    try:
                        if isinstance(g, Polygon):
                            return orient(g, sign=1.0)
                        if isinstance(g, MultiPolygon):
                            return MultiPolygon([orient(pg, sign=1.0) for pg in g.geoms])
                        return g
                    except Exception:
                        return g

                attacked['geometry'] = attacked['geometry'].apply(_fix_geom)

                # 继承/设置 CRS；若源缺失，使用 WGS84 以避免无投影信息告警
                try:
                    base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'
                    attacked.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
                except Exception:
                    pass
                out_path = subdir / f'delete_{pct}pct_vertices.shp'
                # 如果已存在且 KEEP_EXISTING，跳过写入
                if KEEP_EXISTING and out_path.exists():
                    print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                    outputs[base][pct] = out_path
                else:
                    # 保存 Shapefile
                    attacked.to_file(out_path, driver='ESRI Shapefile')
                    print(f'输出: {base}/{out_path.name}')
                    outputs[base][pct] = out_path
            except Exception as exc:
                print('attack_error', base, pct, exc)
                continue

    return outputs


def extract_watermark_from_attacked_vector(attacked_shp_path: str, original_watermark_path: str):
    """从受攻击的矢量文件中提取零水印（沿用 Tan24 方法）。"""
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

        # 1) 父目录直接匹配
        if parent_name in map_names:
            base_name = parent_name

        # 2) 向上迭代祖先目录名匹配
        if base_name is None:
            for anc in p.parents:
                if anc.name in map_names:
                    base_name = anc.name
                    break

        # 3) 在文件名或路径中查找 map 名称
        if base_name is None:
            for map_name in map_names:
                if (map_name in filename) or (map_name in normalized_path):
                    base_name = map_name
                    break

        # 4) 尝试从本模块的 pso_data 目录扫描候选名
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

        # 5) 最终失败则记录并抛错（保留完整路径用于排查）
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
        # 使用加速版构建，避免在特征数量极大时 O(n^2) 过慢
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


def evaluate_nc(attacked_map: Dict[str, Dict[int, Path]]):
    """对删点攻击版本评估NC并保存层次化结果（CSV/XLSX/PNG）。"""
    print('[Step3] 评估NC ->', DIR_RESULTS)
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    # 如果设置了 KEEP_EXISTING 并且结果 CSV 已存在，则跳过评估
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig1 NC 评估: {csv_check}')
            return

    # 收集结果
    rows: List[Dict] = []
    for base, pct_map in attacked_map.items():
        for pct, shp_path in sorted(pct_map.items()):
            try:
                eval_result = extract_watermark_from_attacked_vector(str(shp_path), str(CAT32_PATH))
                nc_value = float(eval_result.get('NC', 0.0))
                rows.append({'base': base, 'attack_pct': pct, 'nc': nc_value})
            except Exception as exc:
                print('nc_eval_error', base, pct, exc)
                continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 组织为与 Fig1 相同的层次结构（删除比例 -> 每图NC -> 平均）
    hierarchical_rows: List[Dict] = []
    base_names = sorted(df['base'].unique())
    attack_pcts = sorted(df['attack_pct'].unique())

    for pct in attack_pcts:
        hierarchical_rows.append({'删除比例': f'{pct}%', 'VGAT': '', '类型': 'header'})
        for base in base_names:
            match = df[(df['base'] == base) & (df['attack_pct'] == pct)]
            nc_value = float(match['nc'].iloc[0]) if len(match) > 0 else 0.0
            hierarchical_rows.append({'删除比例': f'  {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['attack_pct'] == pct]['nc'].mean())
        hierarchical_rows.append({'删除比例': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV
    csv_path = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 保存 Excel（带少量格式）
    xlsx_path = DIR_RESULTS / 'fig1_delete_vertices_nc.xlsx'
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

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    for base, sub in df.groupby('base'):
        sub2 = sub.sort_values('attack_pct')
        plt.plot(sub2['attack_pct'], sub2['nc'], '-o', alpha=0.7, label=base)
    avg = df.groupby('attack_pct')['nc'].mean().reset_index()
    plt.plot(avg['attack_pct'], avg['nc'], 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xlabel('删点比例（%）')
    plt.ylabel('NC')
    plt.title('Fig1：Tan24 删点攻击(10%~90%) 的NC鲁棒性')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig1_delete_vertices_nc.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)
    print('曲线图保存: ', fig_path.name)


def main():
    print('=== Fig1：Tan24 删点攻击对比试验（与 Fig1 逻辑一致）===')

    inputs = discover_inputs()
    if not inputs:
        print('无输入数据，终止。')
        return

    attacked_map = generate_delete_attacks(inputs)
    evaluate_nc(attacked_map)
    print('=== 完成 ===')


if __name__ == '__main__':
    main()
