# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig3：Tan24 对比试验（对象删除 10%→90%）

目标：在保持 Tan24 零水印方法不变的前提下，采用与 zNC-Test/Fig3.py 一致的“对象删除”攻击逻辑、总体流程与结果输出结构，
以 Tan24 数据为输入，生成 `Tan24/NC-Results/Fig3` 下的 NC 结果（CSV/XLSX/PNG）。
"""

from pathlib import Path
from typing import Dict, List
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
import os
KEEP_EXISTING = True
DIR_PSO = SCRIPT_DIR / 'pso_data'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'delete_objects' / 'Fig3_delete_objects'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig3'
KEEP_EXISTING = os.environ.get('KEEP_ATTACKED_FILES', os.environ.get('KEEP_EXISTING', '0')).lower() in ['1', 'true', 'yes']

# 资源
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

# 配置
DELETE_OBJ_PCTS: List[int] = list(range(10, 100, 10))


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


def generate_delete_object_attacks(shp_files: List[Path]) -> Dict[str, Dict[int, Path]]:
    """按 zNC-Test/Fig3 逻辑生成 10%-90% 对象删除攻击，输出 Shapefile。"""
    print('[Step2] 生成对象删除攻击 ->', DIR_ATTACKED)
    outputs: Dict[str, Dict[int, Path]] = {}
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
                import shutil
                shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_shp_error', src.name, exc)
            continue

        outputs[base] = {}
        n_total = len(gdf)
        for pct in DELETE_OBJ_PCTS:
            try:
                attacked = gdf.copy()
                num_to_delete = int(n_total * pct / 100)
                if num_to_delete > 0:
                    indices = list(range(n_total))
                    to_del = random.sample(indices, min(num_to_delete, len(indices)))
                    attacked = attacked.drop(to_del).reset_index(drop=True)
                # 继承/设置 CRS；若源缺失，默认 WGS84
                try:
                    base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'
                    attacked.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
                except Exception:
                    pass
                out_path = subdir / f'delete_{pct}pct_objects.shp'
                if KEEP_EXISTING and out_path.exists():
                    print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
                    outputs[base][pct] = out_path
                else:
                    attacked.to_file(out_path, driver='ESRI Shapefile')
                    print(f'输出: {base}/{out_path.name}')
                    outputs[base][pct] = out_path
            except Exception as exc:
                print('attack_error', base, pct, exc)
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


def evaluate_nc(attacked_map: Dict[str, Dict[int, Path]]):
    """评估对象删除攻击版本的 NC，并保存 CSV/XLSX/PNG。"""
    print('[Step3] 评估NC ->', DIR_RESULTS)
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig3_delete_objects_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig3: {csv_check}')
            return

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

    # 组织层次：删除对象比例 -> 8图 -> 平均
    hierarchical_rows: List[Dict] = []
    base_names = sorted(df['base'].unique())
    attack_pcts = sorted(df['attack_pct'].unique())

    for pct in attack_pcts:
        hierarchical_rows.append({'删除对象比例': f'{pct}%', 'VGAT': '', '类型': 'header'})
        for base in base_names:
            sub = df[(df['base'] == base) & (df['attack_pct'] == pct)]
            nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
            hierarchical_rows.append({'删除对象比例': f'  {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['attack_pct'] == pct]['nc'].mean())
        hierarchical_rows.append({'删除对象比例': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV
    csv_path = DIR_RESULTS / 'fig3_delete_objects_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 保存 Excel
    xlsx_path = DIR_RESULTS / 'fig3_delete_objects_nc.xlsx'
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

    # 绘图
    plt.figure(figsize=(10, 6))
    for base, sub in df.groupby('base'):
        sub2 = sub.sort_values('attack_pct')
        plt.plot(sub2['attack_pct'], sub2['nc'], '-o', alpha=0.7, label=base)
    avg = df.groupby('attack_pct')['nc'].mean().reset_index()
    plt.plot(avg['attack_pct'], avg['nc'], 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xlabel('删除对象比例（%）')
    plt.ylabel('NC')
    plt.title('Fig3：Tan24 对象删除(10%~90%) 的NC鲁棒性')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig3_delete_objects_nc.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)
    print('曲线图保存: ', fig_path.name)


def main():
    print('=== Fig3：Tan24 对象删除鲁棒性测试（与 Fig3 逻辑一致）===')

    inputs = discover_inputs()
    if not inputs:
        print('无输入数据，终止。')
        return

    attacked_map = generate_delete_object_attacks(inputs)
    evaluate_nc(attacked_map)
    print('=== 完成 ===')


if __name__ == '__main__':
    main()