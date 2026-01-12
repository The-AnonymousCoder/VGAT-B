# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig9：Tan24 对比试验（翻转攻击：X轴镜像、Y轴镜像、同时XY镜像）

目标：在保持 Tan24 零水印方法不变的前提下，采用与 zNC-Test/Fig9.py 一致的翻转攻击总体流程，
以 Tan24 数据为输入，输出到 `Tan24/NC-Results/Fig9`（CSV/XLSX/PNG）。
"""

from pathlib import Path
from typing import Dict, List, Tuple
import shutil

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
    from shapely.affinity import scale as shp_scale, rotate as shp_rotate  # type: ignore
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
import os
KEEP_EXISTING = True
DIR_PSO = SCRIPT_DIR / 'pso_data'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'flip' / 'Fig9_flip'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig9'
KEEP_EXISTING = os.environ.get('KEEP_ATTACKED_FILES', os.environ.get('KEEP_EXISTING', '0')).lower() in ['1', 'true', 'yes']

# 资源
CAT32_PATH = SCRIPT_DIR / 'Cat32.png'

# 配置：翻转（key, (sx, sy), label）
# ⚠️ flip_xy必须放在前面，与zNC-Test/Fig9一致
FLIP_CONFIGS: List[Tuple[str, Tuple[float, float], str]] = [
    ("flip_xy", (-1.0, -1.0), "同时X、Y轴镜像翻转"),
    ("flip_x", (-1.0, 1.0), "X轴镜像翻转"),
    ("flip_y", (1.0, -1.0), "Y轴镜像翻转"),
]


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


def generate_flip_attacks(shp_files: List[Path]) -> Dict[str, Dict[str, Path]]:
    """生成翻转攻击（输出 Shapefile）。

    返回: {base: {key: out_path}}
    """
    print('[Step2] 生成翻转攻击 ->', DIR_ATTACKED)
    outputs: Dict[str, Dict[str, Path]] = {}
    DIR_ATTACKED.mkdir(parents=True, exist_ok=True)

    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return outputs

    for src in shp_files:
        base = src.stem
        subdir = DIR_ATTACKED / base
        if subdir.exists():
            shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_shp_error', src.name, exc)
            continue

        outputs[base] = {}
        base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'
        
        # ✅ 使用全局中心作为变换原点，与zNC-Test/Fig9策略一致
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center_x = (bounds[0] + bounds[2]) / 2
        global_center_y = (bounds[1] + bounds[3]) / 2
        global_center = (global_center_x, global_center_y)

        for key, (sx, sy), _label in FLIP_CONFIGS:
            try:
                attacked = gdf.copy()
                
                # ✅ 使用全局中心进行翻转，与zNC-Test一致
                if key == 'flip_xy':
                    # 同时X、Y翻转 = 旋转180°（数学等价）
                    # 使用rotate更稳定，避免scale(-1,-1)导致的顶点顺序反转问题
                    attacked['geometry'] = attacked['geometry'].apply(
                        lambda geom: shp_rotate(geom, 180, origin=global_center)
                    )
                else:
                    # 单轴翻转使用scale，以全局中心为原点
                    attacked['geometry'] = attacked['geometry'].apply(
                        lambda geom: shp_scale(geom, sx, sy, origin=global_center)
                    )
                try:
                    attacked.set_crs(base_crs, allow_override=True, inplace=True)  # type: ignore
                except Exception:
                    pass
                out_path = subdir / f'{key}.shp'
                attacked.to_file(out_path, driver='ESRI Shapefile')
                print(f'输出: {base}/{out_path.name}')
                outputs[base][key] = out_path
            except Exception as exc:
                print('attack_error', base, key, exc)
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
    """评估翻转攻击版本的 NC，并保存 CSV/XLSX/PNG。"""
    print('[Step3] 评估NC ->', DIR_RESULTS)
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    if KEEP_EXISTING:
        csv_check = DIR_RESULTS / 'fig9_flip_nc.csv'
        if csv_check.exists():
            print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig9: {csv_check}')
            return

    key_to_label = {k: lbl for k, _, lbl in FLIP_CONFIGS}
    order_keys = [k for k, _, _ in FLIP_CONFIGS]

    rows: List[Dict] = []  # type: ignore
    for base, key_map in attacked_map.items():
        for key, shp_path in sorted(key_map.items()):
            try:
                eval_result = extract_watermark_from_attacked_vector(str(shp_path), str(CAT32_PATH))
                nc_value = float(eval_result.get('NC', 0.0))
                rows.append({'base': base, 'key': key, 'nc': nc_value})
            except Exception as exc:
                print('nc_eval_error', base, key, exc)
                continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 层次表：翻转方式 -> 8图 -> 平均
    hierarchical_rows: List[Dict] = []  # type: ignore
    base_names = sorted(df['base'].unique())
    for k in order_keys:
        label = key_to_label.get(k, k)
        hierarchical_rows.append({'翻转方式': label, 'VGAT': '', '类型': 'header'})
        for base in base_names:
            sub = df[(df['base'] == base) & (df['key'] == k)]
            nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
            hierarchical_rows.append({'翻转方式': f'  {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = float(df[df['key'] == k]['nc'].mean())
        hierarchical_rows.append({'翻转方式': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存 CSV/Excel
    csv_path = DIR_RESULTS / 'fig9_flip_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    xlsx_path = DIR_RESULTS / 'fig9_flip_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            ws = writer.sheets['NC结果']
            try:
                from copy import copy as _copy
            except Exception:
                _copy = None
            try:
                ws.column_dimensions['A'].width = 24
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
    x = np.arange(len(order_keys))
    labels = [key_to_label.get(k, k) for k in order_keys]
    for base, sub in df.groupby('base'):
        y = []
        for k in order_keys:
            val = sub[sub['key'] == k]['nc']
            y.append(val.iloc[0] if len(val) > 0 else np.nan)
        plt.plot(x, y, '-o', alpha=0.7, label=base)
    avg_vals = [df[df['key'] == k]['nc'].mean() for k in order_keys]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, labels)
    plt.xlabel('翻转方式')
    plt.ylabel('NC')
    plt.title('Fig9：Tan24 翻转攻击 的NC鲁棒性')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig9_flip_nc.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)
    print('曲线图保存: ', fig_path.name)


def main():
    print('=== Fig9：Tan24 翻转攻击鲁棒性测试（与 Fig9 逻辑一致）===')

    inputs = discover_inputs()
    if not inputs:
        print('无输入数据，终止。')
        return

    attacked_map = generate_flip_attacks(inputs)
    evaluate_nc(attacked_map)
    print('=== 完成 ===')


if __name__ == '__main__':
    main()