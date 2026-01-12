#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig11：组合攻击（结合 Fig1~Fig10 的单例配置，共10种组合）

组合清单（每项为一个独立攻击场景）：
1) Fig1 顶点删除 10%
2) Fig2 顶点增加 强度1 比例50%
3) Fig3 对象删除 50%
4) Fig4 噪声扰动 强度0.8 比例50%
5) Fig5 沿Y轴中心裁剪50%
6) Fig6 平移 (dx=20, dy=40)
7) Fig7 缩放 90%
8) Fig8 顺时针旋转 180°
9) Fig9 Y轴镜像翻转
10) Fig10 顺序操作：先反转顶点顺序，再反转对象顺序

流程与 Fig1 对齐：Step1~Step6，带跳步逻辑、目录隔离、层次化CSV/Excel和曲线图。
"""

from pathlib import Path
import sys
from typing import List, Dict, Tuple
import pickle
import shutil
import random

import numpy as np

# 控制台UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# 强制刷新输出（确保实时写入日志）
import builtins
_original_print = builtins.print

def flush_print(*args, **kwargs):
    """带自动刷新的print函数"""
    _original_print(*args, **kwargs)
    sys.stdout.flush()

# 将print替换为flush_print以确保实时输出
builtins.print = flush_print


# 导入共享模块 ⭐
try:
    from fig_common import (
        PROJECT_ROOT, MODEL_PATH, CAT32_PATH, K_FOR_KNN,
        extract_features_20d, gdf_to_graph, load_improved_gat_model,
        load_cat32, features_to_matrix, calc_nc, extract_features_from_graph,
        convert_to_geojson, convert_geojsons_to_graphs
    )
except ImportError as e:
    print(f"无法导入 fig_common 模块: {e}")
    print("请确保 fig_common.py 在同一目录下")
    sys.exit(1)

try:
    import geopandas as gpd  # type: ignore
except Exception as exc:
    print("需要安装 geopandas: pip install geopandas fiona pyproj shapely")
    print("geopandas_import_error", exc)
    gpd = None  # type: ignore

try:
    import torch  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except Exception as exc:
    print("需要安装 torch 和 torch-geometric")
    print("torch_import_error", exc)
    Data = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph  # type: ignore
except Exception as exc:
    print("需要安装 scikit-learn: pip install scikit-learn")
    print("sklearn_import_error", exc)
    StandardScaler = None  # type: ignore
    NearestNeighbors = None  # type: ignore
    kneighbors_graph = None  # type: ignore

try:
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from PIL import Image  # type: ignore
except Exception as exc:
    print("需要安装 pandas, matplotlib, pillow")
    print("pandas_import_error", exc)
    pd = None  # type: ignore
    plt = None  # type: ignore
    Image = None  # type: ignore

try:
    import matplotlib  # type: ignore
    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# 目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

DIR_VECTOR = SCRIPT_DIR / 'vector-data'
DIR_VECTOR_GEOJSON = SCRIPT_DIR / 'vector-data-geojson'
DIR_VECTOR_GEOJSON_ATTACKED = SCRIPT_DIR / 'vector-data-geojson-attacked' / 'compound'
DIR_GRAPH = SCRIPT_DIR / 'vector-data-geojson-attacked-graph'
DIR_GRAPH_ORIGINAL = DIR_GRAPH / 'Original'
DIR_GRAPH_ATTACKED = DIR_GRAPH / 'Attacked' / 'compound'
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig11'


# KNN配置

# 组合定义：key, label
COMBOS: List[Tuple[str, str]] = [
    ("del_vertices_10pct", "Fig1 顶点删除 10%"),
    ("add_vertices_s1_50pct", "Fig2 顶点增加 强度1 比例50%"),
    ("del_objects_50pct", "Fig3 对象删除 50%"),
    ("noise_s0.8_50pct", "Fig4 噪声攻击 强度0.8 比例50%"),
    ("crop_y_center_50pct", "Fig5 沿Y轴中心裁剪50%"),
    ("translate_x20_y40", "Fig6 平移 X20,Y40"),
    ("scale_90pct", "Fig7 缩放 90%"),
    ("rotate_180deg", "Fig8 顺时针旋转180°"),
    ("flip_y", "Fig9 翻转 Y轴镜像翻转"),
    ("order_rev_vertices_rev_objects", "Fig10 顺序: 反转顶点→反转对象"),
]


def step1_discover_inputs() -> List[Path]:
    print('[Step1] 扫描输入数据: ', DIR_VECTOR)
    if not DIR_VECTOR.exists():
        print('未找到目录: ', DIR_VECTOR)
        return []
    files: List[Path] = []
    files.extend(sorted(DIR_VECTOR.glob('*.shp')))
    files.extend(sorted(DIR_VECTOR.glob('*.geojson')))
    selected = files[:6]
    print('发现文件: ', [p.name for p in selected])
    return selected


def step2_convert_to_geojson(inputs: List[Path]) -> List[Path]:
    """按 convertToGeoJson 逻辑转为 GeoJSON。"""
    print('[Step2] 转换为 GeoJSON ->', DIR_VECTOR_GEOJSON)
    return convert_to_geojson(inputs, DIR_VECTOR_GEOJSON)
    for src in inputs:
        try:
            base = src.stem
            out_path = DIR_VECTOR_GEOJSON / f'{base}.geojson'
            gdf = gpd.read_file(src)
            if getattr(gdf, 'crs', None) and str(gdf.crs) != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            gdf.to_file(out_path, driver='GeoJSON', encoding='utf-8')
            print(f'输出: {out_path.name} ({len(gdf)} 要素)')
            outputs.append(out_path)
        except Exception as exc:
            print('convert_error', src.name, exc)
            continue
    return outputs


def step3_generate_compound_attacks(original_geojsons: List[Path]) -> Dict[str, Dict[str, Path]]:
    print('[Step3] 生成组合攻击 ->', DIR_VECTOR_GEOJSON_ATTACKED)
    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return {}

    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    from shapely.geometry import LineString, Polygon  # type: ignore
    from shapely.affinity import translate as shp_translate, scale as shp_scale, rotate as shp_rotate  # type: ignore

    def delete_vertices_from_geom(geom, pct):
        try:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) <= 2:
                    return geom
                n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
                if n_to_delete >= len(coords) - 2:
                    return geom
                indices = list(range(1, len(coords) - 1))
                to_delete = set(random.sample(indices, min(n_to_delete, len(indices))))
                new_coords = [coords[0]] + [coords[i] for i in range(1, len(coords) - 1) if i not in to_delete] + [coords[-1]]
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                if len(ext) <= 4:
                    return geom
                n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
                if n_to_delete >= len(ext) - 4:
                    return geom
                indices = list(range(1, len(ext) - 2))
                to_delete = set(random.sample(indices, min(n_to_delete, len(indices))))
                new_ext = [ext[0]] + [ext[i] for i in range(1, len(ext) - 2) if i not in to_delete] + [ext[-2], ext[-1]]
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

    def add_vertices_to_geom(geom, pct, strength_level=1):
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

    def jitter_vertices(geom, pct, strength):
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
                        new_coords.append(c)
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
                        new_ext.append(c)
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    n2 = len(rc); k2 = max(1, int(n2 * pct / 100))
                    idx2 = list(range(n2)); chosen2 = set(random.sample(idx2, min(k2, len(idx2))))
                    new_rc = []
                    for i, c in enumerate(rc):
                        if i in chosen2:
                            new_rc.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
                        else:
                            new_rc.append(c)
                    holes.append(new_rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    def reverse_vertices_geom(geom):
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

    outputs: Dict[str, Dict[str, Path]] = {}
    DIR_VECTOR_GEOJSON_ATTACKED.mkdir(parents=True, exist_ok=True)

    for src in original_geojsons:
        base = src.stem
        subdir = DIR_VECTOR_GEOJSON_ATTACKED / base
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_geojson_error', src.name, exc)
            continue

        outputs[base] = {}

        # ✅ 计算全局中心（用于缩放、旋转、翻转）和裁剪辅助
        bounds = gdf.total_bounds
        global_center_x = (bounds[0] + bounds[2]) / 2
        global_center_y = (bounds[1] + bounds[3]) / 2
        global_center = (global_center_x, global_center_y)
        mid_x = global_center_x
        mid_y = global_center_y
        bdf = gdf.geometry.bounds

        for key, _label in COMBOS:
            try:
                attacked = gdf.copy()
                if key == 'del_vertices_10pct':
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, 10))
                elif key == 'add_vertices_s1_50pct':
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: add_vertices_to_geom(geom, 50, strength_level=1))
                elif key == 'del_objects_50pct':
                    num = len(attacked)
                    to_del = int(num * 50 / 100)
                    if to_del > 0:
                        idx = random.sample(range(num), min(to_del, num))
                        attacked = attacked.drop(idx).reset_index(drop=True)
                elif key == 'noise_s0.8_50pct':
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: jitter_vertices(geom, 50, 0.8))
                elif key == 'crop_y_center_50pct':
                    attacked = attacked[bdf['miny'] < mid_y].reset_index(drop=True)
                elif key == 'translate_x20_y40':
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: shp_translate(geom, 20, 40))
                elif key == 'scale_90pct':
                    # ✅ 使用全局中心，与Fig7一致
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: shp_scale(geom, 0.9, 0.9, origin=global_center))
                elif key == 'rotate_180deg':
                    # ✅ 使用全局中心，与Fig8一致
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: shp_rotate(geom, 180, origin=global_center))
                elif key == 'flip_y':
                    # ✅ 使用全局中心，与Fig9一致
                    attacked['geometry'] = attacked['geometry'].apply(lambda geom: shp_scale(geom, 1.0, -1.0, origin=global_center))
                elif key == 'order_rev_vertices_rev_objects':
                    attacked['geometry'] = attacked['geometry'].apply(reverse_vertices_geom)
                    attacked = attacked.iloc[::-1].reset_index(drop=True)

                out_path = subdir / f'{key}.geojson'
                attacked.to_file(out_path, driver='GeoJSON')
                print(f'输出: {base}/{out_path.name}')
                outputs[base][key] = out_path
            except Exception as exc:
                print('attack_error', base, key, exc)
                continue

    return outputs


def step4_convert_to_graph(original_geojsons: List[Path], attacked_map: Dict[str, Dict[str, Path]]):
    """使用 fig_common 的标准函数转为图结构（KNN+Delaunay, k=8；20维特征）。"""
    print('[Step4] 转换为图结构 ->', DIR_GRAPH)
    
    if gpd is None or Data is None:
        print('缺少依赖，无法构图。')
        return
    
    # ⭐使用共享函数批量转换（20维特征 + KNN+Delaunay）
    convert_geojsons_to_graphs(
        original_geojsons=original_geojsons,
        attacked_geojson_map=attacked_map,
        output_dir_original=DIR_GRAPH_ORIGINAL,
        output_dir_attacked=DIR_GRAPH_ATTACKED
    )


def step5_generate_zero_watermark():
    """用 VGAT 模型对 Original 生成零水印。"""
    print('[Step5] 生成零水印 ->', DIR_ZEROWM)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_improved_gat_model(device)
    copyright_img = load_cat32()
    
    DIR_ZEROWM.mkdir(parents=True, exist_ok=True)
    
    cnt = 0
    for pkl in sorted(DIR_GRAPH_ORIGINAL.glob('*_graph.pkl')):
        try:
            with open(pkl, 'rb') as f:
                graph = pickle.load(f)
            
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            zwm = np.logical_xor(feat_mat, copyright_img).astype(np.uint8)
            
            base = pkl.stem.replace('_graph', '')
            np.save(DIR_ZEROWM / f'{base}_watermark.npy', zwm)
            try:
                from PIL import Image
                Image.fromarray((zwm * 255).astype(np.uint8)).save(DIR_ZEROWM / f'{base}_watermark.png')
            except Exception:
                pass
            print('零水印已保存: ', f'{base}_watermark.npy')
            cnt += 1
        except Exception as exc:
            print('zero_watermark_error', pkl.name, exc)
            continue
    
    if cnt == 0:
        print('未找到 Original 图，无法生成零水印。')
    else:
        print(f'共生成零水印 {cnt} 个。')


def step6_evaluate_nc():
    print('[Step6] 评估NC ->', DIR_RESULTS)

    if not DIR_GRAPH_ATTACKED.exists():
        print('缺少攻击图目录: ', DIR_GRAPH_ATTACKED)
        print('请先运行 Step4 转换为图结构。')
        return

    if Data is None or not MODEL_PATH.exists():
        if not MODEL_PATH.exists():
            print('模型文件不存在: ', MODEL_PATH)
        else:
            print('缺少 torch/torch-geometric 依赖。')
        return

    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print('需要安装 pandas/matplotlib'); print(exc); return
    # ⭐使用共享函数加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_improved_gat_model(device)
    except Exception as exc:
        print(f'模型加载失败: {exc}')
        return

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    copyright_img = load_cat32()

    key_to_label = {k: lbl for k, lbl in COMBOS}
    order_keys = [k for k, _ in COMBOS]
    rows = []
    for base_dir_path in sorted(DIR_GRAPH_ATTACKED.iterdir()):
        if not base_dir_path.is_dir():
            continue
        base = base_dir_path.name
        zwm_path = DIR_ZEROWM / f'{base}_watermark.npy'
        if not zwm_path.exists():
            print('缺少零水印，跳过: ', base)
            continue
        zwm = np.load(zwm_path)

        for pkl in sorted(base_dir_path.glob('*_graph.pkl')):
            try:
                with open(pkl, 'rb') as f:
                    graph: Data = pickle.load(f)
                feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
                extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
                nc = calc_nc(copyright_img, extracted)

                name = pkl.stem.replace('_graph', '')
                k_found = None
                for k in order_keys:
                    if k in name:
                        k_found = k
                        break
                rows.append({'base': base, 'combo_key': k_found or 'unknown', 'nc': nc})
            except Exception as exc:
                print('nc_eval_error', pkl.name, exc)
                continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 层次表：每个组合下列出8图NC+平均
    hierarchical_rows = []
    base_names = sorted(df['base'].unique())
    for k in order_keys:
        label = key_to_label.get(k, k)
        hierarchical_rows.append({'组合攻击': label, 'VGAT': '', '类型': 'header'})
        for base in base_names:
            sub = df[(df['base'] == base) & (df['combo_key'] == k)]
            nc_value = sub['nc'].iloc[0] if len(sub) > 0 else 0
            hierarchical_rows.append({'组合攻击': f'  {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = df[df['combo_key'] == k]['nc'].mean()
        hierarchical_rows.append({'组合攻击': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig11_compound_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    xlsx_path = DIR_RESULTS / 'fig11_compound_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            ws = writer.sheets['NC结果']
            ws.column_dimensions['A'].width = 34
            ws.column_dimensions['B'].width = 15
            for idx, row in hierarchical_df.iterrows():
                if row['类型'] == 'header':
                    ws[f'A{idx+2}'].font = ws[f'A{idx+2}'].font.copy(bold=True)
                elif row['类型'] == 'average':
                    ws[f'A{idx+2}'].font = ws[f'A{idx+2}'].font.copy(bold=True, italic=True)
                    ws[f'B{idx+2}'].font = ws[f'B{idx+2}'].font.copy(bold=True, italic=True)
                elif row['类型'] == 'data':
                    ws[f'A{idx+2}'].alignment = ws[f'A{idx+2}'].alignment.copy(horizontal='left', indent=1)
    except Exception as e:
        print('Excel保存失败: ', e)
        hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)

    # 曲线图：横轴为10种组合
    plt.figure(figsize=(12, 6))
    x = np.arange(len(order_keys))
    labels = [key_to_label[k] for k in order_keys]
    for base, sub in df.groupby('base'):
        y = []
        for k in order_keys:
            val = sub[sub['combo_key'] == k]['nc']
            y.append(val.iloc[0] if len(val) > 0 else np.nan)
        plt.plot(x, y, '-o', alpha=0.7, label=base)
    avg_vals = [df[df['combo_key'] == k]['nc'].mean() for k in order_keys]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, labels, rotation=15)
    plt.xlabel('组合攻击')
    plt.ylabel('NC')
    plt.title('Fig11：组合攻击(10种) 的NC鲁棒性')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig11_compound_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print('曲线图保存: ', fig_path.name)


def check_existing_files():
    print('[检查] 检查现有文件...')
    geojson_files = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
    step2_done = len(geojson_files) > 0
    attacked_dirs = [d for d in DIR_VECTOR_GEOJSON_ATTACKED.glob('*') if not d.name.startswith('.')]
    step3_done = len(attacked_dirs) > 0 and all(len(list(ad.glob('*.geojson'))) > 0 for ad in attacked_dirs if ad.is_dir())
    original_graphs = [p for p in DIR_GRAPH_ORIGINAL.glob('*_graph.pkl') if not p.name.startswith('.')]
    attacked_graphs = [p for p in DIR_GRAPH_ATTACKED.glob('*/*_graph.pkl') if not p.name.startswith('.')]
    step4_done = len(original_graphs) > 0 and len(attacked_graphs) > 0
    watermark_files = [p for p in DIR_ZEROWM.glob('*_watermark.npy') if not p.name.startswith('.')]
    step5_done = len(watermark_files) > 0
    nc_files = [p for p in DIR_RESULTS.glob('*.csv') if not p.name.startswith('.')] + [p for p in DIR_RESULTS.glob('*.xlsx') if not p.name.startswith('.')] + [p for p in DIR_RESULTS.glob('*.png') if not p.name.startswith('.')]
    step6_done = len(nc_files) > 0
    print(f'  Step2 (GeoJSON): {"✓" if step2_done else "✗"} ({len(geojson_files)} files)')
    print(f'  Step3 (Attacked): {"✓" if step3_done else "✗"} ({len(attacked_dirs)} dirs)')
    print(f'  Step4 (Graphs): {"✓" if step4_done else "✗"} (Original: {len(original_graphs)}, Attacked: {len(attacked_graphs)})')
    print(f'  Step5 (Watermarks): {"✓" if step5_done else "✗"} ({len(watermark_files)} files)')
    print(f'  Step6 (NC Results): {"✓" if step6_done else "✗"} ({len(nc_files)} files)')
    return {'step2': step2_done, 'step3': step3_done, 'step4': step4_done, 'step5': step5_done, 'step6': step6_done}


def main():
    print('=== Fig11：组合攻击鲁棒性测试 ===')
    existing = check_existing_files()
    if existing['step6']:
        print('\n[跳过] 检测到NC结果文件已存在，直接运行Step6...'); step6_evaluate_nc(); print('=== 完成 ==='); return
    # 如果检测到零水印已存在，先验证图结构/攻击文件是否完整
    if existing['step5']:
        print('\n[检查] 检测到零水印文件已存在，验证图结构/攻击文件是否完整...')
        if existing['step4']:
            print('\n[跳过] 图结构文件已存在，直接运行Step6...'); step6_evaluate_nc(); print('=== 完成 ==='); return

        print('\n[补全] 零水印存在但图结构或攻击数据缺失，尝试补齐缺失步骤...')
        inputs = step1_discover_inputs()
        original_geojsons = step2_convert_to_geojson(inputs) if not existing['step2'] else [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]

        attacked_map = {}
        if existing['step3']:
            for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
                if attacked_dir.is_dir():
                    base = attacked_dir.name; attacked_map[base] = {}
                    for key, _lbl in COMBOS:
                        geojson_file = attacked_dir / f'{key}.geojson'
                        if geojson_file.exists(): attacked_map[base][key] = geojson_file
        else:
            attacked_map = step3_generate_compound_attacks(original_geojsons)

        step4_convert_to_graph(original_geojsons, attacked_map); print('补全完成，开始评估Step6...')
        # 不重新生成零水印以避免覆盖，直接评估 NC
        step6_evaluate_nc(); print('=== 完成 ==='); return
    if existing['step4']:
        print('\n[跳过] 检测到图结构文件已存在，直接运行Step5-6...'); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== 完成 ==='); return
    if existing['step3']:
        print('\n[跳过] 检测到攻击文件已存在，直接运行Step4-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map: Dict[str, Dict[str, Path]] = {}
        for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
            if attacked_dir.is_dir():
                base = attacked_dir.name; attacked_map[base] = {}
                for key, _lbl in COMBOS:
                    geojson_file = attacked_dir / f'{key}.geojson'
                    if geojson_file.exists(): attacked_map[base][key] = geojson_file
        step4_convert_to_graph(original_geojsons, attacked_map); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== 完成 ==='); return
    if existing['step2']:
        print('\n[跳过] 检测到GeoJSON文件已存在，直接运行Step3-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = step3_generate_compound_attacks(original_geojsons)
        step4_convert_to_graph(original_geojsons, attacked_map); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== 完成 ==='); return
    print('\n[从头开始] 运行所有步骤...')
    inputs = step1_discover_inputs(); original_geojsons = step2_convert_to_geojson(inputs)
    attacked_map = step3_generate_compound_attacks(original_geojsons)
    step4_convert_to_graph(original_geojsons, attacked_map); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== 完成 ===')


if __name__ == '__main__':
    main()


