#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig12_Ablation2_GraphOnly：使用消融实验2（仅图级特征）的模型测试复合攻击鲁棒性

与Fig12.py逻辑完全一致，仅模型路径和输出目录不同：
- 模型：VGAT/models/gat_model_Ablation2_GraphOnly_best.pth
- 零水印输出：vector-data-zerowatermark-ablation2
- NC结果输出：NC-Results/Fig12_Ablation2_GraphOnly
"""

from pathlib import Path
import sys
from typing import List, Dict
import pickle
import shutil
import random

import numpy as np

# 控制台 UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 强制刷新输出
import builtins
_original_print = builtins.print

def flush_print(*args, **kwargs):
    """带自动刷新的print函数"""
    _original_print(*args, **kwargs)
    sys.stdout.flush()

builtins.print = flush_print


# 导入共享模块
try:
    from fig_common import (
        PROJECT_ROOT, CAT32_PATH, K_FOR_KNN,
        extract_features_20d, gdf_to_graph,
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

try:
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from PIL import Image  # type: ignore
except Exception as exc:
    print("需要安装 pandas, matplotlib, pillow")
    print("pandas_import_error", exc)

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


# ⭐ 消融实验2专属配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
ABLATION_MODEL_PATH = PROJECT_ROOT / 'VGAT' / 'models' / 'gat_model_Ablation2_GraphOnly_best.pth'

DIR_VECTOR = SCRIPT_DIR / 'vector-data'
DIR_VECTOR_GEOJSON = SCRIPT_DIR / 'vector-data-geojson'
DIR_VECTOR_GEOJSON_ATTACKED = SCRIPT_DIR / 'vector-data-geojson-attacked' / 'compound_seq'
DIR_GRAPH = SCRIPT_DIR / 'vector-data-geojson-attacked-graph'
DIR_GRAPH_ORIGINAL = DIR_GRAPH / 'Original'
DIR_GRAPH_ATTACKED = DIR_GRAPH / 'Attacked' / 'compound_seq'
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark-ablation2'  # ⭐ 独立零水印目录
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig12_Ablation2_GraphOnly'  # ⭐ 独立结果目录


def load_ablation2_model(device):
    """加载消融实验2的模型"""
    if not ABLATION_MODEL_PATH.exists():
        raise FileNotFoundError(f"消融实验2模型文件不存在: {ABLATION_MODEL_PATH}")
    
    # 动态导入消融模型
    import importlib.util
    ablation2_path = PROJECT_ROOT / 'VGAT' / 'Ablation2_GraphOnly.py'
    spec = importlib.util.spec_from_file_location("Ablation2_GraphOnly", ablation2_path)
    ablation2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablation2_module)
    
    GraphOnlyModel = ablation2_module.GraphOnlyModel
    
    # 创建模型实例（参数与训练时一致）
    model = GraphOnlyModel(
        input_dim=20,  # 输入20维完整特征，模型内部只使用图级8维
        hidden_dim=256,
        output_dim=1024,
        num_heads=8,  # 不使用，保留兼容性
        dropout=0.3
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(ABLATION_MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ 已加载消融实验2模型: {ABLATION_MODEL_PATH.name}")
    return model, device


# ... (Step1-Step3 与 Ablation1 完全相同，这里省略以节省token)
# 以下是完整代码的占位符，实际会包含完整的 step1-step3 函数

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


def step3_generate_compound_seq_attacks(original_geojsons: List[Path]) -> Dict[str, Dict[str, Path]]:
    """对每个 base，按 Fig1→Fig10 顺序一次性依次执行并仅输出一个最终攻击版本。"""
    print('[Step3] 生成复合(顺序)攻击 ->', DIR_VECTOR_GEOJSON_ATTACKED)
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
                if len(coords) <= 2: return geom
                n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
                if n_to_delete >= len(coords) - 2: return geom
                idx = list(range(1, len(coords) - 1))
                to_del = set(random.sample(idx, min(n_to_delete, len(idx))))
                new_coords = [coords[0]] + [coords[i] for i in idx if i not in to_del] + [coords[-1]]
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                if len(ext) <= 4: return geom
                n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
                if n_to_delete >= len(ext) - 4: return geom
                idx = list(range(1, len(ext) - 2))
                to_del = set(random.sample(idx, min(n_to_delete, len(idx))))
                new_ext = [ext[0]] + [ext[i] for i in range(1, len(ext) - 2) if i not in to_del] + [ext[-2], ext[-1]]
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    if len(rc) > 4:
                        n_h = max(1, int((len(rc) - 4) * pct / 100))
                        if n_h < len(rc) - 4:
                            idx_h = list(range(1, len(rc) - 2))
                            del_h = set(random.sample(idx_h, min(n_h, len(idx_h))))
                            holes.append([rc[0]] + [rc[i] for i in range(1, len(rc) - 2) if i not in del_h] + [rc[-2], rc[-1]])
                        else: holes.append(rc)
                    else: holes.append(rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception: pass
        return geom

    def add_vertices_to_geom(geom, pct, strength_level=1):
        noise_sigma = 0.01 if strength_level == 1 else 0.0
        try:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) < 2: return geom
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
                if len(ext) < 4: return geom
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
                    else: holes.append(rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception: pass
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
        except Exception: pass
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
        except Exception: pass
        return geom

    outputs: Dict[str, Dict[str, Path]] = {}
    DIR_VECTOR_GEOJSON_ATTACKED.mkdir(parents=True, exist_ok=True)

    for src in original_geojsons:
        base = src.stem
        subdir = DIR_VECTOR_GEOJSON_ATTACKED / base
        if subdir.exists(): shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_geojson_error', src.name, exc)
            continue

        bounds = gdf.total_bounds
        global_center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        mid_y = global_center[1]

        gdf['geometry'] = gdf['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, 10))
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: add_vertices_to_geom(geom, 50, strength_level=1))
        n_total = len(gdf)
        n_del = int(n_total * 0.5)
        if n_del > 0 and n_total > 0:
            idx = random.sample(range(n_total), min(n_del, n_total))
            gdf = gdf.drop(idx).reset_index(drop=True)
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: jitter_vertices(geom, 50, 0.8))
        bdf = gdf.geometry.bounds
        gdf = gdf[bdf['miny'] < mid_y].reset_index(drop=True)
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_translate(geom, 20, 40))
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 0.9, 0.9, origin=global_center))
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_rotate(geom, 180, origin=global_center))
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 1.0, -1.0, origin=global_center))
        gdf['geometry'] = gdf['geometry'].apply(reverse_vertices_geom)
        gdf = gdf.iloc[::-1].reset_index(drop=True)

        out_path = subdir / 'compound_seq_all.geojson'
        try:
            gdf.to_file(out_path, driver='GeoJSON')
            print(f'输出: {base}/{out_path.name}')
        except Exception as exc:
            print('attack_save_error', base, exc)
            continue

        outputs.setdefault(base, {})['compound_seq_all'] = out_path

    return outputs


def step4_convert_to_graph(original_geojsons: List[Path], attacked_map: Dict[str, Dict[str, Path]]):
    """使用 fig_common 的标准函数转为图结构。"""
    print('[Step4] 转换为图结构 ->', DIR_GRAPH)
    
    if gpd is None or Data is None:
        print('缺少依赖，无法构图。')
        return
    
    convert_geojsons_to_graphs(
        original_geojsons=original_geojsons,
        attacked_geojson_map=attacked_map,
        output_dir_original=DIR_GRAPH_ORIGINAL,
        output_dir_attacked=DIR_GRAPH_ATTACKED
    )


def step5_generate_zero_watermark():
    """⭐ 用消融实验2模型生成零水印"""
    print('[Step5] 生成零水印 (消融实验2) ->', DIR_ZEROWM)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_ablation2_model(device)
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
    print('[Step6] 评估NC (消融实验2) ->', DIR_RESULTS)
    if not DIR_GRAPH_ATTACKED.exists():
        print('缺少攻击图目录: ', DIR_GRAPH_ATTACKED)
        print('请先运行 Step4 转换为图结构。')
        return
    
    if not ABLATION_MODEL_PATH.exists():
        print('消融实验2模型文件不存在: ', ABLATION_MODEL_PATH)
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_ablation2_model(device)
    except Exception as exc:
        print(f'模型加载失败: {exc}')
        return

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    extract_dir = SCRIPT_DIR / 'extract' / 'watermark' / 'Fig12_Ablation2_GraphOnly'
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    copyright_img = load_cat32()

    rows = []
    for base_dir_path in sorted(DIR_GRAPH_ATTACKED.iterdir()):
        if not base_dir_path.is_dir(): continue
        base = base_dir_path.name
        zwm_path = DIR_ZEROWM / f'{base}_watermark.npy'
        if not zwm_path.exists():
            print('缺少零水印，跳过: ', base)
            continue
        zwm = np.load(zwm_path)
        pkl = base_dir_path / 'compound_seq_all_graph.pkl'
        if not pkl.exists():
            print('缺少复合攻击图: ', pkl)
            continue
        try:
            with open(pkl, 'rb') as f:
                graph: Data = pickle.load(f)
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
            nc = calc_nc(copyright_img, extracted)
            rows.append({'base': base, 'nc': nc})
            
            try:
                out_img_path = extract_dir / f'Cat32_{base}_extracted.png'
                extracted_img = (extracted * 255).astype(np.uint8)
                Image.fromarray(extracted_img).save(out_img_path)
                print(f'已保存提取水印图: {out_img_path.name}')
            except Exception as img_exc:
                print(f'保存提取水印图失败 {base}: {img_exc}')
        except Exception as exc:
            print('nc_eval_error', pkl.name, exc)

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows).sort_values('base')

    hierarchical_rows = []
    hierarchical_rows.append({'复合攻击(顺序)': '复合(Fig1→Fig10)', 'Ablation2_GraphOnly': '', '类型': 'header'})
    for _, row in df.iterrows():
        hierarchical_rows.append({'复合攻击(顺序)': f"  {row['base']}", 'Ablation2_GraphOnly': f"{row['nc']:.6f}", '类型': 'data'})
    hierarchical_rows.append({'复合攻击(顺序)': '  Average', 'Ablation2_GraphOnly': f"{df['nc'].mean():.6f}", '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig12_ablation2_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    xlsx_path = DIR_RESULTS / 'fig12_ablation2_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            ws = writer.sheets['NC结果']
            ws.column_dimensions['A'].width = 30
            ws.column_dimensions['B'].width = 20
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

    plt.figure(figsize=(10, 6))
    bases = list(df['base'])
    vals = list(df['nc'])
    x = np.arange(len(bases))
    plt.bar(x, vals, alpha=0.8, label='Ablation2_GraphOnly')
    plt.axhline(df['nc'].mean(), color='k', linestyle='--', label='平均')
    plt.xticks(x, bases, rotation=20)
    plt.ylabel('NC')
    plt.title('Fig12_Ablation2：复合攻击 NC鲁棒性 (仅图级特征/MLP)')
    plt.legend()
    fig_path = DIR_RESULTS / 'fig12_ablation2_nc.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('柱状图保存: ', fig_path.name)


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
    print('=== Fig12_Ablation2：复合攻击鲁棒性测试 (仅图级特征/MLP) ===')
    existing = check_existing_files()
    if existing['step6']:
        print('\n[跳过] 检测到NC结果文件已存在，直接运行Step6...')
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    if existing['step5'] and existing['step4']:
        print('\n[跳过] 检测到零水印与图结构已存在，直接运行Step6...')
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    if existing['step4']:
        print('\n[跳过] 检测到图结构文件已存在，直接运行Step5-6...')
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    if existing['step3']:
        print('\n[跳过] 检测到攻击文件已存在，直接运行Step4-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map: Dict[str, Dict[str, Path]] = {}
        for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
            if attacked_dir.is_dir():
                base = attacked_dir.name
                attacked_map[base] = {}
                geojson_file = attacked_dir / 'compound_seq_all.geojson'
                if geojson_file.exists():
                    attacked_map[base]['compound_seq_all'] = geojson_file
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    if existing['step2']:
        print('\n[跳过] 检测到GeoJSON文件已存在，直接运行Step3-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = step3_generate_compound_seq_attacks(original_geojsons)
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    print('\n[从头开始] 运行所有步骤...')
    inputs = step1_discover_inputs()
    original_geojsons = step2_convert_to_geojson(inputs)
    attacked_map = step3_generate_compound_seq_attacks(original_geojsons)
    step4_convert_to_graph(original_geojsons, attacked_map)
    step5_generate_zero_watermark()
    step6_evaluate_nc()
    print('=== 完成 ===')


if __name__ == '__main__':
    main()

