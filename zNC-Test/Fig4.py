#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4：噪声攻击鲁棒性测试脚本（强度0.4/0.6/0.8；扰动顶点比例10/30/50/70/90）

步骤对齐 Fig2：
1) 扫描 `vector-data` 下8个矢量数据
2) 转为 `vector-data-geojson`
3) 生成指定强度与比例的噪声攻击到 `vector-data-geojson-attacked/noise`
4) 转图结构到 `vector-data-geojson-attacked-graph/noise/{Original,Attacked}`
5) 用 `VGAT/models/gat_model_best.pth` 对 `Original` 提取零水印到 `vector-data-zerowatermark`
6) 对攻击版本做NC评估，输出到 `zNC-Test/NC-Results/Fig4`

噪声攻击逻辑引用 convertToGeoJson-Attacked-TrainingSet.py 的 jitter 方式：
随机选择指定比例的顶点，将其坐标在[-strength, strength]范围内随机扰动。
"""

from pathlib import Path
import sys
from typing import List
import random
import pickle
import shutil

import numpy as np

# 强制 Python 在控制台使用 UTF-8 编码（Windows/PowerShell 防乱码）
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
    gpd = None  # type: ignore

try:
    import torch  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except Exception as exc:
    print("需要安装 torch 和 torch-geometric")
    Data = None  # type: ignore

try:
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from PIL import Image  # type: ignore
except Exception as exc:
    print("需要安装 pandas, matplotlib, pillow")
    pd = None  # type: ignore
    plt = None  # type: ignore
    Image = None  # type: ignore

# 配置 Matplotlib 使用支持中文的字体，并允许坐标轴显示负号
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

# 脚本目录
SCRIPT_DIR = Path(__file__).resolve().parent

# 数据与输出目录
DIR_VECTOR = SCRIPT_DIR / 'vector-data'
DIR_VECTOR_GEOJSON = SCRIPT_DIR / 'vector-data-geojson'
DIR_VECTOR_GEOJSON_ATTACKED = SCRIPT_DIR / 'vector-data-geojson-attacked' / 'noise'
DIR_GRAPH = SCRIPT_DIR / 'vector-data-geojson-attacked-graph'
DIR_GRAPH_ORIGINAL = DIR_GRAPH / 'Original'
DIR_GRAPH_ATTACKED = DIR_GRAPH / 'Attacked' / 'noise'
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig4'

# 配置
NOISE_STRENGTHS = [0.4, 0.6, 0.8]
NOISE_PCTS = [10, 30, 50, 70, 90]


def step1_discover_inputs() -> List[Path]:
    """扫描 vector-data 下的8个矢量数据（.shp 或 .geojson，最多8个）。"""
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


def jitter_vertices_geom(geom, pct: int, strength: float):
    """对几何体进行顶点抖动（噪声）——与 TrainingSet 逻辑一致。"""
    from shapely.geometry import LineString, Polygon

    if geom.is_empty:
        return geom

    geom_type = getattr(geom, 'geom_type', 'Unknown')

    if geom_type == 'LineString':
        coords = list(geom.coords)
        n = len(coords)
        k = max(1, int(n * pct / 100))
        indices = list(range(n))
        chosen = set(random.sample(indices, min(k, len(indices))))
        new_coords = []
        for i, coord in enumerate(coords):
            if i in chosen:
                new_coords.append((
                    coord[0] + random.uniform(-strength, strength),
                    coord[1] + random.uniform(-strength, strength)
                ))
            else:
                new_coords.append(coord)
        return LineString(new_coords)

    elif geom_type == 'Polygon':
        ext = list(geom.exterior.coords)
        n = len(ext)
        k = max(1, int(n * pct / 100))
        indices = list(range(n))
        chosen = set(random.sample(indices, min(k, len(indices))))
        new_ext = []
        for i, coord in enumerate(ext):
            if i in chosen:
                new_ext.append((
                    coord[0] + random.uniform(-strength, strength),
                    coord[1] + random.uniform(-strength, strength)
                ))
            else:
                new_ext.append(coord)
        holes = []
        for ring in geom.interiors:
            ring_coords = list(ring.coords)
            n_ring = len(ring_coords)
            k_ring = max(1, int(n_ring * pct / 100))
            indices_ring = list(range(n_ring))
            chosen_ring = set(random.sample(indices_ring, min(k_ring, len(indices_ring))))
            new_ring = []
            for i, coord in enumerate(ring_coords):
                if i in chosen_ring:
                    new_ring.append((
                        coord[0] + random.uniform(-strength, strength),
                        coord[1] + random.uniform(-strength, strength)
                    ))
                else:
                    new_ring.append(coord)
            holes.append(new_ring)
        return Polygon(new_ext, holes=holes if holes else None)

    return geom


def step3_generate_noise_attacks(original_geojsons):
    """生成指定强度与比例的噪声攻击。"""
    print('[Step3] 生成噪声攻击 ->', DIR_VECTOR_GEOJSON_ATTACKED)
    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return {}

    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    outputs = {}
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
        for strength in NOISE_STRENGTHS:
            for pct in NOISE_PCTS:
                try:
                    attacked = gdf.copy()
                    attacked['geometry'] = attacked['geometry'].apply(
                        lambda geom: jitter_vertices_geom(geom, pct, float(strength))
                    )
                    s_str = str(strength).replace('.', '_')
                    out_path = subdir / f'noise_{pct}pct_strength_{s_str}.geojson'
                    attacked.to_file(out_path, driver='GeoJSON')
                    print(f'输出: {base}/noise pct={pct} strength={strength}')
                    outputs[base][(pct, strength)] = out_path
                except Exception as exc:
                    print('attack_error', base, pct, strength, exc)
                    continue

    return outputs


def step4_convert_to_graph(original_geojsons, attacked_map):
    """按 convertToGraph 逻辑转为图结构（KNN + 20维特征）。"""
    print('[Step4] 转换为图结构 ->', DIR_GRAPH)
    convert_geojsons_to_graphs(
        original_geojsons=original_geojsons,
        attacked_geojson_map=attacked_map,
        output_dir_original=DIR_GRAPH_ORIGINAL,
        output_dir_attacked=DIR_GRAPH_ATTACKED
    )


def step5_generate_zero_watermark():
    """用 VGAT 模型对 Original 生成零水印。"""
    print('[Step5] 生成零水印 ->', DIR_ZEROWM)
    
    # 若已有零水印则跳过
    if DIR_ZEROWM.exists() and any(DIR_ZEROWM.glob('*_watermark.*')):
        print('零水印已存在，跳过生成步骤。')
        return
    
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
    """评估噪声攻击的NC并生成结果。"""
    print('[Step6] 评估NC ->', DIR_RESULTS)
    
    if not DIR_GRAPH_ATTACKED.exists():
        print('缺少攻击图目录: ', DIR_GRAPH_ATTACKED)
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_improved_gat_model(device)
    copyright_img = load_cat32()
    
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for subdir in DIR_GRAPH_ATTACKED.iterdir():
        if not subdir.is_dir():
            continue
        base = subdir.name
        # 加载对应 base 的零水印：优先 NPY，回退 PNG；缺失则跳过
        zwm_npy = DIR_ZEROWM / f'{base}_watermark.npy'
        zwm_png = DIR_ZEROWM / f'{base}_watermark.png'
        if zwm_npy.exists():
            try:
                zwm = np.load(zwm_npy)
            except Exception:
                print('load_zwm_npy_error', zwm_npy)
                zwm = None
        elif zwm_png.exists():
            try:
                zwm_img = np.array(Image.open(zwm_png).convert('L'))
                zwm = (zwm_img > 128).astype(np.uint8)
            except Exception:
                print('load_zwm_png_error', zwm_png)
                zwm = None
        else:
            print('缺少零水印，跳过: ', base)
            zwm = None

        if zwm is None:
            continue
        for pkl in sorted(subdir.glob('*_graph.pkl')):
            try:
                with open(pkl, 'rb') as f:
                    graph = pickle.load(f)
                
                feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
                extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
                nc = calc_nc(copyright_img, extracted)

                # 解析比例与强度
                name = pkl.stem.replace('_graph', '')
                parsed_pct = -1
                parsed_strength = None
                for p in NOISE_PCTS:
                    if f'noise_{p}pct_strength_' in name:
                        parsed_pct = p
                        break
                for s in NOISE_STRENGTHS:
                    s_str = str(s).replace('.', '_')
                    if f'strength_{s_str}' in name:
                        parsed_strength = s
                        break
                rows.append({'base': base, 'noise_pct': parsed_pct, 'strength': parsed_strength, 'nc': nc})
            except Exception as exc:
                print('nc_eval_error', pkl.name, exc)
                continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 组织层次数据：按强度分组，每个强度下列出不同扰动比例的各图NC与均值
    hierarchical_rows = []
    base_names = sorted(df['base'].unique())
    for strength in NOISE_STRENGTHS:
        hierarchical_rows.append({'噪声': f'强度: {strength}', 'VGAT': '', '类型': 'header'})
        for pct in NOISE_PCTS:
            hierarchical_rows.append({'噪声': f'  {pct}%', 'VGAT': '', '类型': 'subheader'})
            for base in base_names:
                sub = df[(df['base'] == base) & (df['strength'] == strength) & (df['noise_pct'] == pct)]
                nc_value = sub['nc'].iloc[0] if len(sub) > 0 else 0
                hierarchical_rows.append({'噪声': f'    {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
            avg_nc = df[(df['strength'] == strength) & (df['noise_pct'] == pct)]['nc'].mean()
            hierarchical_rows.append({'噪声': '    Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存CSV/Excel
    csv_path = DIR_RESULTS / 'fig4_noise_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    xlsx_path = DIR_RESULTS / 'fig4_noise_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            worksheet = writer.sheets['NC结果']
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 15
            for idx, row in hierarchical_df.iterrows():
                if row['类型'] == 'header':
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True)
                elif row['类型'] == 'subheader':
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True)
                elif row['类型'] == 'average':
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True, italic=True)
                    worksheet[f'B{idx+2}'].font = worksheet[f'B{idx+2}'].font.copy(bold=True, italic=True)
                elif row['类型'] == 'data':
                    worksheet[f'A{idx+2}'].alignment = worksheet[f'A{idx+2}'].alignment.copy(horizontal='left', indent=2)
    except Exception as e:
        print(f'Excel保存失败: {e}')
        hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')

    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)

    # 绘制每个强度的小图 + 综合平均图
    plt.figure(figsize=(12, 8))
    for i, strength in enumerate(NOISE_STRENGTHS):
        plt.subplot(2, 2, i + 1)
        for base, sub in df[df['strength'] == strength].groupby('base'):
            sub2 = sub.sort_values('noise_pct')
            plt.plot(sub2['noise_pct'], sub2['nc'], '-o', alpha=0.7, label=base, markersize=4)
        avg = df[df['strength'] == strength].groupby('noise_pct')['nc'].mean().reset_index()
        plt.plot(avg['noise_pct'], avg['nc'], 'k-o', linewidth=2.5, label='平均', markersize=6)
        plt.grid(True, alpha=0.3)
        plt.xlabel('扰动顶点比例（%）')
        plt.ylabel('NC')
        plt.title(f'强度 {strength}：噪声攻击的NC鲁棒性')
        plt.legend(loc='best', fontsize=6, ncol=2)
        plt.ylim(0, 1.1)

    plt.subplot(2, 2, 4)
    for strength in NOISE_STRENGTHS:
        avg = df[df['strength'] == strength].groupby('noise_pct')['nc'].mean().reset_index()
        plt.plot(avg['noise_pct'], avg['nc'], '-o', linewidth=2, label=f'强度 {strength}', markersize=6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('扰动顶点比例（%）')
    plt.ylabel('NC')
    plt.title('综合对比：不同强度噪声的NC鲁棒性')
    plt.legend(loc='best')
    plt.ylim(0, 1.1)

    plt.tight_layout()
    fig_path = DIR_RESULTS / 'fig4_noise_nc.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('曲线图保存: ', fig_path.name)


def check_existing_files():
    """检查各步骤输出是否已存在，决定是否跳过。"""
    print('[检查] 检查现有文件...')

    geojson_files = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
    step2_done = len(geojson_files) > 0

    attacked_dirs = [d for d in DIR_VECTOR_GEOJSON_ATTACKED.glob('*') if not d.name.startswith('.')]
    step3_done = len(attacked_dirs) > 0 and all(
        len(list(attacked_dir.glob('*.geojson'))) > 0
        for attacked_dir in attacked_dirs if attacked_dir.is_dir()
    )

    original_graphs = [p for p in DIR_GRAPH_ORIGINAL.glob('*_graph.pkl') if not p.name.startswith('.')]
    attacked_graphs = [p for p in DIR_GRAPH_ATTACKED.glob('*/*_graph.pkl') if not p.name.startswith('.')]
    step4_done = len(original_graphs) > 0 and len(attacked_graphs) > 0

    watermark_files = [p for p in DIR_ZEROWM.glob('*_watermark.npy') if not p.name.startswith('.')] + [p for p in DIR_ZEROWM.glob('*_watermark.png') if not p.name.startswith('.')]
    step5_done = len(watermark_files) > 0

    nc_files = [p for p in DIR_RESULTS.glob('*.csv') if not p.name.startswith('.')] + [p for p in DIR_RESULTS.glob('*.xlsx') if not p.name.startswith('.')] + [p for p in DIR_RESULTS.glob('*.png') if not p.name.startswith('.')]
    step6_done = len(nc_files) > 0

    print(f'  Step2 (GeoJSON): {"✓" if step2_done else "✗"} ({len(geojson_files)} files)')
    print(f'  Step3 (Attacked): {"✓" if step3_done else "✗"} ({len(attacked_dirs)} dirs)')
    print(f'  Step4 (Graphs): {"✓" if step4_done else "✗"} (Original: {len(original_graphs)}, Attacked: {len(attacked_graphs)})')
    print(f'  Step5 (Watermarks): {"✓" if step5_done else "✗"} ({len(watermark_files)} files)')
    print(f'  Step6 (NC Results): {"✓" if step6_done else "✗"} ({len(nc_files)} files)')

    return {
        'step2': step2_done,
        'step3': step3_done,
        'step4': step4_done,
        'step5': step5_done,
        'step6': step6_done
    }


def main():
    print('=== Fig4：噪声攻击鲁棒性测试 ===')

    existing = check_existing_files()

    if existing['step6']:
        print('\n[跳过] 检测到NC结果文件已存在，直接运行Step6...')
        step6_evaluate_nc()
        print('=== 完成 ===')
        return

    # 如果检测到零水印已存在，先验证图结构/攻击文件是否完整
    if existing['step5']:
        print('\n[检查] 检测到零水印文件已存在，验证图结构/攻击文件是否完整...')
        if existing['step4']:
            print('\n[跳过] 图结构文件已存在，直接运行Step6...')
            step6_evaluate_nc()
            print('=== 完成 ===')
            return

        print('\n[补全] 零水印存在但图结构或攻击数据缺失，尝试补齐缺失步骤...')
        inputs = step1_discover_inputs()
        original_geojsons = step2_convert_to_geojson(inputs) if not existing['step2'] else [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]

        attacked_map = {}
        if existing['step3']:
            for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
                if attacked_dir.is_dir():
                    base = attacked_dir.name
                    attacked_map[base] = {}
                    for strength in NOISE_STRENGTHS:
                        for pct in NOISE_PCTS:
                            s_str = str(strength).replace('.', '_')
                            geojson_file = attacked_dir / f'noise_{pct}pct_strength_{s_str}.geojson'
                            if geojson_file.exists():
                                attacked_map[base][(pct, strength)] = geojson_file
        else:
            attacked_map = step3_generate_noise_attacks(original_geojsons)

        step4_convert_to_graph(original_geojsons, attacked_map)
        # 直接评估 NC（不重新生成零水印以避免覆盖）
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
        inputs = step1_discover_inputs()
        original_geojsons = step2_convert_to_geojson(inputs) if not existing['step2'] else [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = {}
        for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
            if attacked_dir.is_dir():
                base = attacked_dir.name
                attacked_map[base] = {}
                for strength in NOISE_STRENGTHS:
                    for pct in NOISE_PCTS:
                        s_str = str(strength).replace('.', '_')
                        geojson_file = attacked_dir / f'noise_{pct}pct_strength_{s_str}.geojson'
                        if geojson_file.exists():
                            attacked_map[base][(pct, strength)] = geojson_file
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return

    if existing['step2']:
        print('\n[跳过] 检测到GeoJSON文件已存在，直接运行Step3-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = step3_generate_noise_attacks(original_geojsons)
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return

    print('\n[从头开始] 运行所有步骤...')
    inputs = step1_discover_inputs()
    original_geojsons = step2_convert_to_geojson(inputs)
    attacked_map = step3_generate_noise_attacks(original_geojsons)
    step4_convert_to_graph(original_geojsons, attacked_map)
    step5_generate_zero_watermark()
    step6_evaluate_nc()
    print('=== 完成 ===')


if __name__ == '__main__':
    main()


