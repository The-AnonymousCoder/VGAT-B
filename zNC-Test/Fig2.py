#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig2：增加顶点鲁棒性测试脚本（强度0,1,2，比例10%-90%）

分步实现计划：
1) 扫描 `vector-data` 下8个矢量数据
2) 转为 `vector-data-geojson`
3) 生成不同强度(0,1,2)和比例(10%~90%)的增加顶点攻击到 `vector-data-geojson-attacked`
4) 转图结构到 `vector-data-geojson-attacked-graph/{Original,Attacked}`
5) 用 `VGAT/models/gat_model_best.pth` 对 `Original` 提取零水印到 `vector-data-zerowatermark`
6) 对攻击版本做NC评估，输出到 `zNC-Test/NC-Results/Fig2`

增加顶点攻击策略：
- 强度0：在现有顶点之间随机插入新顶点
- 强度1：在现有顶点周围添加噪声顶点
- 强度2：在几何中心附近添加密集顶点
"""

from pathlib import Path
import sys
from typing import List
import random
import pickle
import shutil

import numpy as np

# 强制 Python 在控制台使用 UTF-8 编码，避免中文输出乱码（Windows/PowerShell）
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
    import cv2  # type: ignore
except Exception as exc:
    print("需要安装 pandas, matplotlib, pillow, opencv-python")
    pd = None  # type: ignore
    plt = None  # type: ignore
    Image = None  # type: ignore
    cv2 = None  # type: ignore

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

# 数据与输出目录（放在 zNC-Test 下）
DIR_VECTOR = SCRIPT_DIR / 'vector-data'
DIR_VECTOR_GEOJSON = SCRIPT_DIR / 'vector-data-geojson'
DIR_VECTOR_GEOJSON_ATTACKED = SCRIPT_DIR / 'vector-data-geojson-attacked' / 'add_vertices'
DIR_GRAPH = SCRIPT_DIR / 'vector-data-geojson-attacked-graph'
DIR_GRAPH_ORIGINAL = DIR_GRAPH / 'Original'
DIR_GRAPH_ATTACKED = DIR_GRAPH / 'Attacked' / 'add_vertices'
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig2'

# 配置
STRENGTHS = [0, 1, 2]  # 强度：0, 1, 2
ADD_PCTS = list(range(10, 100, 10))  # 10..90


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
    # ⭐使用共享函数
    return convert_to_geojson(inputs, DIR_VECTOR_GEOJSON)


def add_vertices_to_geom(geom, strength: int, pct: int):
    """根据强度和比例向几何体添加顶点。与 convertToGeoJson-Attacked-TrainingSet.py 保持一致。"""
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
    
    if geom.is_empty:
        return geom
    
    geom_type = getattr(geom, 'geom_type', 'Unknown')
    
    if geom_type == 'Point':
        return geom  # 点不需要添加顶点
    
    elif geom_type == 'LineString':
        coords = list(geom.coords)
        if len(coords) < 2:
            return geom
        
        # 限制添加的顶点数量，避免过度复杂化（与 convertToGeoJson-Attacked-TrainingSet.py 一致）
        n_to_add = min(3, max(1, int((len(coords) - 1) * pct / 100)))
        
        if strength == 0:
            # 强度0：在现有顶点之间均匀插入新顶点（与 convertToGeoJson-Attacked-TrainingSet.py 一致）
            new_coords = [coords[0]]
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                new_coords.append(p1)
                for j in range(n_to_add):
                    t = (j + 1) / (n_to_add + 1)
                    mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                    new_coords.append(mid_point)
            new_coords.append(coords[-1])
            return LineString(new_coords)
        
        elif strength == 1:
            # 强度1：在现有顶点之间均匀插入新顶点，但添加小幅度噪声
            new_coords = [coords[0]]
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                new_coords.append(p1)
                for j in range(n_to_add):
                    t = (j + 1) / (n_to_add + 1)
                    mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                    # 添加小幅度噪声
                    noise = np.random.normal(0, 0.01, 2)
                    mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                    new_coords.append(mid_point)
            new_coords.append(coords[-1])
            return LineString(new_coords)
        
        elif strength == 2:
            # 强度2：在现有顶点之间均匀插入新顶点，但添加较大幅度噪声
            new_coords = [coords[0]]
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                new_coords.append(p1)
                for j in range(n_to_add):
                    t = (j + 1) / (n_to_add + 1)
                    mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                    # 添加较大幅度噪声
                    noise = np.random.normal(0, 0.05, 2)
                    mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                    new_coords.append(mid_point)
            new_coords.append(coords[-1])
            return LineString(new_coords)
    
    elif geom_type == 'Polygon':
        ext_coords = list(geom.exterior.coords)
        if len(ext_coords) < 4:
            return geom
        
        # 限制添加的顶点数量，避免过度复杂化
        n_to_add = min(3, max(1, int((len(ext_coords) - 1) * pct / 100)))
        
        # 处理外环
        new_ext_coords = [ext_coords[0]]
        for i in range(len(ext_coords) - 1):
            p1, p2 = ext_coords[i], ext_coords[i + 1]
            new_ext_coords.append(p1)
            for j in range(n_to_add):
                t = (j + 1) / (n_to_add + 1)
                mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                # 根据强度添加噪声
                if strength == 1:
                    noise = np.random.normal(0, 0.01, 2)
                    mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                elif strength == 2:
                    noise = np.random.normal(0, 0.05, 2)
                    mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                new_ext_coords.append(mid_point)
        new_ext_coords.append(ext_coords[-1])
        
        # 处理内环（洞）
        new_holes = []
        for ring in geom.interiors:
            ring_coords = list(ring.coords)
            if len(ring_coords) >= 4:
                new_ring_coords = [ring_coords[0]]
                for i in range(len(ring_coords) - 1):
                    p1, p2 = ring_coords[i], ring_coords[i + 1]
                    new_ring_coords.append(p1)
                    for j in range(n_to_add):
                        t = (j + 1) / (n_to_add + 1)
                        mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                        # 根据强度添加噪声
                        if strength == 1:
                            noise = np.random.normal(0, 0.01, 2)
                            mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                        elif strength == 2:
                            noise = np.random.normal(0, 0.05, 2)
                            mid_point = (mid_point[0] + noise[0], mid_point[1] + noise[1])
                        new_ring_coords.append(mid_point)
                new_ring_coords.append(ring_coords[-1])
                new_holes.append(new_ring_coords)
            else:
                new_holes.append(ring_coords)
        
        return Polygon(new_ext_coords, holes=new_holes if new_holes else None)
    
    elif geom_type in ['MultiPoint', 'MultiLineString', 'MultiPolygon']:
        # 对每个子几何体应用相同的处理
        geoms = list(geom.geoms)
        processed_geoms = []
        for sub_geom in geoms:
            processed_geom = add_vertices_to_geom(sub_geom, strength, pct)
            processed_geoms.append(processed_geom)
        
        if geom_type == 'MultiPoint':
            return MultiPoint(processed_geoms)
        elif geom_type == 'MultiLineString':
            return MultiLineString(processed_geoms)
        elif geom_type == 'MultiPolygon':
            return MultiPolygon(processed_geoms)
    
    return geom


def step3_generate_add_vertex_attacks(original_geojsons):
    """生成不同强度和比例的增加顶点攻击。"""
    print('[Step3] 生成增加顶点攻击 ->', DIR_VECTOR_GEOJSON_ATTACKED)

    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return {}

    outputs = {}
    DIR_VECTOR_GEOJSON_ATTACKED.mkdir(parents=True, exist_ok=True)
    
    for src in original_geojsons:
        base = src.stem
        subdir = DIR_VECTOR_GEOJSON_ATTACKED / base
        # 每次覆盖
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_geojson_error', src.name, exc)
            continue

        outputs[base] = {}
        
        for strength in STRENGTHS:
            for pct in ADD_PCTS:
                try:
                    attacked = gdf.copy()
                    attacked['geometry'] = attacked['geometry'].apply(
                        lambda geom: add_vertices_to_geom(geom, strength, pct)
                    )
                    out_path = subdir / f'add_strength{strength}_{pct}pct_vertices.geojson'
                    attacked.to_file(out_path, driver='GeoJSON')
                    print(f'输出: {base}/strength{strength}_{pct}%')
                    outputs[base][(strength, pct)] = out_path
                except Exception as exc:
                    print('attack_error', base, strength, pct, exc)
                    continue

    return outputs


def step4_convert_to_graph(original_geojsons, attacked_map):
    """按 convertToGraph 逻辑转为图结构（KNN+Delaunay，20维特征）。"""
    print('[Step4] 转换为图结构 ->', DIR_GRAPH)

    if gpd is None or Data is None:
        print('缺少依赖，无法构图。')
        return

    # ⭐使用共享函数批量转换
    convert_geojsons_to_graphs(
        original_geojsons=original_geojsons,
        attacked_geojson_map=attacked_map,
        output_dir_original=DIR_GRAPH_ORIGINAL,
        output_dir_attacked=DIR_GRAPH_ATTACKED
    )


def step5_generate_zero_watermark():
    """用 VGAT 模型对 Original 生成零水印。"""
    print('[Step5] 生成零水印 ->', DIR_ZEROWM)

    # 检查零水印是否已经存在
    if DIR_ZEROWM.exists() and any(DIR_ZEROWM.glob('*_watermark.*')):
        print('零水印已存在，跳过生成步骤。')
        return

    if Data is None or not MODEL_PATH.exists():
        print('缺少依赖或模型文件。')
        return

    try:
        from PIL import Image  # type: ignore
    except Exception:
        print('需要安装 pillow: pip install pillow')
        return

    # ⭐使用共享函数加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_improved_gat_model(device)
    except Exception as exc:
        print(f'模型加载失败: {exc}')
        return

    # ⭐使用共享函数加载版权图像
    copyright_img = load_cat32()

    # 准备输出目录（覆盖生成）
    DIR_ZEROWM.mkdir(parents=True, exist_ok=True)

    # 遍历 Original 图
    cnt = 0
    for pkl in sorted(DIR_GRAPH_ORIGINAL.glob('*_graph.pkl')):
        try:
            with open(pkl, 'rb') as f:
                graph: Data = pickle.load(f)
            
            # ⭐使用共享函数提取特征并转为二值矩阵
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            
            # XOR生成零水印
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
    """评估NC并生成结果。"""
    print('[Step6] 评估NC ->', DIR_RESULTS)

    if Data is None or not MODEL_PATH.exists():
        print('缺少依赖或模型文件。')
        return

    if pd is None or plt is None:
        print('缺少pandas/matplotlib依赖。')
        return

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)

    # ⭐使用共享函数加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_improved_gat_model(device)
    except Exception as exc:
        print(f'模型加载失败: {exc}')
        return

    # ⭐使用共享函数加载版权图像
    copyright_img = load_cat32()

    rows = []
    for subdir in DIR_GRAPH_ATTACKED.iterdir():
        if not subdir.is_dir():
            continue
        base = subdir.name
        
        # 加载对应 base 的零水印
        zwm_path = DIR_ZEROWM / f'{base}_watermark.npy'
        if not zwm_path.exists():
            print('缺少零水印，跳过: ', base)
            continue
        zwm = np.load(zwm_path)

        for pkl in sorted(subdir.glob('*_graph.pkl')):
            try:
                with open(pkl, 'rb') as f:
                    graph: Data = pickle.load(f)
                
                # ⭐使用共享函数提取特征
                feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
                extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
                nc = calc_nc(copyright_img, extracted)  # ⭐使用共享NC计算函数

                # 解析强度和百分比
                name = pkl.stem.replace('_graph', '')
                strength = -1
                pct = -1
                for s in STRENGTHS:
                    for p in ADD_PCTS:
                        if f'add_strength{s}_{p}pct_vertices' in name:
                            strength = s
                            pct = p
                            break
                    if strength != -1:
                        break
                rows.append({'base': base, 'strength': strength, 'add_pct': pct, 'nc': nc})
            except Exception as exc:
                print('nc_eval_error', pkl.name, exc)
                continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)
    
    # 重新组织数据为层次结构：每个强度下列出不同比例的NC值
    hierarchical_rows = []
    
    # 获取所有唯一的基础名称（矢量地图）
    base_names = sorted(df['base'].unique())
    
    for strength in STRENGTHS:
        # 添加强度标题行
        hierarchical_rows.append({
            '增加顶点': f'强度: {strength}',
            'VGAT': '',
            '类型': 'header'
        })
        
        # 添加每个比例的数据
        for pct in ADD_PCTS:
            # 添加比例标题行
            hierarchical_rows.append({
                '增加顶点': f'  {pct}%',
                'VGAT': '',
                '类型': 'subheader'
            })
            
            # 添加每个矢量地图的NC值
            for base in base_names:
                nc_value = df[(df['base'] == base) & (df['strength'] == strength) & (df['add_pct'] == pct)]['nc'].iloc[0] if len(df[(df['base'] == base) & (df['strength'] == strength) & (df['add_pct'] == pct)]) > 0 else 0
                hierarchical_rows.append({
                    '增加顶点': f'    {base}',  # 进一步缩进表示子项
                    'VGAT': f'{nc_value:.6f}',
                    '类型': 'data'
                })
            
            # 计算并添加该强度该比例下的平均值
            avg_nc = df[(df['strength'] == strength) & (df['add_pct'] == pct)]['nc'].mean()
            hierarchical_rows.append({
                '增加顶点': '    Average',
                'VGAT': f'{avg_nc:.6f}',
                '类型': 'average'
            })
    
    # 创建层次结构DataFrame
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    
    # 保存CSV文件
    csv_path = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 保存Excel文件
    xlsx_path = DIR_RESULTS / 'fig2_add_vertices_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            
            # 获取工作表以进行格式化
            worksheet = writer.sheets['NC结果']
            
            # 设置列宽
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 15
            
            # 格式化数据
            for idx, row in hierarchical_df.iterrows():
                if row['类型'] == 'header':
                    # 设置强度行为粗体
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True)
                elif row['类型'] == 'subheader':
                    # 设置比例行为粗体
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True)
                elif row['类型'] == 'average':
                    # 设置平均行为粗体和斜体
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True, italic=True)
                    worksheet[f'B{idx+2}'].font = worksheet[f'B{idx+2}'].font.copy(bold=True, italic=True)
                elif row['类型'] == 'data':
                    # 设置数据行缩进
                    worksheet[f'A{idx+2}'].alignment = worksheet[f'A{idx+2}'].alignment.copy(horizontal='left', indent=2)
    except Exception as e:
        print(f'Excel保存失败: {e}')
        # 如果格式化失败，至少保存基本数据
        hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
    
    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)

    # 绘制每张图与平均曲线
    plt.figure(figsize=(12, 8))
    
    # 为每个强度绘制子图
    for i, strength in enumerate(STRENGTHS):
        plt.subplot(2, 2, i + 1)
        
        # 绘制每个矢量地图的曲线
        for base, sub in df[df['strength'] == strength].groupby('base'):
            sub2 = sub.sort_values('add_pct')
            plt.plot(sub2['add_pct'], sub2['nc'], '-o', alpha=0.7, label=base, markersize=4)
        
        # 绘制平均曲线
        avg = df[df['strength'] == strength].groupby('add_pct')['nc'].mean().reset_index()
        plt.plot(avg['add_pct'], avg['nc'], 'k-o', linewidth=2.5, label='平均', markersize=6)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('增加顶点比例（%）')
        plt.ylabel('NC')
        plt.title(f'强度 {strength}：增加顶点攻击的NC鲁棒性')
        plt.legend(loc='best', fontsize=6, ncol=2)
        plt.ylim(0, 1.1)
    
    # 绘制综合对比图
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
    print('曲线图保存: ', fig_path.name)


def check_existing_files():
    """检查各步骤的输出文件是否已存在，决定是否需要跳过某些步骤"""
    print('[检查] 检查现有文件...')
    
    # 检查Step2输出：GeoJSON文件
    geojson_files = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
    step2_done = len(geojson_files) > 0
    
    # 检查Step3输出：攻击后的GeoJSON文件
    attacked_dirs = [d for d in DIR_VECTOR_GEOJSON_ATTACKED.glob('*') if not d.name.startswith('.')]
    step3_done = len(attacked_dirs) > 0 and all(
        len(list(attacked_dir.glob('*.geojson'))) > 0 
        for attacked_dir in attacked_dirs if attacked_dir.is_dir()
    )
    
    # 检查Step4输出：图结构文件
    original_graphs = [p for p in DIR_GRAPH_ORIGINAL.glob('*_graph.pkl') if not p.name.startswith('.')]
    attacked_graphs = [p for p in DIR_GRAPH_ATTACKED.glob('*/*_graph.pkl') if not p.name.startswith('.')]
    step4_done = len(original_graphs) > 0 and len(attacked_graphs) > 0
    
    # 检查Step5输出：零水印文件（支持 NPY 或 PNG）
    watermark_files = [p for p in DIR_ZEROWM.glob('*_watermark.npy') if not p.name.startswith('.')] + [p for p in DIR_ZEROWM.glob('*_watermark.png') if not p.name.startswith('.')]
    step5_done = len(watermark_files) > 0
    
    # 检查Step6输出：NC结果文件
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
    print('=== Fig2：增加顶点鲁棒性测试 ===')
    
    # 检查现有文件
    existing = check_existing_files()
    
    # 根据文件存在情况决定执行哪些步骤
    if existing['step6']:
        print('\n[跳过] 检测到NC结果文件已存在，直接运行Step6...')
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    
    if existing['step5'] and existing['step4']:
        print('\n[跳过] 检测到零水印文件和图结构已存在，直接运行Step6...')
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    
    if existing['step5'] and not existing['step4']:
        print('\n[部分跳过] 检测到零水印文件存在但攻击图不存在，运行Step3-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = {}
        # 重新构建attacked_map
        for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
            if attacked_dir.is_dir():
                base = attacked_dir.name
                attacked_map[base] = {}
                for strength in STRENGTHS:
                    for pct in ADD_PCTS:
                        geojson_file = attacked_dir / f'add_strength{strength}_{pct}pct_vertices.geojson'
                        if geojson_file.exists():
                            attacked_map[base][(strength, pct)] = geojson_file
        if not attacked_map:
            # 如果攻击文件也不存在，重新生成
            attacked_map = step3_generate_add_vertex_attacks(original_geojsons)
        step4_convert_to_graph(original_geojsons, attacked_map)
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
        # 需要重新获取原始GeoJSON文件列表
        inputs = step1_discover_inputs()
        original_geojsons = step2_convert_to_geojson(inputs) if not existing['step2'] else [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        # 重新构建attacked_map用于step4
        attacked_map = {}
        for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
            if attacked_dir.is_dir():
                base = attacked_dir.name
                attacked_map[base] = {}
                for strength in STRENGTHS:
                    for pct in ADD_PCTS:
                        geojson_file = attacked_dir / f'add_strength{strength}_{pct}pct_vertices.geojson'
                        if geojson_file.exists():
                            attacked_map[base][(strength, pct)] = geojson_file
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    
    if existing['step2']:
        print('\n[跳过] 检测到GeoJSON文件已存在，直接运行Step3-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = step3_generate_add_vertex_attacks(original_geojsons)
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return
    
    # 如果没有任何文件存在，从头开始运行所有步骤
    print('\n[从头开始] 运行所有步骤...')
    inputs = step1_discover_inputs()
    original_geojsons = step2_convert_to_geojson(inputs)
    attacked_map = step3_generate_add_vertex_attacks(original_geojsons)
    step4_convert_to_graph(original_geojsons, attacked_map)
    step5_generate_zero_watermark()
    step6_evaluate_nc()
    print('=== 完成 ===')


if __name__ == '__main__':
    main()
