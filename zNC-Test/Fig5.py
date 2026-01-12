#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig5：裁剪攻击鲁棒性测试脚本（5种裁剪方式）

步骤与 Fig1 对齐：
1) 扫描 `vector-data` 下8个矢量数据
2) 转为 `vector-data-geojson`
3) 生成 5 种裁剪攻击到 `vector-data-geojson-attacked/crop/{base}`
4) 转图结构到 `vector-data-geojson-attacked-graph/crop/{Original,Attacked}`
5) 用 `VGAT/models/gat_model_best.pth` 对 `Original` 提取零水印到 `vector-data-zerowatermark`
6) 对攻击版本做NC评估，输出到 `zNC-Test/NC-Results/Fig5`
"""

from pathlib import Path
import sys
from typing import List, Dict
import pickle
import random
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
DIR_VECTOR_GEOJSON_ATTACKED = SCRIPT_DIR / 'vector-data-geojson-attacked' / 'crop'
DIR_GRAPH = SCRIPT_DIR / 'vector-data-geojson-attacked-graph'
DIR_GRAPH_ORIGINAL = DIR_GRAPH / 'Original'
DIR_GRAPH_ATTACKED = DIR_GRAPH / 'Attacked' / 'crop'
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig5'


# 配置：裁剪方式
CROP_TYPES = [
    ("x_center_50pct", "沿X轴中心裁剪50%"),
    ("y_center_50pct", "沿Y轴中心裁剪50%"),
    ("top_left", "裁剪左上角"),
    ("bottom_right", "裁剪右下角"),
    ("random_40pct", "随机裁剪40%"),
]


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


def step3_generate_crop_attacks(original_geojsons: List[Path]) -> Dict[str, Dict[str, Path]]:
    """按 TrainingSet 的裁剪逻辑生成 5 种裁剪攻击。"""
    print('[Step3] 生成裁剪攻击 ->', DIR_VECTOR_GEOJSON_ATTACKED)
    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return {}

    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    outputs: Dict[str, Dict[str, Path]] = {}
    DIR_VECTOR_GEOJSON_ATTACKED.mkdir(parents=True, exist_ok=True)

    for src in original_geojsons:
        base = src.stem
        subdir = DIR_VECTOR_GEOJSON_ATTACKED / base
        # 覆盖生成
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_geojson_error', src.name, exc)
            continue

        outputs[base] = {}

        # 预计算边界与每个对象的 bounds
        bounds = gdf.total_bounds  # minx, miny, maxx, maxy
        mid_x = (bounds[0] + bounds[2]) / 2
        mid_y = (bounds[1] + bounds[3]) / 2
        bdf = gdf.geometry.bounds  # DataFrame: minx,miny,maxx,maxy

        for crop_key, _label in CROP_TYPES:
            try:
                attacked = gdf.copy()
                if crop_key == 'x_center_50pct':
                    attacked = attacked[bdf['minx'] < mid_x].reset_index(drop=True)
                elif crop_key == 'y_center_50pct':
                    attacked = attacked[bdf['miny'] < mid_y].reset_index(drop=True)
                elif crop_key == 'top_left':
                    attacked = attacked[(bdf['minx'] < mid_x) & (bdf['miny'] > mid_y)].reset_index(drop=True)
                elif crop_key == 'bottom_right':
                    attacked = attacked[(bdf['minx'] > mid_x) & (bdf['miny'] < mid_y)].reset_index(drop=True)
                elif crop_key == 'random_40pct':
                    num_objects = len(attacked)
                    num_to_keep = int(num_objects * 0.6)
                    if num_to_keep > 0 and num_objects > 0:
                        indices_to_keep = random.sample(range(num_objects), min(num_to_keep, num_objects))
                        attacked = attacked.iloc[indices_to_keep].reset_index(drop=True)
                out_path = subdir / f'crop_{crop_key}.geojson'
                attacked.to_file(out_path, driver='GeoJSON')
                print(f'输出: {base}/{out_path.name}')
                outputs[base][crop_key] = out_path
            except Exception as exc:
                print('attack_error', base, crop_key, exc)
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
    """对裁剪攻击版本评估NC并保存结果。"""
    print('[Step6] 评估NC ->', DIR_RESULTS)

    # 防御：若攻击图目录不存在，提示先运行Step4
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

    # 依赖
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        print('需要安装 pandas: pip install pandas openpyxl')
        print('pandas_import_error', exc)
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print('需要安装 matplotlib: pip install matplotlib')
        print('matplotlib_import_error', exc)
        return

    # ⭐使用共享函数加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_improved_gat_model(device)
    except Exception as exc:
        print(f'模型加载失败: {exc}')
        return

    # 准备
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    copyright_img = load_cat32()

    # 收集NC
    crop_keys = [k for k, _ in CROP_TYPES]
    crop_key_to_label = {k: lbl for k, lbl in CROP_TYPES}
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

                # 解析裁剪类型
                name = pkl.stem.replace('_graph', '')
                ckey = None
                for k in crop_keys:
                    if f'crop_{k}' in name:
                        ckey = k
                        break
                rows.append({'base': base, 'crop_key': ckey or 'unknown', 'nc': nc})
            except Exception as exc:
                print('nc_eval_error', pkl.name, exc)
                continue

    if not rows:
        print('没有评估结果。')
        return

    df = pd.DataFrame(rows)

    # 重新组织数据为层次结构：每种裁剪方式下列出8个矢量地图的NC值和平均值
    hierarchical_rows = []
    base_names = sorted(df['base'].unique())
    crop_order = [k for k, _ in CROP_TYPES]

    for k in crop_order:
        label = crop_key_to_label.get(k, k)
        # 添加裁剪方式行
        hierarchical_rows.append({'裁剪方式': label, 'VGAT': '', '类型': 'header'})
        # 添加每个矢量地图的NC值
        for base in base_names:
            sub = df[(df['base'] == base) & (df['crop_key'] == k)]
            nc_value = sub['nc'].iloc[0] if len(sub) > 0 else 0
            hierarchical_rows.append({'裁剪方式': f'  {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        # 平均值
        avg_nc = df[df['crop_key'] == k]['nc'].mean()
        hierarchical_rows.append({'裁剪方式': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)

    # 保存CSV/Excel
    csv_path = DIR_RESULTS / 'fig5_crop_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    xlsx_path = DIR_RESULTS / 'fig5_crop_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            worksheet = writer.sheets['NC结果']
            worksheet.column_dimensions['A'].width = 28
            worksheet.column_dimensions['B'].width = 15
            for idx, row in hierarchical_df.iterrows():
                if row['类型'] == 'header':
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True)
                elif row['类型'] == 'average':
                    worksheet[f'A{idx+2}'].font = worksheet[f'A{idx+2}'].font.copy(bold=True, italic=True)
                    worksheet[f'B{idx+2}'].font = worksheet[f'B{idx+2}'].font.copy(bold=True, italic=True)
                elif row['类型'] == 'data':
                    worksheet[f'A{idx+2}'].alignment = worksheet[f'A{idx+2}'].alignment.copy(horizontal='left', indent=1)
    except Exception as e:
        print(f'Excel保存失败: {e}')
        hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')

    print('结果表保存: ', csv_path.name)
    print('Excel文件保存: ', xlsx_path.name)

    # 绘图（类别横轴）
    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(CROP_TYPES))
    labels = [lbl for _, lbl in CROP_TYPES]
    for base, sub in df.groupby('base'):
        # 对应顺序的nc
        y = []
        for k, _ in CROP_TYPES:
            val = sub[sub['crop_key'] == k]['nc']
            y.append(val.iloc[0] if len(val) > 0 else np.nan)
        plt.plot(x_positions, y, '-o', alpha=0.7, label=base)
    # 平均
    avg_vals = []
    for k, _ in CROP_TYPES:
        avg_vals.append(df[df['crop_key'] == k]['nc'].mean())
    plt.plot(x_positions, avg_vals, 'k-o', linewidth=2.5, label='平均')

    plt.grid(True, alpha=0.3)
    plt.xticks(x_positions, labels, rotation=20)
    plt.xlabel('裁剪方式')
    plt.ylabel('NC')
    plt.title('Fig5：裁剪攻击(5种) 的NC鲁棒性')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig5_crop_nc.png'
    plt.tight_layout()
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

    watermark_files = [p for p in DIR_ZEROWM.glob('*_watermark.npy') if not p.name.startswith('.')]
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
    print('=== Fig5：裁剪攻击鲁棒性测试 ===')

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
                    for crop_key, _ in CROP_TYPES:
                        geojson_file = attacked_dir / f'crop_{crop_key}.geojson'
                        if geojson_file.exists():
                            attacked_map[base][crop_key] = geojson_file
        else:
            attacked_map = step3_generate_crop_attacks(original_geojsons)

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
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map: Dict[str, Dict[str, Path]] = {}
        for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
            if attacked_dir.is_dir():
                base = attacked_dir.name
                attacked_map[base] = {}
                for crop_key, _ in CROP_TYPES:
                    geojson_file = attacked_dir / f'crop_{crop_key}.geojson'
                    if geojson_file.exists():
                        attacked_map[base][crop_key] = geojson_file
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return

    if existing['step2']:
        print('\n[跳过] 检测到GeoJSON文件已存在，直接运行Step3-6...')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = step3_generate_crop_attacks(original_geojsons)
        step4_convert_to_graph(original_geojsons, attacked_map)
        step5_generate_zero_watermark()
        step6_evaluate_nc()
        print('=== 完成 ===')
        return

    print('\n[从头开始] 运行所有步骤...')
    inputs = step1_discover_inputs()
    original_geojsons = step2_convert_to_geojson(inputs)
    attacked_map = step3_generate_crop_attacks(original_geojsons)
    step4_convert_to_graph(original_geojsons, attacked_map)
    step5_generate_zero_watermark()
    step6_evaluate_nc()
    print('=== 完成 ===')


if __name__ == '__main__':
    main()


