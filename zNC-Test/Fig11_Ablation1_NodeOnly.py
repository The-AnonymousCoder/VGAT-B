#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig11_Ablation1_NodeOnly：使用消融实验1（仅节点级特征）的模型测试组合攻击鲁棒性"""

# 导入Fig11的所有功能，仅修改模型和目录配置
import sys
from pathlib import Path
import psutil  # ⭐ 内存监控

# 添加当前目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ⭐ 先设置配置，再导入（避免使用 Fig11 的默认配置）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODEL_PATH = PROJECT_ROOT / 'VGAT' / 'models' / 'gat_model_Ablation1_NodeOnly_best.pth'

# 临时保存原始配置模块
import Fig11 as _fig11_module

# 导入所有内容
from Fig11 import *

# ⭐ 强制覆盖配置
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark-ablation1'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig11_Ablation1_NodeOnly'
_fig11_module.DIR_ZEROWM = DIR_ZEROWM
_fig11_module.DIR_RESULTS = DIR_RESULTS


def load_ablation1_model(device):
    """加载消融实验1的模型"""
    if not ABLATION_MODEL_PATH.exists():
        raise FileNotFoundError(f"消融实验1模型文件不存在: {ABLATION_MODEL_PATH}")
    import importlib.util
    ablation1_path = PROJECT_ROOT / 'VGAT' / 'Ablation1_NodeOnly.py'
    spec = importlib.util.spec_from_file_location("Ablation1_NodeOnly", ablation1_path)
    ablation1_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablation1_module)
    NodeOnlyGATModel = ablation1_module.NodeOnlyGATModel
    
    # 创建模型实例（参数与训练时一致）
    model = NodeOnlyGATModel(
        input_dim=20,  # 输入仍然是20维，但内部只使用节点级10维
        hidden_dim=256,
        output_dim=1024,
        num_heads=8,
        dropout=0.3
    ).to(device)
    # 加载权重
    checkpoint = torch.load(ABLATION_MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 已加载消融实验1模型: {ABLATION_MODEL_PATH.name}")
    return model, device


# 覆盖step5和step6以使用消融模型
def step5_generate_zero_watermark():
    print('[Step5] 生成零水印 (消融实验1) ->', DIR_ZEROWM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_ablation1_model(device)
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
    if cnt == 0:
        print('未找到 Original 图，无法生成零水印。')
    else:
        print(f'共生成零水印 {cnt} 个。')


def check_existing_files():
    """覆盖：使用消融实验1的目录"""
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


def step6_evaluate_nc_OLD():  # 已废弃，使用新的分批处理方式
    print('[Step6] 评估NC (消融实验1) ->', DIR_RESULTS)
    if not DIR_GRAPH_ATTACKED.exists():
        print('缺少攻击图目录'); return
    if not ABLATION_MODEL_PATH.exists():
        print('消融实验1模型文件不存在'); return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_ablation1_model(device)
    except Exception as exc:
        print(f'模型加载失败: {exc}'); return
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    copyright_img = load_cat32()
    key_to_label = {k: lbl for k, lbl in COMBOS}
    order_keys = [k for k, _ in COMBOS]
    rows = []
    base_dirs = sorted([d for d in DIR_GRAPH_ATTACKED.iterdir() if d.is_dir()])
    total_bases = len(base_dirs)
    print(f'找到 {total_bases} 个数据集')
    
    for idx, base_dir_path in enumerate(base_dirs, 1):
        base = base_dir_path.name
        print(f'\n[{idx}/{total_bases}] 处理数据集: {base}')
        zwm_path = DIR_ZEROWM / f'{base}_watermark.npy'
        if not zwm_path.exists(): 
            print(f'  ✗ 缺少零水印，跳过: {base}'); 
            continue
        zwm = np.load(zwm_path)
        
        pkl_files = sorted(base_dir_path.glob('*_graph.pkl'))
        print(f'  找到 {len(pkl_files)} 个图文件')
        
        for pkl_idx, pkl in enumerate(pkl_files, 1):
            try:
                with open(pkl, 'rb') as f:
                    graph = pickle.load(f)
                
                # 立即计算，不保留中间变量
                with torch.no_grad():
                    feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
                    extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
                    nc = calc_nc(copyright_img, extracted)
                
                name = pkl.stem.replace('_graph', '')
                k_found = None
                for k in order_keys:
                    if k in name:
                        k_found = k; break
                
                rows.append({'base': base, 'combo_key': k_found or 'unknown', 'nc': nc})
                print(f'    [{pkl_idx}/{len(pkl_files)}] {name}: NC={nc:.4f}')
                
                # ⭐ 立即删除所有对象
                del graph, feat_mat, extracted, nc, name, k_found, pkl
                
                # 每处理3个文件就强制清理
                if pkl_idx % 3 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
            except Exception as exc:
                print(f'    ✗ 错误: {exc}')
                import gc
                gc.collect()
        
        # base处理完，强制清理
        del zwm, pkl_files
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f'  ✓ 完成，已处理 {len([r for r in rows if r["base"] == base])} 个文件')
    if not rows: print('没有评估结果。'); return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    base_names = sorted(df['base'].unique())
    for k in order_keys:
        label = key_to_label.get(k, k)
        hierarchical_rows.append({'组合攻击': label, 'Ablation1_NodeOnly': '', '类型': 'header'})
        for base in base_names:
            sub = df[(df['base'] == base) & (df['combo_key'] == k)]
            nc_value = sub['nc'].iloc[0] if len(sub) > 0 else 0
            hierarchical_rows.append({'组合攻击': f'  {base}', 'Ablation1_NodeOnly': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = df[df['combo_key'] == k]['nc'].mean()
        hierarchical_rows.append({'组合攻击': '  Average', 'Ablation1_NodeOnly': f'{avg_nc:.6f}', '类型': 'average'})
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig11_ablation1_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    xlsx_path = DIR_RESULTS / 'fig11_ablation1_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            ws = writer.sheets['NC结果']
            ws.column_dimensions['A'].width = 34
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
    print('结果表保存: ', csv_path.name); print('Excel文件保存: ', xlsx_path.name)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(order_keys))
    labels = [key_to_label[k] for k in order_keys]
    for base, sub in df.groupby('base'):
        y = [sub[sub['combo_key'] == k]['nc'].iloc[0] if len(sub[sub['combo_key'] == k]) > 0 else np.nan for k in order_keys]
        plt.plot(x, y, '-o', alpha=0.7, label=base)
    avg_vals = [df[df['combo_key'] == k]['nc'].mean() for k in order_keys]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, labels, rotation=15)
    plt.xlabel('组合攻击'); plt.ylabel('NC')
    plt.title('Fig11_Ablation1：组合攻击 NC鲁棒性 (仅节点级特征)')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig11_ablation1_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print('曲线图保存: ', fig_path.name)


def check_memory(threshold=90):
    """检查内存使用率，超过阈值则返回True"""
    mem = psutil.virtual_memory()
    percent = mem.percent
    if percent > threshold:
        print(f'\n⚠️  警告：内存使用率 {percent:.1f}% 超过阈值 {threshold}%！')
        return True
    return False


def process_single_base(base_name, output_file, model, device, copyright_img):
    """处理单个base并保存到临时文件（使用已加载的模型）"""
    import json
    print(f'\n处理单个数据集: {base_name}')
    key_to_label = {k: lbl for k, lbl in COMBOS}
    order_keys = [k for k, _ in COMBOS]
    
    base_dir_path = DIR_GRAPH_ATTACKED / base_name
    if not base_dir_path.exists():
        print(f'目录不存在: {base_dir_path}'); return
    
    zwm_path = DIR_ZEROWM / f'{base_name}_watermark.npy'
    if not zwm_path.exists():
        print(f'缺少零水印: {zwm_path}'); return
    
    zwm = np.load(zwm_path)
    pkl_files = sorted(base_dir_path.glob('*_graph.pkl'))
    print(f'找到 {len(pkl_files)} 个图文件')
    
    rows = []
    for pkl_idx, pkl in enumerate(pkl_files, 1):
        # ⭐ 每处理3个文件检查一次内存
        if pkl_idx % 3 == 0:
            mem = psutil.virtual_memory()
            print(f'  [内存: {mem.percent:.1f}%]', end=' ')
            if check_memory(90):
                print(f'\n❗ 内存不足，终止处理 {base_name}，已处理 {len(rows)}/{len(pkl_files)} 个文件')
                break
        
        try:
            with open(pkl, 'rb') as f:
                graph = pickle.load(f)
            
            # ⭐ 完全模仿 Fig11.py 的逻辑，不添加额外的 torch.no_grad()
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
            nc = calc_nc(copyright_img, extracted)
            
            name = pkl.stem.replace('_graph', '')
            k_found = None
            for k in order_keys:
                if k in name:
                    k_found = k; break
            
            rows.append({'base': base_name, 'combo_key': k_found or 'unknown', 'nc': nc})
            print(f'  [{pkl_idx}/{len(pkl_files)}] {name}: NC={nc:.4f}')
            
        except Exception as exc:
            print(f'  ✗ 错误: {exc}')
            continue
    
    # 保存到临时文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f'✓ 已保存: {output_file}')


if __name__ == '__main__':
    import sys, json
    print('=== Fig11_Ablation1：组合攻击鲁棒性测试 (仅节点级特征) ===')
    
    # 支持命令行参数指定处理单个base
    if len(sys.argv) > 1:
        base_name = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else f'temp_{base_name}.json'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, device = load_ablation1_model(device)
        copyright_img = load_cat32()
        process_single_base(base_name, output_file, model, device, copyright_img)
        sys.exit(0)
    
    # 否则处理所有base（⭐与 Fig11.py 完全一致：加载一次模型，循环处理）
    print('=== Fig11：组合攻击鲁棒性测试 ===')
    existing = check_existing_files()
    if existing['step6']:
        print('\n已有结果'); sys.exit(0)
    elif not (existing['step5'] and existing['step4']):
        print('请先运行 Fig11.py'); sys.exit(1)
    
    # ⭐ 加载模型一次（像 Fig11.py 一样）
    print('\n加载模型...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_ablation1_model(device)
    copyright_img = load_cat32()
    
    # 分批处理
    DIR_TEMP = SCRIPT_DIR / 'temp_ablation1'
    DIR_TEMP.mkdir(exist_ok=True)
    
    base_dirs = sorted([d for d in DIR_GRAPH_ATTACKED.iterdir() if d.is_dir()])
    print(f'找到 {len(base_dirs)} 个数据集，将逐个处理...\n')
    
    for idx, base_dir in enumerate(base_dirs, 1):
        base_name = base_dir.name
        temp_file = DIR_TEMP / f'{base_name}.json'
        
        # ⭐ 每个base开始前检查内存
        mem = psutil.virtual_memory()
        print(f'\n[{idx}/{len(base_dirs)}] 内存使用: {mem.percent:.1f}% ({mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB)')
        if check_memory(90):
            print(f'\n❗❗ 内存不足，终止批处理！已处理 {idx-1}/{len(base_dirs)} 个数据集')
            break
        
        if temp_file.exists():
            print(f'  {base_name} - 跳过（已处理）')
            continue
        
        print(f'  处理 {base_name}...')
        process_single_base(base_name, str(temp_file), model, device, copyright_img)
    
    # 合并所有结果
    print('\n合并结果...')
    all_rows = []
    for temp_file in sorted(DIR_TEMP.glob('*.json')):
        with open(temp_file, 'r', encoding='utf-8') as f:
            all_rows.extend(json.load(f))
    
    if not all_rows:
        print('没有结果'); sys.exit(1)
    
    # 生成最终报告（step6的后半部分）
    df = pd.DataFrame(all_rows)
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    key_to_label = {k: lbl for k, lbl in COMBOS}
    order_keys = [k for k, _ in COMBOS]
    
    hierarchical_rows = []
    base_names = sorted(df['base'].unique())
    for k in order_keys:
        label = key_to_label.get(k, k)
        hierarchical_rows.append({'组合攻击': label, 'Ablation1_NodeOnly': '', '类型': 'header'})
        for base in base_names:
            sub = df[(df['base'] == base) & (df['combo_key'] == k)]
            nc_value = sub['nc'].iloc[0] if len(sub) > 0 else 0
            hierarchical_rows.append({'组合攻击': f'  {base}', 'Ablation1_NodeOnly': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = df[df['combo_key'] == k]['nc'].mean()
        hierarchical_rows.append({'组合攻击': '  Average', 'Ablation1_NodeOnly': f'{avg_nc:.6f}', '类型': 'average'})
    
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig11_ablation1_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'✓ CSV: {csv_path}')
    
    # 画图
    plt.figure(figsize=(12, 8))
    labels = [key_to_label.get(k, k) for k in order_keys]
    x = np.arange(len(labels))
    for base in base_names:
        vals = [df[(df['base'] == base) & (df['combo_key'] == k)]['nc'].iloc[0] if len(df[(df['base'] == base) & (df['combo_key'] == k)]) > 0 else 0 for k in order_keys]
        plt.plot(x, vals, 'o-', label=base, linewidth=1.5, alpha=0.7)
    avg_vals = [df[df['combo_key'] == k]['nc'].mean() for k in order_keys]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3); plt.xticks(x, labels, rotation=15)
    plt.xlabel('组合攻击'); plt.ylabel('NC'); plt.title('Fig11_Ablation1：组合攻击 NC鲁棒性 (仅节点级特征)')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig11_ablation1_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f'✓ 图: {fig_path}')
    print('\n=== 完成 ===')
