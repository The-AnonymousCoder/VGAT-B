#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig11_Ablation5_GCN：使用消融实验5（GCN替代GAT）的模型测试组合攻击鲁棒性"""
import sys
from pathlib import Path
import psutil
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from Fig11 import *

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODEL_PATH = PROJECT_ROOT / 'VGAT' / 'models' / 'gat_model_Ablation5_GCN_best.pth'
import Fig11 as _fig11_module
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark-ablation5'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig11_Ablation5_GCN'
_fig11_module.DIR_ZEROWM = DIR_ZEROWM
_fig11_module.DIR_RESULTS = DIR_RESULTS

def load_ablation5_model(device):
    """加载消融实验5的模型"""
    if not ABLATION_MODEL_PATH.exists():
        raise FileNotFoundError(f"消融实验5模型文件不存在: {ABLATION_MODEL_PATH}")
    
    # 动态导入消融模型
    import importlib.util
    ablation5_path = PROJECT_ROOT / 'VGAT' / 'Ablation5_GCN.py'
    spec = importlib.util.spec_from_file_location("Ablation5_GCN", ablation5_path)
    ablation5_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablation5_module)
    
    GCNModel = ablation5_module.GCNModel
    
    # 创建模型实例（参数与训练时一致）
    model = GCNModel(
        input_dim=20,  # GCN使用完整20维输入（内部自动分离节点级和图级）
        hidden_dim=256,
        output_dim=1024,
        num_heads=8,  # GCN不使用，保留兼容性
        dropout=0.3
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(ABLATION_MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ 已加载消融实验5模型: {ABLATION_MODEL_PATH.name}")
    return model, device

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

def step5_generate_zero_watermark():
    print('[Step5] 生成零水印 (消融实验5) ->', DIR_ZEROWM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_ablation5_model(device)
    copyright_img = load_cat32()
    DIR_ZEROWM.mkdir(parents=True, exist_ok=True)
    cnt = 0
    for pkl in sorted(DIR_GRAPH_ORIGINAL.glob('*_graph.pkl')):
        try:
            with open(pkl, 'rb') as f: graph = pickle.load(f)
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            zwm = np.logical_xor(feat_mat, copyright_img).astype(np.uint8)
            base = pkl.stem.replace('_graph', '')
            np.save(DIR_ZEROWM / f'{base}_watermark.npy', zwm)
            try: Image.fromarray((zwm * 255).astype(np.uint8)).save(DIR_ZEROWM / f'{base}_watermark.png')
            except Exception: pass
            print('零水印已保存: ', f'{base}_watermark.npy')
            cnt += 1
        except Exception as exc: print('zero_watermark_error', pkl.name, exc)
    print(f'共生成零水印 {cnt} 个。' if cnt > 0 else '未找到 Original 图，无法生成零水印。')

def check_memory(t=90):
    m = psutil.virtual_memory()
    if m.percent > t: print(f'\n⚠️  内存{m.percent:.1f}%！'); return True
    return False

def step6_evaluate_nc():
    print('[Step6] 评估NC (消融实验5) ->', DIR_RESULTS)
    if not DIR_GRAPH_ATTACKED.exists() or not ABLATION_MODEL_PATH.exists(): return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try: model, device = load_ablation5_model(device)
    except Exception as exc: print(f'模型加载失败: {exc}'); return
    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    copyright_img = load_cat32()
    key_to_label = {k: lbl for k, lbl in COMBOS}
    order_keys = [k for k, _ in COMBOS]
    rows = []; fc = 0
    for base_dir_path in sorted(DIR_GRAPH_ATTACKED.iterdir()):
        if not base_dir_path.is_dir(): continue
        base = base_dir_path.name
        if check_memory(90): print('\n❗终止'); break
        zwm_path = DIR_ZEROWM / f'{base}_watermark.npy'
        if not zwm_path.exists(): continue
        zwm = np.load(zwm_path)
        for pkl in sorted(base_dir_path.glob('*_graph.pkl')):
            fc += 1
            if fc % 10 == 0 and check_memory(90): print(f'\n❗{fc}个后终止'); break
            try:
                with open(pkl, 'rb') as f: graph = pickle.load(f)
                feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
                extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
                nc = calc_nc(copyright_img, extracted)
                name = pkl.stem.replace('_graph', '')
                k_found = next((k for k in order_keys if k in name), 'unknown')
                rows.append({'base': base, 'combo_key': k_found or 'unknown', 'nc': nc})
            except Exception as exc: 
                print('nc_eval_error', pkl.name, exc)
                continue
    if not rows: return
    df = pd.DataFrame(rows)
    hierarchical_rows = []
    base_names = sorted(df['base'].unique())
    for k in order_keys:
        label = key_to_label.get(k, k)
        hierarchical_rows.append({'组合攻击': label, 'Ablation5_GCN': '', '类型': 'header'})
        for base in base_names:
            sub = df[(df['base'] == base) & (df['combo_key'] == k)]
            nc_value = sub['nc'].iloc[0] if len(sub) > 0 else 0
            hierarchical_rows.append({'组合攻击': f'  {base}', 'Ablation5_GCN': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = df[df['combo_key'] == k]['nc'].mean()
        hierarchical_rows.append({'组合攻击': '  Average', 'Ablation5_GCN': f'{avg_nc:.6f}', '类型': 'average'})
    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig11_ablation5_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    xlsx_path = DIR_RESULTS / 'fig11_ablation5_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
            ws = writer.sheets['NC结果']
            ws.column_dimensions['A'].width = 34; ws.column_dimensions['B'].width = 20
            for idx, row in hierarchical_df.iterrows():
                if row['类型'] == 'header': ws[f'A{idx+2}'].font = ws[f'A{idx+2}'].font.copy(bold=True)
                elif row['类型'] == 'average':
                    ws[f'A{idx+2}'].font = ws[f'A{idx+2}'].font.copy(bold=True, italic=True)
                    ws[f'B{idx+2}'].font = ws[f'B{idx+2}'].font.copy(bold=True, italic=True)
                elif row['类型'] == 'data': ws[f'A{idx+2}'].alignment = ws[f'A{idx+2}'].alignment.copy(horizontal='left', indent=1)
    except Exception as e: print('Excel保存失败: ', e); hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
    print('结果表保存: ', csv_path.name, '|', xlsx_path.name)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(order_keys))
    labels = [key_to_label[k] for k in order_keys]
    for base, sub in df.groupby('base'):
        y = [sub[sub['combo_key'] == k]['nc'].iloc[0] if len(sub[sub['combo_key'] == k]) > 0 else np.nan for k in order_keys]
        plt.plot(x, y, '-o', alpha=0.7, label=base)
    avg_vals = [df[df['combo_key'] == k]['nc'].mean() for k in order_keys]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
    plt.grid(True, alpha=0.3); plt.xticks(x, labels, rotation=15)
    plt.xlabel('组合攻击'); plt.ylabel('NC'); plt.title('Fig11_Ablation5：组合攻击 NC鲁棒性 (GCN)')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig11_ablation5_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print('曲线图保存: ', fig_path.name)

if __name__ == '__main__':
    print('=== Fig11_Ablation5：组合攻击鲁棒性测试 (GCN) ===')
    print('=== Fig11：组合攻击鲁棒性测试 ===')
    existing = check_existing_files()
    if existing['step6']:
        print('\n[跳过] 检测到NC结果文件已存在，直接运行Step6...'); step6_evaluate_nc(); print('=== 完成 ===');
    elif existing['step5'] and existing['step4']:
        print('\n[跳过] 检测到零水印与图结构已存在，直接运行Step6...'); step6_evaluate_nc(); print('=== 完成 ===');
    else:
        print('请先运行 Fig11.py 生成图结构和零水印')
