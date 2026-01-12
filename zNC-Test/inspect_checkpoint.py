#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查模型检查点的内容"""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
checkpoint_path = PROJECT_ROOT / 'VGAT' / 'models' / 'gat_model_Ablation1_NodeOnly_best.pth'

print(f"检查文件: {checkpoint_path}")
print(f"文件存在: {checkpoint_path.exists()}")
print()

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"检查点类型: {type(checkpoint)}")
    print()
    
    if isinstance(checkpoint, dict):
        print(f"检查点键: {list(checkpoint.keys())}")
        print()
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        print(f"模型层名称（前20个）:")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            tensor_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {i+1}. {key}: {tensor_shape}")
        
        print(f"\n总共 {len(state_dict)} 个参数")
        
        # 检查关键层
        print("\n关键层检查:")
        key_layers = ['gat1', 'gat2', 'gcn1', 'gcn2', 'ln1', 'fusion']
        for layer in key_layers:
            found = [k for k in state_dict.keys() if layer in k]
            if found:
                print(f"  ✓ 找到 {layer}: {len(found)} 个参数")
                print(f"    示例: {found[0]}")
            else:
                print(f"  ✗ 未找到 {layer}")
