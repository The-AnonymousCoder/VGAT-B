#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有消融模型是否能正常前向传播
用于在训练前快速发现潜在问题
"""

import os
import sys
import importlib.util
import torch
import numpy as np

# 导入VGAT-IMPROVED.py
current_dir = os.path.dirname(os.path.abspath(__file__))
vgat_improved_path = os.path.join(current_dir, 'VGAT-IMPROVED.py')

spec = importlib.util.spec_from_file_location("VGAT_IMPROVED", vgat_improved_path)
VGAT_IMPROVED = importlib.util.module_from_spec(spec)
sys.modules['VGAT_IMPROVED'] = VGAT_IMPROVED
spec.loader.exec_module(VGAT_IMPROVED)

from torch_geometric.data import Data

print("="*80)
print("消融模型测试脚本")
print("="*80)

def create_test_graph(num_nodes=10):
    """创建测试图数据"""
    # 创建20维节点特征
    x = torch.randn(num_nodes, 20)
    
    # 创建随机边
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    return Data(x=x, edge_index=edge_index)

def create_test_batch(num_graphs=3, nodes_per_graph=10):
    """创建批次测试数据"""
    batch_list = []
    for i in range(num_graphs):
        data = create_test_graph(nodes_per_graph)
        batch_list.append(data)
    
    # 手动创建batch
    x_list = []
    edge_list = []
    batch_idx = []
    node_offset = 0
    
    for i, data in enumerate(batch_list):
        x_list.append(data.x)
        # 调整边索引
        edge_list.append(data.edge_index + node_offset)
        # 创建batch索引
        batch_idx.extend([i] * data.x.size(0))
        node_offset += data.x.size(0)
    
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_list, dim=1)
    batch = torch.tensor(batch_idx)
    
    return Data(x=x, edge_index=edge_index, batch=batch)

# 测试各个模型
models_to_test = [
    ("Ablation1_NodeOnly.py", "NodeOnlyGATModel"),
    ("Ablation2_GraphOnly.py", "GraphOnlyModel"),
    ("Ablation3_MixedSingleStream.py", "MixedSingleStreamGATModel"),
    ("Ablation4_SingleHead.py", "SingleHeadGATModel"),
    ("Ablation5_GCN.py", "GCNModel"),
]

for script_name, model_class_name in models_to_test:
    print(f"\n{'='*80}")
    print(f"测试: {script_name}")
    print(f"模型类: {model_class_name}")
    print(f"{'='*80}")
    
    try:
        # 动态导入模型文件
        module_path = os.path.join(current_dir, script_name)
        spec = importlib.util.spec_from_file_location(f"ablation_module_{script_name}", module_path)
        ablation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ablation_module)
        
        # 获取模型类
        model_class = getattr(ablation_module, model_class_name)
        
        # 创建模型实例
        model = model_class(
            input_dim=20,
            hidden_dim=256,
            output_dim=1024,
            num_heads=8,
            dropout=0.3
        )
        model.eval()
        
        print(f"✅ 模型创建成功")
        
        # 测试1：单图前向传播
        test_data = create_test_graph(10)
        with torch.no_grad():
            output = model(test_data.x, test_data.edge_index, batch=None)
        
        assert output.shape == (1024,), f"单图输出维度错误: {output.shape}"
        assert not torch.isnan(output).any(), "输出包含NaN"
        assert not torch.isinf(output).any(), "输出包含Inf"
        
        print(f"✅ 单图前向传播成功，输出shape: {output.shape}")
        
        # 测试2：批次前向传播
        batch_data = create_test_batch(3, 10)
        with torch.no_grad():
            output = model(batch_data.x, batch_data.edge_index, batch=batch_data.batch)
        
        assert output.shape == (3, 1024), f"批次输出维度错误: {output.shape}"
        assert not torch.isnan(output).any(), "输出包含NaN"
        assert not torch.isinf(output).any(), "输出包含Inf"
        
        print(f"✅ 批次前向传播成功，输出shape: {output.shape}")
        
        # 测试3：边界情况（单节点图）
        single_node_data = create_test_graph(1)
        try:
            with torch.no_grad():
                output = model(single_node_data.x, single_node_data.edge_index, batch=None)
            print(f"✅ 单节点图测试成功")
        except Exception as e:
            print(f"⚠️  单节点图测试失败: {e}")
        
        print(f"\n✅✅✅ {model_class_name} 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌❌❌ {model_class_name} 测试失败!")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("测试完成")
print(f"{'='*80}")

