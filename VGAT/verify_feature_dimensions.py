#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证所有消融模型都正确使用了18维特征（排除维度3、4）
"""

import os
import sys
import importlib.util

print("="*80)
print("特征维度验证脚本")
print("="*80)
print("\n【目标】确保所有模型都使用18维特征，排除维度3(Hu不变矩)和维度4(边界复杂度)\n")

# 正确的特征索引（共18维）
CORRECT_NODE_DIMS = [5, 6, 7, 8, 9, 10, 14, 15, 16, 17]  # 10维
CORRECT_GRAPH_DIMS = [0, 1, 2, 11, 12, 13, 18, 19]  # 8维
CORRECT_ALL_DIMS = sorted(CORRECT_NODE_DIMS + CORRECT_GRAPH_DIMS)  # 18维

print(f"✅ 正确的节点级特征索引（10维）: {CORRECT_NODE_DIMS}")
print(f"✅ 正确的图级特征索引（8维）:   {CORRECT_GRAPH_DIMS}")
print(f"✅ 正确的完整特征索引（18维）:  {CORRECT_ALL_DIMS}")
print(f"❌ 应排除的特征索引：[3, 4]（Hu不变矩 + 边界复杂度）\n")

current_dir = os.path.dirname(os.path.abspath(__file__))

# 导入VGAT-IMPROVED.py
vgat_improved_path = os.path.join(current_dir, 'VGAT-IMPROVED.py')
spec = importlib.util.spec_from_file_location("VGAT_IMPROVED", vgat_improved_path)
VGAT_IMPROVED = importlib.util.module_from_spec(spec)
sys.modules['VGAT_IMPROVED'] = VGAT_IMPROVED
spec.loader.exec_module(VGAT_IMPROVED)

# 测试各个模型
models_to_test = [
    {
        "script": "VGAT-IMPROVED.py",
        "class": "ImprovedGATModel",
        "expected_node": CORRECT_NODE_DIMS,
        "expected_graph": CORRECT_GRAPH_DIMS,
    },
    {
        "script": "Ablation1_NodeOnly.py",
        "class": "NodeOnlyGATModel",
        "expected_node": CORRECT_NODE_DIMS,
        "expected_graph": None,
    },
    {
        "script": "Ablation2_GraphOnly.py",
        "class": "GraphOnlyModel",
        "expected_node": None,
        "expected_graph": CORRECT_GRAPH_DIMS,
    },
    {
        "script": "Ablation3_MixedSingleStream.py",
        "class": "MixedSingleStreamGATModel",
        "expected_all": CORRECT_ALL_DIMS,
    },
    {
        "script": "Ablation4_SingleHead.py",
        "class": "SingleHeadGATModel",
        "expected_node": CORRECT_NODE_DIMS,
        "expected_graph": CORRECT_GRAPH_DIMS,
    },
    {
        "script": "Ablation5_GCN.py",
        "class": "GCNModel",
        "expected_node": CORRECT_NODE_DIMS,
        "expected_graph": CORRECT_GRAPH_DIMS,
    },
]

all_passed = True

for model_info in models_to_test:
    print(f"\n{'='*80}")
    print(f"检查: {model_info['script']} - {model_info['class']}")
    print(f"{'='*80}")
    
    try:
        # 动态导入模型文件
        if model_info['script'] == 'VGAT-IMPROVED.py':
            module = VGAT_IMPROVED
        else:
            module_path = os.path.join(current_dir, model_info['script'])
            spec = importlib.util.spec_from_file_location(f"test_{model_info['script']}", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        
        # 获取模型类
        model_class = getattr(module, model_info['class'])
        
        # 创建模型实例（静默日志）
        import logging
        old_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)
        
        model = model_class(
            input_dim=20,
            hidden_dim=256,
            output_dim=1024,
            num_heads=8 if 'SingleHead' not in model_info['class'] else 1,
            dropout=0.3
        )
        
        logging.root.setLevel(old_level)
        
        passed = True
        
        # 检查节点级特征
        if "expected_node" in model_info and model_info["expected_node"] is not None:
            actual_node = getattr(model, 'node_feature_dims', None)
            if actual_node is None:
                print(f"❌ 未找到 node_feature_dims 属性")
                passed = False
            elif actual_node != model_info["expected_node"]:
                print(f"❌ 节点级特征索引不正确!")
                print(f"   期望: {model_info['expected_node']}")
                print(f"   实际: {actual_node}")
                # 检查是否包含维度3或4
                if 3 in actual_node or 4 in actual_node:
                    print(f"   ⚠️  包含了应排除的维度3或4！")
                passed = False
            else:
                print(f"✅ 节点级特征索引正确: {actual_node}")
        
        # 检查图级特征
        if "expected_graph" in model_info and model_info["expected_graph"] is not None:
            actual_graph = getattr(model, 'graph_feature_dims', None)
            if actual_graph is None:
                print(f"❌ 未找到 graph_feature_dims 属性")
                passed = False
            elif actual_graph != model_info["expected_graph"]:
                print(f"❌ 图级特征索引不正确!")
                print(f"   期望: {model_info['expected_graph']}")
                print(f"   实际: {actual_graph}")
                # 检查是否包含维度3或4
                if 3 in actual_graph or 4 in actual_graph:
                    print(f"   ⚠️  包含了应排除的维度3或4！")
                passed = False
            else:
                print(f"✅ 图级特征索引正确: {actual_graph}")
        
        # 检查混合特征（仅Ablation3）
        if "expected_all" in model_info:
            actual_all = getattr(model, 'all_feature_dims', None)
            if actual_all is None:
                print(f"❌ 未找到 all_feature_dims 属性")
                passed = False
            elif actual_all != model_info["expected_all"]:
                print(f"❌ 混合特征索引不正确!")
                print(f"   期望: {model_info['expected_all']}")
                print(f"   实际: {actual_all}")
                # 检查是否包含维度3或4
                if 3 in actual_all or 4 in actual_all:
                    print(f"   ⚠️  包含了应排除的维度3或4！")
                passed = False
            else:
                print(f"✅ 混合特征索引正确: {actual_all}")
        
        # 检查输入维度
        if hasattr(model, 'node_input_dim'):
            node_dim = model.node_input_dim
            if node_dim != 10:
                print(f"❌ 节点输入维度错误: {node_dim}（应为10）")
                passed = False
            else:
                print(f"✅ 节点输入维度正确: {node_dim}")
        
        if hasattr(model, 'graph_input_dim'):
            graph_dim = model.graph_input_dim
            if graph_dim != 8:
                print(f"❌ 图输入维度错误: {graph_dim}（应为8）")
                passed = False
            else:
                print(f"✅ 图输入维度正确: {graph_dim}")
        
        if hasattr(model, 'input_dim') and 'Mixed' in model_info['class']:
            mixed_dim = model.input_dim
            if mixed_dim != 18:
                print(f"❌ 混合输入维度错误: {mixed_dim}（应为18）")
                passed = False
            else:
                print(f"✅ 混合输入维度正确: {mixed_dim}")
        
        if passed:
            print(f"\n✅✅✅ {model_info['class']} 特征维度检查通过！")
        else:
            print(f"\n❌❌❌ {model_info['class']} 特征维度检查失败！")
            all_passed = False
        
    except Exception as e:
        print(f"\n❌❌❌ {model_info['class']} 检查出错!")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

print(f"\n{'='*80}")
if all_passed:
    print("✅✅✅ 所有模型特征维度验证通过！")
    print("所有消融模型都正确使用了18维特征，排除了维度3和4。")
else:
    print("❌❌❌ 部分模型特征维度验证失败！")
    print("请检查上述错误并修复。")
print(f"{'='*80}\n")

sys.exit(0 if all_passed else 1)

