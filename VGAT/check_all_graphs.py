#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量检查所有训练集图的特征，诊断潜在问题
"""

import pickle
import numpy as np
import os
import glob
from collections import defaultdict

def check_graph_features(graph_path, verbose=False):
    """检查图数据的各项特征统计"""
    try:
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        x = graph_data.x.numpy() if hasattr(graph_data.x, 'numpy') else graph_data.x
        edge_index = graph_data.edge_index.numpy() if hasattr(graph_data.edge_index, 'numpy') else graph_data.edge_index
        
        # 基本统计
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        num_features = x.shape[1]
        
        # 异常值检查
        nan_count = np.isnan(x).sum()
        inf_count = np.isinf(x).sum()
        zero_count = (x == 0).sum()
        total_elements = x.size
        
        # 检查常量特征
        constant_dims = []
        for i in range(x.shape[1]):
            feat = x[:, i]
            if np.std(feat) < 1e-6:
                constant_dims.append(i)
        
        constant_ratio = len(constant_dims) / num_features
        
        # 检查极端值
        max_abs_value = np.max(np.abs(x))
        
        # 检查度分布
        degrees = np.bincount(edge_index[0])
        max_degree = np.max(degrees)
        
        result = {
            'path': graph_path,
            'name': os.path.basename(graph_path).replace('_graph.pkl', ''),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_features': num_features,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'zero_ratio': zero_count / total_elements,
            'constant_dims': constant_dims,
            'constant_ratio': constant_ratio,
            'max_abs_value': max_abs_value,
            'max_degree': max_degree,
            'mean': np.mean(x),
            'std': np.std(x),
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"图: {result['name']}")
            print(f"{'='*70}")
            print(f"📊 基本信息: 节点={num_nodes:,}, 边={num_edges:,}, 特征维度={num_features}")
            print(f"🔍 异常检查: NaN={nan_count}, Inf={inf_count}, 零值占比={result['zero_ratio']:.2%}")
            print(f"📈 常量特征: {len(constant_dims)}/{num_features} ({constant_ratio:.1%})")
            if constant_dims:
                print(f"   常量维度: {constant_dims}")
            print(f"📊 数值范围: max_abs={max_abs_value:.2f}, mean={result['mean']:.4f}, std={result['std']:.4f}")
            print(f"📊 度统计: max_degree={max_degree}")
        
        return result
        
    except Exception as e:
        print(f"❌ 错误处理 {os.path.basename(graph_path)}: {e}")
        return None

def main():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 查找所有原始图
    original_dir = r"e:\Project\VGAT-ZeroWatermark-V5\convertToGraph\Graph\TrainingSet\Original"
    graph_files = glob.glob(os.path.join(original_dir, "*_graph.pkl"))
    
    print(f"\n{'='*70}")
    print(f"批量检查训练集图 - 共 {len(graph_files)} 个原始图")
    print(f"{'='*70}\n")
    
    results = []
    problem_graphs = []
    
    for graph_file in sorted(graph_files):
        result = check_graph_features(graph_file, verbose=False)
        if result:
            results.append(result)
            
            # 标记问题图
            issues = []
            if result['nan_count'] > 0:
                issues.append(f"NaN={result['nan_count']}")
            if result['inf_count'] > 0:
                issues.append(f"Inf={result['inf_count']}")
            if result['constant_ratio'] > 0.4:
                issues.append(f"常量特征={result['constant_ratio']:.1%}")
            if result['max_abs_value'] > 100:
                issues.append(f"极端值={result['max_abs_value']:.1f}")
            if result['max_degree'] > 1000:
                issues.append(f"超高度={result['max_degree']}")
            
            if issues:
                problem_graphs.append({
                    'name': result['name'],
                    'issues': issues,
                    'constant_ratio': result['constant_ratio'],
                    'num_nodes': result['num_nodes'],
                    'num_edges': result['num_edges'],
                })
    
    # 生成报告
    print(f"\n{'='*70}")
    print(f"统计摘要")
    print(f"{'='*70}\n")
    
    print(f"✅ 检查完成: {len(results)} 个图")
    print(f"⚠️  问题图数量: {len(problem_graphs)} 个\n")
    
    if results:
        # 节点数统计
        node_counts = [r['num_nodes'] for r in results]
        print(f"📊 节点数统计:")
        print(f"   最小: {min(node_counts):,}")
        print(f"   最大: {max(node_counts):,}")
        print(f"   平均: {np.mean(node_counts):,.0f}")
        print(f"   中位数: {np.median(node_counts):,.0f}\n")
        
        # 常量特征统计
        constant_ratios = [r['constant_ratio'] for r in results]
        print(f"📊 常量特征比例统计:")
        print(f"   最小: {min(constant_ratios):.1%}")
        print(f"   最大: {max(constant_ratios):.1%}")
        print(f"   平均: {np.mean(constant_ratios):.1%}")
        print(f"   中位数: {np.median(constant_ratios):.1%}\n")
    
    # 输出问题图列表
    if problem_graphs:
        print(f"{'='*70}")
        print(f"⚠️  问题图详细列表 (按常量特征比例排序)")
        print(f"{'='*70}\n")
        
        # 按常量特征比例排序
        problem_graphs.sort(key=lambda x: x['constant_ratio'], reverse=True)
        
        for i, pg in enumerate(problem_graphs, 1):
            print(f"{i}. {pg['name']}")
            print(f"   节点数: {pg['num_nodes']:,}, 边数: {pg['num_edges']:,}")
            print(f"   问题: {', '.join(pg['issues'])}")
            print()
        
        # 生成建议的黑名单
        print(f"{'='*70}")
        print(f"💡 建议加入黑名单的图 (常量特征>40%)")
        print(f"{'='*70}\n")
        
        blacklist = [pg['name'] for pg in problem_graphs if pg['constant_ratio'] > 0.4]
        
        if blacklist:
            print("GRAPH_BLACKLIST = [")
            for name in sorted(blacklist):
                print(f"    '{name}',")
            print("]\n")
            print(f"共 {len(blacklist)} 个图建议加入黑名单")
        else:
            print("✓ 没有图需要加入黑名单（所有图常量特征<40%）")
    else:
        print("✓ 所有图都正常，未发现问题")
    
    # 详细输出常量特征>30%的图
    high_constant_graphs = [r for r in results if r['constant_ratio'] > 0.3]
    if high_constant_graphs:
        print(f"\n{'='*70}")
        print(f"🔍 常量特征>30%的图详细信息")
        print(f"{'='*70}")
        
        for result in sorted(high_constant_graphs, key=lambda x: x['constant_ratio'], reverse=True):
            check_graph_features(result['path'], verbose=True)

if __name__ == "__main__":
    main()
