#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查特定图的特征，诊断NaN问题
"""

import pickle
import numpy as np
import os

def check_graph_features(graph_path):
    """检查图数据的各项特征统计"""
    print(f"\n{'='*70}")
    print(f"检查图: {os.path.basename(graph_path)}")
    print(f"{'='*70}")
    
    try:
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        print(f"\n📊 基本信息:")
        print(f"   节点数: {graph_data.x.shape[0]:,}")
        print(f"   边数: {graph_data.edge_index.shape[1]:,}")
        print(f"   特征维度: {graph_data.x.shape[1]}")
        
        print(f"\n📈 节点特征统计:")
        x = graph_data.x.numpy() if hasattr(graph_data.x, 'numpy') else graph_data.x
        
        print(f"   形状: {x.shape}")
        print(f"   数据类型: {x.dtype}")
        print(f"   最小值: {np.min(x):.6f}")
        print(f"   最大值: {np.max(x):.6f}")
        print(f"   均值: {np.mean(x):.6f}")
        print(f"   标准差: {np.std(x):.6f}")
        print(f"   中位数: {np.median(x):.6f}")
        
        # 检查异常值
        print(f"\n🔍 异常值检查:")
        nan_count = np.isnan(x).sum()
        inf_count = np.isinf(x).sum()
        zero_count = (x == 0).sum()
        total_elements = x.size
        
        print(f"   NaN数量: {nan_count} / {total_elements} ({100*nan_count/total_elements:.2f}%)")
        print(f"   Inf数量: {inf_count} / {total_elements} ({100*inf_count/total_elements:.2f}%)")
        print(f"   零值数量: {zero_count} / {total_elements} ({100*zero_count/total_elements:.2f}%)")
        
        if nan_count > 0:
            print(f"   ⚠️ 警告: 发现NaN值!")
        if inf_count > 0:
            print(f"   ⚠️ 警告: 发现Inf值!")
        
        # 检查每个特征维度
        print(f"\n📊 各特征维度统计:")
        for i in range(x.shape[1]):
            feat = x[:, i]
            print(f"   维度 {i:2d}: min={np.min(feat):8.4f}, max={np.max(feat):8.4f}, "
                  f"mean={np.mean(feat):8.4f}, std={np.std(feat):8.4f}")
            
            # 检查是否有极端值
            if np.max(np.abs(feat)) > 100:
                print(f"           ⚠️ 警告: 特征值过大 (|max|={np.max(np.abs(feat)):.2f})")
            if np.std(feat) < 1e-6:
                print(f"           ⚠️ 警告: 标准差过小 (可能是常量特征)")
        
        # 检查边索引
        print(f"\n📊 边索引统计:")
        edge_index = graph_data.edge_index.numpy() if hasattr(graph_data.edge_index, 'numpy') else graph_data.edge_index
        print(f"   形状: {edge_index.shape}")
        print(f"   最小索引: {np.min(edge_index)}")
        print(f"   最大索引: {np.max(edge_index)}")
        
        # 检查是否有自环
        self_loops = np.sum(edge_index[0] == edge_index[1])
        print(f"   自环数量: {self_loops}")
        
        # 检查度分布
        degrees = np.bincount(edge_index[0])
        print(f"   最小度: {np.min(degrees)}")
        print(f"   最大度: {np.max(degrees)}")
        print(f"   平均度: {np.mean(degrees):.2f}")
        
        if np.max(degrees) > 1000:
            print(f"   ⚠️ 警告: 存在超高度节点 (max_degree={np.max(degrees)})")
        
        # 特征相关性检查
        print(f"\n📊 特征相关性分析:")
        corr_matrix = np.corrcoef(x.T)
        max_corr = np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        print(f"   最大特征间相关系数: {max_corr:.4f}")
        
        if max_corr > 0.95:
            print(f"   ⚠️ 警告: 特征高度相关 (可能冗余)")
        
        # 潜在问题总结
        print(f"\n🎯 潜在问题总结:")
        issues = []
        
        if nan_count > 0:
            issues.append(f"✗ 包含{nan_count}个NaN值")
        if inf_count > 0:
            issues.append(f"✗ 包含{inf_count}个Inf值")
        if np.max(np.abs(x)) > 100:
            issues.append(f"✗ 特征值过大 (max={np.max(np.abs(x)):.2f})")
        if np.max(degrees) > 1000:
            issues.append(f"✗ 超高度节点 (max_degree={np.max(degrees)})")
        if zero_count / total_elements > 0.5:
            issues.append(f"✗ 稀疏特征 (零值占比{100*zero_count/total_elements:.1f}%)")
        
        if issues:
            for issue in issues:
                print(f"   {issue}")
            print(f"\n   💡 建议: 该图可能需要加入黑名单或进行特征预处理")
        else:
            print(f"   ✓ 未发现明显问题")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查问题图
    problem_graph = r"e:\Project\VGAT-ZeroWatermark-V5\convertToGraph\Graph\TrainingSet\Original\tianjin-latest-free.shp-gis_osm_transport_free_1_graph.pkl"
    
    if os.path.exists(problem_graph):
        check_graph_features(problem_graph)
    else:
        print(f"文件不存在: {problem_graph}")
    
    # 对比检查一个正常的图
    print(f"\n\n" + "="*70)
    print("对比：检查一个正常的图")
    print("="*70)
    
    normal_graphs = [
        r"e:\Project\VGAT-ZeroWatermark-V5\convertToGraph\Graph\TrainingSet\Original\H51-RESA_graph.pkl",
        r"e:\Project\VGAT-ZeroWatermark-V5\convertToGraph\Graph\TrainingSet\Original\H51-RESP_graph.pkl",
    ]
    
    for normal_graph in normal_graphs:
        if os.path.exists(normal_graph):
            check_graph_features(normal_graph)
            break
