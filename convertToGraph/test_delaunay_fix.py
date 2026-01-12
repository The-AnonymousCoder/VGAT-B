#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Delaunay修复方案：验证特征值是否在合理范围内
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

def test_single_graph(pkl_path):
    """测试单个pkl文件的特征值范围"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if not hasattr(data, 'x'):
            return None
        
        features = data.x.numpy() if hasattr(data.x, 'numpy') else data.x
        
        # 检查维度15-17
        dim15 = features[:, 15]
        dim16 = features[:, 16]
        dim17 = features[:, 17]
        
        result = {
            'file': pkl_path.name,
            'shape': features.shape,
            'dim15': {
                'min': float(dim15.min()),
                'max': float(dim15.max()),
                'mean': float(dim15.mean()),
                'has_large': np.any(np.abs(dim15) > 1.5)  # 超过1.5就异常（因为clip到1.0）
            },
            'dim16': {
                'min': float(dim16.min()),
                'max': float(dim16.max()),
                'mean': float(dim16.mean()),
                'has_large': np.any(np.abs(dim16) > 1.5)
            },
            'dim17': {
                'min': float(dim17.min()),
                'max': float(dim17.max()),
                'mean': float(dim17.mean()),
                'has_large': np.any(np.abs(dim17) > 1.5)
            },
            'all_features': {
                'min': float(features.min()),
                'max': float(features.max()),
                'has_nan': np.any(np.isnan(features)),
                'has_inf': np.any(np.isinf(features)),
                'has_very_large': np.any(np.abs(features) > 1e6)
            }
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e), 'file': pkl_path.name}

def main():
    """主函数"""
    print("=" * 80)
    print("测试 Delaunay 修复方案")
    print("=" * 80)
    print()
    
    # 检测修改的代码
    script_path = Path(__file__).parent / "convertToGraph-TrainingSet-IMPROVED.py"
    if script_path.exists():
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查关键修复点
        has_clip = 'np.clip' in content
        has_delaunay_edges_list = 'delaunay_edges_list' in content
        has_delaunay_for_features = 'delaunay_edges_for_features' in content
        
        print("代码修改检查:")
        print(f"  [{'OK' if has_clip else 'NO'}] 添加了 np.clip 限制")
        print(f"  [{'OK' if has_delaunay_edges_list else 'NO'}] 保存了 Delaunay 边列表")
        print(f"  [{'OK' if has_delaunay_for_features else 'NO'}] 使用 Delaunay 边计算特征")
        print()
        
        if not (has_clip and has_delaunay_edges_list and has_delaunay_for_features):
            print("[!] 代码修改不完整！")
            return
    
    # 测试已有的问题图（如果存在）
    test_graphs = [
        "Graph/TrainingSet/Attacked/tianjin-latest-free.shp-gis_osm_pois_a_free_1/shuffle_vertices_graph.pkl",
        "Graph/TrainingSet/Attacked/tianjin-latest-free.shp-gis_osm_traffic_a_free_1/shuffle_vertices_graph.pkl",
    ]
    
    print("测试已知问题图（如果已重新生成）:")
    print("-" * 80)
    
    for graph_path in test_graphs:
        full_path = Path(graph_path)
        if full_path.exists():
            result = test_single_graph(full_path)
            if result and 'error' not in result:
                print(f"\n[*] {result['file']}")
                print(f"    形状: {result['shape']}")
                print(f"    维度15: [{result['dim15']['min']:.2e}, {result['dim15']['max']:.2e}]", end="")
                if result['dim15']['has_large']:
                    print(" [X] 超出范围！")
                else:
                    print(" [OK]")
                
                print(f"    维度16: [{result['dim16']['min']:.2e}, {result['dim16']['max']:.2e}]", end="")
                if result['dim16']['has_large']:
                    print(" [X] 超出范围！")
                else:
                    print(" [OK]")
                
                print(f"    维度17: [{result['dim17']['min']:.2e}, {result['dim17']['max']:.2e}]", end="")
                if result['dim17']['has_large']:
                    print(" [X] 超出范围！")
                else:
                    print(" [OK]")
                
                print(f"    全部特征: [{result['all_features']['min']:.2e}, {result['all_features']['max']:.2e}]")
                
                if result['all_features']['has_nan']:
                    print("    [X] 包含 NaN！")
                if result['all_features']['has_inf']:
                    print("    [X] 包含 Inf！")
                if result['all_features']['has_very_large']:
                    print("    [X] 包含超大值 (>1e6)！")
                    
                if not (result['all_features']['has_nan'] or 
                       result['all_features']['has_inf'] or 
                       result['all_features']['has_very_large']):
                    print("    [OK] 特征值范围正常！")
            else:
                print(f"\n[X] {graph_path}: 读取失败或文件不存在")
        else:
            print(f"\n[>] {graph_path}: 文件不存在（未重新生成）")
    
    print()
    print("=" * 80)
    print("[i] 测试说明:")
    print("=" * 80)
    print("1. 如果文件不存在，说明还没有重新运行特征提取")
    print("2. 重新运行:")
    print("   cd convertToGraph")
    print("   python convertToGraph-TrainingSet-IMPROVED.py")
    print()
    print("3. 预期结果:")
    print("   - 维度15-17的值都应该在 [0, 1] 范围内")
    print("   - 没有 NaN、Inf 或超大值")
    print("   - 所有特征值 < 1e6（最好 < 10）")
    print("=" * 80)

if __name__ == '__main__':
    main()

