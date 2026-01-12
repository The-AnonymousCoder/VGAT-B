#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时脚本：检查图数据的节点数和边数
"""

import pickle
import os
from pathlib import Path

def check_graph_info(graph_name):
    """检查指定图的信息"""
    # 构建路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    graph_dir = project_root / 'convertToGraph' / 'Graph' / 'TrainingSet' / 'Original'
    
    graph_file = graph_dir / f'{graph_name}_graph.pkl'
    
    if not graph_file.exists():
        print(f"❌ 文件不存在: {graph_file}")
        return
    
    try:
        with open(graph_file, 'rb') as f:
            data = pickle.load(f)
        
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        num_features = data.x.shape[1]
        
        print(f"📊 图信息：{graph_name}")
        print(f"{'='*60}")
        print(f"节点数: {num_nodes:,}")
        print(f"边数:   {num_edges:,}")
        print(f"特征维度: {num_features}")
        print(f"平均度数: {num_edges*2/num_nodes:.2f}")
        print(f"{'='*60}")
        
        # 判断是否超大图
        if num_nodes > 30000:
            print(f"⚠️  超大图（>30,000节点）- 训练时会被过滤")
        else:
            print(f"✅ 正常大小图 - 可以训练")
        
        return num_nodes, num_edges
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None, None

def check_all_graphs():
    """检查所有原始图的大小"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    graph_dir = project_root / 'convertToGraph' / 'Graph' / 'TrainingSet' / 'Original'
    
    if not graph_dir.exists():
        print(f"❌ 目录不存在: {graph_dir}")
        return
    
    print("\n🔍 扫描所有原始图...")
    print(f"{'='*80}")
    
    graph_info = []
    
    for graph_file in sorted(graph_dir.glob('*_graph.pkl')):
        graph_name = graph_file.stem.replace('_graph', '')
        
        try:
            with open(graph_file, 'rb') as f:
                data = pickle.load(f)
            
            num_nodes = data.x.shape[0]
            num_edges = data.edge_index.shape[1]
            
            graph_info.append({
                'name': graph_name,
                'nodes': num_nodes,
                'edges': num_edges
            })
            
        except Exception as e:
            print(f"❌ {graph_name}: 读取失败 - {e}")
    
    # 按节点数排序
    graph_info.sort(key=lambda x: x['nodes'], reverse=True)
    
    print(f"\n📊 所有图按节点数排序（共{len(graph_info)}个）：")
    print(f"{'='*80}")
    print(f"{'排名':<5} {'图名':<50} {'节点数':<15} {'边数':<15}")
    print(f"{'-'*80}")
    
    large_graphs = []
    
    for idx, info in enumerate(graph_info, 1):
        status = "⚠️超大" if info['nodes'] > 30000 else "  "
        print(f"{idx:<5} {info['name']:<50} {info['nodes']:>10,}  {status:<5} {info['edges']:>12,}")
        
        if info['nodes'] > 30000:
            large_graphs.append(info)
    
    print(f"{'='*80}")
    print(f"\n📈 统计：")
    print(f"   总图数: {len(graph_info)}")
    print(f"   超大图（>30,000节点）: {len(large_graphs)}")
    print(f"   正常图: {len(graph_info) - len(large_graphs)}")
    
    if large_graphs:
        print(f"\n⚠️  被过滤的超大图：")
        for info in large_graphs:
            print(f"   - {info['name']}: {info['nodes']:,} 节点, {info['edges']:,} 边")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 检查指定的图
        graph_name = sys.argv[1]
        check_graph_info(graph_name)
    else:
        # 检查所有图
        check_all_graphs()
