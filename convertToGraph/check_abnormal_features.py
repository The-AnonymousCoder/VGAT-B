#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查图数据中的异常特征值
用于诊断训练时出现的NaN问题
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_graph_file(pkl_path):
    """
    检查单个pkl文件的特征值
    
    Returns:
        dict: 包含统计信息的字典，如果有异常则返回详细信息
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if not hasattr(data, 'x'):
            return {'error': 'No feature matrix (x)'}
        
        features = data.x
        
        # 统计信息
        min_val = features.min().item()
        max_val = features.max().item()
        mean_val = features.mean().item()
        std_val = features.std().item()
        
        # 检查异常值
        has_nan = torch.isnan(features).any().item()
        has_inf = torch.isinf(features).any().item()
        num_large = (features.abs() > 1e6).sum().item()
        num_very_large = (features.abs() > 1e9).sum().item()
        
        # 如果有异常，记录详细信息
        is_abnormal = has_nan or has_inf or num_large > 0
        
        result = {
            'file': str(pkl_path),
            'shape': tuple(features.shape),
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'num_large_values': num_large,
            'num_very_large_values': num_very_large,
            'is_abnormal': is_abnormal
        }
        
        # 如果有超大值，记录每个维度的最大值
        if num_large > 0:
            max_per_dim = features.abs().max(dim=0).values
            result['max_per_dimension'] = max_per_dim.tolist()
            
            # 找出哪些维度有问题
            abnormal_dims = []
            for i, val in enumerate(max_per_dim):
                if val > 1e6:
                    abnormal_dims.append((i, val.item()))
            result['abnormal_dimensions'] = abnormal_dims
        
        return result
        
    except Exception as e:
        return {'error': str(e), 'file': str(pkl_path)}

def scan_directory(directory, pattern='**/*.pkl'):
    """
    扫描目录中的所有pkl文件
    
    Args:
        directory: 要扫描的目录
        pattern: 文件匹配模式
    """
    directory = Path(directory)
    
    print(f"扫描目录: {directory}")
    print(f"匹配模式: {pattern}")
    print("=" * 80)
    
    # 查找所有pkl文件
    pkl_files = list(directory.glob(pattern))
    print(f"找到 {len(pkl_files)} 个pkl文件")
    print()
    
    if len(pkl_files) == 0:
        print("没有找到pkl文件！")
        return
    
    # 检查所有文件
    abnormal_files = []
    error_files = []
    
    print("开始检查特征值...")
    for pkl_file in tqdm(pkl_files, desc="检查进度"):
        result = check_graph_file(pkl_file)
        
        if 'error' in result:
            error_files.append(result)
        elif result['is_abnormal']:
            abnormal_files.append(result)
    
    # 输出结果
    print()
    print("=" * 80)
    print("检查完成！")
    print("=" * 80)
    print(f"总文件数: {len(pkl_files)}")
    print(f"正常文件: {len(pkl_files) - len(abnormal_files) - len(error_files)}")
    print(f"异常文件: {len(abnormal_files)}")
    print(f"错误文件: {len(error_files)}")
    print()
    
    # 详细报告异常文件
    if abnormal_files:
        print("=" * 80)
        print("[!] 异常文件详情:")
        print("=" * 80)
        
        for i, result in enumerate(abnormal_files, 1):
            print(f"\n[{i}] {Path(result['file']).name}")
            print(f"    路径: {result['file']}")
            print(f"    形状: {result['shape']}")
            print(f"    范围: [{result['min']:.2e}, {result['max']:.2e}]")
            print(f"    均值: {result['mean']:.2e}, 标准差: {result['std']:.2e}")
            
            if result['has_nan']:
                print(f"    [!] 包含 NaN 值")
            if result['has_inf']:
                print(f"    [!] 包含 Inf 值")
            if result['num_large_values'] > 0:
                print(f"    [!] 包含 {result['num_large_values']} 个超大值 (>1e6)")
            if result['num_very_large_values'] > 0:
                print(f"    [!] 包含 {result['num_very_large_values']} 个极大值 (>1e9)")
            
            # 显示异常维度
            if 'abnormal_dimensions' in result and result['abnormal_dimensions']:
                print(f"    [?] 异常维度:")
                for dim, val in result['abnormal_dimensions'][:5]:  # 只显示前5个
                    print(f"       维度 {dim}: {val:.2e}")
                if len(result['abnormal_dimensions']) > 5:
                    print(f"       ... 还有 {len(result['abnormal_dimensions']) - 5} 个异常维度")
    
    # 报告错误文件
    if error_files:
        print()
        print("=" * 80)
        print("[X] 读取错误的文件:")
        print("=" * 80)
        for result in error_files:
            print(f"  - {Path(result['file']).name}: {result['error']}")
    
    # 统计建议
    print()
    print("=" * 80)
    print("[*] 诊断建议:")
    print("=" * 80)
    
    if len(abnormal_files) == 0:
        print("[OK] 所有图数据的特征值都在正常范围内！")
        print("     问题可能出在训练过程中的特征组合或模型计算。")
    else:
        print(f"[!] 发现 {len(abnormal_files)} 个异常文件")
        print()
        print("建议操作：")
        print("1. 检查这些文件对应的原始GeoJSON数据")
        print("2. 重新运行特征提取，添加数值范围检查")
        print("3. 在convertToGraph中添加特征值裁剪：")
        print("   features = np.clip(features, -1e6, 1e6)")
        print("4. 或在训练时过滤这些文件")
    
    return abnormal_files

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查图数据中的异常特征值')
    parser.add_argument('--dir', type=str, 
                       default='Graph/TrainingSet',
                       help='要检查的目录 (默认: Graph/TrainingSet)')
    parser.add_argument('--pattern', type=str,
                       default='**/*.pkl',
                       help='文件匹配模式 (默认: **/*.pkl)')
    parser.add_argument('--original-only', action='store_true',
                       help='只检查原始图 (Original目录)')
    parser.add_argument('--attacked-only', action='store_true',
                       help='只检查攻击图 (Attacked目录)')
    parser.add_argument('--specific-graph', type=str,
                       help='只检查特定图名称的文件')
    
    args = parser.parse_args()
    
    # 确定检查目录
    base_dir = Path(args.dir)
    
    if args.original_only:
        check_dir = base_dir / 'Original'
        print("只检查原始图")
    elif args.attacked_only:
        check_dir = base_dir / 'Attacked'
        print("只检查攻击图")
    elif args.specific_graph:
        check_dir = base_dir / 'Attacked' / args.specific_graph
        print(f"只检查特定图: {args.specific_graph}")
    else:
        check_dir = base_dir
        print("检查所有图（原始+攻击）")
    
    if not check_dir.exists():
        print(f"错误: 目录不存在: {check_dir}")
        return
    
    # 执行扫描
    abnormal_files = scan_directory(check_dir, args.pattern)
    
    # 保存异常文件列表
    if abnormal_files:
        output_file = 'abnormal_files.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("异常特征值文件列表\n")
            f.write("=" * 80 + "\n")
            for result in abnormal_files:
                f.write(f"{result['file']}\n")
                f.write(f"  范围: [{result['min']:.2e}, {result['max']:.2e}]\n")
                if 'abnormal_dimensions' in result:
                    f.write(f"  异常维度: {result['abnormal_dimensions']}\n")
                f.write("\n")
        
        print()
        print(f"异常文件列表已保存到: {output_file}")

if __name__ == '__main__':
    main()

