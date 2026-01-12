#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算Fig15.py测试结果中每种攻击类型下6个矢量地图的平均NC值
"""

import pandas as pd
import numpy as np

def calculate_attack_averages():
    """
    从Fig15测试结果中计算每种攻击类型下6个矢量地图的平均NC值
    """
    # 读取详细结果文件
    df = pd.read_csv('Fig15_geometric_attacks_results.csv')
    
    print("Fig15 几何攻击测试 - 6个矢量地图在3种攻击下的平均NC值")
    print("=" * 70)
    
    # 文件列表
    file_names = ['Boundary', 'Road', 'Landuse', 'Railways', 'Building', 'Lake']
    attack_types = ['rotation', 'scaling', 'translation']
    
    # 为每种攻击类型计算平均NC值
    results_summary = []
    
    for attack_type in attack_types:
        print(f"\n{attack_type.upper()}攻击 (6个矢量地图的平均NC值):")
        print("-" * 50)
        
        attack_data = df[df['attack_type'] == attack_type]
        
        if attack_data.empty:
            print(f"❌ 没有找到{attack_type}攻击的数据")
            continue
        
        # 计算每个文件在该攻击类型下的平均NC值
        file_nc_values = []
        
        for file_name in file_names:
            file_data = attack_data[attack_data['file_name'] == file_name]
            if not file_data.empty:
                file_avg_nc = file_data['nc_value'].mean()
                file_nc_values.append(file_avg_nc)
                print(f"  {file_name:>10}: 平均NC = {file_avg_nc:.4f}")
            else:
                print(f"  {file_name:>10}: 无数据")
        
        # 计算所有文件在该攻击类型下的总平均NC值
        if file_nc_values:
            overall_avg_nc = np.mean(file_nc_values)
            std_nc = np.std(file_nc_values)
            min_nc = np.min(file_nc_values)
            max_nc = np.max(file_nc_values)
            
            print(f"\n  📊 {attack_type.upper()}攻击统计:")
            print(f"     总平均NC值: {overall_avg_nc:.4f}")
            print(f"     标准差:     {std_nc:.4f}")
            print(f"     最小NC值:   {min_nc:.4f}")
            print(f"     最大NC值:   {max_nc:.4f}")
            print(f"     测试参数数: {len(attack_data['parameter'].unique())}")
            print(f"     总测试次数: {len(attack_data)}")
            
            results_summary.append({
                '攻击类型': attack_type.upper(),
                '6个矢量地图平均NC值': f'{overall_avg_nc:.4f}',
                '标准差': f'{std_nc:.4f}',
                '最小NC值': f'{min_nc:.4f}',
                '最大NC值': f'{max_nc:.4f}',
                '测试参数数': len(attack_data['parameter'].unique()),
                '总测试次数': len(attack_data)
            })
    
    # 生成汇总表格
    print(f"\n" + "=" * 70)
    print("3种攻击类型下6个矢量地图的平均NC值汇总")
    print("=" * 70)
    
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # 保存汇总结果
    summary_df.to_csv('Fig15_3种攻击平均NC值汇总.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 汇总结果已保存到: Fig15_3种攻击平均NC值汇总.csv")
    
    return summary_df

if __name__ == "__main__":
    calculate_attack_averages()
