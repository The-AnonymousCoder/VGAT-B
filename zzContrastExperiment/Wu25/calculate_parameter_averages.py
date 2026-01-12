#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算Fig15.py测试结果中每种攻击类型在不同参数下6个矢量地图的平均NC值
"""

import pandas as pd
import numpy as np

def calculate_parameter_averages():
    """
    计算每种攻击类型在不同参数下6个矢量地图的平均NC值
    """
    # 读取详细结果文件
    df = pd.read_csv('Fig15_geometric_attacks_results.csv')
    
    print("Fig15 几何攻击测试 - 3种攻击在不同参数下6个矢量地图的平均NC值")
    print("=" * 80)
    
    attack_types = ['rotation', 'scaling', 'translation']
    all_results = []
    
    for attack_type in attack_types:
        print(f"\n{attack_type.upper()}攻击在不同参数下的平均NC值:")
        print("-" * 60)
        
        attack_data = df[df['attack_type'] == attack_type]
        
        if attack_data.empty:
            print(f"❌ 没有找到{attack_type}攻击的数据")
            continue
        
        # 获取该攻击类型的所有参数值
        parameters = sorted(attack_data['parameter'].unique())
        
        attack_results = []
        
        for param in parameters:
            param_data = attack_data[attack_data['parameter'] == param]
            
            if not param_data.empty:
                # 计算该参数下6个矢量地图的平均NC值
                avg_nc = param_data['nc_value'].mean()
                std_nc = param_data['nc_value'].std()
                min_nc = param_data['nc_value'].min()
                max_nc = param_data['nc_value'].max()
                
                # 参数单位显示
                if attack_type == 'rotation':
                    param_str = f"{param:.0f}°"
                    param_name = "角度"
                elif attack_type == 'scaling':
                    param_str = f"{param:.1f}"
                    param_name = "缩放因子"
                else:  # translation
                    param_str = f"{param:.0f}"
                    param_name = "平移距离"
                
                print(f"  {param_name} {param_str:>6}: 平均NC={avg_nc:.4f}, 标准差={std_nc:.4f}, 范围=[{min_nc:.4f}, {max_nc:.4f}]")
                
                attack_results.append({
                    '攻击类型': attack_type.upper(),
                    '参数值': param,
                    '参数显示': param_str,
                    '6个矢量地图平均NC值': f'{avg_nc:.4f}',
                    '标准差': f'{std_nc:.4f}',
                    '最小NC值': f'{min_nc:.4f}',
                    '最大NC值': f'{max_nc:.4f}',
                    '测试文件数': len(param_data)
                })
        
        all_results.extend(attack_results)
        
        # 计算该攻击类型的整体统计
        if attack_results:
            nc_values = [float(r['6个矢量地图平均NC值']) for r in attack_results]
            overall_avg = np.mean(nc_values)
            overall_std = np.std(nc_values)
            
            print(f"\n  📊 {attack_type.upper()}攻击整体统计:")
            print(f"     所有参数下的总体平均NC值: {overall_avg:.4f}")
            print(f"     参数间NC值标准差: {overall_std:.4f}")
            print(f"     测试参数数: {len(parameters)}")
    
    # 生成详细汇总表格
    print(f"\n" + "=" * 80)
    print("详细参数汇总表")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    # 按攻击类型分组显示
    for attack_type in ['ROTATION', 'SCALING', 'TRANSLATION']:
        attack_results = results_df[results_df['攻击类型'] == attack_type]
        if not attack_results.empty:
            print(f"\n{attack_type}攻击:")
            display_cols = ['参数显示', '6个矢量地图平均NC值', '标准差', '最小NC值', '最大NC值']
            print(attack_results[display_cols].to_string(index=False))
    
    # 保存详细结果
    results_df.to_csv('Fig15_参数级平均NC值汇总.csv', index=False, encoding='utf-8-sig')
    
    # 生成简化汇总
    print(f"\n" + "=" * 80)
    print("各攻击类型参数汇总")
    print("=" * 80)
    
    attack_summary = []
    for attack_type in ['ROTATION', 'SCALING', 'TRANSLATION']:
        attack_results = results_df[results_df['攻击类型'] == attack_type]
        if not attack_results.empty:
            nc_values = [float(r) for r in attack_results['6个矢量地图平均NC值']]
            attack_summary.append({
                '攻击类型': attack_type,
                '测试参数数': len(attack_results),
                '所有参数平均NC值': f'{np.mean(nc_values):.4f}',
                '参数间标准差': f'{np.std(nc_values):.4f}',
                '最小平均NC值': f'{np.min(nc_values):.4f}',
                '最大平均NC值': f'{np.max(nc_values):.4f}'
            })
    
    summary_df = pd.DataFrame(attack_summary)
    print(summary_df.to_string(index=False))
    
    # 保存汇总结果
    summary_df.to_csv('Fig15_攻击类型汇总.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 结果已保存到:")
    print(f"   - Fig15_参数级平均NC值汇总.csv (详细参数结果)")
    print(f"   - Fig15_攻击类型汇总.csv (攻击类型汇总)")
    
    return results_df, summary_df

if __name__ == "__main__":
    calculate_parameter_averages()
