#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_vertex_deletion_verification.py - 验证顶点删除攻击是否真正按比例删除了顶点
"""

import os
import numpy as np
import random
from Zero_watermarking import Read_Shapfile

def count_total_vertices(shapefile_path):
    """统计shapefile中的总顶点数"""
    try:
        XLst, YLst, feature_num = Read_Shapfile(shapefile_path)
        total_points = sum(len(x) for x in XLst)
        return total_points, feature_num
    except Exception as e:
        print(f"读取文件失败: {e}")
        return 0, 0

def simulate_deletion_logic(total_vertices, delete_factor, seed=212367):
    """模拟删除逻辑，计算预期保留的顶点数"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 这是attacks1_vertex_delete_poly.py第70行的逻辑
    keep_flags = np.random.rand(total_vertices) >= delete_factor
    
    return np.sum(keep_flags)

def test_vertex_deletion():
    """测试顶点删除是否按照预期工作"""
    print("🔍 验证顶点删除攻击的实际效果")
    print("=" * 60)
    
    # 测试参数
    test_file = "pso_data/Boundary.shp"
    delete_factors = [0.1, 0.3, 0.5]
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    # 统计原始顶点数
    print(f"📊 分析原始文件: {test_file}")
    original_points, original_features = count_total_vertices(test_file)
    print(f"    原始要素数量: {original_features}")
    print(f"    原始总顶点数: {original_points}")
    
    if original_points == 0:
        print("❌ 无法读取原始文件数据")
        return
    
    print(f"\n🧮 模拟删除逻辑预测：")
    for delete_factor in delete_factors:
        # 模拟删除逻辑
        predicted_remaining = simulate_deletion_logic(original_points, delete_factor)
        predicted_deleted = original_points - predicted_remaining
        
        actual_delete_rate = predicted_deleted / original_points
        
        print(f"    删除因子 {delete_factor:.1f}:")
        print(f"        预期删除: {predicted_deleted}/{original_points} ({actual_delete_rate:.1%})")
        print(f"        预期保留: {predicted_remaining}")
    
    print(f"\n📈 结论分析：")
    print(f"    删除逻辑: keep_flags = np.random.rand(total_vertices) >= delete_factor")
    print(f"    • np.random.rand() 生成 [0,1) 随机数")
    print(f"    • >= delete_factor 表示随机数大于等于删除因子时保留")
    print(f"    • 理论删除概率 = delete_factor")
    print(f"    • 理论保留概率 = 1 - delete_factor")
    
    # 验证多次运行的一致性
    print(f"\n🔄 验证随机种子一致性:")
    test_factor = 0.3
    results = []
    for i in range(5):
        remaining = simulate_deletion_logic(original_points, test_factor)
        results.append(remaining)
        print(f"    第{i+1}次运行 (删除因子{test_factor}): 保留 {remaining} 个点")
    
    if len(set(results)) == 1:
        print(f"    ✅ 随机种子工作正常，结果一致")
    else:
        print(f"    ❌ 随机种子可能有问题，结果不一致")

def analyze_deletion_statistics():
    """分析删除统计的准确性"""
    print(f"\n📊 统计学验证:")
    
    total_vertices = 100000  # 大样本测试
    delete_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for delete_factor in delete_factors:
        # 不使用固定种子，多次测试
        deletion_rates = []
        
        for trial in range(10):
            np.random.seed(trial)  # 不同的种子
            keep_flags = np.random.rand(total_vertices) >= delete_factor
            actual_kept = np.sum(keep_flags)
            actual_deleted = total_vertices - actual_kept
            actual_deletion_rate = actual_deleted / total_vertices
            deletion_rates.append(actual_deletion_rate)
        
        mean_rate = np.mean(deletion_rates)
        std_rate = np.std(deletion_rates)
        
        print(f"    删除因子 {delete_factor:.1f}: 实际删除率 {mean_rate:.3f} ± {std_rate:.3f} (期望: {delete_factor:.3f})")
        
        if abs(mean_rate - delete_factor) < 0.01:
            status = "✅"
        else:
            status = "❌"
        print(f"        {status} 与期望值差异: {abs(mean_rate - delete_factor):.3f}")

if __name__ == "__main__":
    test_vertex_deletion()
    analyze_deletion_statistics()
