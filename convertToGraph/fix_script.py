#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复TestSet脚本的工具
"""

import shutil
import os

def main():
    print("开始修复TestSet脚本...")

    # 备份旧文件
    if os.path.exists('convertToGraph-TestSet-IMPROVED.py'):
        shutil.copy2('convertToGraph-TestSet-IMPROVED.py', 'convertToGraph-TestSet-IMPROVED.py.backup')
        print("已备份旧文件")

    # 读取TrainingSet文件作为模板
    with open('convertToGraph-TrainingSet-IMPROVED.py', 'r', encoding='utf-8') as f:
        training_content = f.read()

    # 基本替换
    test_content = training_content.replace(
        'class ImprovedTrainSetVectorToGraphConverter:',
        'class ImprovedTestSetVectorToGraphConverter:'
    ).replace(
        '训练集',
        '测试集'
    ).replace(
        'TrainSet',
        'TestSet'
    ).replace(
        'TrainingSet',
        'TestSet'
    ).replace(
        'ImprovedTrainSetVectorToGraphConverter(',
        'ImprovedTestSetVectorToGraphConverter('
    ).replace(
        'convert_train_set_to_graph',
        'convert_test_set_to_graph'
    )

    # 替换构造函数参数
    old_init = '''    def __init__(self, vector_dir="../convertToGeoJson/GeoJson/TrainingSet",
                 attacked_dir="../convertToGeoJson-Attacked/GeoJson-Attacked/TrainingSet",
                 graph_dir="Graph/TrainingSet",
                 batch_size=100,
                 max_workers=None,
                 use_cache=True):
        self.vector_dir = vector_dir
        self.attacked_dir = attacked_dir
        self.graph_dir = graph_dir
        self.batch_size = batch_size
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.use_cache = use_cache'''

    new_init = '''    def __init__(self, original_dir="../convertToGeoJson/GeoJson/TestSet",
                 attacked_dir="../convertToGeoJson-Attacked/GeoJson-Attacked/TestSet",
                 graph_dir="Graph/TestSet"):
        self.original_dir = original_dir
        self.attacked_dir = attacked_dir
        self.graph_dir = graph_dir'''

    test_content = test_content.replace(old_init, new_init)

    # 移除TrainingSet特有的初始化代码
    lines = test_content.split('\n')
    new_lines = []
    skip_line = False

    for line in lines:
        # 跳过TrainingSet特有的初始化
        if 'self.batch_size = batch_size' in line:
            continue
        elif 'self.max_workers = max_workers or min(8, mp.cpu_count())' in line:
            continue
        elif 'self.use_cache = use_cache' in line:
            continue
        elif 'self.cache_dir =' in line:
            continue
        elif 'self.features_cache_file =' in line:
            continue
        elif 'self.file_hashes_file =' in line:
            continue
        elif 'os.path.join(self.cache_dir,' in line:
            continue

        # 跳过batch processing相关的方法
        if 'def partition_by_hilbert' in line:
            skip_line = True
            continue
        elif skip_line and line.strip().startswith('def '):
            skip_line = False

        if not skip_line:
            new_lines.append(line)

    test_content = '\n'.join(new_lines)

    # 移除不需要的导入
    imports_to_remove = [
        'import multiprocessing as mp',
        'from concurrent.futures import ProcessPoolExecutor, as_completed',
        'import psutil',
        'import gc',
        'import time',
        'import hashlib',
        'import json'
    ]

    for imp in imports_to_remove:
        test_content = test_content.replace(imp + '\n', '')

    # 修改构造函数体
    test_content = test_content.replace(
        '''        self.ensure_graph_dir()

        # 使用全局标准化器
        self.global_scaler = StandardScaler()
        self.scaler_fitted = False

        # 存储全局统计量（用于计算相对位置）
        self.global_bounds = None
        self.global_centroid = None

        # 缓存相关
        self.cache_dir = os.path.join(self.graph_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # ⭐ 方案A：仅用原始图生成scaler，缓存文件名区分
        self.features_cache_file = os.path.join(self.cache_dir, "features_cache_original_only.pkl")
        self.file_hashes_file = os.path.join(self.cache_dir, "file_hashes_original_only.json")''',
        '''        self.ensure_graph_dir()

        # ⭐ 全局标准化器（两遍处理）
        self.global_scaler = StandardScaler()
        self.scaler_fitted = False'''
    )

    # 移除TrainingSet特有的main函数参数
    test_content = test_content.replace(
        '    converter = ImprovedTrainSetVectorToGraphConverter(\n        batch_size=25,      # ⭐降低为25（原50），减少内存占用\n        max_workers=4,      # 可根据CPU核心数调整 (默认8)\n        use_cache=True      # 启用缓存加速\n    )',
        '    converter = ImprovedTestSetVectorToGraphConverter()'
    )

    test_content = test_content.replace(
        '''    # ⭐ 增量更新模式（默认启用）
    # 如需完全重新生成，设置 incremental_mode=False
    incremental_mode = True  # True: 增量更新 | False: 完全重新生成

    print(f"🔧 配置参数:")
    print(f"   - 批次大小: {converter.batch_size}")
    print(f"   - 最大工作进程: {converter.max_workers}")
    print(f"   - 缓存启用: {converter.use_cache}")
    print(f"   - 增量更新: {'启用 🔄' if incremental_mode else '禁用 🔥'}")
    print()

    # 转换训练集数据
    try:
        converter.convert_train_set_to_graph(incremental_mode=incremental_mode)''',
        '''    # 转换测试集数据（两遍处理）
    converter.convert_test_set_to_graph()'''
    )

    # 写回文件
    with open('convertToGraph-TestSet-IMPROVED.py', 'w', encoding='utf-8') as f:
        f.write(test_content)

    print("TestSet脚本已重新创建完成！")

if __name__ == '__main__':
    main()






















