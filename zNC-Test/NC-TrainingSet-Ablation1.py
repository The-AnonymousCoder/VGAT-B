#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集NC测试 - 消融实验1（仅节点级特征）
便捷启动脚本
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # 获取通用脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, 'NC-TrainingSet-Ablation.py')
    
    # 调用通用脚本，传入ablation参数
    subprocess.run([sys.executable, main_script, '--ablation', '1'])

