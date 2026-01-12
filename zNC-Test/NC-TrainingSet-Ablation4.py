#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集NC测试 - 消融实验4（单注意力头）
便捷启动脚本
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, 'NC-TrainingSet-Ablation.py')
    subprocess.run([sys.executable, main_script, '--ablation', '4'])

