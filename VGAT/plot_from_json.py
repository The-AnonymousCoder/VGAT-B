#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练历史JSON文件重新绘制不同风格的曲线图

使用方法:
    python plot_from_json.py logs/training_history_IMPROVED_20241008_*.json --style sci
    python plot_from_json.py logs/training_history_IMPROVED_20241008_*.json --style colorful
    python plot_from_json.py logs/training_history_IMPROVED_20241008_*.json --style minimal
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def load_training_history(json_file):
    """加载训练历史JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return history


def plot_sci_style(history, output_dir):
    """
    SCI学术论文风格（默认）
    
    特点：
    - Times New Roman字体
    - Colorblind-friendly颜色
    - 300 DPI高分辨率
    - PDF矢量格式
    - 双栏布局（7英寸宽）
    """
    # 设置字体为Times New Roman（学术论文标准）
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 9
    
    # 设置线条样式
    rcParams['lines.linewidth'] = 1.5
    rcParams['axes.linewidth'] = 1.0
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linestyle'] = '--'
    rcParams['grid.linewidth'] = 0.5
    
    # Colorblind-friendly颜色
    colors = {
        'blue': '#0173B2',
        'orange': '#DE8F05',
        'green': '#029E73',
        'red': '#CC78BC',
        'cyan': '#56B4E9',
        'magenta': '#CA9161',
    }
    
    epochs = range(1, len(history['epoch_losses']) + 1)
    
    # 创建3x2子图
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.0))
    
    # (a) 总损失
    ax = axes[0, 0]
    ax.plot(epochs, history['epoch_losses'], color=colors['blue'], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Total Training Loss')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # (b) 对比损失
    ax = axes[0, 1]
    ax.plot(epochs, history['contrastive_losses'], color=colors['orange'], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(b) Contrastive Loss (InfoNCE)')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # (c) 二值化损失
    ax = axes[1, 0]
    ax.plot(epochs, history['binary_consistency_losses'], color=colors['green'], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(c) Binary Consistency Loss')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # (d) 验证NC值
    ax = axes[1, 1]
    if history['val_nc_values']:
        val_epochs = range(3, len(epochs)+1, 3)[:len(history['val_nc_values'])]
        ax.plot(val_epochs, history['val_nc_values'], 
               color=colors['red'], linewidth=1.5, 
               marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)
        ax.set_ylim([0, 1.0])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NC Value')
    ax.set_title('(d) Validation NC Value')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # (e) 学习率
    ax = axes[2, 0]
    ax.plot(epochs, history['learning_rates'], color=colors['cyan'], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('(e) Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # (f) 梯度范数
    ax = axes[2, 1]
    ax.plot(epochs, history['gradient_norms'], color=colors['magenta'], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('(f) Gradient Norm')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # 保存
    png_file = os.path.join(output_dir, 'training_curves_SCI.png')
    pdf_file = os.path.join(output_dir, 'training_curves_SCI.pdf')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ SCI风格图已保存:")
    print(f"   PNG: {png_file}")
    print(f"   PDF: {pdf_file}")


def plot_colorful_style(history, output_dir):
    """
    鲜艳彩色风格（适合PPT演示）
    
    特点：
    - 鲜艳的颜色
    - 较粗的线条
    - 白色背景
    - 适合投影展示
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 13
    rcParams['lines.linewidth'] = 2.5
    
    # 鲜艳颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    epochs = range(1, len(history['epoch_losses']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 总损失
    axes[0, 0].plot(epochs, history['epoch_losses'], color=colors[0], linewidth=2.5)
    axes[0, 0].set_title('Total Loss', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.2)
    
    # 对比损失
    axes[0, 1].plot(epochs, history['contrastive_losses'], color=colors[1], linewidth=2.5)
    axes[0, 1].set_title('Contrastive Loss', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.2)
    
    # 二值化损失
    axes[0, 2].plot(epochs, history['binary_consistency_losses'], color=colors[2], linewidth=2.5)
    axes[0, 2].set_title('Binary Consistency Loss', fontsize=13, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.2)
    
    # 验证NC值
    if history['val_nc_values']:
        val_epochs = range(3, len(epochs)+1, 3)[:len(history['val_nc_values'])]
        axes[1, 0].plot(val_epochs, history['val_nc_values'], 
                       color=colors[3], linewidth=2.5, marker='o', markersize=8)
        axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].set_title('Validation NC Value', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('NC')
    axes[1, 0].grid(True, alpha=0.2)
    
    # 学习率
    axes[1, 1].plot(epochs, history['learning_rates'], color=colors[4], linewidth=2.5)
    axes[1, 1].set_title('Learning Rate', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.2)
    
    # 梯度范数
    axes[1, 2].plot(epochs, history['gradient_norms'], color=colors[5], linewidth=2.5)
    axes[1, 2].set_title('Gradient Norm', fontsize=13, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Grad Norm')
    axes[1, 2].grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    png_file = os.path.join(output_dir, 'training_curves_Colorful.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 彩色风格图已保存: {png_file}")


def plot_minimal_style(history, output_dir):
    """
    简约黑白风格（适合打印）
    
    特点：
    - 黑白灰配色
    - 清晰线条
    - 适合黑白打印
    """
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['lines.linewidth'] = 1.5
    
    epochs = range(1, len(history['epoch_losses']) + 1)
    
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.0))
    
    # 使用不同线型区分
    line_styles = ['-', '--', '-.', ':', '-', '--']
    colors = ['black', 'black', 'black', 'black', 'black', 'black']
    
    # (a) 总损失
    axes[0, 0].plot(epochs, history['epoch_losses'], 
                   color=colors[0], linestyle=line_styles[0], linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('(a) Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) 对比损失
    axes[0, 1].plot(epochs, history['contrastive_losses'], 
                   color=colors[1], linestyle=line_styles[1], linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('(b) Contrastive Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # (c) 二值化损失
    axes[1, 0].plot(epochs, history['binary_consistency_losses'], 
                   color=colors[2], linestyle=line_styles[2], linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('(c) Binary Consistency Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # (d) 验证NC值
    if history['val_nc_values']:
        val_epochs = range(3, len(epochs)+1, 3)[:len(history['val_nc_values'])]
        axes[1, 1].plot(val_epochs, history['val_nc_values'], 
                       color='black', linestyle=line_styles[3], linewidth=1.5,
                       marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)
        axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('NC Value')
    axes[1, 1].set_title('(d) Validation NC')
    axes[1, 1].grid(True, alpha=0.3)
    
    # (e) 学习率
    axes[2, 0].plot(epochs, history['learning_rates'], 
                   color=colors[4], linestyle=line_styles[4], linewidth=1.5)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Learning Rate')
    axes[2, 0].set_title('(e) Learning Rate')
    axes[2, 0].set_yscale('log')
    axes[2, 0].grid(True, alpha=0.3)
    
    # (f) 梯度范数
    axes[2, 1].plot(epochs, history['gradient_norms'], 
                   color=colors[5], linestyle=line_styles[5], linewidth=1.5)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Gradient Norm')
    axes[2, 1].set_title('(f) Gradient Norm')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    png_file = os.path.join(output_dir, 'training_curves_Minimal.png')
    pdf_file = os.path.join(output_dir, 'training_curves_Minimal.pdf')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 简约风格图已保存:")
    print(f"   PNG: {png_file}")
    print(f"   PDF: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description='从JSON文件重新绘制训练曲线')
    parser.add_argument('json_file', type=str, help='训练历史JSON文件路径')
    parser.add_argument('--style', type=str, default='sci', 
                       choices=['sci', 'colorful', 'minimal', 'all'],
                       help='绘图风格: sci(学术论文), colorful(彩色PPT), minimal(简约黑白), all(全部)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认为JSON文件所在目录）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.json_file):
        print(f"❌ 错误: 文件不存在 {args.json_file}")
        return
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = os.path.dirname(args.json_file)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 从JSON加载训练历史: {args.json_file}")
    history = load_training_history(args.json_file)
    
    print(f"   Epochs: {len(history['epoch_losses'])}")
    print(f"   输出目录: {output_dir}")
    print("")
    
    # 根据风格绘图
    if args.style == 'sci' or args.style == 'all':
        print("🎨 绘制SCI学术论文风格...")
        plot_sci_style(history, output_dir)
    
    if args.style == 'colorful' or args.style == 'all':
        print("🎨 绘制彩色PPT风格...")
        plot_colorful_style(history, output_dir)
    
    if args.style == 'minimal' or args.style == 'all':
        print("🎨 绘制简约黑白风格...")
        plot_minimal_style(history, output_dir)
    
    print("")
    print("✅ 完成！")


if __name__ == '__main__':
    main()

