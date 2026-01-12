#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融试验结果对比脚本

功能：
1. 从日志文件中提取各消融试验的训练指标
2. 生成对比表格
3. 绘制对比图表

使用方法：
    python compare_ablation_results.py
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class AblationResultsCollector:
    """消融试验结果收集器"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.results = {}
        
    def extract_metrics_from_log(self, log_file):
        """从日志文件中提取关键指标"""
        metrics = {
            'model_name': 'Unknown',
            'total_params': 0,
            'best_epoch': 0,
            'best_total_loss': float('inf'),
            'best_val_nc': 0.0,
            'final_train_time': 0.0,
            'loss_history': [],
            'nc_history': []
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 提取模型名称
                if 'Ablation1' in log_file or '节点级' in content:
                    metrics['model_name'] = '消融1-节点级单流'
                elif 'Ablation2' in log_file or '图级' in content:
                    metrics['model_name'] = '消融2-图级单流'
                elif 'Ablation3' in log_file or '混合' in content:
                    metrics['model_name'] = '消融3-混合单流'
                elif 'Ablation4' in log_file or '单头' in content:
                    metrics['model_name'] = '消融4-单头注意力'
                elif 'Ablation5' in log_file or 'GCN' in content:
                    metrics['model_name'] = '消融5-GCN'
                elif 'IMPROVED' in log_file:
                    metrics['model_name'] = '完整模型（基线）'
                
                # 提取参数数量
                param_match = re.search(r'总参数数量[:\s]+([0-9,]+)', content)
                if param_match:
                    params_str = param_match.group(1).replace(',', '')
                    metrics['total_params'] = int(params_str)
                
                # 提取最佳epoch信息
                best_epoch_match = re.search(r'Epoch\s+(\d+)/\d+.*Total Loss:\s+([\d.]+).*Val NC:\s+([\d.]+)', content)
                if best_epoch_match:
                    metrics['best_epoch'] = int(best_epoch_match.group(1))
                    metrics['best_total_loss'] = float(best_epoch_match.group(2))
                    metrics['best_val_nc'] = float(best_epoch_match.group(3))
                
                # 提取训练时间
                time_match = re.search(r'训练总时间[:\s]+([\d.]+).*?(?:分钟|min)', content)
                if time_match:
                    metrics['final_train_time'] = float(time_match.group(1))
                
                # 提取训练曲线数据
                epoch_pattern = r'Epoch\s+\d+/\d+.*?Total Loss:\s+([\d.]+).*?Val NC:\s+([\d.]+)'
                for match in re.finditer(epoch_pattern, content):
                    loss = float(match.group(1))
                    nc = float(match.group(2))
                    metrics['loss_history'].append(loss)
                    metrics['nc_history'].append(nc)
        
        except Exception as e:
            print(f"警告：无法解析日志文件 {log_file}: {e}")
        
        return metrics
    
    def collect_all_results(self):
        """收集所有消融试验的结果"""
        if not os.path.exists(self.log_dir):
            print(f"日志目录不存在: {self.log_dir}")
            return
        
        # 查找所有相关日志文件
        log_patterns = [
            '*IMPROVED_latest.log',
            '*Ablation1*.log',
            '*Ablation2*.log',
            '*Ablation3*.log',
            '*Ablation4*.log',
            '*Ablation5*.log'
        ]
        
        import glob
        for pattern in log_patterns:
            log_files = glob.glob(os.path.join(self.log_dir, pattern))
            if log_files:
                # 使用最新的日志文件
                latest_log = max(log_files, key=os.path.getmtime)
                print(f"正在解析: {latest_log}")
                metrics = self.extract_metrics_from_log(latest_log)
                self.results[metrics['model_name']] = metrics
        
        return self.results
    
    def generate_comparison_table(self):
        """生成对比表格"""
        print("\n" + "="*100)
        print("消融试验结果对比表")
        print("="*100)
        
        # 表头
        header = f"{'模型名称':<20} {'参数量':<12} {'最佳Epoch':<10} {'总损失':<10} {'验证NC':<10} {'训练时间(min)':<15}"
        print(header)
        print("-"*100)
        
        # 按预定义顺序排序
        model_order = [
            '完整模型（基线）',
            '消融1-节点级单流',
            '消融2-图级单流',
            '消融3-混合单流',
            '消融4-单头注意力',
            '消融5-GCN'
        ]
        
        for model_name in model_order:
            if model_name in self.results:
                m = self.results[model_name]
                params_str = f"{m['total_params']:,}" if m['total_params'] > 0 else "N/A"
                row = f"{model_name:<20} {params_str:<12} {m['best_epoch']:<10} " \
                      f"{m['best_total_loss']:<10.4f} {m['best_val_nc']:<10.4f} " \
                      f"{m['final_train_time']:<15.2f}"
                print(row)
        
        print("="*100)
        print("\n注：验证NC值为训练过程中的验证集NC，最终测试集NC需要运行零水印验证脚本获得")
    
    def plot_training_curves(self, save_path="ablation_comparison.png"):
        """绘制训练曲线对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        model_order = [
            '完整模型（基线）',
            '消融1-节点级单流',
            '消融2-图级单流',
            '消融3-混合单流',
            '消融4-单头注意力',
            '消融5-GCN'
        ]
        
        # 绘制损失曲线
        ax1 = axes[0]
        for i, model_name in enumerate(model_order):
            if model_name in self.results and self.results[model_name]['loss_history']:
                history = self.results[model_name]['loss_history']
                ax1.plot(history, label=model_name, color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Total Loss', fontsize=12)
        ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 绘制NC曲线
        ax2 = axes[1]
        for i, model_name in enumerate(model_order):
            if model_name in self.results and self.results[model_name]['nc_history']:
                history = self.results[model_name]['nc_history']
                ax2.plot(history, label=model_name, color=colors[i % len(colors)], linewidth=2)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation NC', fontsize=12)
        ax2.set_title('验证NC对比', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图表已保存: {save_path}")
        plt.close()
    
    def export_to_csv(self, save_path="ablation_results.csv"):
        """导出结果为CSV格式"""
        import csv
        
        model_order = [
            '完整模型（基线）',
            '消融1-节点级单流',
            '消融2-图级单流',
            '消融3-混合单流',
            '消融4-单头注意力',
            '消融5-GCN'
        ]
        
        with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['模型名称', '参数量', '最佳Epoch', '总损失', '验证NC', '训练时间(分钟)'])
            
            for model_name in model_order:
                if model_name in self.results:
                    m = self.results[model_name]
                    writer.writerow([
                        model_name,
                        m['total_params'],
                        m['best_epoch'],
                        f"{m['best_total_loss']:.4f}",
                        f"{m['best_val_nc']:.4f}",
                        f"{m['final_train_time']:.2f}"
                    ])
        
        print(f"结果已导出为CSV: {save_path}")


def main():
    """主函数"""
    print("="*80)
    print("消融试验结果收集与对比")
    print("="*80)
    
    # 创建收集器
    collector = AblationResultsCollector(log_dir="logs")
    
    # 收集结果
    print("\n正在收集消融试验结果...")
    results = collector.collect_all_results()
    
    if not results:
        print("\n错误：未找到任何消融试验结果！")
        print("请确保：")
        print("1. 已运行消融试验训练脚本")
        print("2. 日志文件位于 VGAT/logs/ 目录下")
        return
    
    print(f"\n成功收集 {len(results)} 个模型的结果")
    
    # 生成对比表格
    collector.generate_comparison_table()
    
    # 绘制训练曲线
    print("\n正在绘制训练曲线对比图...")
    collector.plot_training_curves(save_path="logs/ablation_comparison.png")
    
    # 导出CSV
    print("\n正在导出CSV文件...")
    collector.export_to_csv(save_path="logs/ablation_results.csv")
    
    print("\n" + "="*80)
    print("结果收集完成！")
    print("="*80)
    print("\n下一步：")
    print("1. 查看对比表格（上方输出）")
    print("2. 查看训练曲线对比图：logs/ablation_comparison.png")
    print("3. 查看CSV文件：logs/ablation_results.csv")
    print("4. 运行零水印验证脚本评估最终测试集NC值")
    print("\n详细说明请参考：README_Ablation_Studies.md")


if __name__ == "__main__":
    main()

