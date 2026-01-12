#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征质量对比分析脚本

用于评估：
1. 原始13维特征 vs 改进16维特征
2. 逐图标准化 vs 全局标准化
3. 对不同攻击类型的鲁棒性

运行前需要：
1. 已生成原始图数据（使用 convertToGraph-TrainingSet.py）
2. 已生成改进图数据（使用 convertToGraph-TrainingSet-IMPROVED.py）
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

class FeatureQualityAnalyzer:
    """特征质量分析器"""
    
    def __init__(self, original_graph_dir="Graph/TrainingSet", 
                 improved_graph_dir="Graph-Improved/TrainingSet"):
        self.original_graph_dir = original_graph_dir
        self.improved_graph_dir = improved_graph_dir
        
        # 定义攻击类型分组
        self.attack_groups = {
            '几何变换': ['translate', 'scale', 'rotate', 'flip'],
            '顶点操作': ['delete_', 'add_'],
            '对象操作': ['delete_', 'crop'],
            '噪声扰动': ['noise'],
            '顺序打乱': ['reverse', 'shuffle'],
            '组合攻击': ['combo']
        }
    
    def load_graph(self, graph_path):
        """加载图数据"""
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def load_all_graphs(self, graph_dir, data_type='Original'):
        """加载所有图数据"""
        graphs = {}
        
        if data_type == 'Original':
            original_dir = os.path.join(graph_dir, 'Original')
            if os.path.exists(original_dir):
                for filename in os.listdir(original_dir):
                    if filename.endswith('_graph.pkl'):
                        graph_name = filename.replace('_graph.pkl', '')
                        graph_path = os.path.join(original_dir, filename)
                        graphs[graph_name] = self.load_graph(graph_path)
        
        elif data_type == 'Attacked':
            attacked_dir = os.path.join(graph_dir, 'Attacked')
            if os.path.exists(attacked_dir):
                for subdir in os.listdir(attacked_dir):
                    subdir_path = os.path.join(attacked_dir, subdir)
                    if os.path.isdir(subdir_path):
                        base_name = subdir  # 原图名称
                        graphs[base_name] = {}
                        
                        for filename in os.listdir(subdir_path):
                            if filename.endswith('_graph.pkl'):
                                attack_name = filename.replace('_graph.pkl', '')
                                graph_path = os.path.join(subdir_path, filename)
                                graphs[base_name][attack_name] = self.load_graph(graph_path)
        
        return graphs
    
    def calculate_feature_similarity(self, features1, features2):
        """计算特征相似度"""
        # 全局池化：将节点级特征聚合为图级特征
        if len(features1.shape) == 2:
            # 均值池化
            feat1_mean = torch.mean(features1, dim=0)
            feat1_max, _ = torch.max(features1, dim=0)
            feat1 = torch.cat([feat1_mean, feat1_max])
        else:
            feat1 = features1
        
        if len(features2.shape) == 2:
            feat2_mean = torch.mean(features2, dim=0)
            feat2_max, _ = torch.max(features2, dim=0)
            feat2 = torch.cat([feat2_mean, feat2_max])
        else:
            feat2 = features2
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=1)
        return similarity.item()
    
    def analyze_robustness(self, graph_dir, output_prefix):
        """分析特征对攻击的鲁棒性"""
        print(f"\n分析 {output_prefix} 的特征鲁棒性...")
        
        # 加载原始图和攻击图
        original_graphs = self.load_all_graphs(graph_dir, 'Original')
        attacked_graphs = self.load_all_graphs(graph_dir, 'Attacked')
        
        print(f"加载了 {len(original_graphs)} 个原始图")
        print(f"加载了 {len(attacked_graphs)} 个原始图的攻击版本")
        
        # 分析每种攻击类型的相似度
        attack_type_similarities = {}
        
        for base_name in attacked_graphs:
            if base_name not in original_graphs:
                continue
            
            original_features = original_graphs[base_name].x
            
            for attack_name, attacked_graph in attacked_graphs[base_name].items():
                attacked_features = attacked_graph.x
                
                # 计算相似度
                similarity = self.calculate_feature_similarity(
                    original_features, attacked_features
                )
                
                # 确定攻击类型
                attack_type = self.get_attack_type(attack_name)
                
                if attack_type not in attack_type_similarities:
                    attack_type_similarities[attack_type] = []
                attack_type_similarities[attack_type].append(similarity)
        
        # 计算统计量
        results = {}
        for attack_type, similarities in attack_type_similarities.items():
            results[attack_type] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'count': len(similarities)
            }
        
        # 打印结果
        print(f"\n{output_prefix} - 各攻击类型的特征相似度:")
        print("-" * 80)
        print(f"{'攻击类型':<15} {'平均值':<10} {'标准差':<10} {'最小值':<10} {'最大值':<10} {'样本数':<10}")
        print("-" * 80)
        for attack_type, stats in sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True):
            print(f"{attack_type:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                  f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['count']:<10}")
        print("-" * 80)
        
        return results
    
    def get_attack_type(self, attack_name):
        """根据攻击名称确定攻击类型"""
        attack_name_lower = attack_name.lower()
        
        if 'translate' in attack_name_lower:
            return '平移'
        elif 'scale' in attack_name_lower:
            return '缩放'
        elif 'rotate' in attack_name_lower:
            return '旋转'
        elif 'flip' in attack_name_lower:
            return '翻转'
        elif 'delete' in attack_name_lower and 'vertices' in attack_name_lower:
            return '删除顶点'
        elif 'add' in attack_name_lower and 'vertices' in attack_name_lower:
            return '添加顶点'
        elif 'delete' in attack_name_lower and 'objects' in attack_name_lower:
            return '删除对象'
        elif 'noise' in attack_name_lower:
            return '噪声扰动'
        elif 'crop' in attack_name_lower:
            return '裁剪'
        elif 'reverse' in attack_name_lower or 'shuffle' in attack_name_lower:
            return '顺序打乱'
        elif 'combo' in attack_name_lower:
            return '组合攻击'
        else:
            return '其他'
    
    def compare_features(self):
        """对比原始特征和改进特征"""
        print("=" * 80)
        print("特征质量对比分析")
        print("=" * 80)
        
        # 分析原始特征
        original_results = self.analyze_robustness(
            self.original_graph_dir, "原始13维特征"
        )
        
        # 分析改进特征
        improved_results = self.analyze_robustness(
            self.improved_graph_dir, "改进16维特征"
        )
        
        # 对比分析
        print("\n" + "=" * 80)
        print("改进效果对比（相似度提升）")
        print("=" * 80)
        print(f"{'攻击类型':<15} {'原始特征':<12} {'改进特征':<12} {'提升':<12} {'提升率':<12}")
        print("-" * 80)
        
        all_attack_types = set(original_results.keys()) | set(improved_results.keys())
        
        improvements = []
        for attack_type in sorted(all_attack_types):
            if attack_type in original_results and attack_type in improved_results:
                original_mean = original_results[attack_type]['mean']
                improved_mean = improved_results[attack_type]['mean']
                improvement = improved_mean - original_mean
                improvement_rate = (improvement / original_mean) * 100 if original_mean > 0 else 0
                
                improvements.append({
                    'attack_type': attack_type,
                    'original': original_mean,
                    'improved': improved_mean,
                    'improvement': improvement,
                    'improvement_rate': improvement_rate
                })
                
                print(f"{attack_type:<15} {original_mean:<12.4f} {improved_mean:<12.4f} "
                      f"{improvement:<12.4f} {improvement_rate:<12.2f}%")
        
        print("-" * 80)
        
        # 总体统计
        if improvements:
            avg_improvement = np.mean([x['improvement'] for x in improvements])
            avg_improvement_rate = np.mean([x['improvement_rate'] for x in improvements])
            
            print(f"\n总体平均提升: {avg_improvement:.4f} ({avg_improvement_rate:.2f}%)")
            
            # 找出改进最显著的攻击类型
            best_improvement = max(improvements, key=lambda x: x['improvement'])
            print(f"改进最显著的攻击类型: {best_improvement['attack_type']} "
                  f"(+{best_improvement['improvement']:.4f}, +{best_improvement['improvement_rate']:.2f}%)")
        
        # 可视化对比
        self.visualize_comparison(original_results, improved_results, improvements)
        
        return original_results, improved_results, improvements
    
    def visualize_comparison(self, original_results, improved_results, improvements):
        """可视化对比结果"""
        if not improvements:
            print("没有可对比的数据，跳过可视化")
            return
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 相似度对比柱状图
        ax1 = axes[0, 0]
        attack_types = [x['attack_type'] for x in improvements]
        original_means = [x['original'] for x in improvements]
        improved_means = [x['improved'] for x in improvements]
        
        x = np.arange(len(attack_types))
        width = 0.35
        
        ax1.bar(x - width/2, original_means, width, label='原始13维特征', alpha=0.8)
        ax1.bar(x + width/2, improved_means, width, label='改进16维特征', alpha=0.8)
        
        ax1.set_xlabel('攻击类型')
        ax1.set_ylabel('特征相似度')
        ax1.set_title('各攻击类型的特征相似度对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(attack_types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 提升率柱状图
        ax2 = axes[0, 1]
        improvement_rates = [x['improvement_rate'] for x in improvements]
        colors = ['green' if x > 0 else 'red' for x in improvement_rates]
        
        ax2.bar(attack_types, improvement_rates, color=colors, alpha=0.7)
        ax2.set_xlabel('攻击类型')
        ax2.set_ylabel('提升率 (%)')
        ax2.set_title('改进特征的提升率')
        ax2.set_xticklabels(attack_types, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 相似度分布箱线图
        ax3 = axes[1, 0]
        
        original_data = []
        improved_data = []
        labels = []
        
        for attack_type in sorted(set([x['attack_type'] for x in improvements])):
            if attack_type in original_results:
                # 这里我们没有原始数据，只能用均值和标准差模拟
                # 实际使用时应该保存完整的相似度列表
                labels.append(attack_type[:6])  # 缩短标签
        
        # 使用热力图展示均值
        original_matrix = np.array([[original_results[at]['mean']] for at in sorted(original_results.keys())])
        improved_matrix = np.array([[improved_results[at]['mean']] for at in sorted(improved_results.keys())])
        
        combined_matrix = np.hstack([original_matrix, improved_matrix])
        
        sns.heatmap(combined_matrix, annot=True, fmt='.3f', cmap='YlGnBu', 
                    xticklabels=['原始', '改进'], 
                    yticklabels=[at[:8] for at in sorted(original_results.keys())],
                    ax=ax3, cbar_kws={'label': '相似度'})
        ax3.set_title('特征相似度热力图')
        
        # 4. 改进幅度排序
        ax4 = axes[1, 1]
        sorted_improvements = sorted(improvements, key=lambda x: x['improvement'])
        sorted_attack_types = [x['attack_type'] for x in sorted_improvements]
        sorted_improvements_values = [x['improvement'] for x in sorted_improvements]
        
        colors4 = ['green' if x > 0 else 'red' for x in sorted_improvements_values]
        ax4.barh(sorted_attack_types, sorted_improvements_values, color=colors4, alpha=0.7)
        ax4.set_xlabel('相似度提升')
        ax4.set_title('改进幅度排序（由低到高）')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = 'feature_comparison_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存: {output_path}")
        
        plt.show()
    
    def analyze_feature_dimensionality(self):
        """分析特征维度的重要性"""
        print("\n" + "=" * 80)
        print("特征维度分析")
        print("=" * 80)
        
        # 加载改进特征的一个样本
        improved_graphs = self.load_all_graphs(self.improved_graph_dir, 'Original')
        
        if not improved_graphs:
            print("未找到改进特征数据")
            return
        
        # 收集所有特征
        all_features = []
        for graph_name, graph_data in improved_graphs.items():
            features = graph_data.x.numpy()
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        
        print(f"\n特征维度: {all_features.shape[1]}")
        print(f"节点总数: {all_features.shape[0]}")
        
        # 计算每个维度的统计量
        feature_names = [
            '点类型', '线类型', '面类型',  # 1-3
            '紧凑度', '形状指数',  # 4-5
            '相对X', '相对Y', '距离中心',  # 6-8
            '长宽比', '矩形度',  # 9-10
            '凸包比', '对数顶点数',  # 11-12
            '邻域距离', '邻居数', '邻域密度',  # 13-15
            '伸长率'  # 16
        ]
        
        print(f"\n{'特征名称':<15} {'均值':<10} {'标准差':<10} {'最小值':<10} {'最大值':<10} {'变异系数':<10}")
        print("-" * 80)
        
        for i, name in enumerate(feature_names):
            if i < all_features.shape[1]:
                mean = np.mean(all_features[:, i])
                std = np.std(all_features[:, i])
                min_val = np.min(all_features[:, i])
                max_val = np.max(all_features[:, i])
                cv = std / (abs(mean) + 1e-6)  # 变异系数
                
                print(f"{name:<15} {mean:<10.4f} {std:<10.4f} {min_val:<10.4f} {max_val:<10.4f} {cv:<10.4f}")
        
        print("-" * 80)

def main():
    """主函数"""
    print("=" * 80)
    print("特征质量对比分析工具")
    print("=" * 80)
    
    # 创建分析器
    analyzer = FeatureQualityAnalyzer()
    
    # 检查目录是否存在
    if not os.path.exists(analyzer.original_graph_dir):
        print(f"错误：原始图目录不存在: {analyzer.original_graph_dir}")
        print("请先运行 convertToGraph-TrainingSet.py")
        return
    
    if not os.path.exists(analyzer.improved_graph_dir):
        print(f"警告：改进图目录不存在: {analyzer.improved_graph_dir}")
        print("请先运行 convertToGraph-TrainingSet-IMPROVED.py")
        print("将只分析原始特征...")
        
        # 仅分析原始特征
        original_results = analyzer.analyze_robustness(
            analyzer.original_graph_dir, "原始13维特征"
        )
        return
    
    # 执行完整对比分析
    original_results, improved_results, improvements = analyzer.compare_features()
    
    # 分析特征维度
    analyzer.analyze_feature_dimensionality()
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n建议：")
    print("1. 如果几何变换类攻击（平移、缩放、旋转）的相似度提升显著，说明改进有效")
    print("2. 如果整体平均提升 > 5%，建议采用改进方案")
    print("3. 观察可视化结果，识别仍需改进的攻击类型")

if __name__ == "__main__":
    main()

