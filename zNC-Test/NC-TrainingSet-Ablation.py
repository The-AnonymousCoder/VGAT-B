#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集NC测试 - 消融实验通用脚本
使用被攻击的矢量地图验证零水印的鲁棒性
支持Ablation1-5所有消融实验

用法:
    python NC-TrainingSet-Ablation.py --ablation 1  # 消融实验1（仅节点级特征）
    python NC-TrainingSet-Ablation.py --ablation 2  # 消融实验2（仅图级特征）
    python NC-TrainingSet-Ablation.py --ablation 3  # 消融实验3（混合单流）
    python NC-TrainingSet-Ablation.py --ablation 4  # 消融实验4（单注意力头）
    python NC-TrainingSet-Ablation.py --ablation 5  # 消融实验5（GCN替代GAT）
"""

import os
import sys
import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import shutil
import argparse

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'Songti SC']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'


# ⭐ 消融实验配置字典
ABLATION_CONFIG = {
    1: {
        'name': 'Ablation1_NodeOnly',
        'description': '消融实验1（仅节点级特征）',
        'model_file': 'gat_model_Ablation1_NodeOnly_best.pth',
        'model_script': 'Ablation1_NodeOnly.py',
        'model_class': 'NodeOnlyGATModel',
        'watermark_dir': 'TrainingSet-Ablation1',
        'results_dir': 'TrainingSet-Ablation1',
    },
    2: {
        'name': 'Ablation2_GraphOnly',
        'description': '消融实验2（仅图级特征）',
        'model_file': 'gat_model_Ablation2_GraphOnly_best.pth',
        'model_script': 'Ablation2_GraphOnly.py',
        'model_class': 'GraphOnlyModel',
        'watermark_dir': 'TrainingSet-Ablation2',
        'results_dir': 'TrainingSet-Ablation2',
    },
    3: {
        'name': 'Ablation3_MixedSingle',
        'description': '消融实验3（混合单流）',
        'model_file': 'gat_model_Ablation3_MixedSingle_best.pth',
        'model_script': 'Ablation3_MixedSingleStream.py',
        'model_class': 'MixedSingleStreamGATModel',
        'watermark_dir': 'TrainingSet-Ablation3',
        'results_dir': 'TrainingSet-Ablation3',
    },
    4: {
        'name': 'Ablation4_SingleHead',
        'description': '消融实验4（单注意力头）',
        'model_file': 'gat_model_Ablation4_SingleHead_best.pth',
        'model_script': 'Ablation4_SingleHead.py',
        'model_class': 'SingleHeadGATModel',
        'watermark_dir': 'TrainingSet-Ablation4',
        'results_dir': 'TrainingSet-Ablation4',
    },
    5: {
        'name': 'Ablation5_GCN',
        'description': '消融实验5（GCN替代GAT）',
        'model_file': 'gat_model_Ablation5_GCN_best.pth',
        'model_script': 'Ablation5_GCN.py',
        'model_class': 'GCNModel',
        'watermark_dir': 'TrainingSet-Ablation5',
        'results_dir': 'TrainingSet-Ablation5',
    },
}


class RobustnessVerifier:
    """鲁棒性验证器 - 支持消融实验"""
    
    def __init__(self, ablation_id=1):
        """
        初始化验证器
        
        Args:
            ablation_id: 消融实验编号 (1-5)
        """
        self.ablation_id = ablation_id
        self.config = ABLATION_CONFIG[ablation_id]
        
        # 设置模型路径
        base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = os.path.join(base_dir, 'VGAT', 'models', self.config['model_file'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        print(f"消融实验: {self.config['description']}")
        
        self.model = None
        self.load_model()
        
        # 初始化攻击类型映射字典
        self.attack_type_mapping = self._create_attack_type_mapping()
    
    def _create_attack_type_mapping(self):
        """创建攻击类型映射字典（基于attack200.py）"""
        mapping = {}
        
        # 基础攻击类型
        basic_attacks = {
            "delete_10pct_vertices": "删除10%顶点",
            "delete_20pct_vertices": "删除20%顶点", 
            "delete_30pct_vertices": "删除30%顶点",
            "delete_40pct_vertices": "删除40%顶点",
            "delete_50pct_vertices": "删除50%顶点",
            "delete_10pct_objects": "删除10%图形对象",
            "delete_20pct_objects": "删除20%图形对象",
            "delete_30pct_objects": "删除30%图形对象",
            "delete_40pct_objects": "删除40%图形对象",
            "delete_50pct_objects": "删除50%图形对象",
            "add_10pct_vertices": "添加10%顶点",
            "add_20pct_vertices": "添加20%顶点",
            "add_30pct_vertices": "添加30%顶点",
            "add_40pct_vertices": "添加40%顶点",
            "add_50pct_vertices": "添加50%顶点",
            "noise_10pct_strength_0.2": "噪声扰动10%顶点，强度0.2",
            "noise_10pct_strength_0.4": "噪声扰动10%顶点，强度0.4",
            "noise_10pct_strength_0.6": "噪声扰动10%顶点，强度0.6",
            "noise_20pct_strength_0.3": "噪声扰动20%顶点，强度0.3",
            "noise_30pct_strength_0.8": "噪声扰动30%顶点，强度0.8",
            "crop_x_center_50pct": "沿X轴中心裁剪50%",
            "crop_y_center_50pct": "沿Y轴中心裁剪50%",
            "crop_top_left": "裁剪左上角区域",
            "crop_bottom_right": "裁剪右下角区域",
            "crop_random_40pct": "随机裁剪40%",
            "translate_10_10": "X轴右移10单位，Y轴上移10单位",
            "translate_20_20": "X轴右移20单位，Y轴上移20单位",
            "translate_x_30": "仅X轴右移30单位",
            "translate_y_15": "仅Y轴上移15单位",
            "translate_neg10": "X轴左移10单位，Y轴下移10单位",
            "scale_0.5x": "缩放0.5倍",
            "scale_2x": "缩放2倍",
            "scale_x0.5_y2": "X轴缩小0.5倍，Y轴放大2倍",
            "scale_x2_y0.5": "X轴放大2倍，Y轴缩小0.5倍",
            "scale_random": "随机缩放",
            "rotate_45": "旋转45度",
            "rotate_90": "旋转90度",
            "rotate_135": "旋转135度",
            "rotate_180": "旋转180度",
            "rotate_random": "随机旋转",
            "flip_x": "X轴镜像翻转",
            "flip_y": "Y轴镜像翻转",
            "flip_xy": "同时X、Y轴镜像翻转",
            "reverse_vertex_order": "反转顶点顺序",
            "reverse_object_order": "反转对象顺序",
            "shuffle_objects": "打乱对象顺序",
            "shuffle_vertices": "打乱顶点顺序",
            "jitter_vertices": "顶点顺序随机偏移",
            "merge_objects": "合并对象",
            "split_objects": "拆分对象",
        }
        
        for attack_key, description in basic_attacks.items():
            mapping[attack_key] = description
        
        # 添加扩展攻击类型
        for i in range(55, 101):
            mapping[f"attack_{i:03d}"] = f"扩展攻击策略{i}"
        
        # 添加组合攻击类型
        for i in range(1, 101):
            mapping[f"combo_attack_{i:03d}"] = f"组合攻击策略{i}"
        
        return mapping
    
    def get_attack_type_description(self, filename):
        """根据文件名获取攻击类型描述"""
        base_name = filename.replace('_graph.pkl', '').replace('.geojson', '')
        
        # 移除图名前缀
        for prefix in ['Boundary_', 'Building_', 'Lake_', 'Landuse_', 'Railways_', 'Road_']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        
        # 查找匹配的攻击类型
        for attack_key, description in self.attack_type_mapping.items():
            if attack_key in base_name:
                return description
        
        return base_name
    
    def load_model(self):
        """加载消融实验模型"""
        try:
            print(f"加载{self.config['description']}模型...")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 动态导入模型类
            import importlib.util
            base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
            model_script_path = os.path.join(base_dir, 'VGAT', self.config['model_script'])
            
            spec = importlib.util.spec_from_file_location(self.config['name'], model_script_path)
            model_module = importlib.util.module_from_spec(spec)
            sys.modules[self.config['name']] = model_module
            spec.loader.exec_module(model_module)
            
            ModelClass = getattr(model_module, self.config['model_class'])
            
            # 创建模型实例
            if self.ablation_id == 5:  # GCN没有num_heads参数
                self.model = ModelClass(input_dim=20, hidden_dim=256, output_dim=1024, dropout=0.3)
            elif self.ablation_id == 4:  # 单头注意力
                self.model = ModelClass(input_dim=20, hidden_dim=256, output_dim=1024, num_heads=1, dropout=0.3)
            else:
                self.model = ModelClass(input_dim=20, hidden_dim=256, output_dim=1024, num_heads=8, dropout=0.3)
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ {self.config['description']}模型加载完成")
            print(f"  输入维度: 20")
            print(f"  隐藏维度: 256")
            print(f"  输出维度: 1024")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_robust_features(self, graph_data):
        """提取鲁棒特征"""
        if not hasattr(graph_data, 'x') or not hasattr(graph_data, 'edge_index'):
            raise ValueError("输入图数据无效")
        if self.model is None:
            raise RuntimeError("模型未加载")

        with torch.no_grad():
            features = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            features = features.cpu().numpy()

        if features.ndim > 1:
            features = features.flatten()
        
        if len(features) != 1024:
            if len(features) < 1024:
                features = np.tile(features, (1024 // len(features) + 1,))
            features = features[:1024]
        
        return features
    
    def load_copyright_image(self, image_path=None):
        """加载版权图像"""
        try:
            if image_path is None:
                image_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'ZeroWatermark', 'Cat32.png'))
            image = Image.open(image_path)
            image = image.convert('L')
            image = image.resize((32, 32))
            threshold = 128
            image = image.point(lambda x: 0 if x < threshold else 255, '1')
            return np.array(image)
        except Exception as e:
            print(f"加载版权图像失败: {e}")
            return np.random.randint(0, 2, (32, 32))
    
    def features_to_matrix(self, features, target_shape):
        """将特征向量转换为矩阵（使用中位数阈值二值化）"""
        total_elements = target_shape[0] * target_shape[1]
        
        if len(features) < total_elements:
            features = np.tile(features, (total_elements // len(features) + 1,))
        
        features = features[:total_elements]
        matrix = features.reshape(target_shape)
        
        threshold = np.median(matrix)
        matrix = (matrix > threshold).astype(np.uint8)
        
        return matrix
    
    def verify_copyright(self, graph_data, zero_watermark, original_copyright):
        """验证版权"""
        robust_features = self.extract_robust_features(graph_data)
        feature_matrix = self.features_to_matrix(robust_features, original_copyright.shape)
        extracted_copyright = np.logical_xor(zero_watermark, feature_matrix).astype(np.uint8)
        nc_value = self.calculate_nc(original_copyright, extracted_copyright)
        
        return extracted_copyright, nc_value
    
    def calculate_nc(self, original, extracted):
        """计算归一化相关系数（NC值）"""
        original_vec = original.flatten().astype(float)
        extracted_vec = extracted.flatten().astype(float)
        
        dot_product = np.sum(original_vec * extracted_vec)
        norm_original = np.sqrt(np.sum(original_vec ** 2))
        norm_extracted = np.sqrt(np.sum(extracted_vec ** 2))
        
        if norm_original == 0 or norm_extracted == 0:
            return 0.0
        
        nc = dot_product / (norm_original * norm_extracted)
        return nc
    
    def load_watermark(self, filename):
        """加载对应的零水印"""
        watermark_dir = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', 'ZeroWatermark', 'ZeroWatermark', 
            self.config['watermark_dir']
        ))
        watermark_path = os.path.join(watermark_dir, f"{filename}_watermark.npy")
        
        if os.path.exists(watermark_path):
            return np.load(watermark_path)
        else:
            print(f"零水印文件不存在: {watermark_path}")
            return None
    
    def verify_robustness(self, original_graph, attacked_graphs, zero_watermark, copyright_image, filename, attacked_filenames=None):
        """验证鲁棒性"""
        print(f"验证鲁棒性: {filename}")
        
        results = []
        
        for i, attacked_graph in enumerate(attacked_graphs):
            extracted_copyright, nc_value = self.verify_copyright(
                attacked_graph, zero_watermark, copyright_image
            )
            
            attack_filename = attacked_filenames[i] if attacked_filenames and i < len(attacked_filenames) else f"attack_{i+1}"
            attack_description = self.get_attack_type_description(attack_filename)
            
            results.append({
                'attack_index': i,
                'attack_filename': attack_filename,
                'attack_description': attack_description,
                'nc_value': nc_value,
                'extracted_copyright': extracted_copyright
            })
            
            print(f"  攻击 {i+1}: NC值 = {nc_value:.4f}")
        
        return results
    
    def save_robustness_results(self, results, zero_watermark, copyright_image, filename):
        """保存鲁棒性验证结果"""
        results_dir = os.path.normpath(os.path.join(
            os.path.dirname(__file__), 'NC-Results', self.config['results_dir']
        ))
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        nc_values = [r['nc_value'] for r in results]
        avg_nc = np.mean(nc_values)
        max_nc = np.max(nc_values)
        min_nc = np.min(nc_values)
        
        # 保存结果图像
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 4, 1)
        plt.imshow(zero_watermark, cmap='gray')
        plt.title('Zero Watermark')
        plt.axis('off')
        
        plt.subplot(3, 4, 2)
        plt.imshow(copyright_image, cmap='gray')
        plt.title('Original Copyright')
        plt.axis('off')
        
        for i in range(min(6, len(results))):
            plt.subplot(3, 4, i + 7)
            plt.imshow(results[i]['extracted_copyright'], cmap='gray')
            plt.title(f'Attack {i+1}\nNC: {results[i]["nc_value"]:.4f}')
            plt.axis('off')
        
        plt.subplot(3, 4, 11)
        plt.text(0.5, 0.5, f'Average NC: {avg_nc:.4f}\nMax NC: {max_nc:.4f}\nMin NC: {min_nc:.4f}', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=10)
        plt.title('Statistics')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{filename}_robustness_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存统计信息
        stats_path = os.path.join(results_dir, f'{filename}_robustness_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"鲁棒性验证结果: {filename}\n")
            f.write(f"消融实验: {self.config['description']}\n")
            f.write(f"平均NC值: {avg_nc:.4f}\n")
            f.write(f"最高NC值: {max_nc:.4f}\n")
            f.write(f"最低NC值: {min_nc:.4f}\n")
            f.write(f"验证成功数量 (NC > 0.7): {sum(1 for nc in nc_values if nc > 0.7)}/{len(nc_values)}\n")
            f.write("\n详细结果:\n")
            for i, result in enumerate(results):
                f.write(f"攻击 {i+1}: NC值 = {result['nc_value']:.4f}\n")
        
        print(f"鲁棒性验证结果已保存: {results_dir}")
        return avg_nc, max_nc, min_nc
    
    def save_excel_results(self, all_results):
        """只保存summary.csv（统计汇总）"""
        results_dir = os.path.normpath(os.path.join(
            os.path.dirname(__file__), 'NC-Results', self.config['results_dir']
        ))
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 🆕 保存统计汇总CSV（每个图的平均值）
        summary_data = []
        for graph_name, results in all_results.items():
            nc_values = [r['nc_value'] for r in results]
            summary_data.append({
                '图名称': graph_name,
                '攻击数量': len(nc_values),
                '平均NC值': np.mean(nc_values),
                '最大NC值': np.max(nc_values),
                '最小NC值': np.min(nc_values),
                '标准差': np.std(nc_values)
            })
        
        # ⭐ 添加总体统计行
        if summary_data:
            all_avg_nc = [d['平均NC值'] for d in summary_data]
            all_max_nc = [d['最大NC值'] for d in summary_data]
            all_min_nc = [d['最小NC值'] for d in summary_data]
            all_std = [d['标准差'] for d in summary_data]
            total_attacks = sum(d['攻击数量'] for d in summary_data)
            
            summary_data.append({
                '图名称': 'Overall Average',
                '攻击数量': total_attacks,
                '平均NC值': np.mean(all_avg_nc),
                '最大NC值': np.mean(all_max_nc),
                '最小NC值': np.mean(all_min_nc),
                '标准差': np.mean(all_std)
            })
        
        df_summary = pd.DataFrame(summary_data)
        csv_summary_path = os.path.join(results_dir, f'ablation{self.ablation_id}_nc_summary_{timestamp}.csv')
        df_summary.to_csv(csv_summary_path, index=False, encoding='utf-8-sig')
        print(f"CSV统计汇总已保存: {csv_summary_path}")
        
        return csv_summary_path


class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, graph_dir=None):
        if graph_dir is None:
            base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
            graph_dir = os.path.join(base_dir, 'convertToGraph', 'Graph', 'TrainingSet')
        self.graph_dir = os.path.normpath(graph_dir)
        print(f"图数据加载路径: {self.graph_dir}")
    
    def load_original_graphs_train_only(self):
        """只加载训练集的原始图数据"""
        original_dir = os.path.join(self.graph_dir, 'Original')
        
        if not os.path.exists(original_dir):
            print(f"目录不存在: {original_dir}")
            return [], []
        
        graphs = []
        filenames = []
        
        for filename in os.listdir(original_dir):
            if filename.endswith('_graph.pkl'):
                try:
                    with open(os.path.join(original_dir, filename), 'rb') as f:
                        graph_data = pickle.load(f)
                        graphs.append(graph_data)
                        base_name = filename.replace('_graph.pkl', '')
                        filenames.append(base_name)
                        print(f"成功加载训练图数据: {filename}")
                except Exception as e:
                    print(f"加载图数据失败 {filename}: {e}")
                    continue
        
        print(f"总共加载了 {len(graphs)} 个训练图数据")
        return graphs, filenames
    
    def load_attacked_graphs_for_original(self, original_filename):
        """加载指定原始图对应的所有被攻击图"""
        attacked_dir = os.path.join(self.graph_dir, 'Attacked')
        
        if not os.path.exists(attacked_dir):
            print(f"被攻击图目录不存在: {attacked_dir}")
            return [], []
        
        subdir_path = os.path.join(attacked_dir, original_filename)
        if os.path.exists(subdir_path):
            attacked_graphs = []
            attacked_filenames = []
            for filename in sorted(os.listdir(subdir_path)):
                if filename.endswith('_graph.pkl'):
                    try:
                        with open(os.path.join(subdir_path, filename), 'rb') as f:
                            graph_data = pickle.load(f)
                            attacked_graphs.append(graph_data)
                            attacked_filenames.append(filename)
                            print(f"成功加载被攻击图: {filename}")
                    except Exception as e:
                        print(f"加载被攻击图失败 {filename}: {e}")
                        continue
            
            print(f"为 {original_filename} 加载了 {len(attacked_graphs)} 个被攻击图")
            return attacked_graphs, attacked_filenames
        else:
            print(f"没有找到 {original_filename} 对应的被攻击图目录")
            return [], []


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练集NC测试 - 消融实验')
    parser.add_argument('--ablation', type=int, required=True, choices=[1, 2, 3, 4, 5],
                       help='消融实验编号 (1-5)')
    args = parser.parse_args()
    
    ablation_id = args.ablation
    config = ABLATION_CONFIG[ablation_id]
    
    print("="*70)
    print(f"训练集NC测试 - {config['description']}")
    print("="*70)
    print()
    
    # 清理并准备结果输出目录
    results_root = os.path.normpath(os.path.join(os.path.dirname(__file__), 'NC-Results', config['results_dir']))
    if os.path.exists(results_root):
        print(f"清理旧的结果目录: {results_root}")
        try:
            shutil.rmtree(results_root)
            print("[OK] 旧结果已清理")
        except Exception as e:
            print(f"[WARNING] 清理目录时出错: {e}")
    os.makedirs(results_root, exist_ok=True)
    print()
    
    # 加载训练集的原始图数据
    data_loader = GraphDataLoader()
    original_graphs, filenames = data_loader.load_original_graphs_train_only()
    
    if not original_graphs:
        print("没有找到训练图数据")
        return
    
    # 创建鲁棒性验证器
    verifier = RobustnessVerifier(ablation_id=ablation_id)
    
    # 加载版权图像
    copyright_image = verifier.load_copyright_image()
    print(f"版权图像大小: {copyright_image.shape}")
    
    # 为每个训练图验证鲁棒性
    all_avg_nc = []
    all_max_nc = []
    all_min_nc = []
    all_excel_results = {}
    
    for i, (original_graph, filename) in enumerate(zip(original_graphs, filenames)):
        print(f"\n处理第 {i+1}/{len(original_graphs)} 个图: {filename}")
        
        # 加载对应的零水印
        zero_watermark = verifier.load_watermark(filename)
        if zero_watermark is None:
            print(f"跳过 {filename}，零水印不存在")
            continue
        
        # 加载对应的被攻击图
        attacked_graphs, attacked_filenames = data_loader.load_attacked_graphs_for_original(filename)
        
        if not attacked_graphs:
            print(f"跳过 {filename}，没有找到对应的被攻击图")
            continue
        
        # 验证鲁棒性
        results = verifier.verify_robustness(
            original_graph, attacked_graphs, zero_watermark, copyright_image, filename, attacked_filenames
        )
        
        # 收集Excel数据
        all_excel_results[filename] = results
        
        # 保存结果
        avg_nc, max_nc, min_nc = verifier.save_robustness_results(
            results, zero_watermark, copyright_image, filename
        )
        
        all_avg_nc.append(avg_nc)
        all_max_nc.append(max_nc)
        all_min_nc.append(min_nc)
    
    # 输出总体结果
    if all_avg_nc:
        # 按攻击类型统计
        if all_excel_results:
            print(f"\n按攻击类型统计:")
            print("-" * 50)
            
            attack_nc_values = {}
            for filename, results in all_excel_results.items():
                for result in results:
                    attack_desc = result.get('attack_description', '未知攻击')
                    nc_value = result.get('nc_value', 0)
                    if attack_desc not in attack_nc_values:
                        attack_nc_values[attack_desc] = []
                    attack_nc_values[attack_desc].append(nc_value)
            
            for attack_type, nc_values in attack_nc_values.items():
                avg_nc = np.mean(nc_values)
                max_nc = np.max(nc_values)
                min_nc = np.min(nc_values)
                success_count = sum(1 for nc in nc_values if nc > 0.7)
                success_rate = (success_count / len(nc_values)) * 100
                
                print(f"{attack_type}:")
                print(f"  测试数量: {len(nc_values)}")
                print(f"  成功验证数: {success_count}")
                print(f"  成功率: {success_rate:.2f}%")
                print(f"  平均NC值: {avg_nc:.4f}")
                print(f"  最大NC值: {max_nc:.4f}")
                print(f"  最小NC值: {min_nc:.4f}")
                print()
        
        print(f"\n{'='*70}")
        print(f"{config['description']} - 训练集NC值验证总体结果")
        print(f"{'='*70}")
        print(f"处理的图数量: {len(all_avg_nc)}")
        
        # 按原始地图统计
        print(f"\n按原始地图统计:")
        print("-" * 50)
        for i, filename in enumerate(filenames):
            if i < len(all_avg_nc):
                avg_nc = all_avg_nc[i]
                max_nc = all_max_nc[i]
                min_nc = all_min_nc[i]
                success_status = "成功" if avg_nc > 0.7 else "失败"
                
                print(f"{filename}:")
                print(f"  平均NC值: {avg_nc:.4f}")
                print(f"  最大NC值: {max_nc:.4f}")
                print(f"  最小NC值: {min_nc:.4f}")
                print(f"  验证状态: {success_status}")
        
        # 总体统计
        print(f"\n总体统计:")
        print("-" * 50)
        overall_avg_nc = np.mean(all_avg_nc)
        overall_max_nc = np.mean(all_max_nc)
        overall_min_nc = np.mean(all_min_nc)
        overall_std_nc = np.std(all_avg_nc)
        
        print(f"总体平均NC值: {overall_avg_nc:.4f}")
        print(f"总体最大NC值: {overall_max_nc:.4f}")
        print(f"总体最小NC值: {overall_min_nc:.4f}")
        print(f"总体NC值标准差: {overall_std_nc:.4f}")
        
        # 成功率统计
        success_count = sum(1 for avg_nc in all_avg_nc if avg_nc > 0.7)
        success_rate = (success_count / len(all_avg_nc)) * 100
        
        print(f"\n成功率统计:")
        print("-" * 50)
        print(f"总验证数量: {len(all_avg_nc)}")
        print(f"成功验证数量: {success_count}")
        print(f"总体成功率: {success_rate:.2f}%")
        
        print(f"\n验证成功统计:")
        print("-" * 50)
        print(f"地图验证成功数量: {success_count}/{len(all_avg_nc)}")
        
        # 生成Excel表格
        if all_excel_results:
            verifier.save_excel_results(all_excel_results)
        
        print(f"\n{'='*70}")
        print(f"[OK] 鲁棒性验证完成！")
        print(f"结果保存目录: {results_root}")
        print(f"{'='*70}")
    else:
        print("\n[WARNING] 没有成功处理任何图数据")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()

