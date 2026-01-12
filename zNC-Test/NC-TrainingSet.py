#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁棒性验证脚本
使用被攻击的矢量地图验证零水印的鲁棒性
只测试训练集，每个图都使用对应的零水印
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import shutil

# 设置中文字体 - 适配Windows环境
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'Songti SC']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用默认字体
    plt.rcParams['font.family'] = 'DejaVu Sans'

class RobustnessVerifier:
    """鲁棒性验证器"""
    
    def __init__(self, model_path=None, use_trained_model=True):
        # 模型路径改为VGAT/models/gat_model_IMPROVED_best.pth（使用改进版模型）
        if model_path is None:
            model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'VGAT', 'models', 'gat_model_IMPROVED_best.pth'))
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model = None
        self.use_trained_model = use_trained_model
        if use_trained_model:
            self.load_model()
        
        # 初始化攻击类型映射字典（基于attack200.py）
        self.attack_type_mapping = self._create_attack_type_mapping()
    
    def _create_attack_type_mapping(self):
        """创建攻击类型映射字典（基于attack200.py）"""
        mapping = {}
        
        # 基础攻击类型（前54个，基于attack200.py）
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
        
        # 添加基础攻击类型
        for attack_key, description in basic_attacks.items():
            mapping[attack_key] = description
        
        # 添加扩展攻击类型（attack_055到attack_100）
        for i in range(55, 101):
            mapping[f"attack_{i:03d}"] = f"扩展攻击策略{i}"
        
        # 添加组合攻击类型（combo_attack_001到combo_attack_100）
        for i in range(1, 101):
            mapping[f"combo_attack_{i:03d}"] = f"组合攻击策略{i}"
        
        return mapping
    
    def get_attack_type_description(self, filename):
        """根据文件名获取攻击类型描述"""
        # 移除文件扩展名和图名前缀
        base_name = filename.replace('_graph.pkl', '').replace('.geojson', '')
        
        # 移除图名前缀（如Boundary_、Building_等）
        for prefix in ['Boundary_', 'Building_', 'Lake_', 'Landuse_', 'Railways_', 'Road_']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        
        # 查找匹配的攻击类型
        for attack_key, description in self.attack_type_mapping.items():
            if attack_key in base_name:
                return description
        
        # 如果没有找到匹配的，返回文件名
        return base_name
    
    def load_model(self):
        """加载训练好的模型（强制，失败即报错）"""
        try:
            print("加载训练好的改进版GAT模型...")
            if os.path.exists(self.model_path):
                # 导入模型类（使用importlib因为文件名包含连字符）
                import sys
                import importlib.util
                
                base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
                vgat_improved_path = os.path.join(base_dir, 'VGAT', 'VGAT-IMPROVED.py')
                
                # 动态加载模块
                spec = importlib.util.spec_from_file_location("vgat_improved", vgat_improved_path)
                vgat_module = importlib.util.module_from_spec(spec)
                sys.modules['vgat_improved'] = vgat_module
                spec.loader.exec_module(vgat_module)
                
                ImprovedGATModel = vgat_module.ImprovedGATModel
                
                # 使用IMPROVED版本的模型参数：input_dim=20, hidden_dim=256, num_heads=8
                self.model = ImprovedGATModel(input_dim=20, hidden_dim=256, output_dim=1024, num_heads=8, dropout=0.3)
                
                # 加载权重
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                print("改进版GAT模型加载完成")
                print(f"  输入维度: 20 (20维几何不变特征)")
                print(f"  隐藏维度: 256")
                print(f"  输出维度: 1024")
                print(f"  注意力头数: 8")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def extract_robust_features(self, graph_data):
        """提取鲁棒特征（强制使用训练模型）"""
        if not hasattr(graph_data, 'x') or not hasattr(graph_data, 'edge_index'):
            raise ValueError("输入图数据无效，缺少必要属性 'x' 或 'edge_index'")
        if self.model is None:
            raise RuntimeError("模型未加载，无法提取鲁棒特征")

        with torch.no_grad():
            features = self.model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device))
            features = features.cpu().numpy()

        # 新模型直接输出1024维特征，不需要扩展
        # 确保特征维度正确
        if len(features) != 1024:
            print(f"警告：模型输出特征维度为{len(features)}，期望1024维")
            # 如果维度不对，进行适当调整
            if len(features) < 1024:
                features = np.tile(features, (1024 // len(features) + 1,))
            features = features[:1024]
        
        return features
    
    def load_copyright_image(self, image_path=None):
        """加载版权图像"""
        try:
            if image_path is None:
                # 使用 ZeroWatermark/Cat32.png（脚本相对路径）
                image_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'ZeroWatermark', 'Cat32.png'))
            image = Image.open(image_path)
            image = image.convert('L')
            image = image.resize((32, 32))  # 调整大小为32x32
            threshold = 128
            image = image.point(lambda x: 0 if x < threshold else 255, '1')
            return np.array(image)
        except Exception as e:
            print(f"加载版权图像失败: {e}")
            return np.random.randint(0, 2, (32, 32))
    
    def features_to_matrix(self, features, target_shape):
        """将特征向量转换为矩阵（使用中位数阈值二值化）"""
        total_elements = target_shape[0] * target_shape[1]
        
        # 如果特征数量不足，重复填充
        if len(features) < total_elements:
            features = np.tile(features, (total_elements // len(features) + 1,))
        
        # 取前total_elements个元素
        features = features[:total_elements]
        
        # 重塑为目标形状
        matrix = features.reshape(target_shape)
        
        # 二值化（中位数阈值与第4步保持一致）
        threshold = np.median(matrix)
        matrix = (matrix > threshold).astype(np.uint8)
        
        return matrix
    
    def verify_copyright(self, graph_data, zero_watermark, original_copyright):
        """验证版权"""
        # 提取鲁棒特征
        robust_features = self.extract_robust_features(graph_data)
        
        # 将特征转换为矩阵
        feature_matrix = self.features_to_matrix(robust_features, original_copyright.shape)
        
        # 从零水印中提取版权图像
        extracted_copyright = np.logical_xor(zero_watermark, feature_matrix).astype(np.uint8)
        
        # 计算NC值（归一化相关系数）
        nc_value = self.calculate_nc(original_copyright, extracted_copyright)
        
        return extracted_copyright, nc_value
    
    def calculate_nc(self, original, extracted):
        """计算归一化相关系数（NC值）"""
        # 将图像转换为向量
        original_vec = original.flatten().astype(float)
        extracted_vec = extracted.flatten().astype(float)
        
        # 计算归一化相关系数
        # NC = (A·B) / (||A||·||B||)
        # 其中 A·B 是点积，||A|| 和 ||B|| 是向量的模长
        dot_product = np.sum(original_vec * extracted_vec)
        norm_original = np.sqrt(np.sum(original_vec ** 2))
        norm_extracted = np.sqrt(np.sum(extracted_vec ** 2))
        
        if norm_original == 0 or norm_extracted == 0:
            return 0.0
        
        nc = dot_product / (norm_original * norm_extracted)
        
        return nc
    
    def load_watermark(self, filename, watermark_dir=None):
        """加载对应的零水印"""
        if watermark_dir is None:
            # ZeroWatermark/ZeroWatermark/TrainingSet 下
            watermark_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'ZeroWatermark', 'ZeroWatermark', 'TrainingSet'))
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
        
        # 对每个被攻击的图进行验证
        for i, attacked_graph in enumerate(attacked_graphs):
            # 验证版权
            extracted_copyright, nc_value = self.verify_copyright(
                attacked_graph, zero_watermark, copyright_image
            )
            
            # 获取攻击类型描述
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
    
    def save_robustness_results(self, results, zero_watermark, copyright_image, filename, results_dir=None):
        """保存鲁棒性验证结果"""
        if results_dir is None:
            # 训练集结果目录：zNC-Test/NC-Results/TrainingSet
            results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'NC-Results', 'TrainingSet'))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 计算统计信息
        nc_values = [r['nc_value'] for r in results]
        avg_nc = np.mean(nc_values)
        max_nc = np.max(nc_values)
        min_nc = np.min(nc_values)
        
        # 保存结果图像
        plt.figure(figsize=(15, 10))
        
        # 第一行：原始图像
        plt.subplot(3, 4, 1)
        plt.imshow(zero_watermark, cmap='gray')
        plt.title('Zero Watermark')
        plt.axis('off')
        
        plt.subplot(3, 4, 2)
        plt.imshow(copyright_image, cmap='gray')
        plt.title('Original Copyright')
        plt.axis('off')
        
        # 第二行：前6个攻击结果
        for i in range(min(6, len(results))):
            plt.subplot(3, 4, i + 7)
            plt.imshow(results[i]['extracted_copyright'], cmap='gray')
            plt.title(f'Attack {i+1}\nNC: {results[i]["nc_value"]:.4f}')
            plt.axis('off')
        
        # 第三行：统计信息
        plt.subplot(3, 4, 11)
        plt.text(0.5, 0.5, f'Average NC: {avg_nc:.4f}\nMax NC: {max_nc:.4f}\nMin NC: {min_nc:.4f}', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=10)
        plt.title('Statistics')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{filename}_robustness_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存统计信息到文件
        stats_path = os.path.join(results_dir, f'{filename}_robustness_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"鲁棒性验证结果: {filename}\n")
            f.write(f"平均NC值: {avg_nc:.4f}\n")
            f.write(f"最高NC值: {max_nc:.4f}\n")
            f.write(f"最低NC值: {min_nc:.4f}\n")
            f.write(f"验证成功数量 (NC > 0.7): {sum(1 for nc in nc_values if nc > 0.7)}/{len(nc_values)}\n")
            f.write("\n详细结果:\n")
            for i, result in enumerate(results):
                f.write(f"攻击 {i+1}: NC值 = {result['nc_value']:.4f}\n")
        
        print(f"鲁棒性验证结果已保存: {results_dir}")
        return avg_nc, max_nc, min_nc
    
    def save_excel_results(self, all_results, results_dir=None):
        """只保存summary.csv（统计汇总）"""
        if results_dir is None:
            results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'NC-Results', 'TrainingSet'))
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
        csv_summary_path = os.path.join(results_dir, f'train_set_nc_summary_{timestamp}.csv')
        df_summary.to_csv(csv_summary_path, index=False, encoding='utf-8-sig')
        print(f"CSV统计汇总已保存: {csv_summary_path}")
        
        return csv_summary_path

class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, graph_dir=None):
        # 训练集图数据根目录（使用绝对路径）
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
        
        # 加载所有原始图（训练集）
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
        
        # 检查是否有对应的子目录
        subdir_path = os.path.join(attacked_dir, original_filename)
        if os.path.exists(subdir_path):
            # 如果存在子目录，从子目录中加载所有被攻击图
            attacked_graphs = []
            attacked_filenames = []
            for filename in sorted(os.listdir(subdir_path)):  # 排序以保持一致性
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
    print("="*70)
    print("第五步：鲁棒性验证（改进版）")
    print("="*70)
    print("使用改进的GAT模型（20维特征 + ImprovedGATModel）")
    print("模型配置：input_dim=20, hidden_dim=256, num_heads=8")
    print("="*70)
    print()
    
    # 清理并准备结果输出目录（确保每次运行可完美替换）
    results_root = os.path.normpath(os.path.join(os.path.dirname(__file__), 'NC-Results', 'TrainingSet'))
    if os.path.exists(results_root):
        print(f"清理旧的结果目录: {results_root}")
        try:
            shutil.rmtree(results_root)
            print("[OK] 旧结果已清理")
        except Exception as e:
            print(f"[WARNING] 清理目录时出错: {e}")
            print("尝试继续...")
    os.makedirs(results_root, exist_ok=True)
    print()
    
    # 加载训练集的原始图数据
    data_loader = GraphDataLoader()
    original_graphs, filenames = data_loader.load_original_graphs_train_only()
    
    if not original_graphs:
        print("没有找到训练图数据，请先运行第二步")
        return
    
    # 创建鲁棒性验证器
    verifier = RobustnessVerifier()
    
    # 加载版权图像
    copyright_image = verifier.load_copyright_image()
    print(f"版权图像大小: {copyright_image.shape}")
    
    # 为每个训练图验证鲁棒性
    all_avg_nc = []
    all_max_nc = []
    all_min_nc = []
    all_excel_results = {}  # 收集Excel数据
    
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

        
        # 按攻击类型统计（从Excel结果中提取）
        if all_excel_results:
            print(f"\n按攻击类型统计:")
            print("-" * 50)
            
            # 收集所有攻击类型的NC值
            attack_nc_values = {}
            for filename, results in all_excel_results.items():
                for result in results:
                    attack_desc = result.get('attack_description', '未知攻击')
                    nc_value = result.get('nc_value', 0)
                    if attack_desc not in attack_nc_values:
                        attack_nc_values[attack_desc] = []
                    attack_nc_values[attack_desc].append(nc_value)
            
            # 计算每种攻击类型的统计
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
        print(f"训练集NC值验证总体结果")
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
        
        # 验证成功数量统计
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