#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第六步：测试集NC值验证脚本
测试graph_data/test下所有测试集的NC值
每个测试图都使用对应的零水印进行验证
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

class TestSetNCVerifier:
    """测试集NC值验证器"""
    
    def __init__(self, model_path=None, use_trained_model=True):
        # 模型路径改为VGAT/models/gat_model_best.pth
        if model_path is None:
            model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'VGAT', 'models', 'gat_model_best.pth'))
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model = None
        self.use_trained_model = use_trained_model
        if use_trained_model:
            self.load_model()
        
        # 初始化攻击类型映射字典（基于test100.py）
        self.attack_type_mapping = self._create_attack_type_mapping()
    
    def _create_attack_type_mapping(self):
        """创建攻击类型映射字典（基于test100.py）"""
        mapping = {}
        
        # 基础攻击类型（前50个，基于test100.py）
        basic_attacks = {
            "test_del_vertices_7pct": "随机删除 7% 顶点",
            "test_del_vertices_15pct": "随机删除 15% 顶点",
            "test_del_vertices_28pct": "随机删除 28% 顶点",
            "test_del_vertices_42pct": "随机删除 42% 顶点",
            "test_del_vertices_55pct": "随机删除 55% 顶点",
            "test_del_objects_8pct": "删除 8% 图形对象",
            "test_del_objects_18pct": "删除 18% 图形对象",
            "test_del_objects_33pct": "删除 33% 图形对象",
            "test_del_objects_46pct": "删除 46% 图形对象",
            "test_del_objects_60pct": "删除 60% 图形对象",
            "test_add_vertices_12pct": "添加 12% 顶点",
            "test_add_vertices_25pct": "添加 25% 顶点",
            "test_add_vertices_38pct": "添加 38% 顶点",
            "test_add_vertices_47pct": "添加 47% 顶点",
            "test_add_vertices_65pct": "添加 65% 顶点",
            "test_noise_vertices_8pct_0.25": "扰动 8% 顶点，强度 0.25",
            "test_noise_vertices_14pct_0.45": "扰动 14% 顶点，强度 0.45",
            "test_noise_vertices_21pct_0.55": "扰动 21% 顶点，强度 0.55",
            "test_noise_vertices_26pct_0.75": "扰动 26% 顶点，强度 0.75",
            "test_noise_vertices_33pct_0.85": "扰动 33% 顶点，强度 0.85",
            "test_crop_x_40pct": "沿 X 轴裁剪 40%",
            "test_crop_y_35pct": "沿 Y 轴裁剪 35%",
            "test_crop_top_25pct": "裁剪上部 25% 区域",
            "test_crop_bottom_20pct": "裁剪下部 20% 区域",
            "test_crop_random_30pct": "随机裁剪 30%",
            "test_translate_5_5": "平移 (5, 5)",
            "test_translate_15_-10": "平移 (15, -10)",
            "test_translate_-20_8": "平移 (-20, 8)",
            "test_translate_30_25": "平移 (30, 25)",
            "test_translate_-12_-12": "平移 (-12, -12)",
            "test_scale_0.65": "缩放 0.65 倍",
            "test_scale_1.5": "缩放 1.5 倍",
            "test_scale_x0.8_y1.4": "X 轴缩放 0.8，Y 轴缩放 1.4",
            "test_scale_x1.6_y0.7": "X 轴缩放 1.6，Y 轴缩放 0.7",
            "test_scale_random_0.4-2.5": "随机缩放（0.4–2.5）",
            "test_rotate_30": "旋转 30°",
            "test_rotate_75": "旋转 75°",
            "test_rotate_120": "旋转 120°",
            "test_rotate_225": "旋转 225°",
            "test_rotate_random": "随机旋转（0–360°）",
            "test_flip_x": "X 轴翻转",
            "test_flip_y": "Y 轴翻转",
            "test_flip_xy": "X、Y 轴同时翻转",
            "test_reverse_vertices": "反转顶点顺序",
            "test_reverse_objects": "反转对象顺序",
            "test_shuffle_objects": "打乱对象顺序",
            "test_shuffle_vertices": "打乱顶点顺序",
            "test_jitter_vertices_small": "小幅扰动顶点位置",
            "test_merge_objects_random": "随机合并对象",
            "test_split_objects_random": "随机拆分对象",
        }
        
        # 添加基础攻击类型
        for attack_key, description in basic_attacks.items():
            mapping[attack_key] = description
        
        # 添加组合攻击类型（test_combo_attack_001到test_combo_attack_050）
        for i in range(1, 51):
            mapping[f"test_combo_attack_{i:03d}"] = f"组合攻击策略 {i}"
        
        return mapping
    
    def get_attack_type_description(self, filename):
        """根据文件名获取攻击类型描述"""
        # 移除文件扩展名和图名前缀
        base_name = filename.replace('_graph.pkl', '').replace('.geojson', '')
        
        # 移除图名前缀（如Boundary_、Building_等）和_test后缀
        for prefix in ['Boundary_test_', 'Building_test_', 'Lake_test_', 'Landuse_test_', 'Railways_test_', 'Road_test_']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        
        # 移除_test后缀
        if base_name.endswith('_test'):
            base_name = base_name[:-5]
        
        # 查找匹配的攻击类型
        for attack_key, description in self.attack_type_mapping.items():
            if attack_key in base_name:
                return description
        
        # 如果没有找到匹配的，返回文件名
        return base_name
    
    def load_model(self):
        """加载训练好的模型（强制，失败即报错）"""
        try:
            print("加载训练好的GAT模型...")
            if os.path.exists(self.model_path):
                import sys
                base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
                if base_dir not in sys.path:
                    sys.path.insert(0, base_dir)
                from VGAT.VGAT import GATModel
                # 训练时输入特征维度为13
                self.model = GATModel(input_dim=13, hidden_dim=128, output_dim=1024, num_heads=4, dropout=0.2)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print("模型加载完成")
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

        # 确保特征是一维向量
        if features.ndim > 1:
            features = features.flatten()
        
        # 新模型应该直接输出1024维特征，但为了兼容性，我们确保维度正确
        if len(features) != 1024:
            print(f"警告：模型输出特征维度为{len(features)}，期望1024维")
            # 如果维度不对，进行适当调整
            if len(features) < 1024:
                # 使用重复填充到1024维
                repeat_times = (1024 + len(features) - 1) // len(features)  # 向上取整
                features = np.tile(features, repeat_times)
            # 取前1024维
            features = features[:1024]
        
        return features
    
    def load_copyright_image(self, image_path=None):
        """加载版权图像"""
        try:
            if image_path is None:
                image_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'ZeroWatermark', 'Cat32.png'))
            image = Image.open(image_path)
            image = image.convert('L')
            image = image.resize((32, 32))  # 调整大小为32x32
            threshold = 128
            image = image.point(lambda x: 0 if x < threshold else 255, '1')
            return np.array(image)
        except Exception as e:
            print(f"加载版权图像失败: {e}")
            # 创建一个默认的版权图像
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
        """加载零水印"""
        if watermark_dir is None:
            # ZeroWatermark/ZeroWatermark/TestSet 下
            watermark_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'ZeroWatermark', 'ZeroWatermark', 'TestSet'))
        watermark_path = os.path.join(watermark_dir, f"{filename}_watermark.npy")
        if os.path.exists(watermark_path):
            try:
                watermark = np.load(watermark_path)
                return watermark
            except Exception as e:
                print(f"加载零水印失败 {watermark_path}: {e}")
                return None
        else:
            print(f"零水印文件不存在: {watermark_path}")
            return None
    
    def verify_test_graph(self, test_graph, zero_watermark, copyright_image, filename):
        """验证单个测试图"""
        try:
            extracted_copyright, nc_value = self.verify_copyright(test_graph, zero_watermark, copyright_image)
            
            # 获取攻击类型描述
            attack_description = self.get_attack_type_description(filename)
            
            return {
                'filename': filename,
                'attack_description': attack_description,
                'nc_value': nc_value,
                'extracted_matrix': extracted_copyright
            }
        except Exception as e:
            print(f"验证测试图失败 {filename}: {e}")
            return None
    
    def save_test_results(self, results, zero_watermark, copyright_image, filename, results_dir=None):
        """保存测试结果"""
        if not results:
            return 0.0, 0.0, 0.0
        
        # 确保结果目录存在
        if results_dir is None:
            # 测试集结果目录：zNC-Test/NC-Results/TestSet
            results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'NC-Results', 'TestSet'))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 提取NC值
        nc_values = [result['nc_value'] for result in results if result is not None]
        
        if not nc_values:
            return 0.0, 0.0, 0.0
        
        avg_nc = np.mean(nc_values)
        max_nc = np.max(nc_values)
        min_nc = np.min(nc_values)
        
        # 创建结果图像
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'测试集NC值验证结果 - {filename}', fontsize=16)
        
        # 原始版权图像
        axes[0, 0].imshow(copyright_image, cmap='gray')
        axes[0, 0].set_title('原始版权图像')
        axes[0, 0].axis('off')
        
        # 零水印
        axes[0, 1].imshow(zero_watermark, cmap='gray')
        axes[0, 1].set_title('零水印')
        axes[0, 1].axis('off')
        
        # NC值分布
        axes[0, 2].hist(nc_values, bins=20, alpha=0.7, color='blue')
        axes[0, 2].set_title(f'NC值分布\n平均: {avg_nc:.4f}')
        axes[0, 2].set_xlabel('NC值')
        axes[0, 2].set_ylabel('频次')
        
        # 提取的版权图像（选择第一个结果）
        if results and results[0]:
            axes[1, 0].imshow(results[0]['extracted_matrix'], cmap='gray')
            axes[1, 0].set_title(f'提取的版权图像\nNC: {results[0]["nc_value"]:.4f}')
            axes[1, 0].axis('off')
        
        # NC值统计
        axes[1, 1].text(0.1, 0.8, f'测试图数量: {len(nc_values)}', fontsize=12)
        axes[1, 1].text(0.1, 0.7, f'平均NC值: {avg_nc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'最大NC值: {max_nc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'最小NC值: {min_nc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'标准差: {np.std(nc_values):.4f}', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('NC值统计')
        axes[1, 1].axis('off')
        
        # NC值曲线
        axes[1, 2].plot(range(len(nc_values)), nc_values, 'b-o', alpha=0.7)
        axes[1, 2].axhline(y=avg_nc, color='r', linestyle='--', alpha=0.7, label=f'平均值: {avg_nc:.4f}')
        axes[1, 2].set_title('NC值变化曲线')
        axes[1, 2].set_xlabel('测试图索引')
        axes[1, 2].set_ylabel('NC值')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        image_path = os.path.join(results_dir, f"{filename}_test_nc_results.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存文本结果
        text_path = os.path.join(results_dir, f"{filename}_test_nc_results.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"测试集NC值验证结果 - {filename}\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试图数量: {len(nc_values)}\n")
            f.write(f"平均NC值: {avg_nc:.4f}\n")
            f.write(f"最大NC值: {max_nc:.4f}\n")
            f.write(f"最小NC值: {min_nc:.4f}\n")
            f.write(f"标准差: {np.std(nc_values):.4f}\n")
            f.write("\n详细NC值:\n")
            for i, nc in enumerate(nc_values):
                f.write(f"测试图 {i+1}: {nc:.4f}\n")
        
        print(f"结果已保存: {image_path}")
        print(f"文本结果已保存: {text_path}")
        
        return avg_nc, max_nc, min_nc
    
    def save_excel_results(self, all_results, results_dir=None):
        """保存Excel结果表格"""
        if results_dir is None:
            results_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'NC-Results', 'TestSet'))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 准备Excel数据
        excel_data = []
        
        for graph_name, results in all_results.items():
            for result in results:
                excel_data.append({
                    '图名称': graph_name,
                    '攻击类型': result['attack_description'],
                    'NC值': result['nc_value']
                })
        
        # 创建DataFrame并保存为Excel
        df = pd.DataFrame(excel_data)
        # 添加时间戳后缀（年月日时分秒格式）
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        excel_path = os.path.join(results_dir, f'test_set_nc_results_{timestamp}.xlsx')
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        print(f"Excel结果已保存: {excel_path}")
        return excel_path

class TestSetGraphDataLoader:
    """测试集图数据加载器"""
    
    def __init__(self, graph_dir=None):
        # 测试集图数据根目录
        if graph_dir is None:
            graph_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'convertToGraph', 'Graph', 'TestSet'))
        self.graph_dir = graph_dir
    
    def load_all_test_graphs(self):
        """加载所有测试集图数据（从 Attacked/<原文件名> 子目录加载）"""
        test_dir = os.path.join(self.graph_dir, 'Attacked')
        
        if not os.path.exists(test_dir):
            print(f"测试集目录不存在: {test_dir}")
            return {}, {}
        
        all_test_graphs = {}
        all_filenames = {}
        
        # 遍历所有测试集子目录
        for subdir in os.listdir(test_dir):
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                print(f"处理测试集目录: {subdir}")
                
                graphs = []
                filenames = []
                
                # 加载该子目录下的所有图数据
                for filename in os.listdir(subdir_path):
                    if filename.endswith('_graph.pkl'):
                        try:
                            with open(os.path.join(subdir_path, filename), 'rb') as f:
                                graph_data = pickle.load(f)
                                graphs.append(graph_data)
                                base_name = filename.replace('_graph.pkl', '')
                                filenames.append(base_name)
                                print(f"成功加载测试图数据: {filename}")
                        except Exception as e:
                            print(f"加载测试图数据失败 {filename}: {e}")
                            continue
                
                all_test_graphs[subdir] = graphs
                all_filenames[subdir] = filenames
                
                print(f"目录 {subdir} 总共加载了 {len(graphs)} 个测试图数据")
        
        return all_test_graphs, all_filenames

def main():
    """主函数"""
    print("=== 第六步：测试集NC值验证 ===")
    
    # 清理并准备结果输出目录（确保每次运行可完美替换）
    results_root = os.path.normpath(os.path.join(os.path.dirname(__file__), 'NC-Results', 'TestSet'))
    if os.path.exists(results_root):
        shutil.rmtree(results_root)
    os.makedirs(results_root, exist_ok=True)
    
    # 加载所有测试集图数据
    data_loader = TestSetGraphDataLoader()
    all_test_graphs, all_filenames = data_loader.load_all_test_graphs()
    
    if not all_test_graphs:
        print("没有找到测试集图数据，请先运行第二步")
        return
    
    # 创建测试集NC值验证器
    verifier = TestSetNCVerifier()
    
    # 加载版权图像
    copyright_image = verifier.load_copyright_image()
    print(f"版权图像大小: {copyright_image.shape}")
    
    # 为每个测试集验证NC值
    all_results = {}
    all_excel_results = {}  # 收集Excel数据
    
    for subdir, test_graphs in all_test_graphs.items():
        print(f"\n处理测试集: {subdir}")
        
        if not test_graphs:
            print(f"跳过 {subdir}，没有测试图数据")
            continue
        
        # 获取对应的原始图名称（去掉_test后缀）
        original_name = subdir.replace('_test', '')
        
        # 加载对应的零水印
        zero_watermark = verifier.load_watermark(original_name)
        if zero_watermark is None:
            print(f"跳过 {subdir}，零水印不存在")
            continue
        
        # 验证所有测试图
        results = []
        for i, test_graph in enumerate(test_graphs):
            filename = all_filenames[subdir][i] if i < len(all_filenames[subdir]) else f"test_{i}"
            result = verifier.verify_test_graph(test_graph, zero_watermark, copyright_image, filename)
            if result:
                results.append(result)
        
        if results:
            # 收集Excel数据
            all_excel_results[original_name] = results
            
            # 保存结果
            avg_nc, max_nc, min_nc = verifier.save_test_results(
                results, zero_watermark, copyright_image, original_name
            )
            
            all_results[subdir] = {
                'avg_nc': avg_nc,
                'max_nc': max_nc,
                'min_nc': min_nc,
                'count': len(results)
            }
    
    # 输出总体结果
    if all_results:
        print(f"\n=== 测试集NC值验证总体结果 ===")
        print(f"处理的测试集数量: {len(all_results)}")
        
        # 按原始地图统计
        print(f"\n按原始地图统计:")
        print("-" * 50)
        for subdir, result in all_results.items():
            original_name = subdir.replace('_test', '')
            success_status = "成功" if result['avg_nc'] > 0.7 else "失败"
            
            print(f"{original_name}:")
            print(f"  测试图数量: {result['count']}")
            print(f"  平均NC值: {result['avg_nc']:.4f}")
            print(f"  最大NC值: {result['max_nc']:.4f}")
            print(f"  最小NC值: {result['min_nc']:.4f}")
            print(f"  验证状态: {success_status}")
        
        # 总体统计
        all_avg_nc = [result['avg_nc'] for result in all_results.values()]
        all_max_nc = [result['max_nc'] for result in all_results.values()]
        all_min_nc = [result['min_nc'] for result in all_results.values()]
        
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
        success_rate = (success_count / len(all_results)) * 100
        
        print(f"\n成功率统计:")
        print("-" * 50)
        print(f"总验证数量: {len(all_results)}")
        print(f"成功验证数量: {success_count}")
        print(f"总体成功率: {success_rate:.2f}%")
        
        # 按攻击类型统计（从Excel结果中提取）
        if all_excel_results:
            print(f"\n按攻击类型统计:")
            print("-" * 50)
            
            # 收集所有攻击类型的NC值
            attack_nc_values = {}
            for original_name, results in all_excel_results.items():
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
        
        # 验证成功数量统计
        print(f"\n验证成功统计:")
        print("-" * 50)
        print(f"地图验证成功数量: {success_count}/{len(all_results)}")
        
        # 生成Excel表格
        if all_excel_results:
            verifier.save_excel_results(all_excel_results)
    
    print("测试集NC值验证完成！")

if __name__ == "__main__":
    main() 