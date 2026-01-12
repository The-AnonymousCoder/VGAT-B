#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四步：零水印生成和验证
使用训练好的GAT模型提取鲁棒特征，生成零水印并进行版权验证
为每个原始矢量地图生成对应的零水印
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ✅ 强制UTF-8输出编码（解决PowerShell显示问题）
if sys.platform == 'win32':
    try:
        # 设置标准输出为UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6及以下版本
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ✅ 禁用输出缓冲，确保实时显示
os.environ['PYTHONUNBUFFERED'] = '1'

# 设置中文字体 - 适配Windows环境
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'Songti SC']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用默认字体
    plt.rcParams['font.family'] = 'DejaVu Sans'

class WatermarkGenerator:
    """零水印生成器"""
    
    def __init__(self, model_path=None):
        # 模型路径改为VGAT/models下，使用IMPROVED版本的最佳模型
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'VGAT', 'models', 'gat_model_IMPROVED_best.pth')
        self.model_path = os.path.normpath(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
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
            print(f"模型加载失败: {str(e)}")
            print(f"详细错误信息: {type(e).__name__}: {e}")
            import traceback
            print(f"完整错误堆栈:")
            traceback.print_exc()
            raise
    
    def extract_robust_features(self, graph_data):
        """提取鲁棒特征（大图直接抛出异常）"""
        if not hasattr(graph_data, 'x') or not hasattr(graph_data, 'edge_index'):
            raise ValueError("输入图数据无效，缺少必要属性 'x' 或 'edge_index'")

        if self.model is None:
            raise RuntimeError("模型未加载，无法提取鲁棒特征")

        # 使用训练好的GAT模型提取特征
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
                image_path = os.path.join(os.path.dirname(__file__), 'Cat32.png')
            image_path = os.path.normpath(image_path)
            image = Image.open(image_path)
            # 转换为二值图像
            image = image.convert('L')  # 转为灰度图
            image = image.resize((32, 32))  # 调整大小为32x32
            # 二值化
            threshold = 128
            image = image.point(lambda x: 0 if x < threshold else 255, '1')
            return np.array(image)
        except Exception as e:
            print(f"加载版权图像失败: {e}")
            # 创建一个简单的测试图像
            return np.random.randint(0, 2, (32, 32))
    
    def generate_zero_watermark(self, graph_data, copyright_image):
        """生成零水印"""
        print("生成零水印...")
        
        # 提取鲁棒特征
        robust_features = self.extract_robust_features(graph_data)
        
        # 将特征转换为与版权图像相同大小的矩阵
        # 使用中位数阈值以提升区分度稳定性
        feature_matrix = self.features_to_matrix(robust_features, copyright_image.shape, use_median_threshold=True)
        
        # 生成零水印（特征矩阵与版权图像的异或操作）
        zero_watermark = np.logical_xor(feature_matrix, copyright_image).astype(np.uint8)
        
        return zero_watermark, robust_features
    
    def features_to_matrix(self, features, target_shape, use_median_threshold=False):
        """将特征向量转换为矩阵"""
        # 将特征向量重塑为目标形状
        total_elements = target_shape[0] * target_shape[1]
        
        # 如果特征数量不足，重复填充
        if len(features) < total_elements:
            features = np.tile(features, (total_elements // len(features) + 1,))
        
        # 取前total_elements个元素
        features = features[:total_elements]
        
        # 重塑为目标形状
        matrix = features.reshape(target_shape)
        
        # 二值化
        threshold = np.median(matrix) if use_median_threshold else np.mean(matrix)
        matrix = (matrix > threshold).astype(np.uint8)
        
        return matrix
    
    def verify_copyright(self, graph_data, zero_watermark, original_copyright):
        """验证版权"""
        print("验证版权...")
        
        # 提取鲁棒特征
        robust_features = self.extract_robust_features(graph_data)
        
        # 将特征转换为矩阵（与生成阶段保持相同的二值化策略）
        feature_matrix = self.features_to_matrix(robust_features, original_copyright.shape, use_median_threshold=True)
        
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
    
    def save_watermark(self, zero_watermark, filename, watermark_dir=None):
        """保存零水印"""
        if watermark_dir is None:
            watermark_dir = os.path.join(os.path.dirname(__file__), 'ZeroWatermark', 'TrainingSet')
        watermark_dir = os.path.normpath(watermark_dir)
        if not os.path.exists(watermark_dir):
            os.makedirs(watermark_dir)
        
        # 保存零水印为numpy数组
        watermark_path = os.path.join(watermark_dir, f"{filename}_watermark.npy")
        np.save(watermark_path, zero_watermark)
        
        # 保存零水印图像
        image_path = os.path.join(watermark_dir, f"{filename}_watermark.png")
        watermark_image = Image.fromarray((zero_watermark * 255).astype(np.uint8))
        watermark_image.save(image_path)
        
        print(f"零水印已保存: {watermark_path}")
        return watermark_path
    
    def save_results(self, zero_watermark, extracted_copyright, nc_value, filename, results_dir=None):
        """保存结果"""
        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(__file__), 'ZeroWatermark', 'TrainingSet')
        results_dir = os.path.normpath(results_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 保存零水印
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(zero_watermark, cmap='gray')
        plt.title('Zero Watermark')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(extracted_copyright, cmap='gray')
        plt.title('Extracted Copyright')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, f'NC Value: {nc_value:.4f}', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Copyright Verification Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{filename}_watermark_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存提取的版权图像
        extracted_image = Image.fromarray(extracted_copyright * 255)
        extracted_image.save(os.path.join(results_dir, f'{filename}_Cat32_Extract.png'))
        
        print(f"结果已保存到: {results_dir}")
        print(f"NC值: {nc_value:.4f}")

class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, graph_dir=None):
        if graph_dir is None:
            # 使用绝对路径，确保正确找到图数据目录
            base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
            graph_dir = os.path.join(base_dir, 'convertToGraph', 'Graph', 'TrainingSet')
        self.graph_dir = os.path.normpath(graph_dir)
        print(f"图数据加载路径: {self.graph_dir}")
    
    def load_all_original_graphs(self):
        """加载所有原始图数据"""
        original_dir = os.path.join(self.graph_dir, 'Original')
        
        # 检查目录是否存在
        if not os.path.exists(original_dir):
            print(f"目录不存在: {original_dir}")
            return [], []
        
        graphs = []
        filenames = []
        
        # 加载所有原始图
        for filename in os.listdir(original_dir):
            if filename.endswith('_graph.pkl'):
                try:
                    with open(os.path.join(original_dir, filename), 'rb') as f:
                        graph_data = pickle.load(f)
                        graphs.append(graph_data)
                        # 提取文件名（去掉_graph.pkl后缀）
                        base_name = filename.replace('_graph.pkl', '')
                        filenames.append(base_name)
                        print(f"成功加载图数据: {filename}")
                except Exception as e:
                    print(f"加载图数据失败 {filename}: {e}")
                    continue
        
        print(f"总共加载了 {len(graphs)} 个图数据")
        return graphs, filenames

def main():
    """主函数"""
    # ✅ 立即输出测试，确保终端可见
    print("启动零水印生成脚本...", flush=True)
    sys.stdout.flush()
    
    print("="*70)
    print("第四步：零水印生成和验证（改进版）")
    print("="*70)
    print("使用改进的GAT模型（20维特征 + ImprovedGATModel）")
    print("模型配置：input_dim=20, hidden_dim=256, num_heads=8")
    print("="*70)
    print()
    
    # 清理输出目录，确保每次运行可完美替换
    output_root = os.path.normpath(os.path.join(os.path.dirname(__file__), 'ZeroWatermark', 'TrainingSet'))
    if os.path.exists(output_root):
        print(f"清理旧的输出目录: {output_root}")
        import shutil
        try:
            shutil.rmtree(output_root)
            print("[OK] 旧数据已清理")
        except Exception as e:
            print(f"[WARNING] 清理目录时出错: {e}")
            print("尝试继续...")
    os.makedirs(output_root, exist_ok=True)
    print()
    
    # 加载所有原始图数据
    data_loader = GraphDataLoader()
    original_graphs, filenames = data_loader.load_all_original_graphs()
    
    if not original_graphs:
        print("没有找到图数据，请先运行第二步")
        return
    
    # 创建水印生成器
    watermark_generator = WatermarkGenerator()
    
    # 加载版权图像
    copyright_image = watermark_generator.load_copyright_image()
    print(f"版权图像大小: {copyright_image.shape}")
    
    # 为每个原始图生成零水印
    all_nc_values = []
    
    skipped_graphs = []
    MAX_NODES = 30000  # 节点数阈值，超过则跳过（避免CUDA OOM和系统内存溢出）
    
    for i, (graph_data, filename) in enumerate(zip(original_graphs, filenames)):
        print(f"\n处理第 {i+1}/{len(original_graphs)} 个图: {filename}")
        num_nodes = graph_data.x.size(0)
        num_edges = graph_data.edge_index.size(1)
        print(f"  节点数: {num_nodes}, 边数: {num_edges}")
        
        # 🛡️ 预检查：超大图直接跳过
        if num_nodes > MAX_NODES:
            print(f"⚠️ 节点数超过阈值 ({num_nodes} > {MAX_NODES})，跳过此图避免显存/内存溢出")
            skipped_graphs.append(f"{filename} (节点数: {num_nodes})")
            continue
        
        try:
            # 生成零水印
            zero_watermark, robust_features = watermark_generator.generate_zero_watermark(
                graph_data, copyright_image
            )
            print(f"零水印生成完成: {filename}")
            
            # 验证版权
            extracted_copyright, nc_value = watermark_generator.verify_copyright(
                graph_data, zero_watermark, copyright_image
            )
            print(f"版权验证完成: {filename}, NC值: {nc_value:.4f}")
            
            # 保存零水印
            watermark_generator.save_watermark(zero_watermark, filename)
            
            # 保存验证结果
            watermark_generator.save_results(zero_watermark, extracted_copyright, nc_value, filename)
            
            all_nc_values.append(nc_value)
            
        except Exception as e:
            print(f"❌ 处理失败: {filename}")
            print(f"   错误: {str(e)}")
            skipped_graphs.append(filename)
            continue
        
        finally:
            # ✅ 清理CUDA缓存，防止显存溢出
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    # 输出总体结果
    print(f"\n{'='*70}")
    print(f"零水印生成和验证总体结果")
    print(f"{'='*70}")
    print(f"总图数量: {len(original_graphs)}")
    print(f"成功处理: {len(all_nc_values)}")
    if skipped_graphs:
        print(f"跳过的图: {len(skipped_graphs)}")
        for sg in skipped_graphs:
            print(f"  - {sg}")
    print(f"平均NC值: {np.mean(all_nc_values):.4f}")
    print(f"最高NC值: {np.max(all_nc_values):.4f}")
    print(f"最低NC值: {np.min(all_nc_values):.4f}")
    
    # 按不同阈值统计成功率
    success_count_0_7 = sum(1 for nc in all_nc_values if nc > 0.7)
    success_count_0_8 = sum(1 for nc in all_nc_values if nc > 0.8)
    success_count_0_9 = sum(1 for nc in all_nc_values if nc > 0.9)
    
    print(f"\n版权验证成功率统计:")
    print(f"  NC > 0.7: {success_count_0_7}/{len(original_graphs)} ({success_count_0_7/len(original_graphs)*100:.1f}%)")
    print(f"  NC > 0.8: {success_count_0_8}/{len(original_graphs)} ({success_count_0_8/len(original_graphs)*100:.1f}%)")
    print(f"  NC > 0.9: {success_count_0_9}/{len(original_graphs)} ({success_count_0_9/len(original_graphs)*100:.1f}%)")
    
    print(f"\n输出目录: {output_root}")
    print(f"{'='*70}")
    print("[OK] 所有零水印生成和验证完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    main() 