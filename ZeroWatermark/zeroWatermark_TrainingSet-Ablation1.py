#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零水印生成 - 消融实验1（仅节点级特征）
为训练集生成零水印
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

# ✅ 强制UTF-8输出编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.environ['PYTHONUNBUFFERED'] = '1'

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

class WatermarkGenerator_Ablation1:
    """零水印生成器 - 消融实验1（仅节点级特征）"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'VGAT', 'models', 'gat_model_Ablation1_NodeOnly_best.pth')
        self.model_path = os.path.normpath(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载消融实验1模型"""
        try:
            print("加载消融实验1模型（仅节点级特征）...")
            if os.path.exists(self.model_path):
                import importlib.util
                
                base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
                ablation1_path = os.path.join(base_dir, 'VGAT', 'Ablation1_NodeOnly.py')
                
                spec = importlib.util.spec_from_file_location("Ablation1_NodeOnly", ablation1_path)
                ablation1_module = importlib.util.module_from_spec(spec)
                sys.modules['Ablation1_NodeOnly'] = ablation1_module
                spec.loader.exec_module(ablation1_module)
                
                NodeOnlyGATModel = ablation1_module.NodeOnlyGATModel
                
                self.model = NodeOnlyGATModel(input_dim=20, hidden_dim=256, output_dim=1024, num_heads=8, dropout=0.3)
                
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                print("✅ 消融实验1模型加载完成")
                print(f"  节点级特征维度: 10 (dims: [5,6,7,8,9,10,14,15,16,17])")
                print(f"  隐藏维度: 256")
                print(f"  输出维度: 1024")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
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
                repeat_times = (1024 + len(features) - 1) // len(features)
                features = np.tile(features, repeat_times)
            features = features[:1024]
        
        return features
    
    def load_copyright_image(self, image_path=None):
        """加载版权图像"""
        try:
            if image_path is None:
                image_path = os.path.join(os.path.dirname(__file__), 'Cat32.png')
            image_path = os.path.normpath(image_path)
            image = Image.open(image_path)
            image = image.convert('L')
            image = image.resize((32, 32))
            threshold = 128
            image = image.point(lambda x: 0 if x < threshold else 255, '1')
            return np.array(image)
        except Exception as e:
            print(f"加载版权图像失败: {e}")
            return np.random.randint(0, 2, (32, 32))
    
    def generate_zero_watermark(self, graph_data, copyright_image):
        """生成零水印"""
        print("生成零水印...")
        robust_features = self.extract_robust_features(graph_data)
        binary_features = (robust_features > np.median(robust_features)).astype(np.uint8).reshape(32, 32)
        zero_watermark = np.logical_xor(binary_features, copyright_image).astype(np.uint8)
        print("零水印生成完成")
        return zero_watermark
    
    def verify_watermark(self, graph_data, zero_watermark, copyright_image):
        """验证零水印"""
        print("验证零水印...")
        robust_features = self.extract_robust_features(graph_data)
        binary_features = (robust_features > np.median(robust_features)).astype(np.uint8).reshape(32, 32)
        extracted_image = np.logical_xor(zero_watermark, binary_features).astype(np.uint8)
        nc = self.calculate_nc(copyright_image, extracted_image)
        print(f"归一化相关系数 (NC): {nc:.6f}")
        return extracted_image, nc
    
    def calculate_nc(self, img1, img2):
        """计算归一化相关系数"""
        img1_flat = img1.flatten().astype(float)
        img2_flat = img2.flatten().astype(float)
        numerator = np.sum(img1_flat * img2_flat)
        denominator = np.sqrt(np.sum(img1_flat ** 2) * np.sum(img2_flat ** 2))
        if denominator == 0:
            return 0.0
        return numerator / denominator


def main():
    """主函数"""
    print("="*70)
    print("零水印生成 - 消融实验1（仅节点级特征）")
    print("="*70)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_dir = os.path.join(base_dir, '..', 'convertToGraph', 'Graph', 'TrainingSet', 'Original')
    output_dir = os.path.join(base_dir, 'ZeroWatermark', 'TrainingSet-Ablation1')
    
    graph_dir = os.path.normpath(graph_dir)
    output_dir = os.path.normpath(output_dir)
    
    print(f"图结构目录: {graph_dir}")
    print(f"输出目录: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    try:
        generator = WatermarkGenerator_Ablation1()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载版权图像
    copyright_image = generator.load_copyright_image()
    print(f"版权图像形状: {copyright_image.shape}")
    
    # 获取所有图文件
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('_graph.pkl')])
    print(f"找到 {len(graph_files)} 个图文件")
    
    if len(graph_files) == 0:
        print("❌ 未找到图文件")
        return
    
    # 处理每个图
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for graph_file in tqdm(graph_files, desc="生成零水印"):
        try:
            base_name = graph_file.replace('_graph.pkl', '')
            watermark_path = os.path.join(output_dir, f'{base_name}_watermark.npy')
            image_path = os.path.join(output_dir, f'{base_name}_watermark.png')
            
            # 如果已存在则跳过
            if os.path.exists(watermark_path):
                skip_count += 1
                continue
            
            # 加载图数据
            graph_path = os.path.join(graph_dir, graph_file)
            with open(graph_path, 'rb') as f:
                graph_data = pickle.load(f)
            
            # 生成零水印
            zero_watermark = generator.generate_zero_watermark(graph_data, copyright_image)
            
            # 保存
            np.save(watermark_path, zero_watermark)
            watermark_image = Image.fromarray((zero_watermark * 255).astype(np.uint8))
            watermark_image.save(image_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 处理 {graph_file} 失败: {e}")
            error_count += 1
            continue
    
    print("\n" + "="*70)
    print("零水印生成完成（消融实验1）")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {error_count}")
    print("="*70)


if __name__ == '__main__':
    main()

