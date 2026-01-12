#!/usr/bin/env python3  # 指定用 Python3 解释器执行本脚本
# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8，支持中文
"""
第三步：GAT模型训练 - 矢量地图零水印鲁棒特征提取  # 模块说明：训练 GAT 模型以提取鲁棒特征
使用GAT结合对比学习训练模型，提取抵抗RST攻击的鲁棒特征  # 目标：对抗旋转(R)、缩放(S)、平移(T)等攻击
"""

import os  # 与操作系统交互：路径、环境变量等
import pickle  # 序列化/反序列化 Python 对象（用于加载保存的图数据）
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块（层、损失等）
import torch.nn.functional as F  # 常用函数式 API（激活、损失等）
from torch_geometric.nn import GATConv  # 图注意力卷积层（Graph Attention Network）
from torch.cuda import amp  # 自动混合精度（Automatic Mixed Precision）
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示
import json  # 读写 JSON（保存训练历史等）
import logging  # 日志记录
from datetime import datetime  # 时间戳
import glob  # 通配符文件匹配（本文件未使用，但保留以便扩展）

# 设置日志
def setup_logging():
    """设置日志记录（按时间戳+PID生成唯一文件，并维护 latest 文件）"""
    # 将日志输出到VGAT文件夹下
    base_dir = os.path.dirname(__file__)  # 当前文件所在目录
    log_dir = os.path.join(base_dir, "logs")  # 日志目录路径
    if not os.path.exists(log_dir):  # 若不存在则创建
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳（到秒）
    pid = os.getpid()  # 当前进程 ID
    log_file = os.path.join(log_dir, f"step3_training_{timestamp}_{pid}.log")  # 唯一日志文件
    latest_file = os.path.join(log_dir, "step3_training_latest.log")  # 最新日志快照（覆盖写）

    # 清理可能存在的重复 handler（适配重复初始化的场景）
    root_logger = logging.getLogger()  # 获取根 logger
    if root_logger.handlers:  # 如果已存在 handler，先移除避免重复输出
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)

    # 配置两个文件输出：唯一日志文件 + latest 快照，再加控制台输出
    file_handler_unique = logging.FileHandler(log_file, encoding='utf-8')  # 持久化日志
    file_handler_latest = logging.FileHandler(latest_file, mode='w', encoding='utf-8')  # 覆盖写入
    console_handler = logging.StreamHandler()  # 控制台输出

    for h in (file_handler_unique, file_handler_latest, console_handler):
        h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))  # 统一格式

    root_logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
    root_logger.addHandler(file_handler_unique)  # 添加唯一文件 handler
    root_logger.addHandler(file_handler_latest)  # 添加最新快照 handler
    root_logger.addHandler(console_handler)  # 添加控制台 handler

    # 将当前日志路径暴露为模块级变量，便于外部读取或脚本间传递
    globals()["CURRENT_LOG_FILE"] = log_file
    globals()["CURRENT_LATEST_LOG"] = latest_file
    os.environ["VGAT_CURRENT_LOG"] = log_file  # 环境变量暴露
    os.environ["VGAT_CURRENT_LOG_LATEST"] = latest_file

    logger = logging.getLogger(__name__)  # 获取当前模块专属 logger
    logger.info(f"日志文件: {log_file}")  # 打印唯一日志文件路径
    logger.info(f"最新日志(覆盖): {latest_file}")  # 打印最新快照路径
    return logger  # 返回模块级 logger

logger = setup_logging()  # 初始化日志系统并获取 logger 实例

class GATModel(nn.Module):
    """GAT模型用于提取矢量地图的鲁棒特征"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1024, num_heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        
        # GAT层：提取图结构特征（多头注意力）
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)  # 第一层，输出维度=hidden_dim*num_heads
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True)  # 第二层，继续提取高阶邻域
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=False)  # 第三层，concat=False 表示多头求平均
        
        # 特征融合层：将节点级特征聚合为图级特征（mean+max 池化后拼接）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # 输入是2*hidden_dim（mean和max拼接）
            nn.ReLU(),  # 非线性激活
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(hidden_dim * 2, output_dim),  # 投影到最终输出维度（1024）
            nn.Tanh()  # 压到[-1,1]，便于后续二值化处理
        )
        
        self.dropout = nn.Dropout(dropout)  # 中间层的 dropout
    
    def forward(self, x, edge_index):
        """前向传播：输入节点特征 x 和边索引 edge_index，输出图级鲁棒特征"""
        # GAT特征提取：三层 GAT + ELU 激活 + Dropout
        x1 = F.elu(self.gat1(x, edge_index))  # 第一层注意力聚合
        x1 = self.dropout(x1)  # 随机失活
        
        x2 = F.elu(self.gat2(x1, edge_index))  # 第二层注意力聚合
        x2 = self.dropout(x2)
        
        x3 = F.elu(self.gat3(x2, edge_index))  # 第三层注意力聚合
        
        # 全局池化：将节点特征融合为图级特征
        # 使用均值池化 + 最大池化的组合（对节点维做池化）
        mean_pool = torch.mean(x3, dim=0)  # 所有节点的均值
        max_pool, _ = torch.max(x3, dim=0)  # 所有节点的逐维最大值
        graph_features = torch.cat([mean_pool, max_pool], dim=0)  # 拼接成 2*hidden_dim
        
        # 通过融合层得到最终图级特征表示（长度=output_dim）
        output = self.fusion(graph_features)
        
        return output  # 形状：[output_dim]（单个图的特征）

class ContrastiveTrainer:
    """对比学习训练器"""
    
    def __init__(self, model, device='cpu', temperature=0.07, use_amp=True, batch_size=8):
        self.model = model.to(device)  # 将模型移动到指定设备
        self.device = device  # 训练设备（cpu/cuda）
        self.temperature = temperature  # 对比学习温度系数（缩放相似度）
        self.use_amp = use_amp  # 是否使用自动混合精度（节省显存/提速）
        self.batch_size = batch_size  # 训练批大小
        
        # 优化器与学习率调度器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW 带权重衰减
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)  # 余弦退火 LR
        # AMP缩放器（在 use_amp 时启用）
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        
        # 训练历史记录，用于可视化和诊断
        self.training_history = {
            'epoch_losses': [],  # 每个 epoch 的总损失
            'contrastive_losses': [],  # 对比损失
            'similarity_losses': [],  # 相似性损失
            'diversity_losses': [],  # 多样性损失
            'learning_rates': [],  # 学习率变化
            'gradient_norms': [],  # 梯度范数（用来观测训练稳定性）
            'feature_stats': []  # 特征统计（均值/方差/范围）
        }
    
    def contrastive_loss(self, features_original, features_attacked, labels):
        """
        对比学习损失函数
        - 正样本：同一原图的不同攻击版本（应该相似）
        - 负样本：不同原图的任何版本（应该区分）
        """
        features_original = F.normalize(features_original, p=2, dim=1)  # L2 归一化，消除尺度影响
        features_attacked = F.normalize(features_attacked, p=2, dim=1)
        
        sim_matrix = torch.matmul(features_original, features_attacked.T) / self.temperature  # 余弦相似度近似（因已归一化）并按温度缩放
        
        batch_size = features_original.size(0)  # batch 内样本数
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # 构造同类掩码（同一原图为 True）
        
        total_loss = 0  # 累计损失
        valid_samples = 0  # 参与损失计算的有效样本数
        
        for i in range(batch_size):
            # 正样本：同一原图的攻击版本
            positive_mask = labels_matrix[i]  # 与第 i 个样本同标签
            positive_scores = sim_matrix[i][positive_mask]  # 正样本相似度
            
            # 负样本：不同原图的任何版本（包括原图和被攻击版本）
            negative_mask = ~labels_matrix[i]
            negative_scores = sim_matrix[i][negative_mask]  # 负样本相似度
            
            if len(positive_scores) > 0 and len(negative_scores) > 0:
                # 计算对比损失：拼接正负样本分数
                logits = torch.cat([positive_scores, negative_scores])  # [P+N]
                targets = torch.zeros(len(logits), device=self.device)  # 预留（此实现未直接使用）
                targets[:len(positive_scores)] = 1  # 正样本标签为1（说明性）
                
                # 使用InfoNCE风格的损失（此处简化为交叉熵形式）
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0], device=self.device))  # 目标索引为0（对应最大相似度位置）
                total_loss += loss
                valid_samples += 1
        
        return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=self.device)  # 平均损失
    
    def similarity_loss(self, features_original, features_attacked):
        """相似性损失：确保同一原图的攻击版本特征相似"""
        # 计算余弦相似度（逐样本）
        similarity = F.cosine_similarity(features_original, features_attacked, dim=1)
        # 最大化相似度 => 最小化 (1 - similarity)
        return torch.mean(1 - similarity)
    
    def diversity_loss(self, features):
        """多样性损失：防止特征坍塌，确保不同图有不同特征"""
        # 计算特征矩阵的按维方差（跨样本）
        feature_var = torch.var(features, dim=0)
        # 鼓励每个维度都有足够的方差，否则产生惩罚（margin=0.1）
        diversity_loss = torch.mean(torch.relu(0.1 - feature_var))
        return diversity_loss
    
    def train_epoch(self, original_graphs, attacked_graphs, epoch):
        """训练一个epoch"""
        self.model.train()  # 切换到训练模式（启用Dropout等）
        total_loss = 0.0  # 总损失累计
        total_contrastive_loss = 0.0  # 对比损失累计
        total_similarity_loss = 0.0  # 相似性损失累计
        total_diversity_loss = 0.0  # 多样性损失累计
        num_batches = 0  # batch 计数
        
        # 记录梯度范数
        total_grad_norm = 0.0  # 用于监控训练稳定性
        
        # 准备训练数据：将原图与其多种攻击版本配对
        all_pairs = []  # [(orig_graph, attacked_graph), ...]
        all_labels = []  # 同一原图的样本共享同一标签
        
        for i, (graph_name, original_graph) in enumerate(original_graphs.items()):  # i 作为该原图的标签
            if graph_name in attacked_graphs:  # 仅当有被攻击版本时
                for attacked_graph in attacked_graphs[graph_name]:
                    all_pairs.append((original_graph, attacked_graph))  # 形成一对
                    all_labels.append(i)  # 同一原图用相同标签标识
        
        if len(all_pairs) == 0:  # 没有可训练的数据
            logger.warning("没有找到训练数据对")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 随机打乱数据，确保每个batch标签分布多样
        import random
        combined = list(zip(all_pairs, all_labels))  # 打包
        random.shuffle(combined)  # 打乱
        all_pairs, all_labels = zip(*combined)  # 解包
        
        # 统计训练数据信息（日志友好）
        unique_labels = set(all_labels)  # 原图种类数
        label_counts = {}  # 每个原图对应的样本数
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"训练数据统计:")
        logger.info(f"  总样本对数: {len(all_pairs)}")
        logger.info(f"  原图类型数: {len(unique_labels)}")
        logger.info(f"  各原图样本数: {label_counts}")
        
        # 计算正负样本数量（全局统计，用于了解难度）
        total_positive_pairs = sum(count * (count - 1) // 2 for count in label_counts.values())  # 同类对数
        total_negative_pairs = len(all_pairs) * (len(all_pairs) - 1) // 2 - total_positive_pairs  # 异类对数
        logger.info(f"  正样本对数: {total_positive_pairs} (同一原图的不同攻击版本)")
        logger.info(f"  负样本对数: {total_negative_pairs} (不同原图的任何版本)")
        logger.info("")
        
        # 分批训练
        for i in range(0, len(all_pairs), self.batch_size):
            batch_pairs = all_pairs[i:i + self.batch_size]  # 当前 batch 的样本对
            batch_labels = all_labels[i:i + self.batch_size]  # 对应标签
            
            try:
                # 准备batch数据：逐对提取模型输出特征
                batch_original_features = []  # 原图特征集合
                batch_attacked_features = []  # 攻击图特征集合
                
                for original_graph, attacked_graph in batch_pairs:
                    # 移动到设备（Tensor 属性包含 x 和 edge_index）
                    original_graph = original_graph.to(self.device)
                    attacked_graph = attacked_graph.to(self.device)
                    
                    # 提取特征（AMP 自动混合精度以节省显存/提速）
                    with amp.autocast(enabled=self.use_amp):
                        features_original = self.model(original_graph.x, original_graph.edge_index)
                        features_attacked = self.model(attacked_graph.x, attacked_graph.edge_index)
                    
                    batch_original_features.append(features_original)
                    batch_attacked_features.append(features_attacked)
                
                # 堆叠成 [B, D]
                batch_original = torch.stack(batch_original_features)
                batch_attacked = torch.stack(batch_attacked_features)
                batch_labels = torch.tensor(batch_labels, device=self.device)  # [B]
                
                # 计算损失（AMP）
                with amp.autocast(enabled=self.use_amp):
                    contrastive_loss = self.contrastive_loss(batch_original, batch_attacked, batch_labels)  # 区分不同原图
                    similarity_loss = self.similarity_loss(batch_original, batch_attacked)  # 同原图版本靠近
                    diversity_loss = self.diversity_loss(torch.cat([batch_original, batch_attacked], dim=0))  # 防坍塌
                
                # 计算当前batch的正负样本对数（用于日志）
                batch_labels_np = batch_labels.cpu().numpy()
                unique_batch_labels = set(batch_labels_np)  # 当前 batch 的不同原图数量（未直接使用）
                
                # 逐对统计（组合计数）
                batch_positive_pairs = 0
                batch_negative_pairs = 0
                
                for j in range(len(batch_labels_np)):
                    for k in range(j+1, len(batch_labels_np)):
                        if batch_labels_np[j] == batch_labels_np[k]:
                            batch_positive_pairs += 1  # 同一原图
                        else:
                            batch_negative_pairs += 1  # 不同原图
                
                if epoch % 3 == 0:  # 每3个epoch打印一次详细信息
                    logger.info(f"  Batch {i//self.batch_size + 1}: 标签分布={dict(zip(*np.unique(batch_labels_np, return_counts=True)))}, 正样本对={batch_positive_pairs}, 负样本对={batch_negative_pairs}")
                
                # 总损失：多目标加权
                total_batch_loss = contrastive_loss + 0.5 * similarity_loss + 0.1 * diversity_loss
                
                # 反向传播与优化
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(total_batch_loss).backward()  # 缩放梯度以防止下溢
                    # AMP 下先反缩放再裁剪梯度
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪防梯度爆炸
                    total_grad_norm += grad_norm.item()
                    self.scaler.step(self.optimizer)  # 执行优化步
                    self.scaler.update()  # 更新缩放因子
                else:
                    total_batch_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    total_grad_norm += grad_norm.item()
                    self.optimizer.step()
                
                # 累积损失到 epoch 级统计
                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_similarity_loss += similarity_loss.item()
                total_diversity_loss += diversity_loss.item()
                num_batches += 1
                
                # 清理GPU缓存，缓解显存压力
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                # 针对CUDA OOM等运行时错误进行处理：记录并跳过该 batch
                if 'out of memory' in str(e).lower() or 'cuda out of memory' in str(e).lower():
                    logger.error(f"处理batch {i//self.batch_size + 1} 时出错: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"处理batch {i//self.batch_size + 1} 时出错: {e}")
                    continue
            except Exception as e:
                logger.error(f"处理batch {i//self.batch_size + 1} 时出错: {e}")
                continue
        
        # 更新学习率调度器
        self.scheduler.step()
        
        # 计算平均损失和梯度范数（对齐 batch 数）
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0.0
        avg_similarity_loss = total_similarity_loss / num_batches if num_batches > 0 else 0.0
        avg_diversity_loss = total_diversity_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
        
        # 记录训练历史，用于后续绘图与分析
        self.training_history['epoch_losses'].append(avg_loss)
        self.training_history['contrastive_losses'].append(avg_contrastive_loss)
        self.training_history['similarity_losses'].append(avg_similarity_loss)
        self.training_history['diversity_losses'].append(avg_diversity_loss)
        self.training_history['gradient_norms'].append(avg_grad_norm)
        self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss, avg_contrastive_loss, avg_similarity_loss, avg_diversity_loss, avg_grad_norm
    
    def train(self, original_graphs, attacked_graphs, num_epochs=50):
        """训练模型"""
        logger.info(f"开始训练GAT模型（{num_epochs}个epoch）...")
        logger.info("训练目标：提取矢量地图的鲁棒特征，抵抗RST攻击")
        
        best_loss = float('inf')  # 记录最佳（最低）训练损失
        patience = 10  # 早停耐心值（连续多少个 epoch 无提升则停止）
        patience_counter = 0  # 当前耐心计数
        
        for epoch in tqdm(range(num_epochs), desc="训练进度"):
            # 训练一个 epoch 并返回各项统计
            train_loss, contrastive_loss, similarity_loss, diversity_loss, grad_norm = self.train_epoch(original_graphs, attacked_graphs, epoch)
            
            # 早停机制：若损失改善则保存最佳模型并重置计数，否则递增
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                # 保存最佳模型到VGAT文件夹
                model_best_path = os.path.join(os.path.dirname(__file__), 'models', 'gat_model_best.pth')
                self.save_model(model_best_path)
            else:
                patience_counter += 1
            
            # 每3个epoch打印一次训练详情
            if (epoch + 1) % 3 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']  # 读取当前学习率
                logger.info(f"=== Epoch {epoch+1}/{num_epochs} 训练详情 ===")
                logger.info(f"总损失: {train_loss:.6f} (最佳: {best_loss:.6f})")
                logger.info(f"对比损失: {contrastive_loss:.6f}")
                logger.info(f"相似性损失: {similarity_loss:.6f}")
                logger.info(f"多样性损失: {diversity_loss:.6f}")
                logger.info(f"梯度范数: {grad_norm:.6f}")
                logger.info(f"学习率: {current_lr:.8f}")
                logger.info(f"耐心计数: {patience_counter}/{patience}")
                
                # 记录特征统计信息（抽样）
                if hasattr(self, 'model') and self.model is not None:
                    with torch.no_grad():
                        # 随机选择少量图计算特征的均值/方差/范围
                        sample_features = []
                        for graph_name, original_graph in list(original_graphs.items())[:2]:
                            original_graph = original_graph.to(self.device)
                            features = self.model(original_graph.x, original_graph.edge_index)
                            sample_features.append(features.cpu().numpy())
                        
                        if sample_features:
                            sample_features = np.concatenate(sample_features, axis=0)
                            feature_mean = np.mean(sample_features)
                            feature_std = np.std(sample_features)
                            feature_min = np.min(sample_features)
                            feature_max = np.max(sample_features)
                            
                            logger.info(f"特征统计: 均值={feature_mean:.4f}, 标准差={feature_std:.4f}, 范围=[{feature_min:.4f}, {feature_max:.4f}]")
                            
                            # 保存到训练历史
                            self.training_history['feature_stats'].append({
                                'mean': feature_mean,
                                'std': feature_std,
                                'min': feature_min,
                                'max': feature_max
                            })
                
                logger.info("=" * 50)
            
            # 若长期无提升则提前停止
            if patience_counter >= patience:
                logger.warning(f"连续{patience}个epoch没有改善，提前停止训练")
                break
        
        # 保存训练历史（JSON + 曲线图）
        self.save_training_history()
        
        logger.info(f"训练完成！最佳损失值: {best_loss:.6f}")
        return best_loss
    
    def save_model(self, model_path):
        """保存模型"""
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))  # 确保目录存在
        
        torch.save({
            'model_state_dict': self.model.state_dict(),  # 模型权重
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态（可选）
        }, model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        # 将训练历史保存到VGAT文件夹
        history_dir = os.path.join(os.path.dirname(__file__), "logs")  # 使用 logs 目录
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 文件名带时间戳
        history_file = os.path.join(history_dir, f"training_history_{timestamp}.json")
        
        # 转换numpy数组为列表以便JSON序列化
        history_data = {}  # 序列化后的历史
        for key, value in self.training_history.items():
            if key == 'feature_stats':
                history_data[key] = value  # 已是可序列化的列表[dict]
            else:
                # 确保所有数值都转换为Python原生类型，避免 numpy 类型导致的序列化问题
                history_data[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)  # 保存为UTF-8并保留中文
        
        logger.info(f"训练历史已保存到: {history_file}")
        
        # 生成训练曲线图
        self.plot_training_curves(history_data, history_dir, timestamp)
    
    def plot_training_curves(self, history_data, save_dir, timestamp):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt  # 延迟导入，避免无显示环境报错
            
            # 创建 2x3 子图布局
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('GAT模型训练曲线', fontsize=16)
            
            epochs = range(1, len(history_data['epoch_losses']) + 1)  # x 轴：epoch 序号
            
            # 总损失曲线
            axes[0, 0].plot(epochs, history_data['epoch_losses'], 'b-', label='总损失')
            axes[0, 0].set_title('总损失')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 对比损失
            axes[0, 1].plot(epochs, history_data['contrastive_losses'], 'r-', label='对比损失')
            axes[0, 1].set_title('对比损失')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 相似性损失
            axes[0, 2].plot(epochs, history_data['similarity_losses'], 'g-', label='相似性损失')
            axes[0, 2].set_title('相似性损失')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
            
            # 多样性损失
            axes[1, 0].plot(epochs, history_data['diversity_losses'], 'm-', label='多样性损失')
            axes[1, 0].set_title('多样性损失')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 梯度范数
            axes[1, 1].plot(epochs, history_data['gradient_norms'], 'c-', label='梯度范数')
            axes[1, 1].set_title('梯度范数')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            # 学习率
            axes[1, 2].plot(epochs, history_data['learning_rates'], 'y-', label='学习率')
            axes[1, 2].set_title('学习率变化')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            
            # 保存图片到文件
            plot_file = os.path.join(save_dir, f"training_curves_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"训练曲线图已保存到: {plot_file}")
            
        except ImportError:
            logger.warning("matplotlib未安装，跳过训练曲线绘制")
        except Exception as e:
            logger.error(f"绘制训练曲线时出错: {e}")

class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, graph_dir=os.path.join('convertToGraph', 'Graph', 'TrainingSet')):
        self.graph_dir = graph_dir  # 根目录：包含 Original 和 Attacked 子目录
    
    def load_graph_data(self):
        """加载图数据：原始图与其对应的被攻击图"""
        original_dir = os.path.join(self.graph_dir, 'Original')  # 原始图目录
        attacked_dir = os.path.join(self.graph_dir, 'Attacked')  # 攻击图目录
        
        # 加载原始图（name -> Data）
        original_graphs = {}
        if not os.path.exists(original_dir):  # 若原始目录不存在则返回空
            logger.warning(f"原始数据目录不存在: {original_dir}")
            return {}, {}
        for filename in os.listdir(original_dir):
            if filename.endswith('_graph.pkl'):  # 仅读取指定后缀
                graph_name = filename.replace('_graph.pkl', '')  # 去掉后缀作为键
                with open(os.path.join(original_dir, filename), 'rb') as f:
                    graph_data = pickle.load(f)  # 反序列化为 PyG Data 对象
                    original_graphs[graph_name] = graph_data
        
        # 加载被攻击的图（每个子目录对应一个原图名，内含多种攻击版本）
        attacked_graphs = {}
        if os.path.exists(attacked_dir):
            for subdir in os.listdir(attacked_dir):  # 子目录名与原图名对应
                subdir_path = os.path.join(attacked_dir, subdir)
                if os.path.isdir(subdir_path):
                    attacked_graphs[subdir] = []  # 列表存放该原图的所有攻击版本
                    for filename in os.listdir(subdir_path):
                        if filename.endswith('_graph.pkl'):
                            with open(os.path.join(subdir_path, filename), 'rb') as f:
                                graph_data = pickle.load(f)
                                attacked_graphs[subdir].append(graph_data)
        
        logger.info(f"加载了 {len(original_graphs)} 个原始图")
        total_attacked = sum(len(graphs) for graphs in attacked_graphs.values())
        logger.info(f"加载了 {total_attacked} 个被攻击的图")
        
        return original_graphs, attacked_graphs

def main():
    """主函数"""
    logger.info("=== 第三步：GAT模型训练 - 矢量地图零水印鲁棒特征提取 ===")
    
    # 尝试缓解CUDA显存碎片（仅当前进程生效）
    try:
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # 使用可扩展段分配策略
        logger.info(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    except Exception:
        pass  # 环境变量设置失败则忽略

    # 设置设备（首选GPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载数据（原始图与对应的攻击图集合）
    data_loader = GraphDataLoader()
    original_graphs, attacked_graphs = data_loader.load_graph_data()
    
    # 基本检查：必须同时存在原始图与攻击图
    if len(original_graphs) == 0:
        logger.warning("没有找到原始图数据，请先运行第二步")
        return
    
    if len(attacked_graphs) == 0:
        logger.warning("没有找到被攻击的图数据，请先运行第二步")
        return
    
    # 获取输入维度（节点特征维度）
    first_graph = list(original_graphs.values())[0]
    input_dim = first_graph.x.shape[1]
    logger.info(f"输入特征维度: {input_dim}")
    logger.info(f"目标输出维度: 1024 (32x32)")
    
    # 创建模型（3层GAT + 融合层），输出定长特征向量
    model = GATModel(input_dim=input_dim, hidden_dim=128, output_dim=1024, num_heads=4, dropout=0.2)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 读取可选batch大小（环境变量可覆盖，默认8）
    try:
        configured_bs = int(os.environ.get('VGAT_BATCH_SIZE', '8'))
    except Exception:
        configured_bs = 8
    
    # 创建训练器（GPU上启用AMP，缓解显存压力并提速）
    trainer = ContrastiveTrainer(model, device, use_amp=(device=='cuda'), batch_size=configured_bs)
    
    # 训练模型（返回最佳损失）
    train_loss = trainer.train(original_graphs, attacked_graphs, num_epochs=50)
    
    # 保存最终模型到VGAT文件夹（与最佳模型分开保存）
    final_model_path = os.path.join(os.path.dirname(__file__), 'models', 'gat_model.pth')
    trainer.save_model(final_model_path)
    
    logger.info("模型训练完成！")
    logger.info("模型将用于：")
    logger.info("  1. 从原始矢量地图提取鲁棒特征")
    logger.info("  2. 与版权图像结合生成零水印")
    logger.info("  3. 验证阶段提取特征并与零水印结合恢复版权图像")

if __name__ == "__main__":
    main()