#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三步：GCN模型训练 - 矢量地图零水印鲁棒特征提取
使用GCN结合对比学习训练模型，提取抵抗RST攻击的鲁棒特征
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.cuda import amp
import numpy as np
from tqdm import tqdm
import json
import logging
from datetime import datetime
import glob

# 设置日志
def setup_logging():
    """设置日志记录（按时间戳+PID生成唯一文件，并维护 latest 文件）"""
    # 将日志输出到VGCN文件夹下
    base_dir = os.path.dirname(__file__)
    log_dir = os.path.join(base_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_file = os.path.join(log_dir, f"step3_training_{timestamp}_{pid}.log")
    latest_file = os.path.join(log_dir, "step3_training_latest.log")

    # 清理可能存在的重复 handler（适配重复初始化的场景）
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)

    # 配置两个文件输出：唯一日志文件 + latest 快照
    file_handler_unique = logging.FileHandler(log_file, encoding='utf-8')
    file_handler_latest = logging.FileHandler(latest_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()

    for h in (file_handler_unique, file_handler_latest, console_handler):
        h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler_unique)
    root_logger.addHandler(file_handler_latest)
    root_logger.addHandler(console_handler)

    # 将当前日志路径暴露为模块级变量，便于外部读取
    globals()["CURRENT_LOG_FILE"] = log_file
    globals()["CURRENT_LATEST_LOG"] = latest_file
    os.environ["VGCN_CURRENT_LOG"] = log_file
    os.environ["VGCN_CURRENT_LOG_LATEST"] = latest_file

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    logger.info(f"最新日志(覆盖): {latest_file}")
    return logger

logger = setup_logging()

class GCNModel(nn.Module):
    """使用GCN模型提取矢量地图的鲁棒特征"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1024, dropout=0.2):
        super(GCNModel, self).__init__()
        
        # GCN层：提取图结构特征（使用改进的对称归一化）
        self.gcn1 = GCNConv(input_dim, hidden_dim, improved=True, add_self_loops=True)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
        
        # 特征融合层：将节点特征融合为图级特征
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # 输入是2*hidden_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # 输出范围[-1,1]，便于二值化
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # 使用较小的gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index, batch=None, debug=False):
        """前向传播：提取图级鲁棒特征
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引，用于区分不同图的节点 [num_nodes]，如果为None则假设单图
            debug: 是否输出调试信息
        """
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            if debug:
                logger.error(f"❌ 输入特征x包含NaN/Inf！范围=[{x.min():.4f}, {x.max():.4f}]")
            return torch.full((1, 1024), float('nan'), device=x.device)
        
        # GCN特征提取
        x1 = self.gcn1(x, edge_index)
        if torch.isnan(x1).any() or torch.isinf(x1).any():
            if debug:
                logger.error(f"❌ GCN1输出包含NaN/Inf！")
            return torch.full((1, 1024), float('nan'), device=x.device)
        
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.gcn2(x1, edge_index)
        if torch.isnan(x2).any() or torch.isinf(x2).any():
            if debug:
                logger.error(f"❌ GCN2输出包含NaN/Inf！")
            return torch.full((1, 1024), float('nan'), device=x.device)
        
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.gcn3(x2, edge_index)
        if torch.isnan(x3).any() or torch.isinf(x3).any():
            if debug:
                logger.error(f"❌ GCN3输出包含NaN/Inf！")
            return torch.full((1, 1024), float('nan'), device=x.device)
        
        x3 = F.relu(x3)
        
        # 全局池化：将节点特征融合为图级特征
        # 如果没有提供batch，创建一个全0的batch（单图情况）
        if batch is None:
            batch = torch.zeros(x3.size(0), dtype=torch.long, device=x3.device)
        
        # 使用PyG的全局池化函数，正确处理批图
        mean_pool = global_mean_pool(x3, batch)
        max_pool = global_max_pool(x3, batch)
        
        if torch.isnan(mean_pool).any() or torch.isnan(max_pool).any():
            if debug:
                logger.error(f"❌ 池化输出包含NaN！")
            return torch.full((1, 1024), float('nan'), device=x.device)
        
        graph_features = torch.cat([mean_pool, max_pool], dim=1)  # [batch_size, hidden_dim*2]
        
        # 通过融合层得到最终特征
        output = self.fusion(graph_features)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            if debug:
                logger.error(f"❌ Fusion层输出包含NaN/Inf！范围=[{output.min():.4f}, {output.max():.4f}]")
            return torch.full((1, 1024), float('nan'), device=x.device)
        
        return output

class ContrastiveTrainer:
    """对比学习训练器"""
    
    def __init__(self, model, device='cpu', temperature=0.1, use_amp=False, batch_size=8):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature  # 增大到0.1，提高数值稳定性
        self.use_amp = use_amp  # 默认禁用AMP，避免FP16精度导致的NaN
        self.batch_size = batch_size
        self.initial_batch_size = batch_size  # 保存初始batch_size
        self.min_batch_size = 1  # 最小batch_size
        
        # 优化器（降低学习率提高稳定性）
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        # AMP缩放器
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        
        # 训练历史记录
        self.training_history = {
            'epoch_losses': [],
            'contrastive_losses': [],
            'similarity_losses': [],
            'diversity_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'feature_stats': []
        }
    
    def contrastive_loss(self, features_original, features_attacked, labels):
        """
        对比学习损失函数
        - 正样本：同一原图的不同攻击版本（应该相似）
        - 负样本：不同原图的任何版本（应该区分）
        """
        # 检查输入特征是否有NaN
        if torch.isnan(features_original).any():
            logger.error(f"⚠️ features_original包含NaN！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        if torch.isnan(features_attacked).any():
            logger.error(f"⚠️ features_attacked包含NaN！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # L2归一化（添加eps防止除零）
        features_original = F.normalize(features_original, p=2, dim=1, eps=1e-8)
        features_attacked = F.normalize(features_attacked, p=2, dim=1, eps=1e-8)
        
        sim_matrix = torch.matmul(features_original, features_attacked.T) / self.temperature
        
        batch_size = features_original.size(0)
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        total_loss = 0
        valid_samples = 0
        
        for i in range(batch_size):
            # 正样本：同一原图的攻击版本
            positive_mask = labels_matrix[i]
            positive_scores = sim_matrix[i][positive_mask]
            
            # 负样本：不同原图的任何版本（包括原图和被攻击版本）
            negative_mask = ~labels_matrix[i]
            negative_scores = sim_matrix[i][negative_mask]
            
            if len(positive_scores) > 0 and len(negative_scores) > 0:
                # 计算对比损失：正样本得分应该高，负样本得分应该低
                # 使用数值稳定的InfoNCE损失：-log(exp(pos)/sum(exp(all)))
                pos_score = positive_scores.mean()  # 平均正样本得分
                all_scores = torch.cat([positive_scores, negative_scores])
                
                # 数值稳定版本：log(exp(a)/sum(exp(b))) = a - log_sum_exp(b)
                # 使用PyTorch的logsumexp函数，内部实现了数值稳定的技巧
                loss = -pos_score + torch.logsumexp(all_scores, dim=0)
                
                # 检查损失是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"⚠️ Batch {i}: loss={loss.item()}, pos_score={pos_score.item()}, all_scores范围=[{all_scores.min().item():.2f}, {all_scores.max().item():.2f}]")
                    continue  # 跳过这个样本
                
                total_loss += loss
                valid_samples += 1
        
        return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=self.device)
    
    def similarity_loss(self, features_original, features_attacked):
        """相似性损失：确保同一原图的攻击版本特征相似"""
        # 计算余弦相似度（添加eps防止数值问题）
        similarity = F.cosine_similarity(features_original, features_attacked, dim=1, eps=1e-8)
        # 最大化相似度（最小化1-相似度）
        loss = torch.mean(1 - similarity)
        
        # 检查损失
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"⚠️ similarity_loss is NaN/Inf: {loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss
    
    def diversity_loss(self, features):
        """多样性损失：防止特征坍塌，确保不同图有不同特征"""
        # 计算特征矩阵的方差
        feature_var = torch.var(features, dim=0, unbiased=False)
        # 鼓励每个维度都有足够的方差
        diversity_loss = torch.mean(torch.relu(0.1 - feature_var))
        
        # 检查损失
        if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
            logger.error(f"⚠️ diversity_loss is NaN/Inf: {diversity_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return diversity_loss
    
    def train_epoch(self, original_graphs, attacked_graphs, epoch):
        """训练一个epoch"""
        self.model.train()
        
        # Epoch开始时清理CUDA缓存和同步
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 确保之前的操作完成
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_similarity_loss = 0.0
        total_diversity_loss = 0.0
        num_batches = 0
        
        # 记录梯度范数
        total_grad_norm = 0.0
        
        # 准备训练数据
        all_pairs = []
        all_labels = []
        
        for i, (graph_name, original_graph) in enumerate(original_graphs.items()):
            if graph_name in attacked_graphs:
                for attacked_graph in attacked_graphs[graph_name]:
                    all_pairs.append((original_graph, attacked_graph))
                    all_labels.append(i)  # 同一原图使用相同标签
        
        if len(all_pairs) == 0:
            logger.warning("没有找到训练数据对")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 随机打乱数据，确保每个batch都有不同的标签分布
        import random
        combined = list(zip(all_pairs, all_labels))
        random.shuffle(combined)
        all_pairs, all_labels = zip(*combined)
        
        # 统计训练数据信息
        unique_labels = set(all_labels)
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"训练数据统计:")
        logger.info(f"  总样本对数: {len(all_pairs)}")
        logger.info(f"  原图类型数: {len(unique_labels)}")
        logger.info(f"  各原图样本数: {label_counts}")
        
        # 计算正负样本数量
        total_positive_pairs = sum(count * (count - 1) // 2 for count in label_counts.values())
        total_negative_pairs = len(all_pairs) * (len(all_pairs) - 1) // 2 - total_positive_pairs
        logger.info(f"  正样本对数: {total_positive_pairs} (同一原图的不同攻击版本)")
        logger.info(f"  负样本对数: {total_negative_pairs} (不同原图的任何版本)")
        logger.info("")
        
        # 计算日志间隔（每个epoch记录约20个batch）
        total_batches = (len(all_pairs) + self.batch_size - 1) // self.batch_size
        log_interval = max(1, total_batches // 20)
        logger.info(f"  总batch数: {total_batches}, 日志间隔: 每{log_interval}个batch")
        logger.info("")
        
        # 分批训练
        for i in range(0, len(all_pairs), self.batch_size):
            batch_pairs = all_pairs[i:i + self.batch_size]
            batch_labels = all_labels[i:i + self.batch_size]
            
            try:
                # 准备batch数据
                batch_original_features = []
                batch_attacked_features = []
                
                for original_graph, attacked_graph in batch_pairs:
                    # 移动到设备
                    original_graph_gpu = original_graph.to(self.device)
                    attacked_graph_gpu = attacked_graph.to(self.device)
                    
                    # 提取特征（AMP）
                    with amp.autocast(enabled=self.use_amp):
                        features_original = self.model(original_graph_gpu.x, original_graph_gpu.edge_index)
                        features_attacked = self.model(attacked_graph_gpu.x, attacked_graph_gpu.edge_index)
                    
                    # 检查特征是否有NaN（前5个batch开启调试）
                    debug_mode = (i // self.batch_size + 1) <= 5
                    if debug_mode and (torch.isnan(features_original).any() or torch.isnan(features_attacked).any()):
                        # 重新运行forward开启调试
                        logger.error(f"\n⚠️⚠️⚠️ Batch {i // self.batch_size + 1} 检测到NaN！开始详细调试...")
                        logger.error(f"原始图节点数: {original_graph_gpu.x.shape[0]}, 边数: {original_graph_gpu.edge_index.shape[1]}")
                        logger.error(f"攻击图节点数: {attacked_graph_gpu.x.shape[0]}, 边数: {attacked_graph_gpu.edge_index.shape[1]}")
                        logger.error(f"原始图特征范围: [{original_graph_gpu.x.min():.4f}, {original_graph_gpu.x.max():.4f}]")
                        logger.error(f"攻击图特征范围: [{attacked_graph_gpu.x.min():.4f}, {attacked_graph_gpu.x.max():.4f}]")
                        
                        # 重新forward开启debug
                        with torch.no_grad():
                            _ = self.model(original_graph_gpu.x, original_graph_gpu.edge_index, debug=True)
                            _ = self.model(attacked_graph_gpu.x, attacked_graph_gpu.edge_index, debug=True)
                        
                        logger.error(f"跳过此batch继续训练...\n")
                        continue
                    
                    # 保存特征（不要detach，需要保留梯度！）
                    batch_original_features.append(features_original)
                    batch_attacked_features.append(features_attacked)
                    
                    # 删除GPU上的图数据（计算图已建立，可以安全删除）
                    del original_graph_gpu, attacked_graph_gpu
                    
                # 清理GPU缓存（在特征提取循环后）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 堆叠特征（每个features是[1, 1024]，stack后是[batch_size, 1, 1024]，需要squeeze）
                batch_original = torch.cat(batch_original_features, dim=0)  # [batch_size, 1024]
                batch_attacked = torch.cat(batch_attacked_features, dim=0)  # [batch_size, 1024]
                batch_labels = torch.tensor(batch_labels, device=self.device)
                
                # 计算损失（AMP）
                with amp.autocast(enabled=self.use_amp):
                    contrastive_loss = self.contrastive_loss(batch_original, batch_attacked, batch_labels)
                    similarity_loss = self.similarity_loss(batch_original, batch_attacked)
                    diversity_loss = self.diversity_loss(torch.cat([batch_original, batch_attacked], dim=0))
                
                # 计算当前batch的正负样本对数
                batch_labels_np = batch_labels.cpu().numpy()
                unique_batch_labels = set(batch_labels_np)
                
                # 计算正样本对（同一原图的攻击版本）
                batch_positive_pairs = 0
                batch_negative_pairs = 0
                
                for j in range(len(batch_labels_np)):
                    for k in range(j+1, len(batch_labels_np)):
                        if batch_labels_np[j] == batch_labels_np[k]:
                            batch_positive_pairs += 1
                        else:
                            batch_negative_pairs += 1
                
                # 根据日志间隔记录batch信息
                batch_idx = i // self.batch_size + 1
                if batch_idx % log_interval == 0 or batch_idx == 1 or batch_idx == total_batches:
                    logger.info(f"  Batch {batch_idx}/{total_batches}: 标签分布={dict(zip(*np.unique(batch_labels_np, return_counts=True)))}, 正样本对={batch_positive_pairs}, 负样本对={batch_negative_pairs}")
                
                # 总损失
                total_batch_loss = contrastive_loss + 0.5 * similarity_loss + 0.1 * diversity_loss
                
                # 检查总损失是否有效（在反向传播前检查，避免破坏scaler状态）
                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    logger.warning(f"⚠️ Batch {batch_idx}: total_batch_loss={total_batch_loss.item()}, 跳过此batch")
                    continue  # 跳过此batch，不进行反向传播
                
                # 反向传播
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(total_batch_loss).backward()
                    # AMP下需要先反缩放再裁剪
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    total_grad_norm += grad_norm.item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_batch_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    total_grad_norm += grad_norm.item()
                    self.optimizer.step()
                
                # 累积损失（使用.item()立即转为Python标量，释放tensor）
                total_loss += total_batch_loss.detach().item()
                total_contrastive_loss += contrastive_loss.detach().item()
                total_similarity_loss += similarity_loss.detach().item()
                total_diversity_loss += diversity_loss.detach().item()
                num_batches += 1
                
                # 强制清理中间变量和GPU内存
                del batch_original_features, batch_attacked_features
                del batch_original, batch_attacked
                del contrastive_loss, similarity_loss, diversity_loss, total_batch_loss
                # 清理batch变量的引用
                del batch_pairs, batch_labels
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # 每50个batch强制同步和深度清理
                    if batch_idx % 50 == 0:
                        torch.cuda.synchronize()  # 确保所有操作完成
                        torch.cuda.empty_cache()
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        logger.info(f"  💾 GPU显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
                    
                    # 每100个batch也显示（保留原有逻辑）
                    elif batch_idx % 100 == 0:
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        logger.info(f"  💾 GPU显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
                
            except RuntimeError as e:
                # CUDA错误特殊处理
                if 'cuda' in str(e).lower():
                    logger.error(f"处理batch {i//self.batch_size + 1} 时出错: {e}")
                    logger.error("🛑 检测到CUDA错误，尝试清理并同步...")
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()  # 同步所有流
                            torch.cuda.empty_cache()  # 清理缓存
                            torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
                        except:
                            logger.error("⚠️ CUDA清理失败，可能需要重启训练")
                    # 重置AMP scaler状态
                    if self.use_amp:
                        self.scaler.update()
                    continue
                # OOM特殊处理
                elif 'out of memory' in str(e).lower():
                    logger.error(f"处理batch {i//self.batch_size + 1} 时出错: {e}")
                    logger.error("🚨 显存不足（OOM），清理显存...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if self.use_amp:
                        self.scaler.update()
                    continue
                else:
                    logger.error(f"处理batch {i//self.batch_size + 1} 时出错: {e}")
                    if self.use_amp:
                        self.scaler.update()
                    continue
            except Exception as e:
                logger.error(f"处理batch {i//self.batch_size + 1} 时出错: {e}")
                if self.use_amp:
                    self.scaler.update()
                continue
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失和梯度范数
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0.0
        avg_similarity_loss = total_similarity_loss / num_batches if num_batches > 0 else 0.0
        avg_diversity_loss = total_diversity_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
        
        # 记录训练历史
        self.training_history['epoch_losses'].append(avg_loss)
        self.training_history['contrastive_losses'].append(avg_contrastive_loss)
        self.training_history['similarity_losses'].append(avg_similarity_loss)
        self.training_history['diversity_losses'].append(avg_diversity_loss)
        self.training_history['gradient_norms'].append(avg_grad_norm)
        self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss, avg_contrastive_loss, avg_similarity_loss, avg_diversity_loss, avg_grad_norm
    
    def _train_epoch_with_adaptive_batch_size(self, original_graphs, attacked_graphs, epoch):
        """带自适应batch_size的训练（OOM时自动降低batch_size并重试）"""
        max_retries = 3  # 最多重试3次
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # 尝试使用当前batch_size训练
                return self.train_epoch(original_graphs, attacked_graphs, epoch)
            
            except RuntimeError as e:
                error_str = str(e)
                # 检查是否是OOM错误
                if "out of memory" in error_str.lower() or "cuda" in error_str.lower():
                    retry_count += 1
                    
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # 计算新的batch_size（减半，但不低于min_batch_size）
                    old_batch_size = self.batch_size
                    new_batch_size = max(self.min_batch_size, self.batch_size // 2)
                    
                    if new_batch_size == old_batch_size:
                        # 已经是最小batch_size，无法再降低
                        logger.error(f"❌ 显存不足且batch_size已是最小值({self.min_batch_size})，无法继续训练！")
                        logger.error(f"   错误信息: {error_str}")
                        logger.error(f"   建议: 关闭其他GPU程序或升级显卡")
                        raise e
                    
                    self.batch_size = new_batch_size
                    logger.warning("")
                    logger.warning("⚠️" * 40)
                    logger.warning(f"⚠️ 检测到显存不足！")
                    logger.warning(f"⚠️ 自动降低 batch_size: {old_batch_size} → {new_batch_size}")
                    logger.warning(f"⚠️ 重试第 {retry_count}/{max_retries} 次...")
                    logger.warning("⚠️" * 40)
                    logger.warning("")
                    
                    # 等待一段时间让GPU内存完全释放
                    import time
                    time.sleep(2)
                    
                    # 重新计算batch数量并记录
                    logger.info(f"🔄 使用新的batch_size={self.batch_size}重新训练epoch {epoch+1}")
                    
                else:
                    # 不是OOM错误，直接抛出
                    raise e
        
        # 超过最大重试次数
        logger.error(f"❌ 已重试{max_retries}次，仍然失败！")
        raise RuntimeError("自适应batch_size机制失败，训练终止")
    
    def train(self, original_graphs, attacked_graphs, num_epochs=50):
        """训练模型（支持自适应batch_size）"""
        logger.info(f"开始训练GCN模型（{num_epochs}个epoch）...")
        logger.info("训练目标：提取矢量地图的鲁棒特征，抵抗RST攻击")
        logger.info(f"自适应batch_size策略: 初始={self.initial_batch_size}, 最小={self.min_batch_size}")
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # 创建CSV文件记录损失
        loss_csv_path = os.path.join(os.path.dirname(__file__), 'logs', 'training_loss.csv')
        os.makedirs(os.path.dirname(loss_csv_path), exist_ok=True)
        with open(loss_csv_path, 'w', encoding='utf-8') as f:
            f.write('epoch,total_loss,contrastive_loss,similarity_loss,diversity_loss,grad_norm,learning_rate\n')
        logger.info(f"损失记录文件: {loss_csv_path}")
        
        import time
        for epoch in tqdm(range(num_epochs), desc="训练进度"):
            # Epoch开始标记
            epoch_start_time = time.time()
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"📊 Epoch {epoch+1}/{num_epochs} 开始")
            logger.info("=" * 80)
            
            # 训练（带自适应batch_size机制）
            train_loss, contrastive_loss, similarity_loss, diversity_loss, grad_norm = self._train_epoch_with_adaptive_batch_size(original_graphs, attacked_graphs, epoch)
            
            # Epoch结束，计算耗时
            epoch_time = time.time() - epoch_start_time
            
            # Epoch结束时清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"💾 Epoch结束显存: 当前={allocated:.2f}GB, 峰值={max_allocated:.2f}GB")
                torch.cuda.reset_peak_memory_stats()
            
            # 记录损失到CSV
            current_lr = self.optimizer.param_groups[0]['lr']
            with open(loss_csv_path, 'a', encoding='utf-8') as f:
                f.write(f'{epoch+1},{train_loss:.6f},{contrastive_loss:.6f},{similarity_loss:.6f},{diversity_loss:.6f},{grad_norm:.6f},{current_lr:.8f}\n')
            
            # 早停机制
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                # 保存最佳模型到VGCN文件夹
                model_best_path = os.path.join(os.path.dirname(__file__), 'models', 'gcn_model_best.pth')
                self.save_model(model_best_path)
                logger.info("💾 保存最佳模型")
            else:
                patience_counter += 1
            
            # 每个epoch结束时都打印损失汇总
            logger.info("")
            logger.info("─" * 80)
            logger.info(f"✅ Epoch {epoch+1}/{num_epochs} 完成 | 耗时: {epoch_time/60:.2f}分钟")
            logger.info("─" * 80)
            logger.info(f"📉 总损失    : {train_loss:.6f} (最佳: {best_loss:.6f})")
            logger.info(f"📉 对比损失  : {contrastive_loss:.6f}")
            logger.info(f"📉 相似性损失: {similarity_loss:.6f}")
            logger.info(f"📉 多样性损失: {diversity_loss:.6f}")
            logger.info(f"📐 梯度范数  : {grad_norm:.6f}")
            logger.info(f"📚 学习率    : {current_lr:.8f}")
            logger.info(f"📦 Batch大小 : {self.batch_size} (初始: {self.initial_batch_size})")
            logger.info(f"⏸️  耐心计数  : {patience_counter}/{patience}")
            
            # 每3个epoch打印额外的特征统计
            if (epoch + 1) % 3 == 0:
                # 记录详细的特征统计信息
                if hasattr(self, 'model') and self.model is not None:
                    with torch.no_grad():
                        # 随机选择一个batch计算特征统计
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
                            
                            logger.info(f"🔍 特征统计: 均值={feature_mean:.4f}, 标准差={feature_std:.4f}, 范围=[{feature_min:.4f}, {feature_max:.4f}]")
                            
                            # 记录到训练历史
                            self.training_history['feature_stats'].append({
                                'mean': feature_mean,
                                'std': feature_std,
                                'min': feature_min,
                                'max': feature_max
                            })
            
            # 早停
            if patience_counter >= patience:
                logger.warning(f"连续{patience}个epoch没有改善，提前停止训练")
                break
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves(loss_csv_path)
        
        logger.info(f"训练完成！最佳损失值: {best_loss:.6f}")
        return best_loss
    
    def save_model(self, model_path):
        """保存模型"""
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        # 将训练历史保存到VGCN文件夹
        history_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(history_dir, f"training_history_{timestamp}.json")
        
        # 转换numpy数组为列表以便JSON序列化
        history_data = {}
        for key, value in self.training_history.items():
            if key == 'feature_stats':
                # 转换feature_stats中的numpy类型为Python原生类型
                converted_stats = []
                for stat_dict in value:
                    converted_dict = {}
                    for stat_key, stat_value in stat_dict.items():
                        # 将numpy类型转换为float
                        if hasattr(stat_value, 'item'):
                            converted_dict[stat_key] = float(stat_value.item())
                        else:
                            converted_dict[stat_key] = float(stat_value)
                    converted_stats.append(converted_dict)
                history_data[key] = converted_stats
            else:
                # 确保所有数值都转换为Python原生类型
                history_data[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练历史已保存到: {history_file}")
    
    def plot_training_curves(self, csv_path):
        """绘制SCI风格的训练曲线"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # 读取CSV数据
            df = pd.read_csv(csv_path)
            
            # 设置SCI风格
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.linewidth'] = 1.2
            plt.rcParams['grid.alpha'] = 0.3
            
            # 创建2x3子图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. 总损失曲线
            axes[0, 0].plot(df['epoch'], df['total_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#2E86AB')
            axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Total Loss', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('(a) Total Loss', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, linestyle='--')
            axes[0, 0].tick_params(labelsize=9)
            
            # 2. 对比损失曲线
            axes[0, 1].plot(df['epoch'], df['contrastive_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#A23B72')
            axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('Contrastive Loss', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('(b) Contrastive Loss', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, linestyle='--')
            axes[0, 1].tick_params(labelsize=9)
            
            # 3. 相似性损失曲线
            axes[0, 2].plot(df['epoch'], df['similarity_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#F18F01')
            axes[0, 2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[0, 2].set_ylabel('Similarity Loss', fontsize=11, fontweight='bold')
            axes[0, 2].set_title('(c) Similarity Loss', fontsize=12, fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3, linestyle='--')
            axes[0, 2].tick_params(labelsize=9)
            
            # 4. 多样性损失曲线
            axes[1, 0].plot(df['epoch'], df['diversity_loss'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#C73E1D')
            axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Diversity Loss', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('(d) Diversity Loss', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, linestyle='--')
            axes[1, 0].tick_params(labelsize=9)
            
            # 5. 梯度范数曲线
            axes[1, 1].plot(df['epoch'], df['grad_norm'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#6A994E')
            axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Gradient Norm', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('(e) Gradient Norm', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, linestyle='--')
            axes[1, 1].tick_params(labelsize=9)
            
            # 6. 学习率曲线
            axes[1, 2].plot(df['epoch'], df['learning_rate'], '-o', linewidth=2, markersize=4, alpha=0.8, color='#BC4B51')
            axes[1, 2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1, 2].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
            axes[1, 2].set_title('(f) Learning Rate', fontsize=12, fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3, linestyle='--')
            axes[1, 2].tick_params(labelsize=9)
            axes[1, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            
            # 保存图片
            save_dir = os.path.dirname(csv_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    def __init__(self, graph_dir=os.path.join('..', 'convertToGraph', 'Graph', 'TrainingSet')):
        self.graph_dir = graph_dir
    
    def load_graph_data(self):
        """加载图数据：原始图与其对应的被攻击图"""
        original_dir = os.path.join(self.graph_dir, 'Original')
        attacked_dir = os.path.join(self.graph_dir, 'Attacked')
        
        # 加载原始图
        original_graphs = {}
        if not os.path.exists(original_dir):
            logger.warning(f"原始数据目录不存在: {original_dir}")
            return {}, {}
        for filename in os.listdir(original_dir):
            if filename.endswith('_graph.pkl'):
                graph_name = filename.replace('_graph.pkl', '')
                with open(os.path.join(original_dir, filename), 'rb') as f:
                    graph_data = pickle.load(f)
                    original_graphs[graph_name] = graph_data
        
        # 加载被攻击的图
        attacked_graphs = {}
        if os.path.exists(attacked_dir):
            for subdir in os.listdir(attacked_dir):
                subdir_path = os.path.join(attacked_dir, subdir)
                if os.path.isdir(subdir_path):
                    attacked_graphs[subdir] = []
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
    logger.info("＝＝＝ 第三步：GCN模型训练 - 矢量地图零水印鲁棒特征提取 ＝＝＝")
    
    # 尝试缓解CUDA显存碎片（仅当前进程生效）
    try:
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        logger.info(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    except Exception:
        pass

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    data_loader = GraphDataLoader()
    original_graphs, attacked_graphs = data_loader.load_graph_data()
    
    if len(original_graphs) == 0:
        logger.warning("没有找到原始图数据，请先运行第二步")
        return
    
    if len(attacked_graphs) == 0:
        logger.warning("没有找到被攻击的图数据，请先运行第二步")
        return
    
    # 获取输入维度
    first_graph = list(original_graphs.values())[0]
    input_dim = first_graph.x.shape[1]
    logger.info(f"输入特征维度: {input_dim}")
    logger.info(f"目标输出维度: 1024 (32x32)")
    
    # 创建GCN模型
    model = GCNModel(input_dim=input_dim, hidden_dim=128, output_dim=1024, dropout=0.2)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 读取可选batch大小（默认4，适配16GB显存）
    try:
        configured_bs = int(os.environ.get('VGAT_BATCH_SIZE', '4'))
    except Exception:
        configured_bs = 4
    
    # 创建训练器（禁用AMP使用FP32完整精度，避免FP16数值不稳定）
    trainer = ContrastiveTrainer(model, device, use_amp=False, batch_size=configured_bs)
    
    # 训练模型
    train_loss = trainer.train(original_graphs, attacked_graphs, num_epochs=50)
    
    # 保存最终模型到VGCN文件夹
    final_model_path = os.path.join(os.path.dirname(__file__), 'models', 'gcn_model.pth')
    trainer.save_model(final_model_path)
    
    logger.info("模型训练完成！")
    logger.info("模型将用于：")
    logger.info("  1. 从原始矢量地图提取鲁棒特征")
    logger.info("  2. 与版权图像结合生成零水印")
    logger.info("  3. 验证阶段提取特征并与零水印结合恢复版权图像")

if __name__ == "__main__":
    main()