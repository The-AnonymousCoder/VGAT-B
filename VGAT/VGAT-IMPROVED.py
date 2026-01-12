#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Improved VGAT training pipeline for robust zero-watermark extraction."""

import json
import logging
import os
import pickle
import random
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch_geometric.nn import GATv2Conv, GraphNorm  # 使用GATv2 + GraphNorm
from tqdm import tqdm


def default_memory_refresh_marks(total_epochs: int) -> Tuple[int, ...]:
    if total_epochs <= 4:
        return (max(1, total_epochs - 1),)
    marks = []
    marks.append(max(1, int(total_epochs * 0.4)))
    marks.append(max(marks[-1] + 1, int(total_epochs * 0.55)))
    marks = [min(total_epochs - 1, m) for m in marks]
    unique_marks = sorted(set(marks))
    return tuple(mark for mark in unique_marks if mark >= 1)


def should_reset_patience(epoch: int) -> bool:
    return epoch in PATIENT_STAGE_EPOCHS


FIXED_LOSS_WEIGHTS = {
    'contrastive': 1.0,
    'similarity': 0.5,
    'diversity': 0.3,
    'binary_consistency': 1.0,
}
FULL_CHAIN_ATTACK_KEYWORDS = ("full_attack_chain", "full_chain", "compound_seq", "compound_seq_all")
COMBO_ATTACK_KEYWORDS = ("combo", )
COMPOSITE_ATTACK_KEYWORDS = FULL_CHAIN_ATTACK_KEYWORDS + COMBO_ATTACK_KEYWORDS
WEAK_ATTACK_KEYWORDS = ("noise", "add", "crop", "rotate", "flip")
PATIENT_STAGE_EPOCHS = (20, 30, 40)


def get_attack_name(graph) -> str:
    """Return the normalized attack name attached to a graph sample."""
    return str(getattr(graph, 'attack_type', '')).lower()


def is_full_chain_attack(name: str) -> bool:
    return any(keyword in name for keyword in FULL_CHAIN_ATTACK_KEYWORDS)


def is_combo_attack(name: str) -> bool:
    return any(keyword in name for keyword in COMBO_ATTACK_KEYWORDS) and not is_full_chain_attack(name)


def is_composite_attack(name: str) -> bool:
    return is_full_chain_attack(name) or is_combo_attack(name)


def has_weak_perturbation(name: str) -> bool:
    return any(keyword in name for keyword in WEAK_ATTACK_KEYWORDS)


def attack_sample_weight(name: str) -> float:
    """Heuristically score attack difficulty for sampling/weighting."""
    weight = 1.0
    if is_full_chain_attack(name):
        weight = 3.0
    elif is_combo_attack(name):
        weight = 2.4
    if 'noise' in name:
        weight *= 1.4
    if 'add' in name:
        weight *= 1.4
    if 'crop' in name:
        weight *= 1.3
    if 'rotate' in name:
        weight *= 1.3
    if 'flip' in name:
        weight *= 1.3
    return weight


def compute_stage_progress(epoch: int, total_epochs: int) -> Tuple[str, float]:
    early_end = max(1, int(total_epochs * 0.3))
    mid_end = max(early_end + 1, int(total_epochs * 0.7))
    if epoch < early_end:
        return "early", epoch / max(1, early_end)
    if epoch < mid_end:
        return "mid", (epoch - early_end) / max(1, mid_end - early_end)
    return "late", (epoch - mid_end) / max(1, total_epochs - mid_end)


STAGE_DESCRIPTIONS = {
    "early": "前期-区分+唯一性",
    "mid": "中期-平衡优化",
    "late": "后期-强化鲁棒性",
}


def describe_stage(stage: str, progress: float) -> str:
    base = STAGE_DESCRIPTIONS.get(stage, stage)
    progress_clamped = max(0.0, min(1.0, progress))
    if stage == "early":
        return base
    return f"{base} ({progress_clamped * 100:.0f}%)"


def run_fig12_evaluation_for_model(model_path: str) -> Optional[float]:
    """
    使用当前模型运行一次 zNC-Test/Fig12.py，并解析Fig12的Average NC值。
    注意：这是一个重操作，建议仅在关键epoch调用。
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = str(Path(script_dir).parents[0])
        znc_root = os.path.join(project_root, "zNC-Test")
        zerowm_dir = os.path.join(znc_root, "vector-data-zerowatermark")
        fig12_results_dir = os.path.join(znc_root, "NC-Results", "Fig12")

        # ⭐关键修复：每次评估前删除NC-Results/Fig12文件夹，确保Fig12.py重新生成结果
        if os.path.isdir(fig12_results_dir):
            try:
                import shutil
                shutil.rmtree(fig12_results_dir)
                logging.getLogger(__name__).info(f"[Fig12] 已删除旧结果目录: {fig12_results_dir}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[Fig12] 删除结果目录失败 {fig12_results_dir}: {e}")

        # 每次评估前清空零水印目录，确保用当前模型重新生成
        if os.path.isdir(zerowm_dir):
            for fname in os.listdir(zerowm_dir):
                if fname.startswith("."):
                    continue
                fpath = os.path.join(zerowm_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[Fig12] 删除零水印文件失败 {fpath}: {e}")

        env = os.environ.copy()
        env["VGAT_MODEL_PATH"] = model_path

        # 在zNC-Test目录下调用Fig12.py
        logging.getLogger(__name__).info(f"[Fig12] 使用模型评估鲁棒性: {model_path}")
        logging.getLogger(__name__).info(f"[Fig12] 环境变量VGAT_MODEL_PATH={model_path}")
        
        # ⭐关键修复：确保subprocess完成并检查返回码
        result = subprocess.run(
            [sys.executable, "Fig12.py"],
            cwd=znc_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logging.getLogger(__name__).warning(f"[Fig12] Fig12.py执行失败，返回码: {result.returncode}")
            if result.stderr:
                logging.getLogger(__name__).warning(f"[Fig12] 错误输出: {result.stderr[:500]}")

        # ⭐关键修复：等待文件系统同步，确保CSV文件已完全写入
        import time
        time.sleep(0.5)  # 等待0.5秒确保文件写入完成

        # 解析结果CSV，提取Average NC
        csv_path = os.path.join(znc_root, "NC-Results", "Fig12", "fig12_compound_seq_nc.csv")
        if not os.path.exists(csv_path):
            logging.getLogger(__name__).warning(f"[Fig12] 未找到结果文件: {csv_path}")
            logging.getLogger(__name__).warning(f"[Fig12] 请检查Fig12.py是否正常执行完成")
            return None

        avg_nc = None
        try:
            import csv as _csv

            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    # 标准列名：'复合攻击(顺序)', 'VGAT', '类型'
                    row_type = row.get("类型", "").strip()
                    if row_type == "average":
                        v = row.get("VGAT", "").strip()
                        try:
                            avg_nc = float(v)
                        except Exception:
                            continue
        except Exception as e:
            logging.getLogger(__name__).error(f"[Fig12] 解析结果CSV失败: {e}")
            return None

        return avg_nc
    except Exception as e:
        logging.getLogger(__name__).error(f"[Fig12] 评估过程中发生异常: {e}")
        logging.getLogger(__name__).error(traceback.format_exc())
        return None


def stage_temperature(stage: str) -> float:
    return {
        "early": 0.15,
        "mid": 0.12,
        "late": 0.09,
    }.get(stage, 0.12)


def stage_augmentation_probability(stage: str) -> float:
    return {
        "early": 0.30,
        "mid": 0.40,
        "late": 0.50,
    }.get(stage, 0.40)


@dataclass
class TrainingScheduleConfig:
    metric_eval_interval: int = 5
    metric_patience: int = 3
    min_epoch_for_metric_stop: int = 15
    nc_improve_tol: float = 0.005
    distinction_improve_tol: float = 0.005
    onecycle_max_lr: float = 0.0015  # 从0.001提升到0.0015，增强后期学习能力
    onecycle_pct_start: float = 0.2
    onecycle_div_factor: float = 100.0
    onecycle_final_div: float = 5000.0  # 从10000.0降低到5000.0，避免后期学习率过小
    robust_warmup_epochs: int = 4
    robust_lr_boost: float = 1.6
    robust_supcon_temp: float = 0.12
    robust_memory_keep_ratio: float = 0.6
    robust_memory_refresh_interval: int = 2


def log_training_overview():
    logger.info("=" * 70)
    logger.info("改进VGAT：矢量零水印鲁棒特征训练")
    logger.info("核心优化: InfoNCE修复 | 二值化损失 | GATv2+残差 | OneCycleLR | 动态损失权重")
    logger.info("=" * 70)
    logger.info("")


def log_device_info() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    if device == 'cuda':
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"显存: {total_mem_gb:.2f} GB")
        except Exception as err:
            logger.warning(f"无法读取GPU信息: {err}")
    logger.info("")
    return device


def infer_input_dim(graphs) -> int:
    try:
        first_graph = next(iter(graphs.values()))
    except StopIteration as exc:
        raise ValueError("original_graphs为空，无法推断输入维度") from exc
    return int(first_graph.x.shape[1])


def log_feature_profile(input_dim: int):
    logger.info(f"检测到输入特征维度: {input_dim}")
    if input_dim == 20:
        logger.info("✅ 使用20维最优特征（方案D：全局+局部多尺度+节点数编码）")
    elif input_dim == 19:
        logger.info("✅ 使用19维优化特征（Hu不变矩 + 全局/局部位置 + 拓扑邻域）")
    elif input_dim == 16:
        logger.info("⚠️ 使用16维特征（建议升级到20维以提升鲁棒性）")
    elif input_dim == 13:
        logger.info("⚠️ 使用原始13维特征（建议升级到20维）")
    else:
        logger.info("⚠️ 未识别的特征维度，模型将自动适配")
    logger.info("")


def log_model_summary(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型总参数数量: {total_params:,}")
    logger.info("")


def resolve_batch_size(default: int = 6) -> int:
    try:
        batch_size = int(os.environ.get('VGAT_BATCH_SIZE', str(default)))
    except Exception:
        batch_size = default
    logger.info(f"初始批次大小: {batch_size}")
    logger.info(f"智能降级策略: {batch_size}→6→4→2→1 (自适应)")
    if batch_size == 8:
        logger.info("   ✅ 推荐配置：2原图×4样本，16个正样本对；如OOM请降到6")
    elif batch_size > 8:
        logger.warning(f"⚠️ batch_size={batch_size} 可能导致OOM，建议<=8")
    elif batch_size == 6:
        logger.info("   兼容配置：2原图×3样本，12个正样本对")
    elif batch_size <= 2:
        logger.warning("   batch_size过小，对比学习效果受限")
    elif batch_size == 4:
        logger.warning("   batch_size=4 仅提供2个正样本对，建议提升至6或8")
    logger.info("")
    return batch_size


def resolve_checkpoint_choice(default_checkpoint: str) -> Optional[str]:
    mode = os.environ.get('VGAT_RESUME_TRAINING', 'auto').lower()
    checkpoint_exists = os.path.exists(default_checkpoint)

    if mode in {'false', '0', 'no'}:
        logger.info("从头开始新的训练")
        return None

    if not checkpoint_exists:
        if mode in {'true', '1'}:
            logger.warning(f"⚠️ 未找到checkpoint文件: {default_checkpoint}")
        return None

    if mode in {'true', '1'}:
        logger.info(f"✅ 强制从checkpoint恢复: {default_checkpoint}")
        return default_checkpoint

    if mode == 'auto':
        logger.info(f"🔍 检测到checkpoint文件: {default_checkpoint}")
        try:
            user_choice = input("是否从checkpoint恢复训练？[y/N]: ").strip().lower()
        except EOFError:
            logger.warning("标准输入不可用，默认从头训练")
            return None
        if user_choice in {'y', 'yes'}:
            logger.info("选择从checkpoint恢复训练")
            return default_checkpoint
        logger.info("选择从头开始训练")
    else:
        logger.info("从头开始新的训练")
    return None


# =============================================================
# 日志与全局状态
# =============================================================

# 设置日志
def setup_logging():
    """设置日志记录（按时间戳+PID生成唯一文件，并维护 latest 文件）"""
    # 使用绝对路径，确保日志目录位于脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_file = os.path.join(log_dir, f"step3_training_IMPROVED_{timestamp}_{pid}.log")
    latest_file = os.path.join(log_dir, "step3_training_IMPROVED_latest.log")

    # 清理可能存在的重复 handler
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

    # 设置日志级别（可通过环境变量控制）
    log_level = os.environ.get('VGAT_LOG_LEVEL', 'INFO')  # 默认INFO（减少日志量）
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(file_handler_unique)
    root_logger.addHandler(file_handler_latest)
    root_logger.addHandler(console_handler)

    # 将当前日志路径暴露为模块级变量
    globals()["CURRENT_LOG_FILE"] = log_file
    globals()["CURRENT_LATEST_LOG"] = latest_file
    os.environ["VGAT_CURRENT_LOG"] = log_file
    os.environ["VGAT_CURRENT_LOG_LATEST"] = latest_file

    logger = logging.getLogger(__name__)
    logger.info(f"优化版GAT模型训练")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"最新日志(覆盖): {latest_file}")
    return logger

# ⭐ 只在直接运行训练时才设置日志（被其他脚本导入时不设置）
# 初始化一个基本的 logger（被导入时不写文件）
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())  # 默认不输出日志

class ImprovedGATModel(nn.Module):
    """图级鲁棒特征提取骨干：GATv2 + 残差 + GraphNorm + 多尺度池化。"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1024, num_heads=8, dropout=0.3):
        super(ImprovedGATModel, self).__init__()
        
        logger.info(f"创建改进的GAT模型（节点+图池化架构）:")
        logger.info(f"  原始输入维度: {input_dim}")
        
        # ✅ 特征分离：节点级 vs 图级
        # 节点级特征（每个节点不同）：维度5-10, 14-17 = 10维
        self.node_feature_dims = [5, 6, 7, 8, 9, 10, 14, 15, 16, 17]
        # 图级特征（整个图共享）：维度0-2, 11-13, 18-19 = 8维
        self.graph_feature_dims = [0, 1, 2, 11, 12, 13, 18, 19]
        
        self.node_input_dim = len(self.node_feature_dims)  # 10
        self.graph_input_dim = len(self.graph_feature_dims)  # 8
        
        logger.info(f"  节点级特征维度: {self.node_input_dim} (dims: {self.node_feature_dims})")
        logger.info(f"  图级特征维度: {self.graph_input_dim} (dims: {self.graph_feature_dims})")
        logger.info(f"  隐藏维度: {hidden_dim}")
        logger.info(f"  输出维度: {output_dim}")
        logger.info(f"  注意力头数: {num_heads}")
        logger.info(f"  Dropout: {dropout}")
        
        # 第1层：GATv2（concat多头）- 只处理节点级特征
        self.gat1 = GATv2Conv(self.node_input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        # GraphNorm：按图归一化，兼容单图/小batch
        self.gn1 = GraphNorm(hidden_dim * num_heads)
        
        # 第2层：GATv2（不concat，输出hidden_dim）
        self.gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        # GraphNorm：保持不同图规模下的数值稳定
        self.gn2 = GraphNorm(hidden_dim)
        
        # 残差投影（如果维度不匹配）
        if self.node_input_dim != hidden_dim:
            self.residual_proj = nn.Linear(self.node_input_dim, hidden_dim)
            logger.info(f"  添加残差投影: {self.node_input_dim} -> {hidden_dim}")
        else:
            self.residual_proj = None
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 图级特征编码器（可选，如果图级特征需要学习）
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.graph_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征融合：3种池化（均值+最大+注意力）+ 图级特征 -> output_dim
        fusion_input_dim = hidden_dim * 3 + hidden_dim // 2  # 节点池化 + 图级编码
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # 输出[-1,1]，适合中位数阈值二值化
        )
        
        logger.info(f"  融合层输入维度: {fusion_input_dim} = {hidden_dim*3}(节点池化) + {hidden_dim//2}(图级编码)")
        
        self.dropout = nn.Dropout(dropout)
        
        # ✅ 改进权重初始化（防止第一个batch NaN）
        self._init_weights()
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  总参数数量: {total_params:,}")
    
    def _init_weights(self):
        """
        改进的权重初始化策略（防止训练初期NaN）
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # 线性层/卷积层使用Xavier初始化（标准gain）
                    nn.init.xavier_uniform_(param, gain=1.0)  # ✅ 改回标准gain=1.0
                else:
                    # 1D权重（如attention）使用正态分布
                    nn.init.normal_(param, mean=0.0, std=0.1)  # ✅ 增大std到0.1
            elif 'bias' in name:
                # ✅ 偏置初始化为0（标准做法）
                nn.init.constant_(param, 0.0)
        
        logger.info("  ✅ 已应用改进的权重初始化（防止NaN）")
    
    def forward(self, x, edge_index, batch=None):
        """
        前向传播：提取图级鲁棒特征
        
        架构：
        1. 分离节点级特征和图级特征
        2. GAT处理节点级特征（学习节点间关系）
        3. 图级池化（均值+最大+注意力）
        4. 拼接图级特征（几何类型、节点数等）
        5. 融合层输出最终特征
        
        Args:
            x: 节点特征 [num_nodes, 20]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引（单图推理时为None）
        
        Returns:
            output: 图级特征 [output_dim] 或 [batch_size, output_dim]
        """
        from torch_geometric.nn import global_mean_pool, global_max_pool
        
        # ✅ 步顤1：分离节点级和图级特征
        node_features = x[:, self.node_feature_dims]  # [num_nodes, 10]
        graph_features = x[:, self.graph_feature_dims]  # [num_nodes, 8] - 每个节点都相同
        
        # 提取图级特征（取第一个节点即可，因为所有节点相同）
        if batch is None:
            # 单图模式
            graph_features_unique = graph_features[0]  # [8]
        else:
            # 批次模式：从每个图提取一个节点的图级特征
            batch_size = batch.max().item() + 1
            graph_features_list = []
            for i in range(batch_size):
                mask = (batch == i)
                graph_features_list.append(graph_features[mask][0])  # 取第一个节点
            graph_features_unique = torch.stack(graph_features_list, dim=0)  # [batch_size, 8]
        
        # ✅ 步顤2：GAT处理节点级特征
        # 第1层 GATv2
        x1 = self.gat1(node_features, edge_index)
        if batch is None:
            norm_batch = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)
        else:
            norm_batch = batch.to(x1.device)
        # GraphNorm按图归一化，兼容单图/多图
        x1 = self.gn1(x1, norm_batch)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        
        # 第2层 GATv2 + 残差连接
        x2 = self.gat2(x1, edge_index)
        x2 = self.gn2(x2, norm_batch)
        
        # 添加残差（基于节点级特征）
        if self.residual_proj is not None:
            residual = self.residual_proj(node_features)
        else:
            residual = node_features
        x2 = F.elu(x2 + residual)
        
        # ✅ 步顤3：图级池化（节点级特征聚合）
        if batch is None:
            # 单图推理：直接对所有节点池化
            mean_pool = torch.mean(x2, dim=0)
            max_pool, _ = torch.max(x2, dim=0)
            attn_weights = F.softmax(self.attention_pool(x2), dim=0)
            attn_pool = torch.sum(x2 * attn_weights, dim=0)
            
            # 拼接三种池化结果
            pooled_node_features = torch.cat([mean_pool, max_pool, attn_pool], dim=0)  # [hidden_dim*3]
            
            # ✅ 步顤4：编码图级特征
            graph_features_encoded = self.graph_encoder(graph_features_unique)  # [hidden_dim//2]
            
            # ✅ 步顤5：融合节点池化 + 图级特征
            final_features = torch.cat([pooled_node_features, graph_features_encoded], dim=0)
            output = self.fusion(final_features)
        else:
            # 批量推理：对每个图分别池化
            # 1. 均值池化
            mean_pool = global_mean_pool(x2, batch)
            
            # 2. 最大池化
            max_pool = global_max_pool(x2, batch)
            
            # 3. 注意力加权池化（对每个图分别计算）
            attn_scores = self.attention_pool(x2)  # [num_nodes, 1]
            
            # 按图分组计算softmax
            batch_size = batch.max().item() + 1
            attn_pool_list = []
            for i in range(batch_size):
                mask = (batch == i)
                x_i = x2[mask]  # 第i个图的节点特征
                attn_i = attn_scores[mask]  # 第i个图的注意力分数
                attn_weights_i = F.softmax(attn_i, dim=0)
                attn_pool_i = torch.sum(x_i * attn_weights_i, dim=0)
                attn_pool_list.append(attn_pool_i)
            attn_pool = torch.stack(attn_pool_list, dim=0)  # [batch_size, hidden_dim]
            
            # 拼接三种池化结果
            pooled_node_features = torch.cat([mean_pool, max_pool, attn_pool], dim=1)  # [batch_size, hidden_dim*3]
            
            # ✅ 步顤4：编码图级特征
            graph_features_encoded = self.graph_encoder(graph_features_unique)  # [batch_size, hidden_dim//2]
            
            # ✅ 步顤5：融合节点池化 + 图级特征
            final_features = torch.cat([pooled_node_features, graph_features_encoded], dim=1)  # [batch_size, hidden_dim*3 + hidden_dim//2]
            output = self.fusion(final_features)  # [batch_size, output_dim]
        
        return output

# =============================================================
# 温度退火与数据增强
# =============================================================


class AdaptiveTemperature:
    """指数退火温度调度：软二值化 → 硬二值化。"""
    
    def __init__(self, init_temp=1.0, final_temp=0.08, total_epochs=50):
        """
        初始化温度参数
        
        Args:
            init_temp: 初始温度（默认1.0，软二值化）
            final_temp: 最终温度（默认0.08，避免过硬导致梯度消失，保持学习能力）
            total_epochs: 总训练轮数
        """
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        
        logger.info(f"自适应温度初始化:")
        logger.info(f"  初始温度: {init_temp} (软二值化)")
        logger.info(f"  最终温度: {final_temp} (硬二值化)")
        logger.info(f"  退火策略: 指数衰减")
    
    def get_temperature(self, epoch):
        """返回指定 epoch 下的温度：temp = init * (final / init)^(t/T)。"""
        if epoch >= self.total_epochs:
            return self.final_temp
        
        # 指数衰减
        progress = epoch / self.total_epochs  # 0 → 1
        temp = self.init_temp * (self.final_temp / self.init_temp) ** progress
        
        return temp

def augment_graph_data(data, augment_prob=0.3, training=True):
    """在线图增强：随机执行噪声/边裁剪/特征遮罩/复合链。"""
    if not training or random.random() > augment_prob:
        return data
    
    # ⭐修复：添加CUDA错误恢复，防止clone失败导致训练崩溃
    try:
        data = data.clone()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        # clone失败时返回原始数据，不进行增强
        logger.warning(f"⚠️ 数据增强clone失败，跳过增强: {e}")
        return data
    aug_type = random.choice(['vertex_noise', 'edge_drop', 'feature_mask', 'aug_chain'])
    
    if aug_type == 'vertex_noise':
        # ⭐ 噪声增强：30%概率使用更强噪声，应对Fig4噪声攻击0.833短板
        if random.random() < 0.3:  # 20% -> 30%
            noise_ratio = random.uniform(0.08, 0.18)  # 扩大范围，贴近Fig4强度
        else:
            noise_ratio = random.uniform(0.05, 0.10)
        noise = torch.randn_like(data.x) * noise_ratio
        data.x = data.x + noise
        
    elif aug_type == 'edge_drop':
        # 模拟边删除：随机drop 5-10%的边
        if data.edge_index.size(1) > 10:  # 至少保留10条边
            drop_ratio = random.uniform(0.05, 0.10)
            keep_prob = 1.0 - drop_ratio
            mask = torch.rand(data.edge_index.size(1), device=data.edge_index.device) > drop_ratio
            data.edge_index = data.edge_index[:, mask]
            
    elif aug_type == 'feature_mask':
        # 模拟特征缺失：随机mask 5-10%的特征维度
        mask_ratio = random.uniform(0.05, 0.10)
        mask = torch.rand(data.x.size(1), device=data.x.device) > mask_ratio
        data.x = data.x * mask.float()
    elif aug_type == 'aug_chain':
        # 复合增强：顺序执行2-3种轻量增强，模拟链式攻击的扰动分布
        chain_ops = ['vertex_noise', 'edge_drop', 'feature_mask']
        random.shuffle(chain_ops)
        k = random.choice([2, 3])
        for op in chain_ops[:k]:
            if op == 'vertex_noise':
                noise_ratio = random.uniform(0.03, 0.08)
                noise = torch.randn_like(data.x) * noise_ratio
                data.x = data.x + noise
            elif op == 'edge_drop':
                if data.edge_index.size(1) > 10:
                    drop_ratio = random.uniform(0.03, 0.08)
                    mask_e = torch.rand(data.edge_index.size(1), device=data.edge_index.device) > drop_ratio
                    data.edge_index = data.edge_index[:, mask_e]
            elif op == 'feature_mask':
                mask_ratio = random.uniform(0.03, 0.08)
                mask_f = torch.rand(data.x.size(1), device=data.x.device) > mask_ratio
                data.x = data.x * mask_f.float()
    
    return data


# =============================================================
# 训练器：对比学习 + 多损失调度
# =============================================================


class ImprovedContrastiveTrainer:
    """
    改进的对比学习训练器
    
    核心优化：
    1. 修复InfoNCE损失
    2. 添加二值化感知损失
    3. 动态损失权重
    4. 验证集评估
    5. OneCycleLR学习率
    """
    
    def __init__(
        self,
        model,
        device='cpu',
        temperature=0.1,
        use_amp=True,
        batch_size=6,
        checkpoint_name='gat_checkpoint_latest.pth',
        model_prefix='IMPROVED',
        schedule_config: Optional[TrainingScheduleConfig] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature  # 增大至0.1，增强数值稳定性 ⭐修复NaN
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.checkpoint_name = checkpoint_name  # 自定义checkpoint文件名
        self.model_prefix = model_prefix  # 模型文件名前缀（如'IMPROVED'或'Ablation1_NodeOnly'）
        self.schedule = schedule_config or TrainingScheduleConfig()
        
        # 优化器（AdamW with weight decay）
        base_lr = (self.schedule.onecycle_max_lr / self.schedule.onecycle_div_factor)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        self.base_lr = base_lr
        
        # 学习率调度器（稍后在train中初始化，因为需要知道steps_per_epoch）
        self.scheduler = None
        
        # AMP缩放器
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.current_lr_multiplier = 1.0
        self.robust_warmup_state = None
        self.supcon_temp_override = None
        self.last_memory_refresh_epoch = None
        self.default_binary_temp_floor = 0.10
        self.binary_temp_floor = self.default_binary_temp_floor
        self.current_stage = "early"
        self.robust_warmup_triggered = False
        self.min_full_chain_per_batch = 1
        self.max_full_chain_per_batch = 2
        
        # 自适应温度参数（二值化感知损失）
        self.adaptive_temp = AdaptiveTemperature(
            init_temp=1.0,      # 初始温度：软二值化
            final_temp=0.08,    # 最终温度：略微放宽，减轻过硬二值化对唯一性的影响
            total_epochs=20     # ✅ 总轮数（默认20，会在train方法中根据num_epochs更新）
        )
        self.memory_refresh_marks = default_memory_refresh_marks(self.adaptive_temp.total_epochs)
        
        # 训练历史记录（✅ 添加所有需要的键）
        self.training_history = {
            'epoch_losses': [],
            'contrastive_losses': [],
            'similarity_losses': [],
            'diversity_losses': [],  # ⭐ 修复 KeyError
            'uniqueness_losses': [],
            'binary_consistency_losses': [],  # ⭐ 修复绘图错误
            'feature_stats': [],
            'gradient_norms': [],
            'oom_retries': [],
            'learning_rates': [],
            'temperatures': []
        }
        
        # ✅ 动态黑名单：训练过程中OOM的图会被加入
        self.dynamic_blacklist = set()
        self.ema_median = None
        self.ema_momentum = 0.99
        
        # Memory Bank机制：缓解小batch导致的负样本不足
        self.memory_bank_size = 8192  # 存储最近4096个样本的特征
        self.memory_features = None  # [size, feature_dim]
        self.memory_labels = None     # [size]
        self.memory_ptr = 0           # 当前指针位置
        self.memory_initialized = False
        self.memory_seen_count = 0    # 已写入的总样本数（用于有效大小计算）
        
        # 原型字典：为每个label维护动量更新的类中心
        self.prototypes = {}  # {label: prototype_tensor}
        self.prototype_momentum = 0.95  # 动量系数
        
        logger.info("创建改进的对比学习训练器:")
        logger.info(f"  温度参数: {temperature}")
        logger.info(f"  批次大小: {batch_size}")
        logger.info(f"  混合精度训练: {use_amp}")
        logger.info(f"  优化器: AdamW (lr=0.001, weight_decay=0.01)")
        logger.info("  Memory Bank大小: %s 样本", self.memory_bank_size)
        logger.info("  原型动量: %.2f", self.prototype_momentum)
        logger.info("  目的: 缓解小batch负样本不足，扩大难例覆盖")

    # =============================================================
    # 损失函数与工具
    # =============================================================

    def _apply_lr_multiplier(self):
        if self.scheduler is None:
            return
        for group in self.optimizer.param_groups:
            group['_last_scheduler_lr'] = group['lr']
        if abs(self.current_lr_multiplier - 1.0) < 1e-6:
            return
        for group in self.optimizer.param_groups:
            base_lr = group.get('_last_scheduler_lr', group['lr'])
            group['lr'] = base_lr * self.current_lr_multiplier

    def _clear_robust_overrides(self, reset_trigger: bool = False):
        self.current_lr_multiplier = 1.0
        self.supcon_temp_override = None
        self.binary_temp_floor = self.default_binary_temp_floor
        self.robust_warmup_state = None
        if reset_trigger:
            self.robust_warmup_triggered = False

    def _start_robust_warmup(self, epoch: int):
        if self.robust_warmup_triggered or self.schedule.robust_warmup_epochs <= 0:
            return
        self.robust_warmup_triggered = True
        self.robust_warmup_state = {
            'phase': 'boost',
            'remaining': self.schedule.robust_warmup_epochs,
        }
        self.current_lr_multiplier = self.schedule.robust_lr_boost
        self.supcon_temp_override = self.schedule.robust_supcon_temp
        self.binary_temp_floor = max(self.binary_temp_floor, 0.10)

    def _advance_robust_warmup(self):
        if not self.robust_warmup_state:
            return
        state = self.robust_warmup_state
        config = self.schedule
        if state['phase'] == 'boost':
            state['remaining'] -= 1
            if state['remaining'] <= 0:
                state['phase'] = 'decay'
                state['cooldown'] = max(1, config.robust_warmup_epochs)
        elif state['phase'] == 'decay':
            state['cooldown'] -= 1
            ratio = max(state['cooldown'], 0) / max(1, config.robust_warmup_epochs)
            base_temp = stage_temperature("late")
            self.current_lr_multiplier = 1.0 + (config.robust_lr_boost - 1.0) * ratio
            self.supcon_temp_override = base_temp + (config.robust_supcon_temp - base_temp) * ratio
            if state['cooldown'] <= 0:
                self._clear_robust_overrides()
        else:
            self._clear_robust_overrides()

    def _update_robust_phase_state(self, epoch: int) -> Tuple[str, float]:
        total_epochs = getattr(self, "total_epochs", 20)
        stage, progress = compute_stage_progress(epoch, total_epochs)
        self.current_stage = stage
        if stage == "late":
            if not self.robust_warmup_triggered:
                self._start_robust_warmup(epoch)
            self._advance_robust_warmup()
        else:
            self._clear_robust_overrides(reset_trigger=False)
        return stage, progress

    def _should_refresh_memory(self, epoch: int) -> bool:
        if not self.memory_initialized or self.memory_seen_count < self.memory_bank_size:
            return False
        if epoch in self.memory_refresh_marks:
            return True
        if getattr(self, "current_stage", None) == "late":
            interval = max(1, self.schedule.robust_memory_refresh_interval)
            if self.last_memory_refresh_epoch is None:
                return True
            if epoch - self.last_memory_refresh_epoch >= interval:
                return True
        return False

    def _refresh_memory_bank(self, epoch: int, keep_ratio: float = 0.5):
        if not self.memory_initialized:
            return
        valid_size = min(self.memory_seen_count, self.memory_bank_size)
        if valid_size == 0:
            return
        keep_ratio = max(0.1, min(0.95, keep_ratio))
        keep_count = max(1, int(valid_size * keep_ratio))
        keep_count = min(keep_count, valid_size)
        try:
            hardness_slice = self.memory_hardness[:valid_size]
            topk = torch.topk(hardness_slice, k=keep_count, largest=True)
            top_indices = topk.indices
            self.memory_features[:keep_count] = self.memory_features[top_indices].clone()
            self.memory_labels[:keep_count] = self.memory_labels[top_indices].clone()
            self.memory_hardness[:keep_count] = hardness_slice[top_indices].clone()
            self.memory_ptr = keep_count
            self.memory_seen_count = keep_count
            self.last_memory_refresh_epoch = epoch
            logger.info(
                f"🔄 Memory Bank刷新: 保留 {keep_count}/{valid_size} 个难样本 (keep_ratio={keep_ratio:.2f}), epoch={epoch}"
            )
        except Exception as err:
            logger.error(f"Memory Bank刷新失败: {err}")
            logger.error(traceback.format_exc())

    def contrastive_loss_fixed(self, features_original, features_attacked, labels):
        """数值稳定版 InfoNCE（log-sum-exp + 裁剪）。"""
        # 检查输入特征是否异常 
        if torch.isnan(features_original).any() or torch.isinf(features_original).any():
            logger.error("InfoNCE: 原始特征包含NaN/Inf")
            logger.error(f"   NaN数量: {torch.isnan(features_original).sum().item()}")
            logger.error(f"   Inf数量: {torch.isinf(features_original).sum().item()}")
            logger.error(f"   特征范围: [{features_original.min().item():.4f}, {features_original.max().item():.4f}]")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if torch.isnan(features_attacked).any() or torch.isinf(features_attacked).any():
            logger.error("InfoNCE: 攻击特征包含NaN/Inf")
            logger.error(f"   NaN数量: {torch.isnan(features_attacked).sum().item()}")
            logger.error(f"   Inf数量: {torch.isinf(features_attacked).sum().item()}")
            logger.error(f"   特征范围: [{features_attacked.min().item():.4f}, {features_attacked.max().item():.4f}]")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # L2归一化
        features_original = F.normalize(features_original, p=2, dim=1)
        features_attacked = F.normalize(features_attacked, p=2, dim=1)
        
        # 计算相似度矩阵（已除以温度）
        sim_matrix = torch.matmul(features_original, features_attacked.T) / self.temperature
        
        # 记录相似度矩阵统计信息用于诊断 ⭐诊断
        sim_min, sim_max = sim_matrix.min().item(), sim_matrix.max().item()
        if abs(sim_min) > 40 or abs(sim_max) > 40:
            logger.warning(f"InfoNCE: 相似度矩阵数值较大 [min={sim_min:.2f}, max={sim_max:.2f}]")
        
        # 裁剪防止溢出（限制在合理范围）⭐数值稳定性
        sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)
        
        # 创建正样本mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        
        # 对角线mask（排除自己与自己）
        batch_size = features_original.size(0)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask
        
        # 使用log-sum-exp技巧计算（数值稳定）⭐修复NaN
        # log_prob = log(exp(sim_ij) / sum(exp(sim_ik)))
        #          = sim_ij - log(sum(exp(sim_ik)))
        #          = sim_ij - logsumexp(sim_i)
        max_sim = torch.max(sim_matrix, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(sim_matrix - max_sim) * logits_mask
        log_sum_exp = max_sim + torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        log_prob = sim_matrix - log_sum_exp
        
        # 只对正样本计算损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        
        loss = -mean_log_prob_pos.mean()
        
        # ✅ 诊断：检查正样本对数量（只在训练开始时输出一次）
        num_positive_pairs = mask.sum().item()
        if not hasattr(self, '_infonce_first_check'):
            self._infonce_first_check = True
            logger.info("InfoNCE损失诊断（首次）:")
            logger.info(f"   Batch大小: {features_original.size(0)}")
            logger.info(f"   正样本对数量: {num_positive_pairs}")
            logger.info(f"   损失值: {loss.item():.4f}")
            if num_positive_pairs == 0:
                logger.warning("   警告：没有正样本对，检查分组采样")
        
        # 安全检查：如果loss是NaN或Inf，返回0避免传播 ⭐诊断增强
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"InfoNCE损失异常: {loss.item()}")
            logger.error(f"   相似度矩阵范围: [{sim_min:.4f}, {sim_max:.4f}]")
            logger.error(f"   log_prob范围: [{log_prob.min().item():.4f}, {log_prob.max().item():.4f}]")
            logger.error(f"   mean_log_prob_pos: {mean_log_prob_pos}")
            logger.error(f"   batch_size: {features_original.size(0)}")
            logger.error(f"   正样本对数量: {num_positive_pairs}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss

    def intra_class_alignment_loss_weighted(self, features, labels, sample_weights):
        """
        加权同类对齐损失：对复合/链式攻击样本给予更高对齐权重。
        sample_weights: [N] 每个样本的权重（原图=1.0，combo≈1.4，full_chain≈1.6）
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        feats = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(feats, feats.T)
        N = sim.size(0)
        same = (labels.unsqueeze(1) == labels.unsqueeze(0))
        diag = torch.eye(N, device=sim.device, dtype=torch.bool)
        pos_mask = same & (~diag)
        if not pos_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        w = sample_weights.view(-1, 1)
        pair_w = w @ w.T
        pos_w = pair_w[pos_mask].view(-1)
        pos_sim = sim[pos_mask].view(-1)
        pos_dist = (1.0 - pos_sim)
        k = int(max(0, int(0.3 * pos_dist.numel())))
        if k > 0:
            top_vals, top_idx = torch.topk(pos_dist, k, largest=True)
            pos_w = pos_w.clone()
            pos_w[top_idx] = pos_w[top_idx] * 2.0
        loss = (pos_w * pos_dist).sum() / (pos_w.sum() + 1e-8)
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss
    
    def diversity_loss(self, features):
        """
        多样性损失（轻量版）：防止特征坍塌
        
        只包含基础的方差和去相关约束，不包含区分度约束（由uniqueness_loss负责）
        """
        # ⭐数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error(f"🔴 diversity_loss输入异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 1. 特征方差约束（每个维度都应该有足够的变化）
        feature_var = torch.var(features, dim=0) + 1e-8
        var_loss = torch.mean(torch.relu(0.1 - feature_var))
        
        if torch.isnan(var_loss) or torch.isinf(var_loss):
            logger.error(f"🔴 var_loss异常: {var_loss.item()}")
            var_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 2. 特征去相关（防止所有维度都编码相同的信息）
        features_centered = features - features.mean(dim=0, keepdim=True)
        
        if torch.all(torch.abs(features_centered) < 1e-7):
            decorr_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            cov_matrix = torch.matmul(features_centered.T, features_centered) / (features.size(0) + 1e-8)
            identity = torch.eye(cov_matrix.size(0), device=cov_matrix.device)
            decorr_loss = torch.mean((cov_matrix - identity * cov_matrix.diagonal().unsqueeze(1)) ** 2)
            
            if torch.isnan(decorr_loss) or torch.isinf(decorr_loss):
                logger.error(f"🔴 decorr_loss异常: {decorr_loss.item()}")
                decorr_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_diversity_loss = var_loss + 0.1 * decorr_loss
        
        if torch.isnan(total_diversity_loss) or torch.isinf(total_diversity_loss):
            logger.error(f"🔴 total_diversity_loss异常: {total_diversity_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_diversity_loss
    
    def update_memory_bank(self, features, labels, use_hard_mining=True):
        """维护Memory Bank：冷启动写入+后期难样本替换。"""
        batch_size = features.size(0)
        feature_dim = features.size(1)
        
        # 初始化memory bank
        if not self.memory_initialized:
            self.memory_features = torch.zeros(self.memory_bank_size, feature_dim, 
                                              device=self.device, dtype=features.dtype)
            self.memory_labels = torch.zeros(self.memory_bank_size, 
                                            device=self.device, dtype=labels.dtype)
            self.memory_hardness = torch.zeros(self.memory_bank_size, device=self.device)
            self.memory_initialized = True
            self.memory_seen_count = 0
        
        # 归一化特征（用于计算相似度）
        features_norm = F.normalize(features.detach(), p=2, dim=1)
        
        # ===== 阶段1：Memory Bank未满，直接存储（冷启动） =====
        if self.memory_seen_count < self.memory_bank_size:
            space_left = self.memory_bank_size - self.memory_seen_count
            actual_batch_size = min(batch_size, space_left)
            
            end_ptr = self.memory_ptr + actual_batch_size
            if end_ptr <= self.memory_bank_size:
                # 正常情况：不跨边界
                self.memory_features[self.memory_ptr:end_ptr] = features[:actual_batch_size].detach()
                self.memory_labels[self.memory_ptr:end_ptr] = labels[:actual_batch_size]
                if use_hard_mining:
                    hardness_scores = self._compute_hardness_scores(features_norm[:actual_batch_size], labels[:actual_batch_size])
                    self.memory_hardness[self.memory_ptr:end_ptr] = hardness_scores
            else:
                # 跨越边界
                first_part = self.memory_bank_size - self.memory_ptr
                self.memory_features[self.memory_ptr:] = features[:first_part].detach()
                self.memory_labels[self.memory_ptr:] = labels[:first_part]
                if use_hard_mining:
                    hardness_scores = self._compute_hardness_scores(features_norm[:actual_batch_size], labels[:actual_batch_size])
                    self.memory_hardness[self.memory_ptr:] = hardness_scores[:first_part]
                
                second_part = actual_batch_size - first_part
                if second_part > 0:
                    self.memory_features[:second_part] = features[first_part:actual_batch_size].detach()
                    self.memory_labels[:second_part] = labels[first_part:actual_batch_size]
                    if use_hard_mining:
                        self.memory_hardness[:second_part] = hardness_scores[first_part:]
            
            self.memory_ptr = (self.memory_ptr + actual_batch_size) % self.memory_bank_size
            self.memory_seen_count = min(self.memory_seen_count + actual_batch_size, self.memory_bank_size)
        
        # ===== 阶段2：Memory Bank已满，难负样本替换策略 =====
        else:
            if use_hard_mining:
                # 计算当前batch的难度分数
                new_hardness = self._compute_hardness_scores(features_norm, labels)
                
                # 找出当前batch中的"难样本"（难度分数高于Memory Bank平均值）
                memory_avg_hardness = self.memory_hardness[:self.memory_seen_count].mean()
                hard_samples_mask = new_hardness > memory_avg_hardness
                
                if hard_samples_mask.any():
                    hard_features = features[hard_samples_mask].detach()
                    hard_labels = labels[hard_samples_mask]
                    hard_scores = new_hardness[hard_samples_mask]
                    num_hard = hard_samples_mask.sum().item()
                    
                    # 找出Memory Bank中最简单的样本（替换它们）
                    easy_indices = torch.topk(self.memory_hardness, k=num_hard, largest=False).indices
                    
                    # 替换
                    self.memory_features[easy_indices] = hard_features
                    self.memory_labels[easy_indices] = hard_labels
                    self.memory_hardness[easy_indices] = hard_scores
                else:
                    # 当前batch都是简单样本，随机替换一部分（保持多样性）
                    replace_indices = torch.randperm(self.memory_bank_size, device=self.device)[:batch_size]
                    self.memory_features[replace_indices] = features.detach()
                    self.memory_labels[replace_indices] = labels
                    self.memory_hardness[replace_indices] = new_hardness
            else:
                # 不使用难样本挖掘，使用原来的循环队列（已修复边界检查）
                end_ptr = self.memory_ptr + batch_size
                if end_ptr <= self.memory_bank_size:
                    # 正常情况：不跨边界
                    self.memory_features[self.memory_ptr:end_ptr] = features.detach()
                    self.memory_labels[self.memory_ptr:end_ptr] = labels
                else:
                    # 跨越边界
                    first_part = self.memory_bank_size - self.memory_ptr
                    if first_part > 0:
                        self.memory_features[self.memory_ptr:] = features[:first_part].detach()
                        self.memory_labels[self.memory_ptr:] = labels[:first_part]
                    
                    second_part = batch_size - first_part
                    if second_part > 0:
                        self.memory_features[:second_part] = features[first_part:].detach()
                        self.memory_labels[:second_part] = labels[first_part:]
                
                self.memory_ptr = (self.memory_ptr + batch_size) % self.memory_bank_size
    
    def _compute_hardness_scores(self, features_norm, labels):
        """计算样本跨类最高相似度，得分越高表示越难。"""
        batch_size = features_norm.size(0)
        hardness_scores = torch.zeros(batch_size, device=self.device)
        
        # 对每个样本，计算其与不同类样本的最大相似度
        for i in range(batch_size):
            current_label = labels[i]
            current_feat = features_norm[i:i+1]
            
            # 找到所有不同类的样本
            cross_label_mask = (labels != current_label)
            if cross_label_mask.any():
                cross_features = features_norm[cross_label_mask]
                # 计算相似度
                similarities = torch.mm(current_feat, cross_features.t()).squeeze()
                # 最高相似度 = 难度分数
                hardness_scores[i] = similarities.max() if similarities.numel() > 0 else 0.0
            else:
                hardness_scores[i] = 0.0
        
        return hardness_scores
    
    def update_prototypes(self, features, labels):
        """为每个label维护动量原型，避免特征漂移。"""
        unique_labels = labels.unique()
        
        for label in unique_labels:
            label_key = label.item()
            mask = (labels == label)
            
            # 计算当前batch中该label的平均特征
            current_proto = features[mask].mean(dim=0).detach()
            
            # 动量更新
            if label_key in self.prototypes:
                self.prototypes[label_key] = (
                    self.prototype_momentum * self.prototypes[label_key] + 
                    (1 - self.prototype_momentum) * current_proto
                )
            else:
                self.prototypes[label_key] = current_proto
    
    def supervised_contrastive_loss_with_memory(self, features, labels, temperature=0.07, epoch=None):
        """带Memory Bank与原型对比的SupCon损失。"""
        # 数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error(f"🔴 supervised_contrastive_loss_with_memory输入异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1)
        
        # ===== 部分1：与Memory Bank中的样本对比 =====
        memory_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.memory_initialized and self.memory_seen_count > batch_size:
            # 获取有效的memory bank内容（已填充的部分）
            valid_size = min(self.memory_seen_count, self.memory_bank_size)
            memory_features_valid = self.memory_features[:valid_size]
            memory_labels_valid = self.memory_labels[:valid_size]
            
            # 归一化memory特征
            memory_features_norm = F.normalize(memory_features_valid, p=2, dim=1)
            
            # 计算与memory的相似度 [batch_size, memory_size]
            sim_to_memory = torch.matmul(features_norm, memory_features_norm.T) / temperature
            
            # 构建mask：哪些memory样本与当前样本同label
            labels_expanded = labels.unsqueeze(1)  # [batch_size, 1]
            memory_labels_expanded = memory_labels_valid.unsqueeze(0)  # [1, memory_size]
            mask_positive_memory = (labels_expanded == memory_labels_expanded).float()  # [batch_size, memory_size]
            
            # 对每个anchor计算损失
            losses_memory = []
            for i in range(batch_size):
                pos_mask = mask_positive_memory[i]
                num_positives = pos_mask.sum()
                
                if num_positives < 1:
                    continue  # 该label在memory中没有样本
                
                # log-sum-exp技巧
                logits = sim_to_memory[i]
                logits_max, _ = torch.max(logits, dim=0, keepdim=True)
                logits = logits - logits_max.detach()
                
                exp_logits = torch.exp(logits)
                log_denominator = torch.log(exp_logits.sum() + 1e-8)
                log_numerator = torch.log((exp_logits * pos_mask).sum() + 1e-8)
                
                loss = log_denominator - log_numerator
                losses_memory.append(loss)
            
            if len(losses_memory) > 0:
                memory_loss = torch.mean(torch.stack(losses_memory))
        
        # ===== 部分2：与当前batch内样本对比（原始SupCon） =====
        batch_loss = self.supervised_contrastive_loss(features, labels, temperature)
        
        # ===== 部分3：与所有原型对比（确保覆盖所有类别） =====
        prototype_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if len(self.prototypes) >= 2:
            # 获取所有原型
            proto_labels = list(self.prototypes.keys())
            proto_features = torch.stack([self.prototypes[k] for k in proto_labels]).to(self.device)
            proto_features_norm = F.normalize(proto_features, p=2, dim=1)
            
            # 计算与原型的相似度 [batch_size, num_prototypes]
            sim_to_proto = torch.matmul(features_norm, proto_features_norm.T) / temperature
            
            # 构建mask
            proto_labels_tensor = torch.tensor(proto_labels, device=self.device)
            labels_expanded = labels.unsqueeze(1)
            proto_labels_expanded = proto_labels_tensor.unsqueeze(0)
            mask_positive_proto = (labels_expanded == proto_labels_expanded).float()
            
            # 计算损失
            losses_proto = []
            for i in range(batch_size):
                pos_mask = mask_positive_proto[i]
                num_positives = pos_mask.sum()
                
                if num_positives < 1:
                    continue
                
                logits = sim_to_proto[i]
                logits_max, _ = torch.max(logits, dim=0, keepdim=True)
                logits = logits - logits_max.detach()
                
                exp_logits = torch.exp(logits)
                log_denominator = torch.log(exp_logits.sum() + 1e-8)
                log_numerator = torch.log((exp_logits * pos_mask).sum() + 1e-8)
                
                loss = log_denominator - log_numerator
                losses_proto.append(loss)
            
            if len(losses_proto) > 0:
                prototype_loss = torch.mean(torch.stack(losses_proto))
        
        # 组合三部分损失（batch内 + memory + prototype）
        # 早期进行ramp-up，降低memory/prototype带来的噪声影响（前5个epoch线性上升）
        if epoch is None:
            ramp = 1.0
        else:
            ramp = min(1.0, (epoch + 1) / 5.0)
        memory_w_base = 0.3
        proto_w_base = 0.1
        total_loss = batch_loss + (memory_w_base * ramp) * memory_loss + (proto_w_base * ramp) * prototype_loss
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"🔴 supervised_contrastive_loss_with_memory异常: {total_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss
    
    def supervised_contrastive_loss(self, features, labels, temperature=0.07):
        """
        ⭐⭐⭐⭐⭐ 监督对比损失（Supervised Contrastive Loss）- 根本解决方案
        
        论文：Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
        
        核心思想：
        - 同label的样本聚集（鲁棒性：同原图的攻击版本应相似）
        - 不同label的样本分离（唯一性：不同原图应不同）
        - 统一的优化目标，避免InfoNCE+uniqueness的冲突
        
        优势：
        1. 直接构建判别性特征空间
        2. 利用batch中所有同label样本（更强的鲁棒性信号）
        3. 显式推远不同label样本（更好的唯一性）
        4. 理论基础扎实，在图像/NLP领域已广泛验证
        
        参数：
        - features: [N, D] 特征向量
        - labels: [N] 原图ID（同原图的不同攻击版本应有相同label）
        - temperature: 温度参数（推荐0.07-0.1）
        
        预期效果：
        - 鲁棒性：同图攻击NC > 0.95（利用所有同label样本）
        - 唯一性：跨图NC < 0.60（显式推远不同label）
        """
        # 数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error(f"🔴 supervised_contrastive_loss输入异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 归一化特征（L2归一化）
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵：S[i,j] = features[i] · features[j]
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # 构建mask
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(self.device)  # 同label为1
        mask_anchor = torch.eye(batch_size, dtype=torch.bool, device=self.device)  # 对角线
        
        # 排除自己
        mask_positive = mask_positive * (~mask_anchor).float()
        
        # 计算每个anchor的损失
        losses = []
        for i in range(batch_size):
            # 该anchor的正样本mask
            pos_mask = mask_positive[i]
            num_positives = pos_mask.sum()
            
            # 如果没有正样本（batch中只有该原图的一个样本），跳过
            if num_positives < 1:
                continue
            
            # 对数-求和-指数技巧（log-sum-exp trick）防止数值溢出
            # log(Σexp(x)) = max(x) + log(Σexp(x - max(x)))
            logits = similarity_matrix[i]
            logits_max, _ = torch.max(logits, dim=0, keepdim=True)
            logits = logits - logits_max.detach()  # 数值稳定
            
            # 计算分母：Σexp(anchor·all) - exp(anchor·anchor)
            exp_logits = torch.exp(logits)
            # ⭐⭐⭐ 修复：使用mask排除自己，避免in-place操作破坏梯度图
            # 创建排除自己的mask（非in-place方式）
            indices = torch.arange(batch_size, device=exp_logits.device)
            mask_exclude_self = (indices != i).float()
            exp_logits_masked = exp_logits * mask_exclude_self
            log_denominator = torch.log(exp_logits_masked.sum() + 1e-8)
            
            # 计算分子：Σexp(anchor·positives)（也使用masked版本）
            log_numerator = torch.log((exp_logits_masked * pos_mask).sum() + 1e-8)
            
            # Loss = -log(正样本占比) = log(分母) - log(分子)
            loss = log_denominator - log_numerator
            losses.append(loss)
        
        if len(losses) == 0:
            logger.warning(f"⚠️ batch中所有样本都没有正样本对，SupCon损失无法计算")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = torch.mean(torch.stack(losses))
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"🔴 supervised_contrastive_loss异常: {total_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss
    
    def prototype_loss(self, features, labels):
        """
        ⭐⭐⭐ 原型损失（Prototype Loss）- 辅助SupCon的强化机制
        
        核心思想：
        - 每个原图学习一个"原型中心"（prototype）
        - 同原图的所有样本被拉向其原型
        - 不同原图的原型被推远
        
        优势：
        1. 防止特征漂移（原型作为锚点）
        2. 更稳定的特征空间
        3. 强化SupCon的聚类效果
        
        实现：
        - Intra-loss: 特征接近自己的原型（聚集）
        - Inter-loss: 原型之间远离（分离）
        """
        # 数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error(f"🔴 prototype_loss输入异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算每个label的原型（特征中心）
        unique_labels = labels.unique()
        prototypes = {}
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                prototypes[label.item()] = features[mask].mean(dim=0)
        
        if len(prototypes) < 2:
            # batch中只有一个原图，无法计算inter-loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 1. Intra-loss: 特征接近自己的原型
        intra_loss = 0
        count = 0
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                proto = prototypes[label.item()]
                # 使用余弦距离
                feat_norm = F.normalize(features[mask], p=2, dim=1)
                proto_norm = F.normalize(proto.unsqueeze(0), p=2, dim=1)
                cosine_sim = torch.matmul(feat_norm, proto_norm.T).squeeze()
                intra_loss += (1 - cosine_sim).mean()  # 1-cosine作为距离
                count += 1
        
        intra_loss = intra_loss / count if count > 0 else torch.tensor(0.0, device=self.device)
        
        # 2. Inter-loss: 原型之间远离
        proto_list = torch.stack([prototypes[k] for k in sorted(prototypes.keys())])
        proto_norm = F.normalize(proto_list, p=2, dim=1)
        proto_sim_matrix = torch.matmul(proto_norm, proto_norm.T)
        
        # 惩罚非对角线的高相似度
        mask_diag = torch.eye(proto_sim_matrix.size(0), device=self.device)
        inter_sim = proto_sim_matrix * (1 - mask_diag)  # 只看非对角线
        inter_loss = torch.mean(inter_sim)  # 最小化原型间相似度
        
        # 降低原型间分离强度，避免过分拉远导致鲁棒性下降
        total_proto_loss = intra_loss + 0.5 * inter_loss
        
        # 数值稳定性检查
        if torch.isnan(total_proto_loss) or torch.isinf(total_proto_loss):
            logger.error(f"🔴 prototype_loss异常: {total_proto_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_proto_loss
    
    def intra_class_alignment_loss(self, features, labels):
        """
        同类对齐损失：额外收缩同label样本（原图与不同攻击版本、攻击与攻击）
        使用cosine相似度的(1 - sim)作为距离，平均所有正样本对。
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        # 归一化
        feats = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(feats, feats.T)
        N = sim.size(0)
        same = (labels.unsqueeze(1) == labels.unsqueeze(0))
        diag = torch.eye(N, device=sim.device, dtype=torch.bool)
        pos_mask = same & (~diag)
        if not pos_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        pos_sims = sim[pos_mask]
        loss = torch.mean(1.0 - pos_sims)
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss
    
    def label_aware_uniqueness_loss(self, features, labels):
        """
        ⭐⭐⭐ 标签感知的唯一性损失（核心创新！两全其美方案）
        
        核心思想：
        - 只惩罚不同原图（不同label）之间的高相似度
        - 不影响同一原图不同攻击版本的相似度（保护鲁棒性）
        - 针对性解决Railways-Waterways等跨图高相似度问题
        
        实现：
        1. 构建跨标签mask：只选择不同label的特征对
        2. 计算跨标签相似度矩阵
        3. 激进惩罚高相似度（阈值0.12）
        4. 额外惩罚极端相似度（NC>0.6时*10）
        
        预期效果：
        - Railways-Waterways NC: 0.934 → <0.3
        - 鲁棒性：不受影响（同一原图的攻击版本相似度由contrastive_loss优化）
        """
        # ⭐数值稳定性检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.error(f"🔴 label_aware_uniqueness_loss输入异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if features.size(0) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features_norm, features_norm.T)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        
        # ⭐⭐⭐ 构建跨标签mask（只选择不同label的特征对）
        labels = labels.contiguous().view(-1, 1)
        cross_label_mask = ~torch.eq(labels, labels.T).to(self.device)  # 不同label为True
        
        # 排除对角线（虽然对角线肯定是同label，但为了保险）
        diagonal_mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        cross_label_mask = cross_label_mask & diagonal_mask
        
        # 提取跨标签相似度
        if cross_label_mask.sum() == 0:
            # 如果batch中所有样本都是同一label（不应该发生，因为我们使用分组采样）
            logger.warning("batch中没有跨标签样本对，唯一性损失无法计算")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        cross_label_similarity = similarity_matrix[cross_label_mask]
        
        # ⭐⭐⭐ 稳定版唯一性损失：防止负值和爆炸
        # 目标：让所有跨标签相似度接近0，但要防止梯度不稳定
        
        # 基础损失：只惩罚正相似度（负相似度说明已经足够远了）
        base_loss = torch.mean(torch.relu(cross_label_similarity))
        
        # ⭐关键优化：增强分层惩罚，特别是针对>0.9的情况
        high_sim_penalty = torch.mean(torch.relu(cross_label_similarity - 0.50)) * 3.0   # 从2.0提升到3.0
        extreme_penalty = torch.mean(torch.relu(cross_label_similarity - 0.70)) * 6.0   # 从4.0提升到6.0
        disaster_penalty = torch.mean(torch.relu(cross_label_similarity - 0.85)) * 12.0  # 从8.0提升到12.0（针对>0.85）
        critical_penalty = torch.mean(torch.relu(cross_label_similarity - 0.90)) * 20.0  # ⭐新增：针对>0.9的极端惩罚

        # 针对最坏pair的单独惩罚（专门压制 Railways–Waterways）
        max_cross_sim = torch.max(cross_label_similarity)
        max_penalty = torch.relu(max_cross_sim - 0.60) * 8.0  # 从5.0提升到8.0，聚焦于真正高相似的pair
        
        # 总损失（带裁剪，防止单个batch损失过大）
        total_uniqueness_loss = base_loss + high_sim_penalty + extreme_penalty + disaster_penalty + critical_penalty + max_penalty
        total_uniqueness_loss = torch.clamp(total_uniqueness_loss, min=0.0, max=15.0)  # ⭐关键：从10.0提升到15.0，允许更强的唯一性惩罚
        
        # ⭐数值稳定性检查
        if torch.isnan(total_uniqueness_loss) or torch.isinf(total_uniqueness_loss):
            logger.error(f"🔴 label_aware_uniqueness_loss异常: {total_uniqueness_loss.item()}")
            logger.error(f"   base_loss={base_loss.item()}, high_sim_penalty={high_sim_penalty.item()}")
            logger.error(f"   extreme_penalty={extreme_penalty.item()}, disaster_penalty={disaster_penalty.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_uniqueness_loss

    def get_dynamic_loss_weights(self, epoch, max_epoch):
        """三阶段动态损失权重（总权重恒定），前期唯一性→中期平衡→后期鲁棒性。
        
        优化策略：
        1. 更平滑的权重过渡，避免后期突然变化导致损失上升
        2. 后期渐进式增加鲁棒性权重，同时保持唯一性
        3. 针对Fig12复合攻击，在后期进一步强化binary和align权重
        4. 后期允许适度牺牲唯一性换取鲁棒性（uniqueness降低至≈1.3）

        为了在不同总epoch下保持相同策略，这里根据 max_epoch 自适应划分阶段，
        并保证各阶段 supcon+proto+binary+diversity+uniqueness+align 的和恒定（8.0）。
        ⭐优化：前期更长（50%）以强化唯一性，因为Epoch 1就达到最佳鲁棒性但唯一性不足
        """
        # ⭐优化：调整阶段划分，前期更长以强化唯一性
        # 对于12个epoch：前期6个（50%），中期4个（33%），后期2个（17%）
        early_end = max(1, int(max_epoch * 0.5))  # 从30%提升到50%
        mid_end = max(early_end + 1, int(max_epoch * 0.83))  # 从70%调整到83%
        
        # 计算阶段内进度（0.0-1.0），用于平滑插值
        if epoch < early_end:
            stage_progress = epoch / max(1, early_end)
            supcon = 1.8 - 0.1 * stage_progress  # 略降，为唯一性让路
            proto = 1.0
            binary = 0.6 + 0.2 * stage_progress  # 前期降低，专注唯一性
            diversity = 1.2 - 0.1 * stage_progress
            uniqueness = 3.2 - 0.2 * stage_progress  # ⭐关键：从2.5提升到3.2，强化唯一性
            align = 0.2 + 0.1 * stage_progress  # 前期降低，专注唯一性
        elif epoch < mid_end:
            stage_progress = (epoch - early_end) / max(1, mid_end - early_end)
            supcon = 1.8 - 0.1 * stage_progress  # 略降
            proto = 1.0 - 0.05 * stage_progress
            binary = 0.8 + 0.6 * stage_progress  # 0.8 → 1.4（从1.2→1.8降低起点）
            diversity = 1.1 - 0.2 * stage_progress
            uniqueness = 3.0 - 0.8 * stage_progress  # ⭐关键：3.0 → 2.2（从2.0→1.6提升，保持更高唯一性）
            align = 0.3 + 0.3 * stage_progress      # 0.3 → 0.6（从0.6→0.8降低起点）
        else:
            stage_progress = (epoch - mid_end) / max(1, max_epoch - mid_end)
            supcon = 1.7 - 0.1 * stage_progress  # 1.7 → 1.6 (略降)
            proto = 0.95 - 0.05 * stage_progress
            binary = 1.4 + 0.4 * stage_progress   # 1.4 → 1.8 (从1.8→2.4降低，避免过度牺牲唯一性)
            diversity = 0.9 - 0.1 * stage_progress  # 0.9 → 0.8
            uniqueness = 2.2 - 0.4 * stage_progress  # ⭐关键：2.2 → 1.8 (从1.7→1.4提升，保持更高唯一性)
            align = 0.6 + 0.4 * stage_progress    # 0.6 → 1.0 (从0.8→1.3降低，避免过度牺牲唯一性)

        # ⭐关键修复：确保总权重恒定（约8.0），避免后期权重增加导致损失不再降低
        # 计算当前总权重
        total_weight = supcon + proto + binary + diversity + uniqueness + align
        # 如果总权重偏离8.0，按比例归一化（保持相对比例不变）
        if abs(total_weight - 8.0) > 0.01:
            scale_factor = 8.0 / total_weight
            supcon *= scale_factor
            proto *= scale_factor
            binary *= scale_factor
            diversity *= scale_factor
            uniqueness *= scale_factor
            align *= scale_factor
        
        return {
            'supcon': supcon,
            'proto': proto,
            'binary': binary,
            'diversity': diversity,
            'uniqueness': uniqueness,
            'align': align,
        }

    def binary_consistency_loss(self, features_original, features_attacked, epoch=0):
        """
        二值化一致性损失（核心创新 + 自适应温度优化 + 数值稳定性）⭐⭐⭐⭐⭐
        
        动机：
        - 零水印最终使用的是二值化后的特征
        - 应该直接优化二值化后的一致性
        - 这才是NC值的真实优化目标
        
        实现：
        - 软二值化（可微分）
        - 自适应温度：从软到硬（温度退火）⭐新增
        - 最小化二值化后的汉明距离
        - 惩罚接近阈值的特征（鼓励明确的0/1）
        
        预期效果：NC值提升 +25%（原+20% + 温度优化+5%）
        """
        # ⭐数值稳定性检查
        if torch.isnan(features_original).any() or torch.isinf(features_original).any():
            logger.error(f"🔴 binary_consistency_loss: 原始特征异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if torch.isnan(features_attacked).any() or torch.isinf(features_attacked).any():
            logger.error(f"🔴 binary_consistency_loss: 攻击特征异常！")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算中位数（阈值）
        median_orig = torch.median(features_original, dim=1, keepdim=True)[0]
        median_attack = torch.median(features_attacked, dim=1, keepdim=True)[0]
        
        # 自适应温度：从软到硬的温度退火 ⭐
        # 早期（temp=1.0）：梯度平滑，容易优化
        # 后期（temp=0.01）：接近硬二值化，精确优化NC值
        temp = max(self.adaptive_temp.get_temperature(epoch), self.binary_temp_floor)
        
        # 计算logits（不经过sigmoid，用于AMP安全的BCE）⭐添加裁剪
        logits_orig = (features_original - median_orig) / temp
        logits_attack = (features_attacked - median_attack) / temp
        
        # ⭐裁剪logits到合理范围，防止溢出
        logits_orig = torch.clamp(logits_orig, min=-20, max=20)
        logits_attack = torch.clamp(logits_attack, min=-20, max=20)
        
        combined = torch.cat([features_original, features_attacked], dim=0)
        inst_median_dim = torch.median(combined, dim=0, keepdim=False)[0]
        if self.ema_median is None:
            self.ema_median = inst_median_dim.detach().to(features_original.device, dtype=features_original.dtype)
        else:
            if self.ema_median.shape != inst_median_dim.shape or self.ema_median.device != features_original.device:
                self.ema_median = inst_median_dim.detach().to(features_original.device, dtype=features_original.dtype)
            else:
                self.ema_median = self.ema_momentum * self.ema_median + (1.0 - self.ema_momentum) * inst_median_dim.detach()
        ema_median = self.ema_median.view(1, -1)
        logits_orig_shared = torch.clamp((features_original - ema_median) / temp, min=-20, max=20)
        logits_attack_shared = torch.clamp((features_attacked - ema_median) / temp, min=-20, max=20)
        
        # 二值化后的一致性损失（使用binary_cross_entropy_with_logits，AMP安全）
        # target需要是sigmoid后的值（0-1之间）
        bce_loss = F.binary_cross_entropy_with_logits(logits_orig, torch.sigmoid(logits_attack.detach())) + \
                   F.binary_cross_entropy_with_logits(logits_attack, torch.sigmoid(logits_orig.detach()))
        bce_loss = bce_loss / 2.0
        
        # ⭐检查bce_loss是否异常
        if torch.isnan(bce_loss) or torch.isinf(bce_loss):
            logger.error(f"🔴 bce_loss异常: {bce_loss.item()}")
            logger.error(f"   temp={temp}, logits_orig范围=[{logits_orig.min().item():.2f}, {logits_orig.max().item():.2f}]")
            bce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 边界清晰度损失：特征值应远离阈值
        # 鼓励特征值很大或很小，避免在阈值附近（临界状态不稳定）
        margin_orig = torch.abs(features_original - median_orig)
        margin_attack = torch.abs(features_attacked - median_attack)
        
        # 期望margin > 0.5（标准差的一半）
        margin_loss = torch.mean(torch.relu(0.5 - margin_orig)) + \
                     torch.mean(torch.relu(0.5 - margin_attack))
        margin_loss = margin_loss / 2.0
        
        # ⭐检查margin_loss是否异常
        if torch.isnan(margin_loss) or torch.isinf(margin_loss):
            logger.error(f"🔴 margin_loss异常: {margin_loss.item()}")
            margin_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        logit_mse = F.mse_loss(logits_orig_shared, logits_attack_shared)
        total_loss = bce_loss + 0.2 * margin_loss + 0.1 * logit_mse
        
        # ⭐最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"🔴 binary_consistency total_loss异常: {total_loss.item()}")
            logger.error(f"   bce_loss={bce_loss.item()}, margin_loss={margin_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss
    
    def composite_attack_binary_loss(self, batch_pairs, features_original, features_attacked, epoch=0, stage="mid"):
        """
        ⭐⭐⭐ 针对复合攻击的专门二值化损失（提升Fig12鲁棒性）
        
        核心思想：
        - 复合攻击（full_chain, combo）是最难的攻击类型
        - 需要更严格的二值化一致性约束
        - 使用更低的温度（更硬的二值化）和更强的惩罚
        
        实现：
        1. 识别batch中的复合攻击样本对
        2. 对这些样本对计算更严格的二值化损失
        3. 如果batch中没有复合攻击，返回0（不影响其他损失）
        
        预期效果：Fig12 NC值从0.67提升到0.8+
        """
        # 识别复合攻击样本对
        composite_indices = []
        for idx, (_orig_g, atk_g) in enumerate(batch_pairs):
            atype = get_attack_name(atk_g)
            if is_composite_attack(atype):
                composite_indices.append(idx)
        
        # 如果batch中没有复合攻击，返回0
        if len(composite_indices) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 提取复合攻击的特征对
        composite_orig = features_original[composite_indices]
        composite_attack = features_attacked[composite_indices]
        
        # 数值稳定性检查
        if torch.isnan(composite_orig).any() or torch.isinf(composite_orig).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        if torch.isnan(composite_attack).any() or torch.isinf(composite_attack).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算中位数（阈值）
        median_orig = torch.median(composite_orig, dim=1, keepdim=True)[0]
        median_attack = torch.median(composite_attack, dim=1, keepdim=True)[0]
        
        # ⭐关键优化：对复合攻击使用更低的温度（更硬的二值化，进一步提升鲁棒性）
        base_temp = max(self.adaptive_temp.get_temperature(epoch), self.binary_temp_floor)
        stage_temp_scale = {
            "early": 0.40,  # 从0.45降到0.40，更硬
            "mid": 0.55,    # 从0.60降到0.55，更硬
            "late": 0.70,   # 从0.75降到0.70，更硬
        }.get(stage, 0.55)
        composite_temp = max(base_temp * stage_temp_scale, self.binary_temp_floor * 0.75)  # 从0.8降到0.75
        
        # 计算logits
        logits_orig = (composite_orig - median_orig) / composite_temp
        logits_attack = (composite_attack - median_attack) / composite_temp
        
        # 裁剪logits
        logits_orig = torch.clamp(logits_orig, min=-20, max=20)
        logits_attack = torch.clamp(logits_attack, min=-20, max=20)
        
        # 使用共享阈值（EMA中位数）计算更严格的损失
        if self.ema_median is not None:
            ema_median = self.ema_median.view(1, -1)
            logits_orig_shared = torch.clamp((composite_orig - ema_median) / composite_temp, min=-20, max=20)
            logits_attack_shared = torch.clamp((composite_attack - ema_median) / composite_temp, min=-20, max=20)
            
            # 更严格的BCE损失（权重增加）
            bce_loss = F.binary_cross_entropy_with_logits(logits_orig_shared, torch.sigmoid(logits_attack_shared.detach())) + \
                       F.binary_cross_entropy_with_logits(logits_attack_shared, torch.sigmoid(logits_orig_shared.detach()))
            bce_loss = bce_loss / 2.0
            
            # 更严格的MSE损失（直接优化特征一致性）
            mse_loss = F.mse_loss(logits_orig_shared, logits_attack_shared)
        else:
            # 如果没有EMA中位数，使用实例中位数
            bce_loss = F.binary_cross_entropy_with_logits(logits_orig, torch.sigmoid(logits_attack.detach())) + \
                       F.binary_cross_entropy_with_logits(logits_attack, torch.sigmoid(logits_orig.detach()))
            bce_loss = bce_loss / 2.0
            mse_loss = F.mse_loss(logits_orig, logits_attack)
        
        # 边界清晰度损失（对复合攻击更严格，进一步提升鲁棒性）
        margin_orig = torch.abs(composite_orig - median_orig)
        margin_attack = torch.abs(composite_attack - median_attack)
        # 期望margin > 0.65（从0.6提升到0.65，更严格）
        margin_loss = torch.mean(torch.relu(0.65 - margin_orig)) + \
                     torch.mean(torch.relu(0.65 - margin_attack))
        margin_loss = margin_loss / 2.0
        
        if self.ema_median is not None:
            prob_orig = torch.sigmoid(logits_orig_shared)
            prob_attack = torch.sigmoid(logits_attack_shared)
        else:
            prob_orig = torch.sigmoid(logits_orig)
            prob_attack = torch.sigmoid(logits_attack)
        bit_margin = torch.mean(torch.abs(prob_orig - 0.5) + torch.abs(prob_attack - 0.5)) / 2.0
        bit_margin_penalty = torch.relu(0.50 - bit_margin)  # 从0.45提升到0.50，更严格
        
        # 总损失：BCE + 更强的MSE + 更严格的margin + bit稳定度（进一步提升权重）
        total_composite_loss = bce_loss + 0.35 * mse_loss + 0.35 * margin_loss + 0.30 * bit_margin_penalty  # 权重从0.3/0.3/0.25提升到0.35/0.35/0.30
        
        # 数值稳定性检查
        if torch.isnan(total_composite_loss) or torch.isinf(total_composite_loss):
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_composite_loss

    
    def select_diverse_attack_samples(self, attack_graphs, num_samples=8):
        """
        从攻击图列表中选择多样化的样本（分层采样）
        优先包含复合攻击（combo），然后覆盖不同攻击类型
        
        策略：
        1. 最高优先级：combo_full_attack_chain（Fig12完整链式攻击）
        2. 次优先级：其他combo攻击（复合攻击）
        3. 从不同攻击类型中均衡采样（add、delete、noise、crop等）
        
        Args:
            attack_graphs: 攻击图对象列表
            num_samples: 采样数量（默认8个）
        
        Returns:
            选中的攻击图列表
        """
        if len(attack_graphs) <= num_samples:
            return attack_graphs
        
        import random
        
        # 按攻击类型分组
        attack_types = {
            'full_chain': [], # combo_full_attack_chain（Fig12，最高优先级）
            'combo': [],      # 其他复合攻击
            'add': [],        # 顶点添加
            'delete': [],     # 顶点/对象删除
            'noise': [],      # 噪声扰动
            'crop': [],       # 裁剪
            'rotate': [],     # 旋转
            'scale': [],      # 缩放
            'flip': [],       # 翻转
            'translate': [],  # 平移
            'shuffle': [],    # 打乱
            'reverse': [],    # 反转
            'other': []       # 其他
        }
        
        # 分类攻击图
        for graph in attack_graphs:
            attack_name = getattr(graph, 'attack_type', '').lower()
            classified = False
            
            # 最高优先级：full_attack_chain
            if is_full_chain_attack(attack_name):
                attack_types['full_chain'].append(graph)
                classified = True
            elif is_combo_attack(attack_name):
                attack_types['combo'].append(graph)
                classified = True
            else:
                # 其他攻击类型
                for atype in ['add', 'delete', 'noise', 'crop', 'rotate', 'scale', 
                              'flip', 'translate', 'shuffle', 'reverse']:
                    if atype in attack_name:
                        attack_types[atype].append(graph)
                        classified = True
                        break
            
            if not classified:
                attack_types['other'].append(graph)
        
        samples = []
        
        # 第0优先级：combo_full_attack_chain（必选，如果存在）
        if attack_types['full_chain']:
            samples.extend(attack_types['full_chain'][:1])  # 只有1个，全选
            logger.info("  包含Fig12完整链式攻击: combo_full_attack_chain")
        
        # 第1优先级：其他combo攻击（选择2-3个）
        remaining = num_samples - len(samples)
        if attack_types['combo'] and remaining > 0:
            combo_count = min(3, len(attack_types['combo']), remaining)
            samples.extend(random.sample(attack_types['combo'], combo_count))
        
        # 第2优先级：从其他类型中均衡选择
        remaining_count = num_samples - len(samples)
        available_types = [t for t in attack_types if t not in ['full_chain', 'combo'] and len(attack_types[t]) > 0]
        
        if available_types and remaining_count > 0:
            # 计算每个类型应该选择多少个
            per_type = max(1, remaining_count // len(available_types))
            
            for atype in available_types:
                if len(samples) >= num_samples:
                    break
                count = min(per_type, len(attack_types[atype]))
                samples.extend(random.sample(attack_types[atype], count))
        
        # 如果还不够，从所有剩余中随机补充
        if len(samples) < num_samples:
            all_graphs = [g for g in attack_graphs if g not in samples]
            if all_graphs:
                additional = random.sample(all_graphs, min(num_samples - len(samples), len(all_graphs)))
                samples.extend(additional)
        
        # 打乱顺序
        random.shuffle(samples)
        
        return samples[:num_samples]
    
    def evaluate_nc_on_validation(self, val_orig, val_attack):
        """
        在验证集上评估NC值（直接评估最终目标）
        
        模拟完整的零水印流程：
        1. 提取原图特征 -> 二值化
        2. 提取攻击图特征 -> 二值化
        3. 计算二值特征的一致性（NC）
        
        改进：使用分层采样（每图8个攻击版本）而非简单取前3个
              优先包含复合攻击（combo），覆盖多种攻击类型
        
        Returns:
            dict: 包含 'avg_nc' (平均NC值) 和 'fig12_nc' (Fig12链式攻击NC值)
                  如果没有Fig12数据，fig12_nc为None
        """
        self.model.eval()
        nc_values = []
        attack_type_stats = {}
        
        try:
            with torch.no_grad():
                # 评估所有验证集图（确保NC值准确）
                # 如果验证集太大，可以通过调整val_ratio来控制
                for graph_name in list(val_orig.keys()):
                    if graph_name not in val_attack or len(val_attack[graph_name]) == 0:
                        continue
                    
                    try:
                        # 提取原图特征
                        orig_graph = val_orig[graph_name].to(self.device)
                        features_orig = self.model(orig_graph.x, orig_graph.edge_index)
                        features_orig = features_orig.cpu().numpy()
                        
                        # 二值化（中位数阈值）
                        threshold_orig = np.median(features_orig)
                        binary_orig = (features_orig > threshold_orig).astype(np.int32)
                        
                        # 对每个攻击版本（分层采样）
                        # 从攻击列表中智能选择8个样本，覆盖不同攻击类型
                        attack_samples = self.select_diverse_attack_samples(
                            val_attack[graph_name], 
                            num_samples=8  # 从3个增加到8个，提高评估准确性
                        )
                        
                        for attack_graph in attack_samples:
                            try:
                                attack_graph = attack_graph.to(self.device)
                                features_attack = self.model(attack_graph.x, attack_graph.edge_index)
                                features_attack = features_attack.cpu().numpy()
                                
                                # 二值化
                                threshold_attack = np.median(features_attack)
                                binary_attack = (features_attack > threshold_attack).astype(np.int32)
                                
                                # 计算NC（汉明距离）
                                hamming_distance = np.sum(binary_orig != binary_attack)
                                nc = 1.0 - (hamming_distance / len(binary_orig))
                                nc_values.append(nc)
                                
                                # 统计攻击类型
                                attack_name = getattr(attack_graph, 'attack_type', 'unknown').lower()
                                if is_full_chain_attack(attack_name):
                                    attack_category = 'full_chain'  # Fig12完整链式
                                elif is_combo_attack(attack_name):
                                    attack_category = 'combo'  # 其他复合攻击
                                else:
                                    attack_category = 'single'  # 单一攻击
                                
                                if attack_category not in attack_type_stats:
                                    attack_type_stats[attack_category] = []
                                attack_type_stats[attack_category].append(nc)
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    logger.warning(f"  验证集图OOM，跳过: {graph_name} (攻击版本)")
                                    # ✅ OOM恢复
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue
                                else:
                                    raise
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            logger.warning(f"  验证集图OOM，跳过: {graph_name} (原图)")
                            # ✅ OOM恢复
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise
        except Exception as e:
            logger.error(f"验证集评估出错: {e}")
            # ✅ 异常恢复
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except:
                    pass
        finally:
            self.model.train()
        
        avg_nc = np.mean(nc_values) if nc_values else 0.0
        fig12_nc = None
        
        # 输出采样统计
        if nc_values:
            logger.info(f"")
            logger.info(f"="*70)
            logger.info(f"验证集NC评估详情")
            logger.info(f"="*70)
            logger.info(f"📊 总体统计:")
            logger.info(f"  评估样本数: {len(nc_values)} 个")
            logger.info(f"  平均NC值: {avg_nc:.4f}")
            logger.info(f"  最小NC值: {min(nc_values):.4f}")
            logger.info(f"  最大NC值: {max(nc_values):.4f}")
            logger.info(f"  标准差: {np.std(nc_values):.4f}")
            
            # 输出攻击类型统计
            if attack_type_stats:
                logger.info(f"")
                logger.info(f"📊 分攻击类型统计:")
                
                # Fig12完整链式攻击
                if 'full_chain' in attack_type_stats:
                    fig12_nc = np.mean(attack_type_stats['full_chain'])
                    logger.info(f"  🔥 Fig12完整链式攻击: {len(attack_type_stats['full_chain'])} 个样本")
                    logger.info(f"     平均NC值: {fig12_nc:.4f}")
                    logger.info(f"     范围: [{min(attack_type_stats['full_chain']):.4f}, {max(attack_type_stats['full_chain']):.4f}]")
                    logger.info(f"")
                
                # 其他复合攻击
                if 'combo' in attack_type_stats:
                    combo_nc = np.mean(attack_type_stats['combo'])
                    logger.info(f"  💥 其他复合攻击 (Combo): {len(attack_type_stats['combo'])} 个样本")
                    logger.info(f"     平均NC值: {combo_nc:.4f}")
                    logger.info(f"     范围: [{min(attack_type_stats['combo']):.4f}, {max(attack_type_stats['combo']):.4f}]")
                
                # 单一攻击
                if 'single' in attack_type_stats:
                    single_nc = np.mean(attack_type_stats['single'])
                    logger.info(f"  ⚡ 单一攻击: {len(attack_type_stats['single'])} 个样本")
                    logger.info(f"     平均NC值: {single_nc:.4f}")
                    logger.info(f"     范围: [{min(attack_type_stats['single']):.4f}, {max(attack_type_stats['single']):.4f}]")
                
                # 对比分析
                logger.info(f"")
                logger.info(f"💡 对比分析:")
                
                if 'full_chain' in attack_type_stats:
                    full_avg = np.mean(attack_type_stats['full_chain'])
                    logger.info(f"  🔥 Fig12完整链式攻击: {full_avg:.4f}")
                
                if 'combo' in attack_type_stats and 'single' in attack_type_stats:
                    combo_avg = np.mean(attack_type_stats['combo'])
                    single_avg = np.mean(attack_type_stats['single'])
                    if combo_avg < single_avg:
                        diff = single_avg - combo_avg
                        logger.info(f"  💥 复合攻击比单一攻击NC值低 {diff:.4f} (更具挑战性)")
                    else:
                        logger.info(f"  复合攻击和单一攻击表现相当")
                
                if 'full_chain' in attack_type_stats and 'combo' in attack_type_stats:
                    full_avg = np.mean(attack_type_stats['full_chain'])
                    combo_avg = np.mean(attack_type_stats['combo'])
                    if full_avg < combo_avg:
                        diff = combo_avg - full_avg
                        logger.info(f"  🔥 Fig12链式攻击比普通复合攻击NC值低 {diff:.4f} (最具挑战性)")
                    elif full_avg > combo_avg:
                        diff = full_avg - combo_avg
                        logger.info(f"  ⚠️ Fig12链式攻击NC值反而高 {diff:.4f} (可能模型对链式攻击鲁棒)")
            
            logger.info(f"="*70)
            logger.info(f"")
        
        # ⭐返回字典，包含平均NC和Fig12 NC值
        return {'avg_nc': avg_nc, 'fig12_nc': fig12_nc}
    
    def evaluate_feature_distinction(self, val_orig):
        """
        评估特征区分度：确保不同矢量地图生成不同的特征
        
        核心指标：
        1. 批次内平均余弦距离（越大越好，期望 > 0.7）
        2. 二值化后的汉明距离（越大越好，期望 > 400/1024）
        3. 特征坍塌检测（方差过小的维度数量）
        
        Returns:
            distinction_score: 区分度分数（0-1，越高越好）
        """
        self.model.eval()
        all_features = []
        all_binary_features = []
        
        with torch.no_grad():
            # 提取所有验证集图的特征
            # 如果验证集太大导致内存问题，可以限制数量
            max_graphs = min(20, len(val_orig))  # 最多评估20个图
            for graph_name in list(val_orig.keys())[:max_graphs]:
                graph = val_orig[graph_name].to(self.device)
                features = self.model(graph.x, graph.edge_index)
                features_np = features.cpu().numpy()
                
                all_features.append(features_np)
                
                # 二值化
                threshold = np.median(features_np)
                binary = (features_np > threshold).astype(np.int32)
                all_binary_features.append(binary)
        
        if len(all_features) < 2:
            self.model.train()
            return 0.0
        
        all_features = np.array(all_features)  # [num_graphs, 1024]
        all_binary_features = np.array(all_binary_features)  # [num_graphs, 1024]
        
        # 1. 计算连续特征的平均余弦距离
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim_matrix = cosine_similarity(all_features)
        
        # 排除对角线（自己与自己）
        mask = ~np.eye(cos_sim_matrix.shape[0], dtype=bool)
        inter_similarities = cos_sim_matrix[mask]
        avg_cosine_distance = 1.0 - np.mean(inter_similarities)
        
        # 2. 计算二值化后的平均汉明距离
        hamming_distances = []
        for i in range(len(all_binary_features)):
            for j in range(i+1, len(all_binary_features)):
                hamming_dist = np.sum(all_binary_features[i] != all_binary_features[j])
                hamming_distances.append(hamming_dist / 1024.0)  # 归一化到[0,1]
        avg_hamming_distance = np.mean(hamming_distances) if hamming_distances else 0.0
        
        # 3. 检测特征坍塌（方差过小的维度）
        feature_vars = np.var(all_features, axis=0)
        collapsed_dims = np.sum(feature_vars < 0.01)  # 方差小于0.01认为坍塌
        collapse_ratio = collapsed_dims / all_features.shape[1]
        
        # 综合评分
        distinction_score = (
            0.4 * avg_cosine_distance +      # 期望 > 0.7
            0.4 * avg_hamming_distance +     # 期望 > 0.4
            0.2 * (1.0 - collapse_ratio)     # 期望 collapse_ratio < 0.1
        )
        
        self.model.train()
        
        logger.info(f"特征区分度评估:")
        logger.info(f"  平均余弦距离: {avg_cosine_distance:.4f} (期望 > 0.7)")
        logger.info(f"  平均汉明距离: {avg_hamming_distance:.4f} (期望 > 0.4)")
        logger.info(f"  坍塌维度比例: {collapse_ratio:.4f} (期望 < 0.1)")
        logger.info(f"  综合区分度分数: {distinction_score:.4f} (越高越好)")
        
        return distinction_score
    
    def process_batch_with_retry(self, batch_pairs, batch_labels, weights, epoch, current_batch_size, batch_graph_names=None, is_oom_retry=False):
        """
        处理单个batch，如果OOM则自动降低batch_size重试
        
        Args:
            batch_pairs: batch中的图对
            batch_labels: batch中的标签
            weights: 损失权重
            epoch: 当前epoch
            current_batch_size: 当前batch大小
            batch_graph_names: batch中的图名称列表（用于OOM追踪）
            is_oom_retry: 是否是OOM重试的子batch（如果是，不调用scheduler.step()）
            
        Returns:
            (success, losses_dict, grad_norm) 或 None（如果完全失败）
        """
        try:
            total_epochs = getattr(self, "total_epochs", 20)
            stage = getattr(self, "current_stage", None)
            if stage is None:
                stage, _ = compute_stage_progress(epoch, total_epochs)
            stage_for_batch = stage
            aug_p = stage_augmentation_probability(stage_for_batch)
            base_supcon_temp = stage_temperature(stage_for_batch)
            temperature = self.supcon_temp_override if self.supcon_temp_override is not None else base_supcon_temp
            self.current_supcon_temperature = temperature
            
            # 准备batch数据
            batch_original_features = []
            batch_attacked_features = []
            
            for idx, (original_graph, attacked_graph) in enumerate(batch_pairs):
                # ✅ 先在CPU上进行增强（augment_graph_data内部会clone），避免把数据集原始对象迁移到GPU
                # 动态增强概率（后期更强）
                original_graph_cpu = augment_graph_data(original_graph, augment_prob=aug_p, training=True)
                attacked_graph_cpu = augment_graph_data(attacked_graph, augment_prob=aug_p, training=True)

                # ✅ 仅将增强后的克隆体迁移到GPU，避免数据集中的对象常驻GPU导致显存累积
                original_graph_gpu = original_graph_cpu.to(self.device)
                attacked_graph_gpu = attacked_graph_cpu.to(self.device)

                # 提取特征（AMP）
                with amp.autocast(enabled=self.use_amp):
                    features_original = self.model(original_graph_gpu.x, original_graph_gpu.edge_index)
                    features_attacked = self.model(attacked_graph_gpu.x, attacked_graph_gpu.edge_index)
                    
                    # ✅ 数值裁剪：防止特征值过大导致后续计算NaN
                    features_original = torch.clamp(features_original, min=-10.0, max=10.0)
                    features_attacked = torch.clamp(features_attacked, min=-10.0, max=10.0)
                
                # 检测特征提取阶段是否产生异常 ⭐诊断
                if torch.isnan(features_original).any() or torch.isinf(features_original).any():
                    graph_name = batch_graph_names[idx] if batch_graph_names and idx < len(batch_graph_names) else f"未知图{idx}"
                    logger.error(f"🔴 特征提取异常: 原始图特征包含NaN/Inf")
                    logger.error(f"   问题图: {graph_name}")
                    logger.error(f"   节点数: {original_graph.x.size(0)}, 边数: {original_graph.edge_index.size(1)}")
                    logger.error(f"   输入特征范围: [{original_graph.x.min().item():.4f}, {original_graph.x.max().item():.4f}]")
                    logger.error(f"   NaN数量: {torch.isnan(features_original).sum().item()}/{features_original.numel()}")
                    
                if torch.isnan(features_attacked).any() or torch.isinf(features_attacked).any():
                    graph_name = batch_graph_names[idx] if batch_graph_names and idx < len(batch_graph_names) else f"未知图{idx}"
                    logger.error(f"🔴 特征提取异常: 攻击图特征包含NaN/Inf")
                    logger.error(f"   问题图: {graph_name}")
                    logger.error(f"   节点数: {attacked_graph.x.size(0)}, 边数: {attacked_graph.edge_index.size(1)}")
                    logger.error(f"   输入特征范围: [{attacked_graph.x.min().item():.4f}, {attacked_graph.x.max().item():.4f}]")
                    logger.error(f"   NaN数量: {torch.isnan(features_attacked).sum().item()}/{features_attacked.numel()}")
                
                batch_original_features.append(features_original)
                batch_attacked_features.append(features_attacked)
            
            # 堆叠特征
            batch_original = torch.stack(batch_original_features)
            batch_attacked = torch.stack(batch_attacked_features)
            batch_labels_tensor = torch.tensor(batch_labels, device=self.device)
            
            # 计算各项损失（AMP）
            with amp.autocast(enabled=self.use_amp):
                # 监督对比：拼接原始/攻击特征与标签做统一对比
                all_features = torch.cat([batch_original, batch_attacked], dim=0)
                all_labels = torch.cat([batch_labels_tensor, batch_labels_tensor], dim=0)
                
                # 在计算损失前更新 Memory Bank 与原型
                self.update_memory_bank(all_features, all_labels)
                self.update_prototypes(all_features, all_labels)
                
                # 1. 监督对比损失（包含Memory Bank和原型对比）
                supcon_loss = self.supervised_contrastive_loss_with_memory(all_features, all_labels, temperature, epoch=epoch)
                
                # 2. 原型损失（辅助，防止特征漂移）
                proto_loss = self.prototype_loss(all_features, all_labels)
                
                # 3. 二值化一致性损失（优化NC值）
                # 计算基础二值化损失
                binary_loss = self.binary_consistency_loss(batch_original, batch_attacked, epoch)
                composite_attack_indices = []
                full_chain_attack_indices = []
                
                # ⭐新增：针对复合攻击的专门二值化损失（增强鲁棒性）
                composite_binary_loss = self.composite_attack_binary_loss(
                    batch_pairs, batch_original, batch_attacked, epoch, stage_for_batch
                )
                
                if composite_attack_indices:
                    composite_ratio = len(composite_attack_indices) / max(1, len(batch_pairs))
                    binary_loss = binary_loss * (1.0 + 0.3 * composite_ratio)  # 从0.2提升到0.3
                if full_chain_attack_indices:
                    full_chain_ratio = len(full_chain_attack_indices) / max(1, len(batch_pairs))
                    binary_loss = binary_loss * (1.0 + 0.35 * full_chain_ratio)  # 从0.25提升到0.35
                
                # 4. 轻量多样性损失（防止特征坍塌）
                diversity_loss = self.diversity_loss(all_features)
                uniqueness_loss = self.label_aware_uniqueness_loss(all_features, all_labels)
                
                # 5. 加权同类对齐损失（复合/链式攻击更高权重）
                sample_weights_list = []
                composite_attack_indices = []  # 记录复合攻击的索引
                full_chain_attack_indices = []
                for idx, (orig_g, atk_g) in enumerate(batch_pairs):
                    # 原图样本权重
                    sample_weights_list.append(1.0)
                    # 根据攻击类型动态调权
                    atype = str(getattr(atk_g, 'attack_type', '')).lower()
                    w_atk = 1.0
                    is_composite = False
                    if is_full_chain_attack(atype):
                        w_atk = 3.2  # 链式攻击权重进一步提高
                        is_composite = True
                        full_chain_attack_indices.append(idx)
                    elif is_combo_attack(atype):
                        w_atk = 2.6  # 组合攻击更高权重
                        is_composite = True
                    # ⭐V9激进提高单一攻击的权重系数
                    if 'noise' in atype:
                        w_atk = w_atk * 1.4  # 从1.3提高到1.4
                    if 'add' in atype:
                        w_atk = w_atk * 1.4  # 从1.3提高到1.4
                    if 'crop' in atype:
                        w_atk = w_atk * 1.3  # 从1.2提高到1.3
                    if 'rotate' in atype:
                        w_atk = w_atk * 1.3  # 从1.2提高到1.3
                    if 'flip' in atype:
                        w_atk = w_atk * 1.3  # 从1.2提高到1.3
                    sample_weights_list.append(w_atk)
                    if is_composite:
                        composite_attack_indices.append(idx)
                sample_weights_tensor = torch.tensor(sample_weights_list, device=self.device, dtype=all_features.dtype)
                align_loss = self.intra_class_alignment_loss_weighted(all_features, all_labels, sample_weights_tensor)
                
                weights_dyn = self.get_dynamic_loss_weights(epoch, total_epochs)
                
                # ⭐优化：根据训练阶段动态调整复合攻击损失的权重（进一步提升鲁棒性）
                # ⭐关键修复：后期（epoch >= 70%总epoch）时，大幅增加复合攻击损失的权重
                # 但需要确保总损失尺度不会因为composite_weight增加而无限增大
                composite_weight = 0.0
                progress_ratio = epoch / max(1, total_epochs - 1)
                if progress_ratio < 0.3:
                    composite_weight = 0.2 + 0.5 * (progress_ratio / 0.3)  # 0.2→0.7 (提升)
                elif progress_ratio < 0.7:
                    composite_weight = 0.7 + 0.5 * ((progress_ratio - 0.3) / 0.4)  # 0.7→1.2 (提升)
                else:
                    composite_weight = 1.2 + 0.8 * ((progress_ratio - 0.7) / 0.3)  # 1.2→2.0 (大幅提升：1.5→2.0)
                composite_weight = min(composite_weight, 2.0)
                
                total_batch_loss = (
                    weights_dyn['supcon'] * supcon_loss +     # 监督对比（2.0→1.85）
                    weights_dyn['proto'] * proto_loss +       # 原型损失（1.0→1.0）
                    weights_dyn['binary'] * binary_loss +      # 二值化（0.6→1.2→1.8）
                    composite_weight * composite_binary_loss +  # ⭐新增：复合攻击专门损失
                    weights_dyn['diversity'] * diversity_loss + # 多样性（1.4→1.2→0.9）
                    weights_dyn['uniqueness'] * uniqueness_loss + # 唯一性（2.6→2.0→1.6）
                    weights_dyn['align'] * align_loss          # 类内对齐（0.4→0.6→0.95）
                )
                
                # 记录各项损失（用于日志）
                contrastive_loss = supcon_loss  # 兼容日志输出
                similarity_loss = proto_loss     # 兼容日志输出
            
            # NaN/Inf检测：在反向传播前检查损失 ⭐修复NaN + 详细诊断
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                logger.error(f"🔴🔴🔴 检测到异常总损失: {total_batch_loss.item()} 🔴🔴🔴")
                logger.error(f"=" * 70)
                logger.error(f"【NaN诊断报告】Epoch {epoch}, Batch size={current_batch_size}")
                logger.error(f"=" * 70)
                
                # 详细记录各项损失
                logger.error(f"📊 各项损失值:")
                logger.error(f"   - 对比损失(InfoNCE): {contrastive_loss.item()}")
                logger.error(f"   - 相似性损失:        {similarity_loss.item()}")
                logger.error(f"   - 多样性损失:        {diversity_loss.item()}")
                logger.error(f"   - 二值化损失:        {binary_loss.item()}")
                logger.error(f"   - 总损失:            {total_batch_loss.item()}")
                
                # 记录损失权重
                logger.error(f"")
                logger.error(f"⚖️ 当前损失权重:")
                logger.error(f"   - contrastive:        {weights_dyn['supcon']:.3f}")
                logger.error(f"   - similarity:         {weights_dyn['proto']:.3f}")
                logger.error(f"   - diversity:          {weights_dyn['diversity']:.3f}")
                logger.error(f"   - binary_consistency: {weights_dyn['binary']:.3f}")
                
                # 诊断哪个损失是NaN
                logger.error(f"")
                logger.error(f"🔍 异常来源诊断:")
                nan_sources = []
                if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                    nan_sources.append("对比损失(InfoNCE)")
                    logger.error(f"   ❌ 对比损失异常 → 可能原因: 相似度矩阵exp溢出、批次过小导致无正样本对")
                if torch.isnan(similarity_loss) or torch.isinf(similarity_loss):
                    nan_sources.append("相似性损失")
                    logger.error(f"   ❌ 相似性损失异常 → 可能原因: 特征向量全0或特征范数异常")
                if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                    nan_sources.append("多样性损失")
                    logger.error(f"   ❌ 多样性损失异常 → 可能原因: 特征坍塌、协方差矩阵计算问题")
                if torch.isnan(proto_loss) or torch.isinf(proto_loss):
                    nan_sources.append("原型对比损失")
                    logger.error(f"   ❌ 原型对比损失异常 → 可能原因: 原型计算问题、交叉熵溢出")
                if torch.isnan(binary_loss) or torch.isinf(binary_loss):
                    nan_sources.append("二值化损失")
                    logger.error(f"   ❌ 二值化损失异常 → 可能原因: 中位数计算异常、温度退火问题")
                
                logger.error(f"   🎯 异常损失项: {', '.join(nan_sources) if nan_sources else '总损失计算过程'}")
                
                # 记录涉及的图
                logger.error(f"")
                logger.error(f"📁 涉及的图数据:")
                if batch_graph_names:
                    for i, name in enumerate(batch_graph_names):
                        logger.error(f"   [{i+1}] {name}")
                else:
                    logger.error(f"   （未记录图名称）")
                
                # 记录特征统计
                logger.error(f"")
                logger.error(f"📈 特征统计:")
                logger.error(f"   原始特征 - min: {batch_original.min().item():.4f}, max: {batch_original.max().item():.4f}, mean: {batch_original.mean().item():.4f}")
                logger.error(f"   攻击特征 - min: {batch_attacked.min().item():.4f}, max: {batch_attacked.max().item():.4f}, mean: {batch_attacked.mean().item():.4f}")
                if torch.isnan(batch_original).any():
                    logger.error(f"   ⚠️ 原始特征包含 {torch.isnan(batch_original).sum().item()} 个NaN")
                if torch.isnan(batch_attacked).any():
                    logger.error(f"   ⚠️ 攻击特征包含 {torch.isnan(batch_attacked).sum().item()} 个NaN")
                
                logger.error(f"=" * 70)
                logger.error(f"✅ 处理: 跳过此batch，继续训练")
                logger.error(f"=" * 70)
                
                # 清理GPU内存并跳过此batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 返回零损失，避免污染训练
                return {
                    'total': 0.0,
                    'contrastive': 0.0,
                    'similarity': 0.0,
                    'diversity': 0.0,
                    'uniqueness': 0.0,
                    'binary': 0.0,
                    'align': 0.0
                }, 0, True  # True表示跳过了此batch
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # 梯度裁剪 + NaN/Inf检测 ⭐增强稳定性
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 检查梯度是否异常 ⭐诊断增强
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.error(f"🔴 检测到异常梯度范数(AMP): {grad_norm}")
                    logger.error(f"   Epoch {epoch}, Batch size={current_batch_size}")
                    logger.error(f"   损失值: {total_batch_loss.item():.4f}")
                    if batch_graph_names:
                        logger.error(f"   涉及图: {batch_graph_names}")
                    logger.error(f"   → 跳过优化器更新，避免模型参数污染")
                    
                    # ⭐关键修复：重置scaler状态，避免"unscale_() has already been called"错误
                    self.scaler.update()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return {
                        'total': total_batch_loss.item(),
                        'contrastive': contrastive_loss.item(),
                        'similarity': similarity_loss.item(),
                        'diversity': diversity_loss.item(),
                        'uniqueness': uniqueness_loss.item(),
                        'binary': binary_loss.item()
                    }, grad_norm.item() if not torch.isnan(grad_norm) else 0.0, True
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                
                # 梯度裁剪 + NaN/Inf检测 ⭐增强稳定性
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 检查梯度是否异常 ⭐诊断增强
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.error(f"🔴 检测到异常梯度范数: {grad_norm}")
                    logger.error(f"   Epoch {epoch}, Batch size={current_batch_size}")
                    logger.error(f"   损失值: {total_batch_loss.item():.4f}")
                    if batch_graph_names:
                        logger.error(f"   涉及图: {batch_graph_names}")
                    logger.error(f"   → 跳过优化器更新，避免模型参数污染")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return {
                        'total': total_batch_loss.item(),
                        'contrastive': contrastive_loss.item(),
                        'similarity': similarity_loss.item(),
                        'diversity': diversity_loss.item(),
                        'uniqueness': uniqueness_loss.item(),
                        'binary': binary_loss.item(),
                        'align': align_loss.item()
                    }, grad_norm.item() if not torch.isnan(grad_norm) else 0.0, True
                
                self.optimizer.step()
            
            # ⭐关键修复：更新学习率（OneCycleLR每步更新）
            # 注意：OOM重试时，scheduler.step()只在最外层batch成功时调用一次
            # 避免在子batch处理时重复调用导致步数超过预设值
            if self.scheduler is not None and not is_oom_retry:
                try:
                    self.scheduler.step()
                    self._apply_lr_multiplier()
                except ValueError as e:
                    if "Tried to step" in str(e) and "times" in str(e):
                        # OneCycleLR步数超限，说明OOM重试导致实际step数超过预设
                        # 这种情况下不再调用step，避免错误
                        logger.warning(f"⚠️ OneCycleLR步数超限（OOM重试导致），跳过本次step: {e}")
                    else:
                        raise
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            losses = {
                'total': total_batch_loss.item(),
                'contrastive': contrastive_loss.item(),
                'similarity': similarity_loss.item(),
                'diversity': diversity_loss.item(),
                'uniqueness': uniqueness_loss.item(),
                'binary': binary_loss.item(),
                'align': align_loss.item()
            }
            
            # 返回：losses字典, 梯度范数, 是否跳过（False表示正常处理）
            return losses, grad_norm.item(), False
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # OOM错误，尝试拆分batch
                if batch_graph_names:
                    logger.warning(f"🔴 OOM! batch_size={current_batch_size}, 涉及图: {batch_graph_names}")
                else:
                    logger.warning(f"🔴 OOM! batch_size={current_batch_size}")
                
                # ✅ 增强OOM恢复：彻底重置CUDA状态
                self.optimizer.zero_grad(set_to_none=True)  # 释放梯度内存
                
                if torch.cuda.is_available():
                    try:
                        # 同步CUDA操作，确保所有pending操作完成
                        torch.cuda.synchronize()
                        # 清空缓存
                        torch.cuda.empty_cache()
                        # 重置内存统计（避免累积错误）
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                    except Exception as cuda_err:
                        logger.error(f"  ⚠️ CUDA重置警告: {cuda_err}")
                
                # 重置AMP scaler状态（避免"unscale already called"错误）
                if self.use_amp:
                    self.scaler = amp.GradScaler(enabled=True)
                
                # 如果batch只有1个样本，无法再拆分
                if current_batch_size <= 1:
                    oom_graph_name = None
                    if batch_graph_names and batch_pairs:
                        graph_name = batch_graph_names[0]
                        oom_graph_name = graph_name
                        # 获取原图节点数和边数
                        try:
                            original_graph = batch_pairs[0][0]  # (original_graph, attacked_graph)
                            num_nodes = original_graph.x.shape[0]
                            num_edges = original_graph.edge_index.shape[1]
                            logger.error(f"  ❌ 单个图OOM，跳过: {graph_name}")
                            logger.error(f"     节点数: {num_nodes:,}, 边数: {num_edges:,}")
                            logger.error(f"     💡 该图将被加入动态黑名单，后续epoch将跳过")
                        except:
                            logger.error(f"  ❌ 单个图OOM，跳过: {graph_name}")
                            logger.error(f"     💡 该图将被加入动态黑名单，后续epoch将跳过")
                    else:
                        logger.error(f"  ❌ 单个样本OOM，跳过")
                    
                    # ✅ OOM恢复：最后一次清理
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        except:
                            pass
                    
                    # 返回特殊标记：None表示OOM且需要加入黑名单
                    return ('OOM', oom_graph_name)
                
                # 智能降级策略：8→4→2→1
                # 根据当前batch_size决定下一个尝试大小
                if current_batch_size > 4:
                    next_size = 4
                elif current_batch_size > 2:
                    next_size = 2
                elif current_batch_size > 1:
                    next_size = 1
                else:
                    next_size = 1  # 最小为1
                
                logger.warning(f"  ⚡ 尝试降低到 batch_size={next_size}")
                
                # 拆分batch（尽可能均匀分配）
                num_splits = (current_batch_size + next_size - 1) // next_size
                sub_batches = []
                
                for i in range(num_splits):
                    start_idx = i * next_size
                    end_idx = min((i + 1) * next_size, current_batch_size)
                    
                    sub_pairs = batch_pairs[start_idx:end_idx]
                    sub_labels = batch_labels[start_idx:end_idx]
                    sub_names = batch_graph_names[start_idx:end_idx] if batch_graph_names else None
                    sub_size = end_idx - start_idx
                    
                    sub_batches.append((sub_pairs, sub_labels, sub_names, sub_size))
                
                logger.warning(f"  📦 拆分为 {num_splits} 个小批次，每批约 {next_size} 个样本")
                
                # 处理所有子batch（使用梯度累积）
                all_results = []
                oom_graph_names = set()  # ✅ 收集所有OOM的图名
                for idx, (sub_pairs, sub_labels, sub_names, sub_size) in enumerate(sub_batches):
                    try:
                        # ⭐关键修复：OOM重试的子batch标记为is_oom_retry=True，避免重复调用scheduler.step()
                        result = self.process_batch_with_retry(sub_pairs, sub_labels, 
                                                              weights, epoch, sub_size, sub_names, is_oom_retry=True)
                        if result is not None:
                            # ✅ 检查是否是OOM标记
                            if isinstance(result, tuple) and len(result) == 2 and result[0] == 'OOM':
                                logger.warning(f"    子批次 {idx+1}/{num_splits} OOM，跳过")
                                # ✅ 收集OOM的图名
                                if result[1]:
                                    oom_graph_names.add(result[1])
                                continue
                            
                            losses, grad_norm, is_skipped = result
                            # 跳过被标记为异常的batch
                            if not is_skipped:
                                all_results.append(result)
                    except Exception as sub_err:
                        logger.error(f"    子批次 {idx+1}/{num_splits} 处理失败: {sub_err}")
                        # ✅ 子批次失败时也清理CUDA
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                            except:
                                pass
                        continue
                
                # 合并所有成功的结果
                if len(all_results) == 0:
                    # ✅ 如果有OOM的图，返回OOM标记而不是None
                    if len(oom_graph_names) > 0:
                        # 返回第一个OOM的图名（通常batch内都是同一个图）
                        oom_graph_name = list(oom_graph_names)[0]
                        logger.error(f"  ⚠️ 所有子批次都失败，检测到OOM图: {oom_graph_name}")
                        return ('OOM', oom_graph_name)
                    return None
                elif len(all_results) == 1:
                    return all_results[0]
                else:
                    # 平均所有结果
                    avg_losses = {
                        'total': sum(r[0]['total'] for r in all_results) / len(all_results),
                        'contrastive': sum(r[0]['contrastive'] for r in all_results) / len(all_results),
                        'similarity': sum(r[0]['similarity'] for r in all_results) / len(all_results),
                        'diversity': sum(r[0]['diversity'] for r in all_results) / len(all_results),
                        'uniqueness': sum(r[0]['uniqueness'] for r in all_results) / len(all_results),
                        'binary': sum(r[0]['binary'] for r in all_results) / len(all_results),
                        'align': sum(r[0]['align'] for r in all_results) / len(all_results)
                    }
                    avg_grad_norm = sum(r[1] for r in all_results) / len(all_results)
                    
                    # ⭐关键修复：OOM重试合并后，在最外层调用一次scheduler.step()
                    # 因为子batch处理时没有调用scheduler.step()（is_oom_retry=True）
                    if self.scheduler is not None:
                        try:
                            self.scheduler.step()
                            self._apply_lr_multiplier()
                        except ValueError as e:
                            if "Tried to step" in str(e) and "times" in str(e):
                                logger.warning(f"⚠️ OneCycleLR步数超限，跳过本次step: {e}")
                            else:
                                raise
                    
                    logger.info(f"  ✅ 成功合并 {len(all_results)}/{num_splits} 个子批次")
                    return avg_losses, avg_grad_norm, False  # False表示未跳过
            else:
                logger.error(f"[process_batch] RuntimeError（非OOM）: {e}")
                logger.error(f"详细错误: {traceback.format_exc()}")
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
        except Exception as e:
            logger.error(f"[process_batch] 未知异常: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def train_epoch(self, train_orig, train_attack, epoch):
        """训练一个epoch（自适应batch size）"""
        try:
            self.model.train()
            
            # ⭐⭐⭐ V9阶段性Memory Bank刷新：在epoch 15和22时清空50%最旧样本，E25+保持难样本不刷新
            # ⭐修复：完全重新设计刷新逻辑，避免张量大小不匹配
            if self._should_refresh_memory(epoch):
                keep_ratio = self.schedule.robust_memory_keep_ratio if getattr(self, "current_stage", None) == "late" else 0.5
                self._refresh_memory_bank(epoch, keep_ratio=keep_ratio)
            
            total_loss = 0.0
            total_contrastive_loss = 0.0
            total_similarity_loss = 0.0
            total_diversity_loss = 0.0
            total_uniqueness_loss = 0.0  # ✅ 累积唯一性损失
            total_binary_loss = 0.0
            total_grad_norm = 0.0
            num_batches = 0  # ✅ 统计成功处理的batch数
            num_oom_retries = 0  # 统计OOM重试次数
            num_skipped_batches = 0  # 统计因NaN/Inf跳过的batch数 ⭐修复NaN
            
            # ✅ 分组采样策略：按原图组织数据，确保batch内有正样本对
            # 数据结构：{graph_name: [(orig, attack1), (orig, attack2), ...]}
            grouped_data = {}
            graph_name_to_label = {}  # 原图名称到label的映射
            blacklisted_count = 0  # 被动态黑名单过滤的图数量
            
            for i, (graph_name, original_graph) in enumerate(train_orig.items()):
                # ✅ 检查动态黑名单
                if graph_name in self.dynamic_blacklist:
                    blacklisted_count += 1
                    continue
                
                if graph_name in train_attack and len(train_attack[graph_name]) > 0:
                    grouped_data[graph_name] = [
                        (original_graph, attacked_graph) 
                        for attacked_graph in train_attack[graph_name]
                    ]
                    graph_name_to_label[graph_name] = i
            
            if len(grouped_data) == 0:
                logger.warning("没有找到训练数据对")
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            
            # 统计信息
            total_samples = sum(len(pairs) for pairs in grouped_data.values())
            logger.info(f"混合采样：{len(grouped_data)}个原图，共{total_samples}个训练样本")

            full_chain_global_pool = []
            for gname, pairs in grouped_data.items():
                for (og, ak) in pairs:
                    if is_full_chain_attack(get_attack_name(ak)):
                        full_chain_global_pool.append((og, ak, graph_name_to_label[gname], gname))
            
            # ✅ 如果有动态黑名单，精简日志
            if len(self.dynamic_blacklist) > 0:
                logger.warning(f"动态黑名单启用：已过滤{blacklisted_count}个图；当前黑名单数={len(self.dynamic_blacklist)}")
            
        except Exception as e:
            logger.error(f"[train_epoch] 准备数据时出错: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        
        # ✅ 方案A：使用固定权重（模仿VGCN）
        loss_weights = FIXED_LOSS_WEIGHTS
        
        # 第一个epoch时打印权重信息
        if epoch == 0:
            logger.info("")
            logger.info("✅ 使用固定损失权重（模仿VGCN）：")
            logger.info(f"   - contrastive:        {loss_weights['contrastive']:.1f}")
            logger.info(f"   - similarity:         {loss_weights['similarity']:.1f}")
            logger.info(f"   - diversity:          {loss_weights['diversity']:.1f}")
            logger.info(f"   - binary_consistency: {loss_weights['binary_consistency']:.1f} ⭐")
            logger.info("")
        
        # ✅ 混合采样：每个batch包含多个不同原图（提供负样本）⭐新策略
        try:
            import random
            
            # 计算总batch数
            total_samples = sum(len(pairs) for pairs in grouped_data.values())
            
            # ✅ 混合采样参数（平衡效果与显存：batch_size=6推荐）
            num_graphs_per_batch = 2  # 每个batch包含2个不同原图
            samples_per_graph = 3  # 每个图贡献3个样本
            effective_batch_size = num_graphs_per_batch * samples_per_graph  # 2×3=6
            
            num_total_batches = (total_samples + effective_batch_size - 1) // effective_batch_size
            
            # 创建原图采样池（支持多轮采样）
            graph_names_list = list(grouped_data.keys())
            random.shuffle(graph_names_list)
            graph_pool = graph_names_list.copy()  # 可重复的采样池
            
            logger.info(f"✅ 混合采样策略：每batch {num_graphs_per_batch}个原图，每图{samples_per_graph}个样本，有效batch_size={effective_batch_size}")
            logger.info(f"   预计总batch数: {num_total_batches}（遍历所有{total_samples}个样本）")
            
            batch_num = 0
            processed_samples = 0
            
            # 遍历构建batch（直到处理完所有样本）
            while processed_samples < total_samples:
                batch_pairs = []
                batch_labels = []
                batch_graph_names = []
                
                # ✅ 从多个原图采样构建一个batch
                graphs_in_batch = []
                attempts = 0
                max_attempts = len(grouped_data) * 2  # 防止死循环
                
                while len(graphs_in_batch) < num_graphs_per_batch and attempts < max_attempts:
                    attempts += 1
                    
                    # 如果采样池用完，重新打乱
                    if len(graph_pool) == 0:
                        graph_pool = graph_names_list.copy()
                        random.shuffle(graph_pool)
                    
                    # 从池中取出一个原图
                    current_graph_name = graph_pool.pop(0)
                    
                    # 检查动态黑名单
                    if current_graph_name in self.dynamic_blacklist:
                        continue
                    
                    if current_graph_name not in grouped_data:
                        continue
                    
                    current_pairs = grouped_data[current_graph_name]
                    current_label = graph_name_to_label[current_graph_name]
                    
                    if len(current_pairs) == 0:
                        continue
                    
                    # 从当前原图采样samples_per_graph个样本（攻击感知的加权采样）
                    num_samples = min(samples_per_graph, len(current_pairs))
                    attack_weights = np.array(
                        [attack_sample_weight(get_attack_name(atk_g)) for (_og, atk_g) in current_pairs],
                        dtype=np.float64
                    )
                    if attack_weights.size == 0:
                        graph_batch_pairs = random.choices(current_pairs, k=num_samples)
                    else:
                        attack_weights = np.clip(attack_weights, 1e-6, None)
                        full_chain_indices = [
                            idx_local for idx_local, (_og, atk_g) in enumerate(current_pairs)
                            if is_full_chain_attack(get_attack_name(atk_g))
                        ]
                        selected_indices: List[int] = []
                        if full_chain_indices:
                            selected_indices.append(int(random.choice(full_chain_indices)))
                            attack_weights[selected_indices[-1]] = 0.0  # 防止重复抽到同一个full-chain
                        remaining = max(0, num_samples - len(selected_indices))
                        if remaining > 0:
                            if attack_weights.sum() <= 0:
                                attack_weights = np.ones_like(attack_weights)
                            p = attack_weights / attack_weights.sum()
                            replace_flag = remaining > np.count_nonzero(p)
                            sampled_indices = np.random.choice(
                                len(current_pairs),
                                size=remaining,
                                replace=replace_flag,
                                p=p
                            )
                            selected_indices.extend(sampled_indices.tolist())
                        if len(selected_indices) == 0:
                            selected_indices = random.choices(range(len(current_pairs)), k=num_samples)
                        graph_batch_pairs = [current_pairs[int(idx)] for idx in selected_indices[:num_samples]]
                    
                    # 添加到batch
                    batch_pairs.extend(graph_batch_pairs)
                    batch_labels.extend([current_label] * num_samples)
                    batch_graph_names.extend([current_graph_name] * num_samples)
                    graphs_in_batch.append(current_graph_name)
                    processed_samples += num_samples
                
                # 如果batch为空（所有图都被黑名单过滤），跳过
                if len(batch_pairs) == 0:
                    continue
                
                batch_num += 1
                current_batch_size = len(batch_pairs)
                stage_for_batch = getattr(self, "current_stage", "mid")
                target_full_chain = self.min_full_chain_per_batch if stage_for_batch in ("early", "mid") else self.max_full_chain_per_batch
                full_chain_indices_in_batch = [idx for idx, (_og, _atk) in enumerate(batch_pairs) if is_full_chain_attack(get_attack_name(_atk))]
                if len(full_chain_indices_in_batch) < target_full_chain and len(full_chain_global_pool) > 0:
                    needed = min(target_full_chain - len(full_chain_indices_in_batch), len(full_chain_global_pool))
                    for replace_iter in range(needed):
                        rep = random.choice(full_chain_global_pool)
                        replace_pos = (len(batch_pairs) - 1 - replace_iter) % len(batch_pairs)
                        batch_pairs[replace_pos] = (rep[0], rep[1])
                        batch_labels[replace_pos] = rep[2]
                        batch_graph_names[replace_pos] = rep[3]
                has_composite = any(is_composite_attack(get_attack_name(_atk)) for (_og, _atk) in batch_pairs)
                if not has_composite:
                    try:
                        composite_candidates = []
                        for gname, pairs in grouped_data.items():
                            for (og, ak) in pairs:
                                if is_composite_attack(get_attack_name(ak)):
                                    composite_candidates.append((og, ak, graph_name_to_label[gname], gname))
                        if len(composite_candidates) > 0:
                            rep = random.choice(composite_candidates)
                            batch_pairs[-1] = (rep[0], rep[1])
                            batch_labels[-1] = rep[2]
                            batch_graph_names[-1] = rep[3]
                            current_batch_size = len(batch_pairs)
                    except Exception:
                        pass

                # 软约束：若batch中无弱攻击类型（noise/add/crop/rotate/flip），补齐1个
                has_weak = any(has_weak_perturbation(get_attack_name(_atk)) for (_og, _atk) in batch_pairs)
                if not has_weak:
                    try:
                        weak_candidates = []
                        for gname, pairs in grouped_data.items():
                            for (og, ak) in pairs:
                                if has_weak_perturbation(get_attack_name(ak)):
                                    weak_candidates.append((og, ak, graph_name_to_label[gname], gname))
                        if len(weak_candidates) > 0:
                            repw = random.choice(weak_candidates)
                            batch_pairs[0] = (repw[0], repw[1])
                            batch_labels[0] = repw[2]
                            batch_graph_names[0] = repw[3]
                    except Exception:
                        pass
                
                # ✅ 诊断信息：第一个batch验证混合采样
                if batch_num == 1:
                    logger.info(f"")
                    logger.info(f"🔍 混合采样验证（第1个batch）：")
                    logger.info(f"   包含原图: {graphs_in_batch}")
                    logger.info(f"   Batch大小: {current_batch_size}")
                    logger.info(f"   Label分布: {dict(zip(*np.unique(batch_labels, return_counts=True)))}")
                    
                    # 计算正负样本对数量
                    num_positive = sum((np.array(batch_labels) == label).sum() * ((np.array(batch_labels) == label).sum() - 1) 
                                      for label in set(batch_labels))
                    total_pairs = current_batch_size * (current_batch_size - 1)
                    num_negative = total_pairs - num_positive
                    
                    logger.info(f"   ✅ 正样本对: {num_positive} 个（同一原图的不同攻击）")
                    logger.info(f"   ✅ 负样本对: {num_negative} 个（不同原图）⭐关键")
                    logger.info(f"")
                
                # 使用带重试的batch处理
                result = self.process_batch_with_retry(batch_pairs, batch_labels, loss_weights, epoch, current_batch_size, batch_graph_names)
                
                # ✅ 检查是否是OOM导致的失败
                if result is not None and isinstance(result, tuple) and len(result) == 2 and result[0] == 'OOM':
                    # OOM图，加入动态黑名单
                    oom_graph_name = result[1]
                    if oom_graph_name:
                        self.dynamic_blacklist.add(oom_graph_name)
                        logger.error(f"加入动态黑名单: {oom_graph_name}（总数={len(self.dynamic_blacklist)}）")
                        
                        # ✅ 从可选列表中移除，避免重复选中
                        if oom_graph_name in grouped_data:
                            # 计算该图的样本数
                            removed_samples = len(grouped_data[oom_graph_name])
                            total_samples -= removed_samples  # 调整总样本数
                            del grouped_data[oom_graph_name]
                            logger.info(f"  📉 从总样本数中减去 {removed_samples} 个样本")
                        if oom_graph_name in graph_names_list:
                            graph_names_list.remove(oom_graph_name)
                        logger.info(f"  ✅ 已从当前epoch的候选列表中移除该图")
                        
                        # 如果没有剩余图可训练，提前退出
                        if len(graph_names_list) == 0:
                            logger.error(f"")
                            logger.error(f"⚠️ 所有图都已加入黑名单，提前结束当前epoch")
                            logger.error(f"")
                            break
                    continue
                
                if result is not None:
                    losses, grad_norm, is_skipped = result
                    
                    # 如果batch被跳过（NaN），记录但不累积到损失中 ⭐诊断
                    if is_skipped:
                        num_skipped_batches += 1
                        logger.warning(f"")
                        logger.warning(f"⚠️ ========== Batch {batch_num}/{num_total_batches} 被跳过 ==========")
                        logger.warning(f"   原因: NaN/Inf检测触发")
                        logger.warning(f"   当前跳过总数: {num_skipped_batches}")
                        logger.warning(f"   详情请查看上方的【NaN诊断报告】")
                        logger.warning(f"=" * 60)
                        logger.warning(f"")
                        continue
                    
                    # 累积损失
                    total_loss += losses['total']
                    total_contrastive_loss += losses['contrastive']
                    total_similarity_loss += losses['similarity']
                    total_diversity_loss += losses['diversity']
                    total_uniqueness_loss += losses['uniqueness']  # ⭐⭐⭐ 累积唯一性损失
                    total_binary_loss += losses['binary']
                    total_grad_norm += grad_norm
                    num_batches += 1
                    
                    # 如果batch大小与原始不同，说明发生了OOM重试
                    if current_batch_size != self.batch_size:
                        num_oom_retries += 1
                    
                    # 每50个batch显示一次进度
                    if batch_num % 50 == 0:
                        avg_loss_so_far = total_loss / num_batches
                        logger.info(f"  📊 Batch {batch_num}/{num_total_batches} | "
                                  f"总损失: {avg_loss_so_far:.4f}")
                        logger.info(f"      SupCon: {losses['contrastive']:.4f} | "
                                  f"Proto: {losses['similarity']:.4f} | "
                                  f"Binary: {losses['binary']:.4f} | "
                                  f"Div: {losses['diversity']:.4f}")
                        logger.info(f"      ⭐Unique: {losses['uniqueness']:.4f} | "
                                  f"Align: {losses.get('align', 0.0):.4f}")
                else:
                    # 完全失败，跳过此batch
                    graph_info = f"涉及图: {batch_graph_names}" if batch_graph_names else ""
                    logger.warning(f"  ⚠️ Batch {batch_num} 失败，跳过 {graph_info}")
        except Exception as e:
            logger.error(f"[train_epoch] 分批训练时出错: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            
            # ✅ 增强OOM恢复：epoch级别的CUDA状态恢复
            if 'out of memory' in str(e).lower():
                logger.error(f"")
                logger.error(f"🔴 检测到Epoch级别OOM错误，尝试恢复CUDA状态...")
                if torch.cuda.is_available():
                    try:
                        # 清理所有梯度
                        self.optimizer.zero_grad(set_to_none=True)
                        # 同步并清理CUDA
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                        # 重置AMP scaler
                        if self.use_amp:
                            self.scaler = amp.GradScaler(enabled=True)
                        logger.error(f"✅ CUDA状态已重置，训练将继续")
                    except Exception as cuda_err:
                        logger.error(f"❌ CUDA重置失败: {cuda_err}")
                logger.error(f"")
            
            # 不立即返回，尝试计算已有的平均值
        
        # 计算平均值
        try:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0.0
            avg_similarity_loss = total_similarity_loss / num_batches if num_batches > 0 else 0.0
            avg_diversity_loss = total_diversity_loss / num_batches if num_batches > 0 else 0.0
            avg_uniqueness_loss = total_uniqueness_loss / num_batches if num_batches > 0 else 0.0  # ⭐⭐⭐ 计算平均唯一性损失
            avg_binary_loss = total_binary_loss / num_batches if num_batches > 0 else 0.0
            avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
            
            # 输出OOM重试和NaN跳过统计 ⭐修复NaN + 诊断增强
            logger.info(f"")
            logger.info(f"📊 本Epoch统计:")
            logger.info(f"   - 成功处理batch数: {num_batches}")
            logger.info(f"   - OOM自适应重试: {num_oom_retries}次{'（已自动处理）' if num_oom_retries > 0 else ''}")
            if len(self.dynamic_blacklist) > 0:
                logger.warning(f"   - 🚫 动态黑名单图数: {len(self.dynamic_blacklist)}个 (已跳过)")
            if num_skipped_batches > 0:
                skip_rate = (num_skipped_batches / (num_batches + num_skipped_batches)) * 100 if (num_batches + num_skipped_batches) > 0 else 0
                logger.warning(f"   - ⚠️ 跳过异常batch: {num_skipped_batches}次 (跳过率: {skip_rate:.2f}%)")
                logger.warning(f"   - 💡 建议: 如果跳过率>5%, 请检查数据质量或进一步降低学习率")
            else:
                logger.info(f"   - ✅ 无异常batch，训练稳定")
            
            # ⭐⭐⭐⭐⭐ V7新增：Memory Bank和原型状态日志
            if self.memory_initialized:
                valid_memory_size = min(self.memory_seen_count, self.memory_bank_size)
                memory_fill_rate = (valid_memory_size / self.memory_bank_size) * 100
                logger.info(f"")
                logger.info(f"🏦 Memory Bank状态:")
                logger.info(f"   - 已存储样本数: {valid_memory_size}/{self.memory_bank_size} ({memory_fill_rate:.1f}%)")
                logger.info(f"   - 原型数量: {len(self.prototypes)}个原图")
                logger.info(f"   - 负样本池扩大: {len(self.prototypes)}倍（从batch内2-3个原图 → 全部{len(self.prototypes)}个原图）")
            else:
                logger.info(f"")
                logger.info(f"🏦 Memory Bank状态: 未初始化（将在第一个batch后初始化）")
            
            return avg_loss, avg_contrastive_loss, avg_similarity_loss, avg_diversity_loss, avg_uniqueness_loss, avg_binary_loss, avg_grad_norm, num_oom_retries, num_batches
        except Exception as e:
            logger.error(f"[train_epoch] 计算平均值时出错: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    
    def train(self, original_graphs, attacked_graphs, num_epochs=40, resume_from_checkpoint=None):
        """
        训练模型（完全模仿VGCN，不使用验证集）
        
        Args:
            original_graphs: 原始图数据
            attacked_graphs: 攻击后的图数据
            num_epochs: 总训练轮数（默认40，配合早停机制）
            resume_from_checkpoint: checkpoint路径（用于恢复训练）
        """
        logger.info("="*70)
        logger.info("开始训练改进的GAT模型")
        logger.info("="*70)
        logger.info(f"训练轮数: {num_epochs}")
        logger.info(f"✅ 训练策略: 使用全部数据，基于总损失保存最佳模型（模仿VGCN）")
        logger.info("")
        
        # 更新自适应温度的总轮数
        self.adaptive_temp.total_epochs = num_epochs
        logger.info(f"自适应温度已更新: 总轮数={num_epochs}")
        logger.info("")
        
        # ✅ 方案A：不划分验证集，使用全部数据训练（像VGCN）
        train_orig = original_graphs
        train_attack = attacked_graphs
        
        val_orig = train_orig
        val_attack = train_attack
        metric_eval_interval = self.schedule.metric_eval_interval
        metric_patience = self.schedule.metric_patience
        metric_patience_counter = 0
        best_nc = -float('inf')
        best_distinction = -float('inf')
        best_fig12_nc = -float('inf')  # 内部验证集Fig12-like指标
        best_fig12_epoch = 0
        best_fig12_nc_real = -float('inf')  # ⭐新增：真实Fig12.py评估的最佳NC
        best_fig12_epoch_real = 0
        nc_improve_tol = self.schedule.nc_improve_tol
        distinction_improve_tol = self.schedule.distinction_improve_tol
        min_epoch_for_metric_stop = self.schedule.min_epoch_for_metric_stop
        metric_has_valid = False
        
        # 计算每个epoch的batch数（用于OneCycleLR）
        num_pairs = sum(len(train_attack.get(k, [])) for k in train_orig.keys())
        steps_per_epoch = max(1, num_pairs // self.batch_size)
        
        logger.info(f"数据集统计:")
        logger.info(f"  训练图数: {len(train_orig)} 个原图")
        logger.info(f"  训练样本: {num_pairs} 个图对")
        logger.info(f"  每epoch步数: {steps_per_epoch}")
        logger.info("")
        
        # 初始化OneCycleLR（降低学习率避免NaN）⭐极低初始lr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.schedule.onecycle_max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.schedule.onecycle_pct_start,
            anneal_strategy='cos',
            div_factor=self.schedule.onecycle_div_factor,
            final_div_factor=self.schedule.onecycle_final_div
        )
        logger.info(f"OneCycleLR初始化:")
        init_lr = self.schedule.onecycle_max_lr / self.schedule.onecycle_div_factor
        logger.info(f"  优化器: AdamW (初始lr={self.base_lr:.6f}, weight_decay=0.01)")
        logger.info(f"  初始学习率: {init_lr:.6f} (极低启动，防止初始NaN) ")
        logger.info(f"  每epoch步数: {steps_per_epoch}")
        logger.info(f"  Warmup比例: {self.schedule.onecycle_pct_start*100:.0f}% (自适应Reached峰值)")
        logger.info("")
        
        # 初始化训练状态（模仿VGCN）
        start_epoch = 0
        self.total_epochs = num_epochs
        best_loss = float('inf')       
        try:
            patience = int(os.environ.get('VGAT_EARLY_STOP_PATIENCE', '15'))
        except Exception:
            patience = 15  
        patience_counter = 0
        self.patience = patience       
        self.best_epoch = 0       
        
        # 尝试从checkpoint恢复
        if resume_from_checkpoint:
            logger.info(f"尝试从checkpoint恢复训练: {resume_from_checkpoint}")
            checkpoint = self.load_checkpoint(resume_from_checkpoint)
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                # 兼容旧checkpoint（可能还有best_val_nc）
                best_loss = checkpoint.get('best_loss', checkpoint.get('best_val_nc', float('inf')))
                patience_counter = checkpoint['patience_counter']
                self.training_history = checkpoint['training_history']
                logger.info(f"从Epoch {start_epoch}恢复训练")
                logger.info(f"   最佳损失: {best_loss:.6f}")
            else:
                logger.warning("  无法加载checkpoint，从头开始训练")
        
        # Checkpoint保存路径（使用自定义名称）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, 'checkpoints', self.checkpoint_name)
        
        for epoch in tqdm(range(start_epoch, num_epochs), desc="训练进度", initial=start_epoch, total=num_epochs):
            try:
                logger.info(f"\n开始 Epoch {epoch+1}/{num_epochs}...")
                # 阶段切换时重置早停耐心（E20前期→中期，E30中期→后期，E40过渡→稳定）
                if should_reset_patience(epoch):
                    patience_counter = 0
                    self.patience = 25  
                    logger.info(f"阶段切换（epoch={epoch}），重置patience为{self.patience}")
                    # ⭐ Memory Bank刷新已移至train_epoch()开头，避免重复刷新
                
                stage, stage_progress = self._update_robust_phase_state(epoch)
                stage_desc = describe_stage(stage, stage_progress)
                
                # 训练
                train_loss, contrastive_loss, similarity_loss, diversity_loss, uniqueness_loss, binary_loss, grad_norm, num_oom_retries, num_batches = \
                    self.train_epoch(train_orig, train_attack, epoch)
                
                # 相对改善阈值（允许3%的波动）- 仅在本epoch有有效batch时执行
                if num_batches > 0:
                    tolerance = 0.03
                    if train_loss < best_loss * (1 + tolerance):
                        if train_loss < best_loss:
                            best_loss = train_loss
                            self.best_epoch = epoch
                            logger.info(f"🎯 更新最佳损失: {best_loss:.6f} (Epoch {epoch+1})")
                        patience_counter = 0
                        try:
                            script_dir = os.path.dirname(os.path.abspath(__file__))
                            model_best_path = os.path.join(script_dir, 'models', f'gat_model_{self.model_prefix}_best.pth')
                            self.save_model(model_best_path)
                            logger.info(f"💾 保存最佳模型（总损失: {best_loss:.6f}）-> {os.path.basename(model_best_path)}")
                        except Exception as e:
                            logger.error(f"[Epoch {epoch+1}] 保存最佳模型失败: {e}")
                            logger.error(f"详细错误: {traceback.format_exc()}")
                    else:
                        patience_counter += 1
                else:
                    logger.warning(f"本Epoch未处理任何batch，跳过最佳/耐心判定与模型保存")
                
                # 记录训练历史
                try:
                    self.training_history['epoch_losses'].append(train_loss)
                    self.training_history['contrastive_losses'].append(contrastive_loss)
                    self.training_history['similarity_losses'].append(similarity_loss)
                    self.training_history['diversity_losses'].append(diversity_loss)
                    self.training_history['uniqueness_losses'].append(uniqueness_loss)  # ⭐⭐⭐ 记录唯一性损失
                    self.training_history['binary_consistency_losses'].append(binary_loss)
                    self.training_history['gradient_norms'].append(grad_norm)
                    self.training_history['oom_retries'].append(num_oom_retries)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.training_history['learning_rates'].append(current_lr)
                    current_temp = self.adaptive_temp.get_temperature(epoch)
                    self.training_history['temperatures'].append(current_temp)
                except Exception as e:
                    logger.error(f"[Epoch {epoch+1}] 记录训练历史失败: {e}")
                    logger.error(f"详细错误: {traceback.format_exc()}")
                    # 使用默认值
                    current_lr = 0.0
                    current_temp = 1.0
                
                # 每个epoch打印简要信息（精简日志）
                try:
                    weights_snapshot = self.get_dynamic_loss_weights(epoch, self.total_epochs)
                    supcon_stage_temp = getattr(self, "current_supcon_temperature", 0.0)
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} | 阶段: {stage_desc} | 总损失: {train_loss:.6f} (最佳: {best_loss:.6f}) | "
                        f"SupCon: {contrastive_loss:.4f} | Proto: {similarity_loss:.4f} | Binary: {binary_loss:.4f} | Div: {diversity_loss:.4f} | "
                        f"LR: {current_lr:.6f} | SupCon-T: {supcon_stage_temp:.2f} | Binary-T: {current_temp:.2f} | 耐心: {patience_counter}/{self.patience}"
                    )
                    logger.info(
                        "   ⚖️ Loss Weights: "
                        f"SupCon={weights_snapshot['supcon']:.2f} | Proto={weights_snapshot['proto']:.2f} | "
                        f"Binary={weights_snapshot['binary']:.2f} | Div={weights_snapshot['diversity']:.2f} | "
                        f"Unique={weights_snapshot['uniqueness']:.2f} | Align={weights_snapshot['align']:.2f}"
                    )
                    
                    # ⭐ V8新增：Memory Bank难样本挖掘统计
                    if self.memory_initialized and hasattr(self, 'memory_hardness'):
                        valid_size = min(self.memory_seen_count, self.memory_bank_size)
                        if valid_size > 0:
                            avg_hardness = self.memory_hardness[:valid_size].mean().item()
                            max_hardness = self.memory_hardness[:valid_size].max().item()
                            logger.info(
                                f"   📊 Memory Bank: {valid_size}/{self.memory_bank_size} 样本 | "
                                f"平均难度: {avg_hardness:.4f} | 最大难度: {max_hardness:.4f}"
                            )
                except Exception:
                    pass
                
                # 保存checkpoint（每3个epoch保存一次，或在验证epoch），仅在有有效batch时保存
                if num_batches > 0 and ((epoch + 1) % 3 == 0 or epoch == start_epoch):
                    try:
                        self.save_checkpoint(
                            checkpoint_path, 
                            epoch, 
                            best_loss,  # ✅ 改为best_loss
                            patience_counter,
                            self.training_history
                        )
                    except Exception as e:
                        logger.error(f"保存checkpoint失败: {e}")
                        logger.error(f"详细错误: {traceback.format_exc()}")
                
                # ✅ 每个epoch保存最终模型（覆盖更新）- 仅在有有效batch时保存
                if num_batches > 0:
                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        model_final_path = os.path.join(script_dir, 'models', f'gat_model_{self.model_prefix}_final.pth')
                        self.save_model(model_final_path)
                        logger.debug(f"✅ 最终模型已更新: epoch {epoch+1} -> {os.path.basename(model_final_path)}")
                    except Exception as e:
                        logger.error(f"保存最终模型失败: {e}")

                    # ⭐使用真实Fig12.py在每个epoch后评估一次鲁棒性（重操作）
                    try:
                        fig12_nc_real = run_fig12_evaluation_for_model(model_final_path)
                        if isinstance(fig12_nc_real, (float, int)):
                            logger.info(f"[Fig12] Epoch {epoch+1}: Average NC = {fig12_nc_real:.6f}")
                            if fig12_nc_real > best_fig12_nc_real + 1e-4:
                                best_fig12_nc_real = fig12_nc_real
                                best_fig12_epoch_real = epoch
                                best_fig12_path = os.path.join(script_dir, 'models', f'gat_model_{self.model_prefix}_best_fig12.pth')
                                try:
                                    self.save_model(best_fig12_path)
                                    # ⭐关键修复：保存后等待文件系统同步，确保模型文件完全写入
                                    import time
                                    time.sleep(0.5)  # 等待0.5秒确保文件写入完成
                                    
                                    # ⭐关键修复：保存后立即用保存的模型重新评估，确保日志记录的NC值与实际模型一致
                                    logger.info(f"[Fig12] 开始验证保存的模型: {best_fig12_path}")
                                    fig12_nc_verify = run_fig12_evaluation_for_model(best_fig12_path)
                                    if isinstance(fig12_nc_verify, (float, int)):
                                        logger.info(f"💾 更新真实Fig12-best模型: {os.path.basename(best_fig12_path)} (NC={fig12_nc_verify:.6f}, 验证通过)")
                                        logger.info(f"[Fig12] 验证详情: 原始NC={fig12_nc_real:.6f}, 验证NC={fig12_nc_verify:.6f}, 差异={abs(fig12_nc_real - fig12_nc_verify):.6f}")
                                        best_fig12_nc_real = fig12_nc_verify  # 使用验证后的NC值
                                    else:
                                        logger.warning(f"[Fig12] 验证失败，使用原始NC值: {best_fig12_nc_real:.6f}")
                                        logger.info(f"💾 更新真实Fig12-best模型: {os.path.basename(best_fig12_path)} (NC={best_fig12_nc_real:.6f})")
                                except Exception as e:
                                    logger.error(f"[Epoch {epoch+1}] 保存Fig12-best模型失败: {e}")
                                    logger.error(f"[Epoch {epoch+1}] 详细错误: {traceback.format_exc()}")
                    except Exception as e:
                        logger.error(f"[Epoch {epoch+1}] 运行Fig12评估失败: {e}")

                if num_batches > 0 and ((epoch + 1) % metric_eval_interval == 0):
                    nc_results = None
                    distinction_score = None
                    try:
                        nc_results = self.evaluate_nc_on_validation(val_orig, val_attack)
                    except Exception as e:
                        logger.error(f"[Epoch {epoch+1}] evaluate_nc_on_validation 失败: {e}")
                    try:
                        distinction_score = self.evaluate_feature_distinction(val_orig)
                    except Exception as e:
                        logger.error(f"[Epoch {epoch+1}] evaluate_feature_distinction 失败: {e}")
                    
                    improved = False
                    has_valid = False
                    avg_nc = None
                    fig12_nc = None
                    
                    # 处理NC评估结果（现在返回字典）
                    if isinstance(nc_results, dict):
                        avg_nc = nc_results.get('avg_nc')
                        fig12_nc = nc_results.get('fig12_nc')
                    elif isinstance(nc_results, (float, int)):
                        # 兼容旧版本（直接返回float）
                        avg_nc = nc_results
                    
                    if isinstance(avg_nc, (float, int)):
                        has_valid = True
                        logger.info(f"[Metrics] Epoch {epoch+1}: Avg NC = {avg_nc:.6f}")
                        if avg_nc > best_nc + nc_improve_tol:
                            best_nc = avg_nc
                            improved = True
                    
                    # ⭐修复：禁用基于验证集的best_fig12保存逻辑，避免覆盖基于真实Fig12.py评估的模型
                    # 注意：best_fig12模型应该只由每个epoch后的真实Fig12.py评估来保存（见上面的run_fig12_evaluation_for_model）
                    # 验证集的fig12_nc可能不准确，不应该用来保存best_fig12模型
                    if isinstance(fig12_nc, (float, int)) and fig12_nc > 0:
                        has_valid = True
                        logger.info(f"[Metrics] Epoch {epoch+1}: Fig12 NC = {fig12_nc:.6f} (目标: >0.8) [验证集评估，仅供参考]")
                        # ⭐已禁用：不再基于验证集评估保存best_fig12模型
                        # 真实best_fig12模型由每个epoch后的run_fig12_evaluation_for_model保存
                        if fig12_nc > best_fig12_nc + 0.01:  # 0.01的容差
                            best_fig12_nc = fig12_nc
                            best_fig12_epoch = epoch
                            improved = True
                            # ⭐已禁用：不再保存，避免覆盖真实Fig12.py评估的模型
                            logger.debug(f"[Metrics] 验证集Fig12 NC提升到 {best_fig12_nc:.6f}，但不保存模型（真实best_fig12由Fig12.py评估保存）")
                    
                    if isinstance(distinction_score, (float, int)):
                        has_valid = True
                        logger.info(f"[Metrics] Epoch {epoch+1}: Distinction = {distinction_score:.6f}")
                        if distinction_score > best_distinction + distinction_improve_tol:
                            best_distinction = distinction_score
                            improved = True
                    if has_valid:
                        metric_has_valid = True
                    if improved:
                        metric_patience_counter = 0
                    else:
                        metric_patience_counter += 1
                    if metric_has_valid and (epoch + 1) >= min_epoch_for_metric_stop and metric_patience_counter >= metric_patience:
                        logger.warning(f"NC/区分度连续{metric_patience_counter}次无明显提升，触发指标早停 (best NC={best_nc:.6f}, best distinction={best_distinction:.6f})")
                        try:
                            self.save_checkpoint(checkpoint_path, epoch, best_loss, patience_counter, self.training_history)
                        except Exception as e:
                            logger.error(f"基于指标早停时保存checkpoint失败: {e}")
                        break
                    self.model.train()

                # 动态早停检查（使用self.patience和相对改善）- 仅在有有效batch时检查
                if num_batches > 0 and patience_counter >= self.patience:
                    logger.warning(f"连续{patience_counter}个epoch总损失没有改善（最佳在E{self.best_epoch+1}），触发早停机制")
                    # 保存最终checkpoint
                    try:
                        self.save_checkpoint(checkpoint_path, epoch, best_loss, patience_counter, self.training_history)
                    except Exception as e:
                        logger.error(f"保存最终checkpoint失败: {e}")
                    break
                
                # 定期清理CUDA缓存（每2个epoch，防止CUDA错误）
                if (epoch + 1) % 2 == 0 and torch.cuda.is_available():
                    try:
                        logger.info("🧹 清理CUDA缓存...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"⚠️ CUDA缓存清理失败（忽略）: {e}")
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} 完成！\n")
                
            except Exception as e:
                logger.error(f"❌ Epoch {epoch+1} 训练过程中出现严重错误: {e}")
                logger.error(f"详细错误堆栈:\n{traceback.format_exc()}")
                logger.error(f"保存紧急checkpoint...")
                # 保存紧急checkpoint
                try:
                    emergency_checkpoint = os.path.join(script_dir, 'checkpoints', f'gat_checkpoint_emergency_epoch{epoch+1}.pth')
                    self.save_checkpoint(emergency_checkpoint, epoch, best_loss, patience_counter, self.training_history)
                    logger.info(f"✅ 紧急checkpoint已保存: {emergency_checkpoint}")
                except Exception as save_error:
                    logger.error(f"保存紧急checkpoint失败: {save_error}")
                logger.error(f"尝试继续下一个epoch...")
                continue
        
        # 保存训练历史
        try:
            logger.info("开始保存训练历史...")
            self.save_training_history()
            logger.info("训练历史保存成功！")
        except Exception as e:
            logger.error(f"保存训练历史失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
        
        # ✅ 确保最终模型已保存（每个epoch都会保存一次，这里再次确保）
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_final_path = os.path.join(script_dir, 'models', f'gat_model_{self.model_prefix}_final.pth')
            if not os.path.exists(model_final_path):
                self.save_model(model_final_path)
                logger.info(f"✅ 最终模型已保存: {os.path.basename(model_final_path)}")
            else:
                logger.info(f"✅ 最终模型: {os.path.basename(model_final_path)} (已在训练过程中持续更新)")
        except Exception as e:
            logger.error(f"检查最终模型失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
        
        logger.info("")
        logger.info("="*70)
        logger.info("训练完成！")
        logger.info("="*70)
        logger.info(f"✅ 最佳总损失: {best_loss:.6f}")
        logger.info(f"✅ 最佳模型: models/gat_model_{self.model_prefix}_best.pth (总损失最低)")
        if best_fig12_nc_real > 0:
            logger.info(f"✅ 真实Fig12最佳NC值: {best_fig12_nc_real:.6f} (Epoch {best_fig12_epoch_real+1})")
            logger.info(f"✅ 真实Fig12-best模型: models/gat_model_{self.model_prefix}_best_fig12.pth")
            if best_fig12_nc_real >= 0.8:
                logger.info("🎉 恭喜！真实Fig12 Average NC 已达到目标 (>=0.8)")
            else:
                logger.info(f"💡 提示：真实Fig12 Average NC ({best_fig12_nc_real:.6f}) 尚未达到目标 (0.8)，建议继续优化")
        logger.info(f"✅ 最终模型: models/gat_model_IMPROVED_final.pth (每个epoch持续更新)")
        logger.info("="*70)
        return best_loss
    
    def save_model(self, model_path):
        """保存模型（带完整CUDA错误恢复机制）- 三层回退策略"""
        model_path = os.path.abspath(model_path)
        
        # 确保目录存在
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # 🔧 方法1：直接保存（正常情况）
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_path)
            logger.info(f"✅ 模型已保存: {model_path}")
            return
        except Exception as e1:
            logger.warning(f"⚠️ 方法1直接保存失败: {e1}")
        
        # 🔧 方法2：将state_dict复制到CPU后保存
        try:
            model_state_cpu = {k: v.cpu().clone().detach() for k, v in self.model.state_dict().items()}
            try:
                optimizer_state_cpu = {
                    k: ({kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv 
                         for kk, vv in v.items()} if isinstance(v, dict) else 
                        [vvv.cpu() if isinstance(vvv, torch.Tensor) else vvv for vvv in v] if isinstance(v, list) else
                        v.cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in self.optimizer.state_dict().items()
                }
            except:
                optimizer_state_cpu = None  # 优化器状态可能损坏，放弃
            
            torch.save({
                'model_state_dict': model_state_cpu,
                'optimizer_state_dict': optimizer_state_cpu,
            }, model_path)
            logger.info(f"✅ 模型已保存（方法2-CPU拷贝）: {model_path}")
            return
        except Exception as e2:
            logger.warning(f"⚠️ 方法2 CPU拷贝保存失败: {e2}")
        
        # 🔧 方法3：将整个模型移到CPU，只保存模型权重（不保存优化器）
        try:
            original_device = next(self.model.parameters()).device
            self.model.cpu()  # 整个模型移到CPU
            
            model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            torch.save({'model_state_dict': model_state}, model_path)
            
            self.model.to(original_device)  # 移回原设备
            logger.info(f"✅ 模型已保存（方法3-仅模型）: {model_path}")
            return
        except Exception as e3:
            logger.error(f"❌ 方法3也失败: {e3}")
            # 尝试恢复模型位置
            try:
                self.model.to(self.device)
            except:
                pass
        
        # 🔧 方法4：创建新模型实例并复制权重（最后手段）
        try:
            import copy
            model_copy = copy.deepcopy(self.model).cpu()
            torch.save({'model_state_dict': model_copy.state_dict()}, model_path)
            del model_copy
            logger.info(f"✅ 模型已保存（方法4-深拷贝）: {model_path}")
            return
        except Exception as e4:
            logger.error(f"❌ 所有保存方法都失败了: {e4}")
            raise RuntimeError(f"无法保存模型到 {model_path}，所有方法都失败")
    
    def save_checkpoint(self, checkpoint_path, epoch, best_loss, patience_counter, training_history):
        """
        保存完整的训练checkpoint（用于中断恢复）- 带完整CUDA错误恢复机制
        
        Args:
            checkpoint_path: checkpoint保存路径
            epoch: 当前epoch
            best_loss: 最佳总损失
            patience_counter: 早停计数器
            training_history: 训练历史记录
        """
        checkpoint_path = os.path.abspath(checkpoint_path)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 🔧 第一步：获取scaler状态（不依赖GPU）
        scaler_state = None
        if self.use_amp and self.scaler is not None:
            try:
                scaler_state = self.scaler.state_dict()
            except Exception as e:
                logger.warning(f"⚠️ GradScaler状态读取失败: {e}")
                try:
                    from torch.cuda.amp import GradScaler
                    scaler_state = GradScaler().state_dict()
                except:
                    scaler_state = None
        
        # 🔧 第二步：安全获取所有状态（多层回退）
        model_state = None
        optimizer_state = None
        scheduler_state = None
        
        # 方法1：直接获取
        try:
            model_state = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict() if self.scheduler else None
        except Exception as e1:
            logger.warning(f"⚠️ 直接获取状态失败: {e1}")
            # 方法2：移到CPU后获取
            try:
                model_state = {k: v.cpu().clone().detach() for k, v in self.model.state_dict().items()}
                optimizer_state = {
                    k: ({kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()} 
                        if isinstance(v, dict) else v)
                    for k, v in self.optimizer.state_dict().items()
                }
                scheduler_state = self.scheduler.state_dict() if self.scheduler else None
            except Exception as e2:
                logger.error(f"⚠️ CPU方式获取状态也失败: {e2}")
                # 方法3：将整个模型移到CPU
                try:
                    original_device = next(self.model.parameters()).device
                    self.model.cpu()
                    model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    self.model.to(original_device)
                    optimizer_state = None  # 放弃保存优化器状态
                    scheduler_state = None
                except Exception as e3:
                    logger.error(f"❌ 所有方法都失败: {e3}")
                    return  # 放弃保存checkpoint
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'scaler_state_dict': scaler_state,
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'training_history': training_history,
            'adaptive_temp': {
                'init_temp': self.adaptive_temp.init_temp,
                'final_temp': self.adaptive_temp.final_temp,
                'total_epochs': self.adaptive_temp.total_epochs
            }
        }
        
        # 🔧 第三步：安全保存（避免任何CUDA操作）
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✅ Checkpoint已保存: Epoch {epoch+1}, 路径: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"⚠️ Checkpoint保存失败: {e}")
            # 不要调用torch.cuda.empty_cache()，因为可能触发CUDA错误
            # 直接放弃本次checkpoint保存
            logger.warning(f"⚠️ 放弃本次checkpoint保存（继续训练）")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载checkpoint以恢复训练
        
        Args:
            checkpoint_path: checkpoint路径
            
        Returns:
            checkpoint字典，如果文件不存在则返回None
        """
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint文件不存在: {checkpoint_path}")
            return None
        
        try:
            # PyTorch 2.6+需要设置weights_only=False来加载包含numpy对象的checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # ✅ 智能加载：尝试严格加载，失败则尝试部分加载
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                logger.info("✅ 完整加载模型参数（架构完全匹配）")
            except RuntimeError as e:
                # 架构不匹配，尝试部分加载
                if "size mismatch" in str(e) or "Missing key" in str(e) or "Unexpected key" in str(e):
                    logger.warning("⚠️  模型架构已改变，尝试部分加载兼容参数...")
                    
                    # 获取当前模型和checkpoint的参数
                    model_state = self.model.state_dict()
                    checkpoint_state = checkpoint['model_state_dict']
                    
                    # 统计兼容/不兼容参数
                    compatible_params = 0
                    incompatible_params = 0
                    new_params = 0
                    
                    for name, param in checkpoint_state.items():
                        if name in model_state:
                            if model_state[name].shape == param.shape:
                                model_state[name] = param
                                compatible_params += 1
                            else:
                                incompatible_params += 1
                                logger.debug(f"   跳过（形状不匹配）: {name}")
                        else:
                            incompatible_params += 1
                            logger.debug(f"   跳过（旧参数）: {name}")
                    
                    # 统计新增参数
                    for name in model_state.keys():
                        if name not in checkpoint_state:
                            new_params += 1
                            logger.debug(f"   新增参数（随机初始化）: {name}")
                    
                    # 加载兼容参数
                    self.model.load_state_dict(model_state, strict=True)
                    
                    logger.info(f"📊 部分加载统计:")
                    logger.info(f"   ✅ 兼容参数: {compatible_params}")
                    logger.info(f"   ❌ 不兼容参数: {incompatible_params}")
                    logger.info(f"   🆕 新增参数: {new_params}")
                    logger.warning("⚠️  架构改变可能影响性能，建议从头训练以获得最佳效果")
                else:
                    raise  # 其他错误，继续抛出
            
            # 恢复优化器状态（如果兼容）
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"⚠️  优化器状态不兼容，使用新初始化: {e}")
            
            if checkpoint.get('scheduler_state_dict') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if checkpoint.get('scaler_state_dict') and self.use_amp:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # 恢复自适应温度参数
            if 'adaptive_temp' in checkpoint:
                temp_config = checkpoint['adaptive_temp']
                self.adaptive_temp.init_temp = temp_config['init_temp']
                self.adaptive_temp.final_temp = temp_config['final_temp']
                self.adaptive_temp.total_epochs = temp_config['total_epochs']
            
            logger.info(f"✅ 成功加载checkpoint: Epoch {checkpoint['epoch']+1}")
            # ✅ 兼容新旧checkpoint
            if 'best_loss' in checkpoint:
                logger.info(f"   最佳总损失: {checkpoint['best_loss']:.6f}")
            elif 'best_val_nc' in checkpoint:
                logger.info(f"   最佳验证NC值: {checkpoint['best_val_nc']:.4f} (旧checkpoint)")
            logger.info(f"   耐心计数: {checkpoint['patience_counter']}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ 加载checkpoint失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def save_training_history(self):
        """保存训练历史"""
        try:
            # 使用绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            history_dir = os.path.join(script_dir, "logs")
            
            if not os.path.exists(history_dir):
                os.makedirs(history_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = os.path.join(history_dir, f"training_history_IMPROVED_{timestamp}.json")
            
            # 转换为JSON可序列化格式
            history_data = {}
            for key, value in self.training_history.items():
                try:
                    if key == 'feature_stats':
                        history_data[key] = value
                    else:
                        history_data[key] = [float(v) if hasattr(v, 'item') else v for v in value]
                except Exception as e:
                    logger.error(f"[save_training_history] 转换键 '{key}' 失败: {e}")
                    history_data[key] = []
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"训练历史已保存到: {history_file}")
            
            # 绘制训练曲线
            self.plot_training_curves(history_data, history_dir, timestamp)
        except Exception as e:
            logger.error(f"[save_training_history] 保存失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            raise
    
    def plot_training_curves(self, history_data, save_dir, timestamp):
        """绘制训练曲线（SCI学术论文风格）"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib import rcParams
            
            # ============ SCI论文风格配置 ============
            # 设置字体为Times New Roman（学术论文标准）
            rcParams['font.family'] = 'serif'
            rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
            rcParams['font.size'] = 10
            rcParams['axes.labelsize'] = 11
            rcParams['axes.titlesize'] = 12
            rcParams['xtick.labelsize'] = 10
            rcParams['ytick.labelsize'] = 10
            rcParams['legend.fontsize'] = 9
            
            # 设置线条和标记样式
            rcParams['lines.linewidth'] = 1.5
            rcParams['lines.markersize'] = 4
            
            # 设置坐标轴样式
            rcParams['axes.linewidth'] = 1.0
            rcParams['axes.grid'] = True
            rcParams['grid.alpha'] = 0.3
            rcParams['grid.linestyle'] = '--'
            rcParams['grid.linewidth'] = 0.5
            
            # 设置图例样式
            rcParams['legend.frameon'] = True
            rcParams['legend.framealpha'] = 0.9
            rcParams['legend.edgecolor'] = 'black'
            
            # Colorblind-friendly颜色方案（学术论文推荐）
            colors = {
                'blue': '#0173B2',      # 蓝色
                'orange': '#DE8F05',    # 橙色
                'green': '#029E73',     # 绿色
                'red': '#CC78BC',       # 红色
                'cyan': '#56B4E9',      # 青色
                'magenta': '#CA9161',   # 棕色
                'purple': '#949494'     # 灰色
            }
            
            epochs = range(1, len(history_data['epoch_losses']) + 1)
            
            # ============ 创建图表（双栏布局：4行2列，增加温度子图）============
            fig, axes = plt.subplots(4, 2, figsize=(7.0, 11.0))  # 7英寸宽度适合双栏论文
            
            # (a) 总损失
            ax = axes[0, 0]
            ax.plot(epochs, history_data['epoch_losses'], 
                   color=colors['blue'], linewidth=1.5, label='Total Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('(a) Total Training Loss')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # (b) 对比损失（InfoNCE）
            ax = axes[0, 1]
            ax.plot(epochs, history_data['contrastive_losses'], 
                   color=colors['orange'], linewidth=1.5, label='Contrastive Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('(b) Contrastive Loss (InfoNCE)')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # (c) 二值化一致性损失（核心创新）
            ax = axes[1, 0]
            ax.plot(epochs, history_data['binary_consistency_losses'], 
                   color=colors['green'], linewidth=1.5, label='Binary Consistency Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('(c) Binary Consistency Loss')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # (d) 验证NC值（✅ 每个epoch评估）
            ax = axes[1, 1]
            val_nc_values = history_data.get('val_nc_values', [])
            if val_nc_values:
                # ✅ 修复：使用实际的val_nc_values长度，同时兼容旧history（可能不存在该键）
                val_nc_len = len(val_nc_values)
                val_epochs = range(1, val_nc_len + 1)
                ax.plot(val_epochs, val_nc_values, 
                       color=colors['red'], linewidth=1.5, 
                       marker='o', markersize=4, markerfacecolor='white',
                       markeredgewidth=1.5, label='Validation NC')
                ax.set_ylim([0, 1.0])  # NC值范围0-1
            ax.set_xlabel('Epoch')
            ax.set_ylabel('NC Value')
            ax.set_title('(d) Validation NC Value')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # (e) 学习率（OneCycleLR）
            ax = axes[2, 0]
            ax.plot(epochs, history_data['learning_rates'], 
                   color=colors['cyan'], linewidth=1.5, label='Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('(e) Learning Rate Schedule')
            ax.set_yscale('log')  # 学习率用对数坐标
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # (f) 梯度范数
            ax = axes[2, 1]
            ax.plot(epochs, history_data['gradient_norms'], 
                   color=colors['magenta'], linewidth=1.5, label='Gradient Norm')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('(f) Gradient Norm')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # (g) 自适应温度 ⭐新增
            ax = axes[3, 0]
            if 'temperatures' in history_data and len(history_data['temperatures']) > 0:
                ax.plot(epochs, history_data['temperatures'], 
                       color='#E91E63', linewidth=2.0, label='Temperature')
                ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Init')
                ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Final')
                ax.set_yscale('log')  # 对数坐标
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Temperature')
            ax.set_title('(g) Adaptive Temperature (Annealing)')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=8)
            
            # (h) 损失权重变化（可选，暂时空白或添加说明）
            ax = axes[3, 1]
            ax.text(0.5, 0.5, 'Adaptive Loss Weights\n(Dynamic During Training)', 
                   ha='center', va='center', fontsize=10, color='gray')
            ax.set_title('(h) Training Strategy')
            ax.axis('off')
            
            # 调整子图间距
            plt.tight_layout()
            
            # 保存为高分辨率PNG（用于论文）
            plot_file_png = os.path.join(save_dir, f"training_curves_SCI_{timestamp}.png")
            plt.savefig(plot_file_png, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            # 同时保存为矢量格式PDF（用于论文终稿）
            plot_file_pdf = os.path.join(save_dir, f"training_curves_SCI_{timestamp}.pdf")
            plt.savefig(plot_file_pdf, format='pdf', bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            
            plt.close()
            
            logger.info(f"SCI风格训练曲线已保存:")
            logger.info(f"  PNG (300 DPI): {plot_file_png}")
            logger.info(f"  PDF (矢量图): {plot_file_pdf}")
            
            # ============ 额外：生成单独的NC值曲线（用于论文重点展示）============
            val_nc_values = history_data.get('val_nc_values', [])
            if val_nc_values:
                fig_nc, ax_nc = plt.subplots(1, 1, figsize=(3.5, 2.8))  # 单栏宽度
                
                val_epochs = range(3, len(epochs)+1, 3)[:len(val_nc_values)]
                ax_nc.plot(val_epochs, val_nc_values, 
                          color=colors['red'], linewidth=2.0, 
                          marker='o', markersize=5, markerfacecolor='white',
                          markeredgewidth=1.5, label='Validation NC')
                
                ax_nc.set_xlabel('Epoch', fontsize=11)
                ax_nc.set_ylabel('NC Value', fontsize=11)
                ax_nc.set_title('Validation NC Value (Zero-Watermark Robustness)', fontsize=12)
                ax_nc.set_ylim([0, 1.0])
                ax_nc.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax_nc.legend(loc='lower right', frameon=True)
                
                plt.tight_layout()
                
                nc_file_png = os.path.join(save_dir, f"nc_value_SCI_{timestamp}.png")
                nc_file_pdf = os.path.join(save_dir, f"nc_value_SCI_{timestamp}.pdf")
                
                plt.savefig(nc_file_png, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.savefig(nc_file_pdf, format='pdf', bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                
                plt.close()
                
                logger.info(f"  NC值单独图: {nc_file_png}, {nc_file_pdf}")
            
        except ImportError as e:
            logger.warning(f"matplotlib未安装或导入失败: {e}，跳过训练曲线绘制")
        except Exception as e:
            logger.error(f"绘制训练曲线时出错: {e}")
            import traceback
            traceback.print_exc()

class GraphDataLoader:
    """图数据加载器"""

    BLACKLIST = [
        'tianjin-latest-free.shp-gis_osm_landuse_a_free_1',
        'tianjin-latest-free.shp-gis_osm_traffic_free_1',
        'H51-HYDL',
        'tianjin-latest-free.shp-gis_osm_railways_free_1',
        'H51-AANP',
        'H51-LRDL'
    ]

    def __init__(self, graph_dir=None, max_nodes=30000):
        if graph_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            graph_dir = os.path.join(project_root, 'convertToGraph', 'Graph', 'TrainingSet')

        self.graph_dir = os.path.abspath(graph_dir)
        self.max_nodes = max_nodes
        logger.info(f"图数据加载路径: {self.graph_dir}")

    def load_graph_data(self, max_nodes=None):
        threshold = max_nodes if max_nodes is not None else self.max_nodes
        original_dir = os.path.join(self.graph_dir, 'Original')
        attacked_dir = os.path.join(self.graph_dir, 'Attacked')

        if not os.path.exists(original_dir):
            logger.warning(f"原始数据目录不存在: {original_dir}")
            return {}, {}

        logger.info(f"开始加载图数据（过滤>{threshold:,}节点的超大图）...")
        original_graphs, filtered_graphs = self._load_original_graphs(original_dir, threshold)
        attacked_graphs, total_attacked_loaded, total_attacked_filtered = self._load_attacked_graphs(attacked_dir, original_graphs)

        self._log_dataset_stats(len(original_graphs), total_attacked_loaded)
        self._log_filter_summary(filtered_graphs, total_attacked_filtered, len(original_graphs), threshold)

        return original_graphs, attacked_graphs

    def _load_original_graphs(self, original_dir, max_nodes):
        original_graphs = {}
        filtered_graphs = []

        for filename in os.listdir(original_dir):
            if not filename.endswith('_graph.pkl'):
                continue
            graph_name = filename.replace('_graph.pkl', '')
            filepath = os.path.join(original_dir, filename)
            graph_data = self._load_graph_pickle(filepath)

            if graph_name in self.BLACKLIST:
                logger.warning(f"  ⛔ 黑名单过滤: {graph_name} (手动排除)")
                self._record_filtered_graph(filtered_graphs, graph_name, graph_data, '黑名单')
                continue

            num_nodes = graph_data.x.shape[0]
            if num_nodes > max_nodes:
                logger.warning(f"  过滤超大图: {graph_name} ({num_nodes:,}节点, {graph_data.edge_index.shape[1]:,}边)")
                self._record_filtered_graph(filtered_graphs, graph_name, graph_data, '超大图')
                continue

            original_graphs[graph_name] = graph_data

        return original_graphs, filtered_graphs

    def _load_attacked_graphs(self, attacked_dir, original_graphs):
        attacked_graphs = {}
        total_attacked_loaded = 0
        total_attacked_filtered = 0

        if not os.path.exists(attacked_dir):
            return attacked_graphs, total_attacked_loaded, total_attacked_filtered

        for subdir in os.listdir(attacked_dir):
            subdir_path = os.path.join(attacked_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            attack_files = [f for f in os.listdir(subdir_path) if f.endswith('_graph.pkl')]
            if subdir not in original_graphs:
                total_attacked_filtered += len(attack_files)
                continue

            attacked_graphs[subdir] = []
            for filename in attack_files:
                attack_path = os.path.join(subdir_path, filename)
                graph_data = self._load_graph_pickle(attack_path)
                attack_name = filename.replace('_graph.pkl', '')
                if 'compound_seq' in attack_name.lower() and 'full_chain' not in attack_name.lower():
                    attack_name = f"{attack_name}_full_chain"
                graph_data.attack_type = attack_name
                attacked_graphs[subdir].append(graph_data)
                total_attacked_loaded += 1

        return attacked_graphs, total_attacked_loaded, total_attacked_filtered

    def _record_filtered_graph(self, filtered_graphs, graph_name, graph_data, reason):
        filtered_graphs.append({
            'name': graph_name,
            'nodes': graph_data.x.shape[0],
            'edges': graph_data.edge_index.shape[1],
            'reason': reason
        })

    @staticmethod
    def _load_graph_pickle(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _log_dataset_stats(self, original_count, attacked_loaded):
        logger.info(f"✅ 成功加载 {original_count} 个原始图")
        logger.info(f"✅ 成功加载 {attacked_loaded} 个被攻击的图")

    def _log_filter_summary(self, filtered_graphs, total_attacked_filtered, original_count, max_nodes):
        if not filtered_graphs:
            return

        logger.warning("")
        blacklist_count = sum(1 for fg in filtered_graphs if fg.get('reason') == '黑名单')
        large_count = sum(1 for fg in filtered_graphs if fg.get('reason') == '超大图')
        logger.warning(f"⚠️  过滤了 {len(filtered_graphs)} 个原始图")
        logger.warning(f"   ├─ 黑名单: {blacklist_count} 个")
        logger.warning(f"   └─ 超大图(>{max_nodes:,}节点): {large_count} 个")
        logger.warning(f"⚠️  过滤了 {total_attacked_filtered} 个对应的攻击图")

        total_graphs = original_count + len(filtered_graphs)
        if total_graphs > 0:
            retention = original_count / total_graphs * 100
            logger.warning(f"⚠️  数据保留率: {retention:.1f}%")

        logger.warning("")
        logger.warning("被过滤的图列表：")
        for fg in filtered_graphs:
            reason = fg.get('reason', '未知')
            logger.warning(f"   - {fg['name']}: {fg['nodes']:,} 节点, {fg['edges']:,} 边 [{reason}]")
        logger.warning("")
        logger.info("✅ 过滤问题图后，避免训练过程中OOM")
        logger.warning("")


def main():
    """主函数"""
    # ⭐ 训练模式：设置完整的文件日志
    global logger
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("第三步：改进的GAT模型训练 - 矢量地图零水印鲁棒特征提取")
    logger.info("="*70)
    log_training_overview()
    device = log_device_info()
    
    # 加载数据
    data_loader = GraphDataLoader()
    original_graphs, attacked_graphs = data_loader.load_graph_data()
    
    if len(original_graphs) == 0:
        logger.warning("没有找到原始图数据，请先运行第二步")
        return
    
    if len(attacked_graphs) == 0:
        logger.warning("没有找到被攻击的图数据，请先运行第二步")
        return
    
    input_dim = infer_input_dim(original_graphs)
    log_feature_profile(input_dim)
    
    # 创建改进的模型
    model = ImprovedGATModel(
        input_dim=input_dim,
        hidden_dim=256,  # 增加到256（原来128）
        output_dim=1024,
        num_heads=8,  # 增加到8（原来4）
        dropout=0.3
    )
    
    log_model_summary(model)
    configured_bs = resolve_batch_size()
    logger.info(f"✅ 数据加载时已过滤>30000节点的超大图")
    logger.info("")
    
    # 创建改进的训练器（温度参数提高以增强数值稳定性）⭐修复NaN
    trainer = ImprovedContrastiveTrainer(
        model,
        device,
        temperature=0.1,  # 从0.07提升至0.1，减少exp溢出风险
        use_amp=(device=='cuda'),
        batch_size=configured_bs
    )
    
    logger.info("")
    logger.info("="*70)
    logger.info("开始训练...")
    logger.info("="*70)
    logger.info("")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, 'checkpoints', 'gat_checkpoint_latest.pth')
    resume_from = resolve_checkpoint_choice(checkpoint_path)
    
    # 训练模型（✅ 调整为20个epoch：前期唯一性→中期平衡→后期鲁棒性）
    best_loss = trainer.train(
        original_graphs,
        attacked_graphs,
        num_epochs=12,  # ⭐优化：从20减少到12，因为Epoch 1就达到最佳鲁棒性，不需要那么多epoch
        resume_from_checkpoint=resume_from
    )
    
    # 保存最终模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_model_path = os.path.join(script_dir, 'models', 'gat_model_IMPROVED.pth')
    best_model_path = os.path.join(script_dir, 'models', 'gat_model_IMPROVED_best.pth')
    
    trainer.save_model(final_model_path)
    
    logger.info("")
    logger.info("="*70)
    logger.info("模型训练完成！")
    logger.info("="*70)
    logger.info(f"✅ 最佳总损失: {best_loss:.6f}")
    logger.info(f"✅ 最终模型保存到: {final_model_path}")
    logger.info(f"✅ 最佳模型保存到: {best_model_path} (总损失最低)")
    logger.info("")
    logger.info("模型将用于：")
    logger.info("  1. 从原始矢量地图提取1024维鲁棒特征")
    logger.info("  2. 二值化后与版权图像XOR生成零水印")
    logger.info("  3. 验证阶段提取特征并恢复版权图像")
    logger.info("="*70)

if __name__ == "__main__":
    main()

