#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融试验4：单头注意力(heads=1)模型训练

架构特点：
- 保持完整的双流架构
- 将多头注意力(heads=8)改为单头(heads=1)
- 用于验证多头注意力机制的贡献

对比完整模型：只修改注意力头数，其他架构保持不变
"""

import os
import sys
import logging
from datetime import datetime
import importlib.util

# 导入VGAT-IMPROVED.py（文件名带连字符，需要动态导入）
current_dir = os.path.dirname(os.path.abspath(__file__))
vgat_improved_path = os.path.join(current_dir, 'VGAT-IMPROVED.py')

spec = importlib.util.spec_from_file_location("VGAT_IMPROVED", vgat_improved_path)
VGAT_IMPROVED = importlib.util.module_from_spec(spec)
sys.modules['VGAT_IMPROVED'] = VGAT_IMPROVED
spec.loader.exec_module(VGAT_IMPROVED)

# 导入所有需要的类和函数（除了logger，我们自己创建）
from VGAT_IMPROVED import (
    ImprovedGATModel, ImprovedContrastiveTrainer, GraphDataLoader,
    torch, nn, F, np, GATv2Conv
)

# 设置独立的日志记录器
def setup_ablation_logging(ablation_name="Ablation4_SingleHead"):
    """设置消融实验的独立日志"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_file = os.path.join(log_dir, f"{ablation_name}_{timestamp}_{pid}.log")
    latest_file = os.path.join(log_dir, f"{ablation_name}_latest.log")
    
    # 清理可能存在的重复 handler
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
    
    # 配置两个文件输出
    file_handler_unique = logging.FileHandler(log_file, encoding='utf-8')
    file_handler_latest = logging.FileHandler(latest_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    for h in (file_handler_unique, file_handler_latest, console_handler):
        h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    log_level = os.environ.get('VGAT_LOG_LEVEL', 'INFO')
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(file_handler_unique)
    root_logger.addHandler(file_handler_latest)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"消融试验4：单头注意力模型训练")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"最新日志(覆盖): {latest_file}")
    return logger

logger = setup_ablation_logging("Ablation4_SingleHead")

class SingleHeadGATModel(nn.Module):
    """
    消融试验4：单头注意力GAT模型
    
    架构：
    - 完整的双流架构（节点级 + 图级）
    - ⭐ 关键差异：单头注意力（heads=1）
    - 其他设计保持不变
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1024, num_heads=1, dropout=0.3):
        super(SingleHeadGATModel, self).__init__()
        
        logger.info(f"创建【消融4：单头注意力】GAT模型:")
        logger.info(f"  ⭐ 注意力头数: {num_heads} (单头)")
        
        # 特征分离：节点级 vs 图级
        self.node_feature_dims = [5, 6, 7, 8, 9, 10, 14, 15, 16, 17]
        self.graph_feature_dims = [0, 1, 2, 11, 12, 13, 18, 19]
        
        self.node_input_dim = len(self.node_feature_dims)  # 10
        self.graph_input_dim = len(self.graph_feature_dims)  # 8
        
        logger.info(f"  节点级特征维度: {self.node_input_dim}")
        logger.info(f"  图级特征维度: {self.graph_input_dim}")
        logger.info(f"  隐藏维度: {hidden_dim}")
        logger.info(f"  输出维度: {output_dim}")
        
        # ⭐ 关键差异：GAT层使用单头注意力
        # 注意：单头时concat对输出维度无影响
        self.gat1 = GATv2Conv(self.node_input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.ln1 = nn.LayerNorm(hidden_dim * num_heads)  # num_heads=1时就是hidden_dim
        
        self.gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=False)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # 残差投影
        if self.node_input_dim != hidden_dim:
            self.residual_proj = nn.Linear(self.node_input_dim, hidden_dim)
        else:
            self.residual_proj = None
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 图级特征编码器
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.graph_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征融合
        fusion_input_dim = hidden_dim * 3 + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )
        
        logger.info(f"  融合层输入维度: {fusion_input_dim}")
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  总参数数量: {total_params:,}")
        logger.info(f"  对比：多头(8头)模型参数量会更多")
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=1.0)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, edge_index, batch=None):
        """
        前向传播：与完整模型相同，只是使用单头注意力
        """
        from torch_geometric.nn import global_mean_pool, global_max_pool
        
        # 步骤1：分离节点级和图级特征
        node_features = x[:, self.node_feature_dims]
        graph_features = x[:, self.graph_feature_dims]
        
        # 提取图级特征
        if batch is None:
            graph_features_unique = graph_features[0]
        else:
            batch_size = batch.max().item() + 1
            graph_features_list = []
            for i in range(batch_size):
                mask = (batch == i)
                graph_features_list.append(graph_features[mask][0])
            graph_features_unique = torch.stack(graph_features_list, dim=0)
        
        # 步骤2：GAT处理节点级特征（单头注意力）
        x1 = self.gat1(node_features, edge_index)
        x1 = self.ln1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.gat2(x1, edge_index)
        x2 = self.ln2(x2)
        
        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(node_features)
        else:
            residual = node_features
        x2 = F.elu(x2 + residual)
        
        # 步骤3：图级池化
        if batch is None:
            mean_pool = torch.mean(x2, dim=0)
            max_pool, _ = torch.max(x2, dim=0)
            attn_weights = F.softmax(self.attention_pool(x2), dim=0)
            attn_pool = torch.sum(x2 * attn_weights, dim=0)
            
            pooled_node_features = torch.cat([mean_pool, max_pool, attn_pool], dim=0)
            graph_features_encoded = self.graph_encoder(graph_features_unique)
            final_features = torch.cat([pooled_node_features, graph_features_encoded], dim=0)
            output = self.fusion(final_features)
        else:
            mean_pool = global_mean_pool(x2, batch)
            max_pool = global_max_pool(x2, batch)
            
            attn_scores = self.attention_pool(x2)
            batch_size = batch.max().item() + 1
            attn_pool_list = []
            for i in range(batch_size):
                mask = (batch == i)
                x_i = x2[mask]
                attn_i = attn_scores[mask]
                attn_weights_i = F.softmax(attn_i, dim=0)
                attn_pool_i = torch.sum(x_i * attn_weights_i, dim=0)
                attn_pool_list.append(attn_pool_i)
            attn_pool = torch.stack(attn_pool_list, dim=0)
            
            pooled_node_features = torch.cat([mean_pool, max_pool, attn_pool], dim=1)
            graph_features_encoded = self.graph_encoder(graph_features_unique)
            final_features = torch.cat([pooled_node_features, graph_features_encoded], dim=1)
            output = self.fusion(final_features)
        
        return output


def main():
    """主训练流程 - 使用单头注意力模型"""
    logger.info("="*80)
    logger.info("消融试验4：单头注意力(heads=1)模型训练")
    logger.info("="*80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("\n加载训练数据...")
    data_loader = GraphDataLoader()
    original_graphs, attacked_graphs = data_loader.load_graph_data()
    
    if not original_graphs:
        logger.error("没有找到训练数据！")
        return
    
    # 创建单头注意力模型
    model = SingleHeadGATModel(
        input_dim=20,
        hidden_dim=256,
        output_dim=1024,
        num_heads=1,  # ⭐ 关键：单头注意力
        dropout=0.3
    ).to(device)
    
    # 创建训练器（使用独立的checkpoint和模型名）
    trainer = ImprovedContrastiveTrainer(
        model=model,
        device=device,
        temperature=0.1,
        use_amp=True,
        batch_size=int(os.environ.get('VGAT_BATCH_SIZE', '6')),
        checkpoint_name='ablation4_singlehead_checkpoint.pth',  # ⭐ 独立checkpoint，避免覆盖
        model_prefix='Ablation4_SingleHead'  # ⭐ 独立模型文件名，避免覆盖
    )
    
    # 开始训练
    logger.info("\n开始训练...")
    trainer.train(
        original_graphs,
        attacked_graphs,
        num_epochs=int(os.environ.get('VGAT_NUM_EPOCHS', '12'))
    )
    
    # ✅ 模型已直接保存为 gat_model_Ablation4_SingleHead_best.pth 和 _final.pth
    # 不需要重命名（训练器已使用model_prefix='Ablation4_SingleHead'）
    logger.info("\n✅ 消融试验4训练完成！")
    logger.info(f"   最佳模型: VGAT/models/gat_model_Ablation4_SingleHead_best.pth")
    logger.info(f"   最终模型: VGAT/models/gat_model_Ablation4_SingleHead_final.pth")


if __name__ == "__main__":
    main()

