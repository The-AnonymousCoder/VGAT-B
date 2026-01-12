#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融试验2：单流（图级特征）模型训练

架构特点：
- 只使用图级特征（8维）
- 移除节点级特征流和GAT层
- 直接通过MLP处理图级特征
- 用于验证特征解耦策略中图级特征的贡献

对比完整模型：移除节点级特征处理和图池化机制
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
    torch, nn, F, np
)

# 设置独立的日志记录器
def setup_ablation_logging(ablation_name="Ablation2_GraphOnly"):
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
    logger.info(f"消融试验2：图级特征单流模型训练")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"最新日志(覆盖): {latest_file}")
    return logger

logger = setup_ablation_logging("Ablation2_GraphOnly")

class GraphOnlyModel(nn.Module):
    """
    消融试验2：只使用图级特征的模型
    
    架构：
    - 输入：图级特征（8维）
    - 多层MLP处理
    - 直接输出（无节点级GAT和池化）
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=1024, num_heads=8, dropout=0.3):
        super(GraphOnlyModel, self).__init__()
        
        logger.info(f"创建【消融2：图级特征单流】MLP模型:")
        
        # 图级特征维度：维度0-2, 11-13, 18-19 = 8维
        self.graph_feature_dims = [0, 1, 2, 11, 12, 13, 18, 19]
        self.graph_input_dim = len(self.graph_feature_dims)  # 8
        
        logger.info(f"  图级特征维度: {self.graph_input_dim} (dims: {self.graph_feature_dims})")
        logger.info(f"  隐藏维度: {hidden_dim}")
        logger.info(f"  输出维度: {output_dim}")
        logger.info("  注意：不使用GAT，采用纯MLP架构")
        
        # ⭐ 关键差异：纯MLP架构处理图级特征
        self.encoder = nn.Sequential(
            nn.Linear(self.graph_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )
        
        logger.info(f"  MLP架构: {self.graph_input_dim} -> {hidden_dim} -> {hidden_dim*2} -> {hidden_dim*2} -> {output_dim}")
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  总参数数量: {total_params:,}")
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, edge_index, batch=None):
        """
        前向传播：只使用图级特征
        
        注意：edge_index不使用，保留参数以兼容训练接口
        """
        # ⭐ 步骤1：只提取图级特征
        graph_features = x[:, self.graph_feature_dims]  # [num_nodes, 8]
        
        # ⭐ 步骤2：取第一个节点的图级特征（所有节点相同）
        if batch is None:
            # 单图模式
            graph_features_unique = graph_features[0]  # [8]
            output = self.encoder(graph_features_unique)  # [output_dim]
        else:
            # 批次模式：从每个图提取一个节点的图级特征
            batch_size = batch.max().item() + 1
            graph_features_list = []
            for i in range(batch_size):
                mask = (batch == i)
                graph_features_list.append(graph_features[mask][0])
            graph_features_unique = torch.stack(graph_features_list, dim=0)  # [batch_size, 8]
            output = self.encoder(graph_features_unique)  # [batch_size, output_dim]
        
        return output


def main():
    """主训练流程 - 使用图级特征单流模型"""
    logger.info("="*80)
    logger.info("消融试验2：单流（图级特征）模型训练")
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
    
    # 创建图级特征单流模型
    model = GraphOnlyModel(
        input_dim=20,  # 输入仍然是20维，但内部只使用图级8维
        hidden_dim=256,
        output_dim=1024,
        num_heads=8,  # 不使用，保留兼容性
        dropout=0.3
    ).to(device)
    
    # 创建训练器（使用独立的checkpoint和模型名）
    trainer = ImprovedContrastiveTrainer(
        model=model,
        device=device,
        temperature=0.1,
        use_amp=True,
        batch_size=int(os.environ.get('VGAT_BATCH_SIZE', '6')),
        checkpoint_name='ablation2_graphonly_checkpoint.pth',  # ⭐ 独立checkpoint，避免覆盖
        model_prefix='Ablation2_GraphOnly'  # ⭐ 独立模型文件名，避免覆盖
    )
    
    # 开始训练
    logger.info("\n开始训练...")
    trainer.train(
        original_graphs,
        attacked_graphs,
        num_epochs=int(os.environ.get('VGAT_NUM_EPOCHS', '12'))
    )
    
    # ✅ 模型已直接保存为 gat_model_Ablation2_GraphOnly_best.pth 和 _final.pth
    # 不需要重命名（训练器已使用model_prefix='Ablation2_GraphOnly'）
    logger.info("\n✅ 消融试验2训练完成！")
    logger.info(f"   最佳模型: VGAT/models/gat_model_Ablation2_GraphOnly_best.pth")
    logger.info(f"   最终模型: VGAT/models/gat_model_Ablation2_GraphOnly_final.pth")


if __name__ == "__main__":
    main()

