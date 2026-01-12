"""
检查checkpoint的epoch信息
"""
import torch
import os

checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'gat_checkpoint_latest.pth')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint信息：")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Loss: {checkpoint.get('best_loss', checkpoint.get('best_val_nc', 'N/A'))}")
    print(f"  Patience Counter: {checkpoint.get('patience_counter', 'N/A')}")
    print(f"  Training History Length: {len(checkpoint.get('training_history', []))}")
    print(f"\n下一个训练epoch将从: {checkpoint.get('epoch', -1) + 1}")
else:
    print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
