"""
验证checkpoint文件内容
"""
import torch
import os

checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'gat_checkpoint_latest.pth')

print(f"检查文件: {checkpoint_path}")
print(f"文件存在: {os.path.exists(checkpoint_path)}")

if os.path.exists(checkpoint_path):
    print(f"文件大小: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("\n✅ Checkpoint信息：")
        print(f"  当前Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  最佳损失: {checkpoint.get('best_loss', checkpoint.get('best_val_nc', 'N/A'))}")
        print(f"  耐心计数: {checkpoint.get('patience_counter', 'N/A')}")
        print(f"  训练历史长度: {len(checkpoint.get('training_history', []))}")
        print(f"\n📌 下一个训练epoch将从: Epoch {checkpoint.get('epoch', -1) + 1}")
        print(f"   （即从Epoch {checkpoint.get('epoch', -1) + 1}继续训练）")
    except Exception as e:
        print(f"\n❌ 加载checkpoint失败: {e}")
else:
    print("\n❌ Checkpoint文件不存在")
