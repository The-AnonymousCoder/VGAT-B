#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：测试GAT模型和训练流程
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

print("=" * 70)
print("GAT模型诊断脚本")
print("=" * 70)

# 导入模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("vgat_improved", "VGAT-IMPROVED.py")
vgat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vgat_module)

ImprovedGATModel = vgat_module.ImprovedGATModel

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n" + "=" * 70)
print("测试1：模型创建和权重初始化")
print("=" * 70)

try:
    model = ImprovedGATModel(input_dim=20, hidden_dim=256, output_dim=1024, num_heads=8, dropout=0.3)
    model = model.to(device)
    model.eval()
    print("✅ 模型创建成功")
    print(f"   总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查权重初始化
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"❌ 警告：{name} 包含NaN/Inf")
        else:
            print(f"   ✅ {name}: shape={list(param.shape)}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("测试2：单图前向传播（batch=None）")
print("=" * 70)

try:
    # 创建测试图1
    num_nodes = 100
    x1 = torch.randn(num_nodes, 20).to(device)
    edge_index1 = torch.randint(0, num_nodes, (2, 200)).to(device)
    
    with torch.no_grad():
        output1 = model(x1, edge_index1, batch=None)
    
    print(f"✅ 单图前向传播成功")
    print(f"   输入: {num_nodes}节点, 200边")
    print(f"   输出shape: {output1.shape}")
    print(f"   输出范围: [{output1.min().item():.4f}, {output1.max().item():.4f}]")
    print(f"   输出均值: {output1.mean().item():.4f}")
    print(f"   输出标准差: {output1.std().item():.4f}")
    
    if torch.isnan(output1).any() or torch.isinf(output1).any():
        print(f"❌ 输出包含NaN/Inf！")
    else:
        print(f"   ✅ 输出数值正常")
        
except Exception as e:
    print(f"❌ 单图前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("测试3：不同图是否产生不同特征")
print("=" * 70)

try:
    # 创建3个不同的图
    outputs = []
    for i in range(3):
        num_nodes = 50 + i * 20
        x = torch.randn(num_nodes, 20).to(device)
        edge_index = torch.randint(0, num_nodes, (2, 100 + i * 50)).to(device)
        
        with torch.no_grad():
            output = model(x, edge_index, batch=None)
        outputs.append(output)
        print(f"   图{i+1}: {num_nodes}节点, 输出前5维={output[:5].cpu().numpy()}")
    
    # 计算两两之间的距离
    print(f"\n特征差异分析：")
    for i in range(3):
        for j in range(i+1, 3):
            dist = torch.norm(outputs[i] - outputs[j]).item()
            similarity = F.cosine_similarity(outputs[i].unsqueeze(0), outputs[j].unsqueeze(0)).item()
            print(f"   图{i+1} vs 图{j+1}: 欧氏距离={dist:.4f}, 余弦相似度={similarity:.4f}")
            
            if dist < 0.001:
                print(f"      ❌ 警告：距离太小，特征几乎相同！")
            elif similarity > 0.99:
                print(f"      ❌ 警告：相似度太高，特征几乎相同！")
            else:
                print(f"      ✅ 特征有明显差异")
                
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试4：梯度计算")
print("=" * 70)

try:
    model.train()
    
    # 创建测试数据
    x = torch.randn(50, 20, requires_grad=False).to(device)
    edge_index = torch.randint(0, 50, (2, 100)).to(device)
    
    # 前向传播
    output = model(x, edge_index, batch=None)
    
    # 简单的损失
    target = torch.randn_like(output).to(device)
    loss = F.mse_loss(output, target)
    
    print(f"✅ 前向传播成功")
    print(f"   输出shape: {output.shape}")
    print(f"   损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    print(f"✅ 反向传播成功")
    
    # 检查梯度
    has_grad = False
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_stats.append((name, grad_norm))
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"   ❌ {name}: 梯度包含NaN/Inf")
            elif grad_norm < 1e-10:
                print(f"   ⚠️  {name}: 梯度几乎为0 (norm={grad_norm:.2e})")
            else:
                print(f"   ✅ {name}: grad_norm={grad_norm:.4f}")
    
    if not has_grad:
        print(f"❌ 警告：没有参数有梯度！")
    else:
        total_grad_norm = sum(g for _, g in grad_stats)
        print(f"\n   总梯度范数: {total_grad_norm:.4f}")
        
except Exception as e:
    print(f"❌ 梯度计算失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试5：模拟一个训练步骤")
print("=" * 70)

try:
    from torch.cuda import amp
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())
    
    # 创建batch（4个图对）
    batch_size = 4
    features_orig = []
    features_attack = []
    
    for i in range(batch_size):
        x_orig = torch.randn(50, 20).to(device)
        x_attack = x_orig + torch.randn(50, 20).to(device) * 0.1  # 添加小扰动
        edge_index = torch.randint(0, 50, (2, 100)).to(device)
        
        with amp.autocast(enabled=torch.cuda.is_available()):
            feat_orig = model(x_orig, edge_index, batch=None)
            feat_attack = model(x_attack, edge_index, batch=None)
        
        features_orig.append(feat_orig)
        features_attack.append(feat_attack)
    
    # 堆叠特征
    features_orig = torch.stack(features_orig)
    features_attack = torch.stack(features_attack)
    
    print(f"✅ 特征提取成功")
    print(f"   原图特征shape: {features_orig.shape}")
    print(f"   攻击图特征shape: {features_attack.shape}")
    
    # 计算损失
    with amp.autocast(enabled=torch.cuda.is_available()):
        # 简单的相似性损失
        loss = F.mse_loss(features_orig, features_attack)
    
    print(f"✅ 损失计算成功: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    
    # 检查梯度
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    print(f"✅ 反向传播成功")
    print(f"   梯度范数: {grad_norm.item():.4f}")
    
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        print(f"   ❌ 梯度范数是NaN/Inf！")
    else:
        print(f"   ✅ 梯度正常")
        
        # 优化器更新
        scaler.step(optimizer)
        scaler.update()
        print(f"   ✅ 优化器更新成功")
    
except Exception as e:
    print(f"❌ 训练步骤失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("诊断完成")
print("=" * 70)
print("\n建议:")
print("1. 如果所有测试通过 → 问题可能在训练数据或损失函数")
print("2. 如果特征相同 → forward函数有问题")
print("3. 如果梯度NaN → 权重初始化或学习率有问题")
print("4. 如果CUDA错误 → 可能需要重启或更新驱动")
