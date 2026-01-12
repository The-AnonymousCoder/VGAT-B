#!/usr/bin/env python3
# 统计convertToGeoJson-Attacked-TrainingSet.py中的攻击数量

# 单体攻击统计
single_attacks = {
    "Fig1_删除顶点": 9 + 5,  # 9个基础(10%-90%) + 5个85%变体
    "Fig2_添加顶点": 27 + 16 + 5,  # 27基础 + 16额外比例 + 5个变体
    "Fig3_删除对象": 9 + 5,  # 9个基础 + 5个85%变体
    "Fig4_噪声": 15 + 30 + 12,  # 15基础 + 30额外 + 12高强度变体
    "Fig5_裁剪": 5 + 10,  # 5基础 + 10变体
    "Fig6_平移": 5,
    "Fig7_缩放": 6,
    "Fig8_旋转": 8,
    "Fig9_翻转": 3,
    "Fig10_打乱": 4 + 14,  # 4基础 + 14个shuffle变体
}

# 组合攻击统计
combo_attacks = {
    "全攻击链": 1,
    "重度组合(6-8种)": 30,
    "中度组合(4-5种)": 50,
    "轻度组合(2-3种)": 69,
}

print("=== 单体攻击统计 ===")
single_total = 0
for key, count in single_attacks.items():
    print(f"{key}: {count}个")
    single_total += count
print(f"\n单体攻击总计: {single_total}个")

print("\n=== 组合攻击统计 ===")
combo_total = 0
for key, count in combo_attacks.items():
    print(f"{key}: {count}个")
    combo_total += count
print(f"\n组合攻击总计: {combo_total}个")

print(f"\n=== 总计 ===")
print(f"攻击总数: {single_total + combo_total}个")
print(f"  - 单体攻击: {single_total}个")
print(f"  - 组合攻击: {combo_total}个")

# 攻击类型统计
attack_types = len(single_attacks)
print(f"\n攻击类型: {attack_types}种 (Fig1-Fig10)")
