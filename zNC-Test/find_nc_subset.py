#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find subsets of watermark vectors whose pairwise NC is below a threshold.
Default: threshold=0.82, target_size=8, input=zNC-Test/NC_Matrix.csv.
"""

import itertools
import pandas as pd
from pathlib import Path


def load_nc_matrix(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0)
    # Ensure we use the index order for both rows and columns
    labels = list(df.index)
    df = df.loc[labels, labels]
    return labels, df.values


def find_subsets(mat, labels, target_size=8, threshold=0.82, max_list=10):
    n = len(labels)
    combos_ok = []
    for combo in itertools.combinations(range(n), target_size):
        ok = True
        for i in range(target_size):
            for j in range(i + 1, target_size):
                if mat[combo[i], combo[j]] >= threshold:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            combos_ok.append(combo)
            if len(combos_ok) >= max_list:
                # Stop early if we already collected enough
                break
    return combos_ok


def main():
    base_dir = Path(__file__).resolve().parent
    # 优先使用全称矩阵（如果存在），否则退回短标签矩阵
    csv_full = base_dir / "NC_Matrix_full.csv"
    csv_short = base_dir / "NC_Matrix.csv"
    csv_path = csv_full if csv_full.exists() else csv_short

    threshold = 0.85
    target_size = 7
    max_list = 10

    if not csv_path.exists():
        print(f"❌ 未找到 NC 矩阵文件: {csv_path}")
        return

    labels, mat = load_nc_matrix(csv_path)
    combos_ok = find_subsets(mat, labels, target_size=target_size, threshold=threshold, max_list=max_list)

    print(f"使用矩阵文件: {csv_path.name}")
    print(f"总节点数: {len(labels)}, 目标子集大小: {target_size}, 阈值: {threshold}")
    if combos_ok:
        print(f"找到 {len(combos_ok)} 组满足条件的子集（最多展示 {max_list} 组）:")
        for idx, combo in enumerate(combos_ok, 1):
            names = [labels[i] for i in combo]
            print(f"  #{idx}: {names}")
    else:
        print("未找到满足条件的子集。")


if __name__ == "__main__":
    main()


