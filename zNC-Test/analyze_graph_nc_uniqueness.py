#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze NC uniqueness for graph zero-watermarks.
This is modeled after analyze_nc_uniqueness.py but targets the
vector-data-zerowatermark-ablation2 folder by default.
"""

import argparse
from pathlib import Path
from typing import Optional
import json
import re
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def generate_label_from_filename(filename: str) -> str:
    """Generate a short, human-friendly label from a filename."""
    base = filename.replace('_watermark.png', '').replace('_watermark.npy', '')
    # If contains 'ablation2' or similar, strip common prefixes
    base = re.sub(r'^(ablation2_|ablation2-)', '', base, flags=re.I)
    # If too long, take last meaningful part
    if len(base) > 20:
        parts = re.split(r'[_\-\.]+', base)
        if parts:
            return parts[-1]
    return base


def load_vector(file_path: Path) -> np.ndarray:
    """Load a watermark vector from .png or .npy and return 1-D uint8 array."""
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    if file_path.suffix.lower() == '.png':
        img = cv2.imread(str(file_path), 0)
        if img is None:
            raise IOError(f"Cannot read image: {file_path}")
        vec = (img.flatten() / 255).astype(np.uint8)
    else:
        vec = np.load(file_path)
        vec = vec.astype(np.uint8)
        if vec.ndim > 1:
            vec = vec.flatten()
    return vec


def analyze_graph_nc_uniqueness(target_dir: Path, out_dir: Optional[Path] = None):
    """Compute NC matrix and basic uniqueness statistics for watermark vectors in target_dir."""
    target_dir = target_dir.resolve()
    if not target_dir.exists():
        print(f"ERROR: target directory does not exist: {target_dir}")
        return

    # find files *_watermark.png or *_watermark.npy
    watermark_files = {}
    for ext in ('.png', '.npy'):
        for p in sorted(target_dir.glob(f'*_watermark{ext}')):
            base = p.stem.replace('_watermark', '')
            if base not in watermark_files:
                watermark_files[base] = p

    if not watermark_files:
        print("No watermark files found in:", target_dir)
        return

    vectors = []
    short_labels = []
    full_names = []

    print(f"Found {len(watermark_files)} watermark files, loading...")
    for base, p in watermark_files.items():
        label = generate_label_from_filename(p.name)
        try:
            vec = load_vector(p)
        except Exception as e:
            print(f"Warning: failed to load {p.name}: {e}")
            continue
        vectors.append(vec.astype(float))
        short_labels.append(label)
        full_names.append(base)
        print(f"Loaded {p.name}: length={vec.size}")

    if not vectors:
        print("No vectors successfully loaded.")
        return

    # normalize lengths: require all same length
    lengths = [v.size for v in vectors]
    if len(set(lengths)) != 1:
        min_len = min(lengths)
        print(f"Warning: inconsistent vector lengths found, truncating to {min_len}")
        vectors = [v[:min_len] for v in vectors]

    n = len(vectors)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        v1 = vectors[i]
        for j in range(n):
            v2 = vectors[j]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                mat[i, j] = float(np.dot(v1, v2) / (n1 * n2))
            else:
                mat[i, j] = 0.0

    mask = ~np.eye(n, dtype=bool)
    off_diag = mat[mask]
    total_pairs = off_diag.size

    stats = {
        "max_off_diag_nc": float(np.max(off_diag)) if off_diag.size else 0.0,
        "min_off_diag_nc": float(np.min(off_diag)) if off_diag.size else 0.0,
        "mean_off_diag_nc": float(np.mean(off_diag)) if off_diag.size else 0.0,
        "std_off_diag_nc": float(np.std(off_diag)) if off_diag.size else 0.0,
        "median_off_diag_nc": float(np.median(off_diag)) if off_diag.size else 0.0,
        "total_pairs": int(total_pairs)
    }

    # thresholds
    thresholds = [0.80, 0.82, 0.85, 0.90]
    for t in thresholds:
        stats[f"pairs_ge_{int(t*100)}"] = int(np.sum(off_diag >= t))

    # list highest pairs
    indices = np.triu_indices(n, k=1)
    pairs = []
    for i, j in zip(indices[0], indices[1]):
        pairs.append((short_labels[i], short_labels[j], float(mat[i, j]), full_names[i], full_names[j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    stats["top_pairs"] = [{"label1": p[0], "label2": p[1], "nc": p[2], "file1": p[3], "file2": p[4]} for p in pairs[:50]]

    # prepare output paths
    script_dir = Path(__file__).resolve().parent
    if out_dir is None:
        out_dir = script_dir
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_dir / 'graph_nc_uniqueness_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("Saved stats to", stats_path)

    # save matrices
    nc_df = pd.DataFrame(mat, index=short_labels, columns=short_labels)
    nc_csv = out_dir / 'NC_Matrix_graph.csv'
    nc_df.to_csv(nc_csv, float_format='%.6f')
    print("Saved NC matrix csv to", nc_csv)

    nc_df_full = pd.DataFrame(mat, index=full_names, columns=full_names)
    nc_csv_full = out_dir / 'NC_Matrix_graph_full.csv'
    nc_df_full.to_csv(nc_csv_full, float_format='%.6f')
    print("Saved NC matrix (full) csv to", nc_csv_full)

    # heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NC Value', rotation=270, labelpad=20)
    ax.set_xticks(np.arange(len(full_names)))
    ax.set_yticks(np.arange(len(full_names)))
    ax.set_xticklabels(full_names, fontsize=8)
    ax.set_yticklabels(full_names, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha="center", va="center", color="black", fontsize=7)
    plt.tight_layout()
    heatmap_path = out_dir / 'NC_Matrix_graph_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved heatmap to", heatmap_path)

    print("\nSummary:")
    print(f"  files analyzed: {len(full_names)}")
    print(f"  total pairs: {total_pairs}")
    print(f"  max off-diag NC: {stats['max_off_diag_nc']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze NC uniqueness for graph zero-watermarks.")
    parser.add_argument('--dir', '-d', type=str, default=None,
                        help="Target directory containing *_watermark.png/.npy files. If omitted, uses './vector-data-zerowatermark-ablation2' next to this script.")
    parser.add_argument('--out', type=str, default=None, help="Output directory for stats and plots.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir / 'vector-data-zerowatermark-ablation2'
    target = Path(args.dir).resolve() if args.dir else default_dir
    analyze_graph_nc_uniqueness(target, out_dir=args.out or script_dir)


if __name__ == '__main__':
    main()

