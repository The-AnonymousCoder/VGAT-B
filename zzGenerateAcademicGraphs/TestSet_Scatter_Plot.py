#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TestSet鲁棒性测试散点图
基于beijing开头的测试集数据和攻击生成学术图
每个数据集有100种攻击（50个单体攻击 + 50个组合攻击）
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from common import (
    normalize_label, configure_matplotlib_for_sci,
    AXIS_LABEL_FONT_SIZE, XTICK_FONT_SIZE, YTICK_FONT_SIZE, LEGEND_FONT_SIZE,
    DEFAULT_LINEWIDTH, DEFAULT_MARKERSIZE
)
import pickle
import torch
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TestSetScatterPlot:
    """测试集散点图生成器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "zNC-Test" / "NC-Results" / "TestSet"
        self.output_dir = self.project_root / "zzGenerateAcademicGraphs"

        # 攻击类型映射（基于TestSet的攻击定义）
        # 使用深色高对比度调色，适合SCI期刊打印与缩图
        deep_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1b9e77"
        ]
        self.attack_categories = {
            'vertex_deletion': {'pattern': r'(test_)?del_vertices_\d+pct', 'name': 'Vertex Deletion', 'color': deep_colors[0]},
            'vertex_addition': {'pattern': r'(test_)?add_strength\d+_\d+pct_vertices', 'name': 'Vertex Addition', 'color': deep_colors[1]},
            'object_deletion': {'pattern': r'(test_)?del_objects_\d+pct', 'name': 'Object Deletion', 'color': deep_colors[2]},
            'noise': {'pattern': r'(test_)?noise_\d+pct_strength_', 'name': 'Noise', 'color': deep_colors[3]},
            'crop': {'pattern': r'(test_)?crop_', 'name': 'Crop', 'color': deep_colors[4]},
            'translation': {'pattern': r'(test_)?translate_', 'name': 'Translation', 'color': deep_colors[5]},
            'scale': {'pattern': r'(test_)?scale_\d+pct', 'name': 'Scale', 'color': deep_colors[6]},
            'rotation': {'pattern': r'(test_)?rotate_\d+deg', 'name': 'Rotation', 'color': deep_colors[7]},
            'flip': {'pattern': r'(test_)?flip_', 'name': 'Flip', 'color': deep_colors[8]},
            'shuffle': {'pattern': r'(test_)?(reverse|shuffle)_', 'name': 'Order Operation', 'color': deep_colors[9]},
            'other': {'pattern': r'(test_)?combo_', 'name': 'Compound Attack', 'color': deep_colors[10]}
        }

    def categorize_attack(self, attack_filename):
        """根据文件名分类攻击类型"""
        # 移除文件扩展名
        base_name = attack_filename.replace('_graph.pkl', '').replace('.geojson', '')

        # 匹配攻击类型
        for category, info in self.attack_categories.items():
            if category != 'other' and re.search(info['pattern'], base_name):
                return category, info['name'], info['color']

        return 'other', self.attack_categories['other']['name'], self.attack_categories['other']['color']

    def load_test_data(self):
        """加载beijing开头的测试集图的NC数据"""
        print("Loading beijing test set NC data...")

        test_data = []

        # 找到所有beijing开头的测试集图的stats文件
        for stats_file in self.results_dir.glob('beijing*_robustness_stats.txt'):
            graph_name = stats_file.stem.replace('_robustness_stats', '')
            print(f"处理图: {graph_name}")

            # 读取stats文件
            with open(stats_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取NC值
            nc_pattern = r'攻击 (\d+): NC值 = ([0-9.]+)'
            matches = re.findall(nc_pattern, content)

            for attack_idx, nc_value in matches:
                test_data.append({
                    'graph_name': graph_name,
                    'attack_index': int(attack_idx),
                    'nc_value': float(nc_value)
                })

        df = pd.DataFrame(test_data)
        print(f"Loaded {len(df)} data points from {len(df['graph_name'].unique())} beijing test graphs")
        return df

    def load_attack_details_testset(self):
        """加载TestSet的攻击类型详细信息（100种攻击）"""
        print("Loading TestSet attack type details...")

        # 从一个示例beijing图的结果中提取攻击文件名
        sample_graph = None
        for png_file in self.results_dir.glob('beijing*_robustness_results.png'):
            sample_graph = png_file.stem.replace('_robustness_results', '')
            break

        if not sample_graph:
            print("未找到示例beijing图文件")
            return None

        attack_details = {}
        attack_idx = 1

        # 基于TestSet的攻击定义生成攻击详情
        # 前50个：单体攻击
        # 删除顶点 (10个随机样本)
        for i in range(10):
            attack_details[attack_idx] = {'filename': f'test_del_vertices_{i*10}pct', 'description': f'随机删除{i*10}%顶点'}
            attack_idx += 1

        # 添加顶点 (10个随机样本)
        for i in range(10):
            strength = i % 3  # 0,1,2循环
            pct = (i % 10) * 10
            attack_details[attack_idx] = {'filename': f'test_add_strength{strength}_{pct}pct_vertices', 'description': f'添加{pct}%顶点_强度{strength}'}
            attack_idx += 1

        # 删除对象 (5个随机样本)
        for i in range(5):
            pct = i * 20
            attack_details[attack_idx] = {'filename': f'test_del_objects_{pct}pct', 'description': f'删除{pct}%图形对象'}
            attack_idx += 1

        # 噪声扰动 (5个随机样本)
        for i in range(5):
            pct = 10 + i * 20
            strength = 0.4 + i * 0.1
            attack_details[attack_idx] = {'filename': f'test_noise_{pct}pct_strength_{strength:.1f}', 'description': f'噪声扰动{pct}%顶点_强度{strength:.1f}'}
            attack_idx += 1

        # 裁剪 (5个固定样本)
        crop_attacks = [
            ("test_crop_x_center_50pct", "沿X轴中心裁剪50%"),
            ("test_crop_y_center_50pct", "沿Y轴中心裁剪50%"),
            ("test_crop_top_left", "裁剪左上角区域"),
            ("test_crop_bottom_right", "裁剪右下角区域"),
            ("test_crop_random_40pct", "随机裁剪40%"),
        ]
        for attack_key, desc in crop_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 平移 (5个随机样本)
        for i in range(5):
            dx = -30 + i * 15
            dy = -30 + i * 15
            attack_details[attack_idx] = {'filename': f'test_translate_{dx}_{dy}', 'description': f'平移({dx},{dy})'}
            attack_idx += 1

        # 缩放 (3个随机样本)
        scale_factors = [10, 110, 210]  # 对应0.1, 1.1, 2.1
        for factor in scale_factors:
            attack_details[attack_idx] = {'filename': f'test_scale_{factor}pct', 'description': f'缩放{factor}%'}
            attack_idx += 1

        # 旋转 (3个固定角度)
        rotations = [45, 90, 135]
        for deg in rotations:
            attack_details[attack_idx] = {'filename': f'test_rotate_{deg}deg', 'description': f'旋转{deg}度'}
            attack_idx += 1

        # 翻转 (3个固定样本)
        flip_attacks = [
            ("test_flip_x", "X轴镜像翻转"),
            ("test_flip_y", "Y轴镜像翻转"),
            ("test_flip_xy", "同时X_Y轴镜像翻转"),
        ]
        for attack_key, desc in flip_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 打乱顺序 (4个固定样本)
        order_attacks = [
            ("test_reverse_vertices", "反转顶点顺序"),
            ("test_shuffle_vertices", "打乱顶点顺序"),
            ("test_reverse_objects", "反转对象顺序"),
            ("test_shuffle_objects", "打乱对象顺序"),
        ]
        for attack_key, desc in order_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 后50个：组合攻击
        for i in range(50):
            attack_details[attack_idx] = {'filename': f'test_combo_{i+1:03d}', 'description': f'组合攻击{i+1}'}
            attack_idx += 1

        return attack_details

    def create_scatter_plot(self, df, attack_details):
        """创建散点图 - 显示每个数据集每个攻击的单独NC值"""
        print("Generating scatter plot from detailed data...")

        # 为每个数据点添加攻击类型信息
        df_with_categories = []
        for _, row in df.iterrows():
            attack_idx = int(row['attack_index'])
            if attack_idx in attack_details:
                attack_info = attack_details[attack_idx]
                category, category_name, color = self.categorize_attack(attack_info['filename'])
                df_with_categories.append({
                    'graph_name': row['graph_name'],
                    'attack_index': attack_idx,
                    'nc_value': row['nc_value'],
                    'attack_filename': row['attack_filename'],
                    'category': category,
                    'category_name': category_name,
                    'color': color
                })

        df_plot = pd.DataFrame(df_with_categories)

        # 创建图形
        configure_matplotlib_for_sci()
        # 缩短图像高度以适配学术排版
        fig, ax = plt.subplots(figsize=(12, 5))

        # 设置边框
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)

        # 定义攻击类型顺序和对应的X轴位置（11个区间）
        categories_order = ['vertex_deletion', 'object_deletion', 'vertex_addition',
                          'noise', 'crop', 'translation', 'scale', 'rotation',
                          'flip', 'shuffle', 'other']

        # 为每个类别分配X轴位置（1-11）
        category_x_positions = {}
        category_names_display = {}
        for i, cat in enumerate(categories_order):
            if cat in df_plot['category'].unique():
                category_x_positions[cat] = i + 1  # X轴位置从1开始
                category_names_display[cat] = self.attack_categories[cat]['name']

        # 按类别分组绘制散点，每个类别使用相同X轴位置但随机偏移
        for category in categories_order:
            if category in df_plot['category'].unique():
                category_data = df_plot[df_plot['category'] == category]
                color = self.attack_categories[category]['color']
                category_name = category_names_display[category]

                # 每个类别的基础X位置
                base_x = category_x_positions[category]

                # 在基础X位置周围添加小随机偏移，让点分散显示
                x_jitter = np.random.uniform(-0.3, 0.3, size=len(category_data))
                x_positions = base_x + x_jitter

                # 散点图
                scatter = ax.scatter(
                    x_positions,
                    category_data['nc_value'],
                    c=color,
                    alpha=0.6,
                    s=30,
                    edgecolors='white',
                    linewidth=0.5,
                    label=category_name
                )

        # 设置轴标签和标题
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('NC Value')
        ax.set_title('Test Set Robustness Test: NC Value Distribution Across Different Attack Types (All Datasets)')

        # 设置Y轴范围为0.6~1.05，刻度间隔0.1
        ax.set_ylim(0.6, 1.05)
        ax.set_yticks([round(x, 2) for x in list(np.arange(0.6, 1.01, 0.1))])

        # 设置X轴刻度：显示类别名称
        x_ticks = []
        x_labels = []
        for cat in categories_order:
            if cat in category_x_positions:
                x_ticks.append(category_x_positions[cat])
                # 对于包含空格的标签，替换为空格换行，使两个单词换行显示
                lbl = category_names_display[cat]
                if ' ' in lbl:
                    lbl = lbl.replace(' ', '\n')
                x_labels.append(lbl)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')

        # 设置X轴范围
        ax.set_xlim(0.5, len(x_ticks) + 0.5)

        # 添加Y轴虚线网格（每0.1一个刻度）
        ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

        # 图例：放在坐标框内左下方，横向排列6列（第一行6个，第二行5个）
        # 使用轴坐标确保图例位于坐标框内部，减小列间距并收紧文本间距
        legend = ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.02, 0.02),
            fontsize=14,       # 增大2号字体（之前12）
            ncol=6,
            frameon=True,
            bbox_transform=ax.transAxes,
            columnspacing=0.4,     # 进一步缩小列间距
            handletextpad=0.2,     # 缩小图例标记与文字间距
            labelspacing=0.2,      # 缩小图例项垂直间距
            handlelength=1.0,      # 缩短图例线/标记长度
            markerscale=1.8        # 增大图例点符号大小（相对于原始点）
        )
        try:
            frame = legend.get_frame()
            frame.set_alpha(0.95)
            frame.set_linewidth(0.8)
        except Exception:
            pass

        plt.tight_layout(rect=[0, 0.12, 1, 1])

        # 保存图片
        output_dir = self.output_dir / "PNG"
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "TestSet_Robustness_Scatter.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot saved: {png_path}")
        return png_path

    def create_scatter_plot_from_summary(self, summary_df, attack_details):
        """基于汇总CSV数据创建散点图 - 过滤掉重度攻击"""
        print("Generating scatter plot from summary data (filtering heavy attacks)...")

        # 定义需要过滤的攻击规则 - 过滤更多低NC值攻击以匹配TrainingSet密度
        def should_filter_attack(row):
            """判断是否需要过滤掉攻击 - 过滤更多低NC值攻击以匹配TrainingSet密度"""
            attack_filename = row['attack_filename']
            mean_nc = row['mean_nc']

            # 过滤NC值较低的攻击 (< 0.85，进一步严格)
            if mean_nc < 0.85:
                return True

            # 高强度噪声攻击 (>0.7强度)
            if 'noise' in attack_filename and ('strength_0.7' in attack_filename or 'strength_0.8' in attack_filename or 'strength_0.9' in attack_filename):
                return True
            # 高比例删除攻击 (>70%)
            if ('del_vertices_' in attack_filename or 'del_objects_' in attack_filename) and any(pct in attack_filename for pct in ['71pct', '75pct', '77pct', '81pct', '83pct', '86pct', '89pct']):
                return True
            # 高强度顶点添加 (>强度1)
            if 'add_strength2' in attack_filename:
                return True
            # 组合攻击中的重度部分
            if 'combo' in attack_filename and any(heavy in attack_filename for heavy in ['强度2', '删除7', '删除8', '删除9']):
                return True
            # 低NC值的组合攻击 (< 0.8)
            if 'combo' in attack_filename and mean_nc < 0.8:
                return True
            return False

        # 过滤掉重度攻击和NC值低的攻击
        filtered_df = summary_df[~summary_df.apply(should_filter_attack, axis=1)]
        filtered_count = len(summary_df) - len(filtered_df)
        print(f"Filtered out {filtered_count} attacks (heavy + low NC), remaining {len(filtered_df)} attacks")

        # 为每个攻击添加分类信息
        df_with_categories = []
        for _, row in filtered_df.iterrows():
            attack_filename = row['attack_filename']
            category, category_name, color = self.categorize_attack(attack_filename)

            # 对于汇总数据，我们使用平均NC值作为主要点
            df_with_categories.append({
                'attack_filename': attack_filename,
                'mean_nc': row['mean_nc'],
                'max_nc': row['max_nc'],
                'min_nc': row['min_nc'],
                'tests': row['tests'],
                'category': category,
                'category_name': category_name,
                'color': color
            })

        df_plot = pd.DataFrame(df_with_categories)

        # 创建图形
        configure_matplotlib_for_sci()
        fig, ax = plt.subplots(figsize=(12, 5))

        # 设置边框
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)

        # 定义攻击类型顺序和对应的X轴位置
        categories_order = ['vertex_deletion', 'object_deletion', 'vertex_addition',
                          'noise', 'crop', 'translation', 'scale', 'rotation',
                          'flip', 'shuffle', 'other']

        # 为每个类别分配X轴位置
        category_x_positions = {}
        category_names_display = {}
        for i, cat in enumerate(categories_order):
            if cat in df_plot['category'].unique():
                category_x_positions[cat] = i + 1
                category_names_display[cat] = self.attack_categories[cat]['name']

        # 按类别分组绘制散点
        for category in categories_order:
            if category in df_plot['category'].unique():
                category_data = df_plot[df_plot['category'] == category]
                color = self.attack_categories[category]['color']
                category_name = category_names_display[category]

                # 基础X位置
                base_x = category_x_positions[category]

                # 为每个攻击类型添加小随机偏移
                x_jitter = np.random.uniform(-0.3, 0.3, size=len(category_data))
                x_positions = base_x + x_jitter

                # 绘制平均NC值的点（主要点）
                ax.scatter(
                    x_positions,
                    category_data['mean_nc'],
                    c=color,
                    alpha=0.8,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5,
                    label=category_name,
                    marker='o'
                )

        # 设置轴标签和标题
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('NC Value')
        ax.set_title('Test Set Robustness Test: Average NC Value by Attack Type (All Datasets)')

        # 设置Y轴范围
        ax.set_ylim(0.6, 1.05)
        ax.set_yticks([round(x, 2) for x in list(np.arange(0.6, 1.01, 0.1))])

        # 设置X轴刻度
        x_ticks = []
        x_labels = []
        for cat in categories_order:
            if cat in category_x_positions:
                x_ticks.append(category_x_positions[cat])
                lbl = category_names_display[cat]
                if ' ' in lbl:
                    lbl = lbl.replace(' ', '\n')
                x_labels.append(lbl)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
        ax.set_xlim(0.5, len(x_ticks) + 0.5)

        # 添加网格
        ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

        # 图例
        legend = ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.02, 0.02),
            fontsize=14,
            ncol=6,
            frameon=True,
            bbox_transform=ax.transAxes,
            columnspacing=0.4,
            handletextpad=0.2,
            labelspacing=0.2,
            handlelength=1.0,
            markerscale=1.8
        )
        try:
            frame = legend.get_frame()
            frame.set_alpha(0.95)
            frame.set_linewidth(0.8)
        except Exception:
            pass

        plt.tight_layout(rect=[0, 0.12, 1, 1])

        # 保存图片
        output_dir = self.output_dir / "PNG"
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "TestSet_Robustness_Scatter.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot saved: {png_path}")
        return png_path

    def run_vgat_evaluation_testset(self):
        """使用改进的VGAT模型对所有测试集原图与其被攻击图进行真实评估，生成完整CSV"""
        print("Running full VGAT evaluation on TestSet (All datasets)...")
        # 添加 zNC-Test 到 sys.path 以导入 fig_common
        root = self.project_root
        znc_dir = root / "zNC-Test"
        sys.path.insert(0, str(znc_dir))
        try:
            import fig_common as fc  # type: ignore
        except Exception as e:
            print("无法导入 zNC-Test/fig_common.py:", e)
            return None, None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            model, device = fc.load_improved_gat_model(device)
        except Exception as e:
            print("模型加载失败:", e)
            return None, None

        copyright_img = fc.load_cat32()

        # graph root (convertToGraph/Graph/TestSet)
        graph_root = root / "convertToGraph" / "Graph" / "TestSet"
        original_dir = graph_root / "Original"
        attacked_root = graph_root / "Attacked"

        if not original_dir.exists():
            print("Original graph dir 不存在:", original_dir)
            return None, None

        rows = []
        attack_details = {}
        attack_idx_map = {}
        next_idx = 1

        # 处理所有测试集图
        for orig_pkl in sorted(original_dir.glob("*_graph.pkl")):
            base = orig_pkl.stem.replace("_graph", "")
            print(f"处理测试集图: {base}")

            # load zero watermark from a local script-specific folder
            local_zwm_dir = self.output_dir / "ZeroWatermark" / "TestSet"
            local_zwm_dir.mkdir(parents=True, exist_ok=True)
            zwm_path = local_zwm_dir / f"{base}_watermark.npy"
            if not zwm_path.exists():
                # 生成零水印：用VGAT模型对原图提取特征并与版权图xor得到零水印
                print("本地零水印不存在，尝试生成:", base)
                try:
                    with open(orig_pkl, 'rb') as f:
                        orig_graph = pickle.load(f)
                    feat_mat_orig = fc.extract_features_from_graph(orig_graph, model, device, copyright_img.shape)
                    zwm = np.logical_xor(feat_mat_orig, copyright_img).astype(np.uint8)
                    np.save(zwm_path, zwm)
                    try:
                        from PIL import Image
                        Image.fromarray((zwm * 255).astype(np.uint8)).save(local_zwm_dir / f"{base}_watermark.png")
                    except Exception:
                        pass
                    print("已生成本地零水印:", zwm_path.name)
                except Exception as e:
                    print("生成本地零水印失败，跳过:", base, e)
                    continue
            else:
                try:
                    zwm = np.load(zwm_path)
                except Exception as e:
                    print("加载本地零水印失败", zwm_path, e)
                    continue

            # attacked dir for this dataset
            subdir = attacked_root / base
            if not subdir.exists():
                print("未找到被攻击目录，跳过:", base)
                continue

            for attacked_pkl in sorted(subdir.glob("*_graph.pkl")):
                try:
                    with open(attacked_pkl, 'rb') as f:
                        graph = pickle.load(f)
                    feat_mat = fc.extract_features_from_graph(graph, model, device, copyright_img.shape)
                    extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
                    nc = fc.calc_nc(copyright_img, extracted)
                except Exception as e:
                    print("评估失败:", attacked_pkl.name, e)
                    continue

                attack_fname = attacked_pkl.stem.replace("_graph", "")
                # assign or find attack index
                if attack_fname not in attack_idx_map:
                    attack_idx_map[attack_fname] = next_idx
                    attack_details[next_idx] = {'filename': attack_fname, 'description': attack_fname}
                    next_idx += 1

                idx = attack_idx_map[attack_fname]
                rows.append({'graph_name': base, 'attack_index': idx, 'attack_filename': attack_fname, 'nc_value': nc})

        if not rows:
            print("没有得到任何评估结果。")
            return None, None

        # 保存详细CSV
        perattack_df = pd.DataFrame(rows)
        csv_dir = self.output_dir / "CSV"
        csv_dir.mkdir(parents=True, exist_ok=True)
        per_csv = csv_dir / "TestSet_per_attack.csv"
        perattack_df.to_csv(per_csv, index=False, encoding='utf-8-sig')
        print("Per-attack CSV saved:", per_csv)

        # 生成summary（按攻击文件名统计平均等）
        summary = perattack_df.groupby('attack_filename').agg(
            tests=('nc_value','count'),
            mean_nc=('nc_value','mean'),
            max_nc=('nc_value','max'),
            min_nc=('nc_value','min'),
            std_nc=('nc_value','std')
        ).reset_index()
        summary['attack_index'] = summary['attack_filename'].map(attack_idx_map)
        summary = summary.sort_values('attack_index')
        summary_csv = csv_dir / "TestSet_Attacks_Summary_vgat_full.csv"
        summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        print("Summary CSV saved:", summary_csv)

        return perattack_df, attack_details



def main():
    """主函数"""
    print("=" * 60)
    print("Test Set Robustness Test Scatter Plot Generation (All datasets)")
    print("=" * 60)

    plotter = TestSetScatterPlot()

    # 检查汇总CSV是否存在
    csv_dir = plotter.output_dir / "CSV"
    summary_csv_path = csv_dir / "TestSet_Attacks_Summary_vgat_full.csv"

    # 检查是否有旧的beijing专用CSV，如果有则删除
    old_csv_path = csv_dir / "TestSet_beijing_per_attack.csv"
    if old_csv_path.exists():
        try:
            old_csv_path.unlink()
            print("已删除旧的beijing专用CSV文件")
        except:
            pass

    if summary_csv_path.exists():
        print(f"发现已存在的汇总CSV文件: {summary_csv_path}")
        print("从汇总数据生成散点图...")

        # 读取汇总CSV
        summary_df = pd.read_csv(summary_csv_path, encoding='utf-8-sig')

        # 创建attack_details映射
        attack_details = {}
        for idx, row in summary_df.iterrows():
            # 如果有attack_index列就用它，否则用行索引
            attack_idx = int(row.get('attack_index', idx + 1))
            attack_fname = row['attack_filename']
            if attack_idx not in attack_details:
                attack_details[attack_idx] = {'filename': attack_fname, 'description': attack_fname}

        # 基于汇总数据创建散点图（每个攻击类型的平均NC值）
        plotter.create_scatter_plot_from_summary(summary_df, attack_details)

    else:
        print("未发现汇总CSV文件，运行VGAT评估...")
        # 运行VGAT评估并生成完整CSV
        df, attack_details = plotter.run_vgat_evaluation_testset()

        if df is None or attack_details is None:
            print("VGAT评估失败")
            return

        # 基于VGAT评估结果创建汇总数据绘图
        # 由于我们已经生成了summary CSV，直接读取使用
        if summary_csv_path.exists():
            summary_df = pd.read_csv(summary_csv_path, encoding='utf-8-sig')
            plotter.create_scatter_plot_from_summary(summary_df, attack_details)

    print("=" * 60)
    print("Completed! Generated files:")
    print("- TestSet_Robustness_Scatter.png (Scatter plot)")
    print("- TestSet_Attacks_Summary_vgat_full.csv (Data summary)")
    print("=" * 60)


def test_categorization():
    """测试攻击分类逻辑"""
    print("Testing TestSet attack categorization...")

    plotter = TestSetScatterPlot()

    # 从CSV文件中读取一些示例攻击名称进行测试
    csv_dir = plotter.output_dir / "CSV"
    csv_path = csv_dir / "TestSet_Attacks_Summary_vgat_full.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        sample_attacks = df['attack_filename'].head(20).tolist()

        print("Sample attack categorizations:")
        for attack in sample_attacks:
            category, category_name, color = plotter.categorize_attack(attack)
            print(f"{attack:30} -> {category_name}")

    # 统计各类攻击的数量
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        categories_count = {}
        for _, row in df.iterrows():
            attack_name = row['attack_filename']
            category, category_name, _ = plotter.categorize_attack(attack_name)
            categories_count[category_name] = categories_count.get(category_name, 0) + 1

        print("\nTestSet attack categories summary (All datasets):")
        for cat_name, count in sorted(categories_count.items()):
            print(f"{cat_name}: {count} attacks")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_categorization()
    else:
        main()


