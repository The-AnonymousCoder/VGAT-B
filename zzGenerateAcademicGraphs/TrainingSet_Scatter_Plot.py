#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集鲁棒性测试散点图
基于所有训练集数据和攻击生成学术图
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

class TrainingSetScatterPlot:
    """训练集散点图生成器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "zNC-Test" / "NC-Results" / "TrainingSet"
        self.output_dir = self.project_root / "zzGenerateAcademicGraphs"

        # 攻击类型映射（基于NC-TrainingSet.py）
        # 使用深色高对比度调色，适合SCI期刊打印与缩图
        deep_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1b9e77"
        ]
        self.attack_categories = {
            'vertex_deletion': {'pattern': r'delete_\d+pct_vertices', 'name': 'Vertex Deletion', 'color': deep_colors[0]},
            'object_deletion': {'pattern': r'delete_\d+pct_objects', 'name': 'Object Deletion', 'color': deep_colors[1]},
            'vertex_addition': {'pattern': r'add_strength\d+_\d+pct_vertices', 'name': 'Vertex Addition', 'color': deep_colors[2]},
            'noise': {'pattern': r'noise_\d+pct_strength_', 'name': 'Noise', 'color': deep_colors[3]},
            'crop': {'pattern': r'crop_', 'name': 'Crop', 'color': deep_colors[4]},
            'translation': {'pattern': r'translate_', 'name': 'Translation', 'color': deep_colors[5]},
            'scale': {'pattern': r'scale_\d+pct', 'name': 'Scale', 'color': deep_colors[6]},
            'rotation': {'pattern': r'rotate_\d+deg', 'name': 'Rotation', 'color': deep_colors[7]},
            'flip': {'pattern': r'flip_', 'name': 'Flip', 'color': deep_colors[8]},
            'shuffle': {'pattern': r'shuffle_|reverse_', 'name': 'Order Operation', 'color': deep_colors[9]},
            'other': {'pattern': r'.*', 'name': 'Compound Attack', 'color': deep_colors[10]}
        }

    def categorize_attack(self, attack_filename):
        """根据文件名分类攻击类型"""
        # 移除文件扩展名
        base_name = attack_filename.replace('_graph.pkl', '').replace('.geojson', '')

        # 移除图名前缀
        for prefix in ['Boundary_', 'Building_', 'Lake_', 'Landuse_', 'Railways_', 'Road_']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break

        # 匹配攻击类型
        for category, info in self.attack_categories.items():
            if category != 'other' and re.search(info['pattern'], base_name):
                return category, info['name'], info['color']

        return 'other', self.attack_categories['other']['name'], self.attack_categories['other']['color']

    def load_training_data(self):
        """加载所有训练集图的NC数据"""
        print("Loading training set NC data...")

        training_data = []

        # 找到所有训练集图的stats文件
        for stats_file in self.results_dir.glob('*_robustness_stats.txt'):
            graph_name = stats_file.stem.replace('_robustness_stats', '')
            print(f"处理图: {graph_name}")

            # 读取stats文件
            with open(stats_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取NC值
            nc_pattern = r'攻击 (\d+): NC值 = ([0-9.]+)'
            matches = re.findall(nc_pattern, content)

            for attack_idx, nc_value in matches:
                training_data.append({
                    'graph_name': graph_name,
                    'attack_index': int(attack_idx),
                    'nc_value': float(nc_value)
                })

        df = pd.DataFrame(training_data)
        print(f"Loaded {len(df)} data points from {len(df['graph_name'].unique())} training graphs")
        return df

    def load_attack_details(self):
        """加载攻击类型详细信息"""
        print("Loading attack type details...")

        # 从一个示例图的结果中提取攻击文件名
        sample_graph = None
        for png_file in self.results_dir.glob('*_robustness_results.png'):
            sample_graph = png_file.stem.replace('_robustness_results', '')
            break

        if not sample_graph:
            print("未找到示例图文件")
            return None

        # 读取对应的零水印提取结果文件（如果存在）
        extract_dir = self.project_root / "zzManuscript" / "ExtractWatermark_Fig12" / "Proposed"
        if extract_dir.exists():
            # 尝试读取一个提取结果来获取攻击文件名映射
            pass

        # 从NC-TrainingSet.py的攻击映射中推断
        attack_details = {}
        attack_idx = 1

        # 基础攻击类型
        basic_attacks = [
            ("delete_10pct_vertices", "删除10%顶点"),
            ("delete_20pct_vertices", "删除20%顶点"),
            ("delete_30pct_vertices", "删除30%顶点"),
            ("delete_40pct_vertices", "删除40%顶点"),
            ("delete_50pct_vertices", "删除50%顶点"),
            ("delete_10pct_objects", "删除10%对象"),
            ("delete_20pct_objects", "删除20%对象"),
            ("delete_30pct_objects", "删除30%对象"),
            ("delete_40pct_objects", "删除40%对象"),
            ("delete_50pct_objects", "删除50%对象"),
            ("add_10pct_vertices", "添加10%顶点"),
            ("add_20pct_vertices", "添加20%顶点"),
            ("add_30pct_vertices", "添加30%顶点"),
            ("add_40pct_vertices", "添加40%顶点"),
            ("add_50pct_vertices", "添加50%顶点"),
            ("noise_10pct_strength_0.2", "噪声10%强度0.2"),
            ("noise_10pct_strength_0.4", "噪声10%强度0.4"),
            ("noise_10pct_strength_0.6", "噪声10%强度0.6"),
            ("noise_20pct_strength_0.3", "噪声20%强度0.3"),
            ("noise_30pct_strength_0.8", "噪声30%强度0.8"),
        ]

        for attack_key, desc in basic_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 裁剪攻击
        crop_attacks = [
            ("crop_x_center_50pct", "X轴中心裁剪50%"),
            ("crop_y_center_50pct", "Y轴中心裁剪50%"),
            ("crop_top_left", "左上角裁剪"),
            ("crop_bottom_right", "右下角裁剪"),
            ("crop_random_40pct", "随机裁剪40%"),
        ]

        for attack_key, desc in crop_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 几何变换攻击
        geom_attacks = [
            ("translate_10_10", "平移(10,10)"),
            ("translate_20_20", "平移(20,20)"),
            ("translate_x_30", "X轴平移30"),
            ("translate_y_15", "Y轴平移15"),
            ("translate_neg10", "平移(-10,-10)"),
            ("scale_0.5x", "缩放0.5倍"),
            ("scale_2x", "缩放2倍"),
            ("scale_x0.5_y2", "X0.5倍Y2倍"),
            ("scale_x2_y0.5", "X2倍Y0.5倍"),
            ("scale_random", "随机缩放"),
            ("rotate_45", "旋转45°"),
            ("rotate_90", "旋转90°"),
            ("rotate_135", "旋转135°"),
            ("rotate_180", "旋转180°"),
            ("rotate_random", "随机旋转"),
            ("flip_x", "X轴翻转"),
            ("flip_y", "Y轴翻转"),
            ("flip_xy", "XY轴翻转"),
        ]

        for attack_key, desc in geom_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 顺序操作
        order_attacks = [
            ("reverse_vertex_order", "反转顶点顺序"),
            ("reverse_object_order", "反转对象顺序"),
            ("shuffle_objects", "打乱对象顺序"),
            ("shuffle_vertices", "打乱顶点顺序"),
            ("jitter_vertices", "顶点随机偏移"),
        ]

        for attack_key, desc in order_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 其他操作
        other_attacks = [
            ("merge_objects", "合并对象"),
            ("split_objects", "拆分对象"),
        ]

        for attack_key, desc in other_attacks:
            attack_details[attack_idx] = {'filename': attack_key, 'description': desc}
            attack_idx += 1

        # 扩展攻击（55-100）
        for i in range(55, 101):
            attack_details[attack_idx] = {'filename': f"attack_{i:03d}", 'description': f"扩展攻击{i}"}
            attack_idx += 1

        # 组合攻击（1-100）
        for i in range(1, 101):
            attack_details[attack_idx] = {'filename': f"combo_attack_{i:03d}", 'description': f"组合攻击{i}"}
            attack_idx += 1

        return attack_details

    def create_scatter_plot(self, df, attack_details):
        """创建散点图"""
        print("Generating scatter plot...")

        # 为每个数据点添加攻击类型信息
        df_with_categories = []
        for _, row in df.iterrows():
            attack_idx = row['attack_index']
            if attack_idx in attack_details:
                attack_info = attack_details[attack_idx]
                category, category_name, color = self.categorize_attack(attack_info['filename'])
                df_with_categories.append({
                    'graph_name': row['graph_name'],
                    'attack_index': attack_idx,
                    'nc_value': row['nc_value'],
                    'attack_description': attack_info['description'],
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
        ax.set_title('Training Set Robustness Test: NC Value Distribution Across Different Attack Types')

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
        png_path = output_dir / "TrainingSet_Robustness_Scatter.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot saved: {png_path}")

        return png_path

    def create_scatter_plot_from_summary(self, summary_df, attack_details):
        """基于汇总CSV数据创建散点图"""
        print("Generating scatter plot from summary data...")

        # 为每个攻击添加分类信息
        df_with_categories = []
        for _, row in summary_df.iterrows():
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
        ax.set_title('Training Set Robustness Test: Average NC Value by Attack Type')

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
        png_path = output_dir / "TrainingSet_Robustness_Scatter.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot saved: {png_path}")
        return png_path

    def run_vgat_evaluation(self):
        """使用改进的VGAT模型对所有训练集原图与其被攻击图进行真实评估，生成完整CSV"""
        print("Running full VGAT evaluation on TrainingSet...")
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

        # graph root (convertToGraph/Graph/TrainingSet)
        graph_root = root / "convertToGraph" / "Graph" / "TrainingSet"
        original_dir = graph_root / "Original"
        attacked_root = graph_root / "Attacked"

        if not original_dir.exists():
            print("Original graph dir 不存在:", original_dir)
            return None, None

        rows = []
        attack_details = {}
        attack_idx_map = {}
        next_idx = 1

        for orig_pkl in sorted(original_dir.glob("*_graph.pkl")):
            base = orig_pkl.stem.replace("_graph", "")
            # load zero watermark from a local script-specific folder instead of old global folder
            local_zwm_dir = self.output_dir / "ZeroWatermark" / "TrainingSet"
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

            # attacked dir
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
        per_csv = csv_dir / "vgat_full_per_attack.csv"
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
        summary_csv = csv_dir / "TrainingSet_Attacks_Summary_vgat_full.csv"
        summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        print("Summary CSV saved:", summary_csv)

        return perattack_df, attack_details

def main():
    """主函数"""
    print("=" * 60)
    print("Training Set Robustness Test Scatter Plot Generation")
    print("=" * 60)

    plotter = TrainingSetScatterPlot()

    # 检查汇总CSV是否已存在
    csv_dir = plotter.output_dir / "CSV"
    summary_csv_path = csv_dir / "TrainingSet_Attacks_Summary_vgat_full.csv"

    if summary_csv_path.exists():
        print(f"发现已存在的CSV文件: {summary_csv_path}")
        print("直接从CSV文件生成图表...")

        # 读取汇总CSV
        summary_df = pd.read_csv(summary_csv_path, encoding='utf-8-sig')

        # 创建attack_details映射（这里我们不需要attack_index，因为直接用文件名分类）
        attack_details = {}

        # 基于汇总数据创建散点图
        plotter.create_scatter_plot_from_summary(summary_df, attack_details)

    else:
        print("未发现CSV文件，运行VGAT评估...")
        # 运行VGAT评估并生成完整CSV
        df, attack_details = plotter.run_vgat_evaluation()

        if df is None or attack_details is None:
            print("VGAT评估失败")
            return

        # 基于VGAT评估结果创建散点图
        plotter.create_scatter_plot(df, attack_details)

    print("=" * 60)
    print("Completed! Generated files:")
    print("- TrainingSet_Robustness_Scatter.png (Scatter plot)")
    print("- TrainingSet_Attacks_Summary_vgat_full.csv (Data summary)")
    print("=" * 60)

def test_categorization():
    """测试攻击分类逻辑"""
    print("Testing attack categorization...")

    plotter = TrainingSetScatterPlot()

    # 从CSV文件中读取一些示例攻击名称进行测试
    import pandas as pd
    csv_path = plotter.output_dir / "CSV" / "TrainingSet_Attacks_Summary_vgat_full.csv"
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

        print("\nAttack categories summary:")
        for cat_name, count in sorted(categories_count.items()):
            print(f"{cat_name}: {count} attacks")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_categorization()
    else:
        main()
