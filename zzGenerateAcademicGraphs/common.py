"""Common utilities for generating academic comparison figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil


METHOD_DISPLAY_NAME_MAP = {
	"Proposed": "VGAT-B",
	"Tan24": "Tan et al. [17]",
	"Wu25": "Wu et al. [24]",
	"Xi24": "Xi et al. [22]",
	"Xi25": "Xi et al. [23]",
	"Lin18": "Lin et al. [25]",
}

METHOD_PLOT_ORDER = [
	"Proposed",  # VGAT - 第一个
	"Tan24",     # Tan24 - 第二个
	"Wu25",      # Wu25 - 第三个
	"Xi24",      # Xi24 - 第四个
	"Xi25",      # Xi25 - 第五个
	"Lin18",     # Lin18 - 第六个
]

METHOD_COLOR_MAP = {
	"Proposed": "#d62728",
	"Tan24": "#1f77b4",
	"Wu25": "#ff7f0e",
	"Xi24": "#2ca02c",
	"Xi25": "#9467bd",
	"Lin18": "#17becf",
}

METHOD_MARKER_MAP = {
	"Proposed": "o",
	"Tan24": "D",
	"Wu25": "^",
	"Xi24": "s",
	"Xi25": "v",
	"Lin18": "*",
}

LEGEND_DISPLAY_ORDER = [
	METHOD_DISPLAY_NAME_MAP[name]
	for name in METHOD_PLOT_ORDER
	if name in METHOD_DISPLAY_NAME_MAP
]

DEFAULT_LINEWIDTH = 1.2
DEFAULT_MARKERSIZE = 4.5
LEGEND_FONT_SIZE = 12  # 图例字号增大
AXIS_LABEL_FONT_SIZE = 15
XTICK_FONT_SIZE = 13
YTICK_FONT_SIZE = 14


def normalize_label(label: str) -> str:
	"""Normalize labels to match across different experiment formats and translate to English.
	
	Examples:
	- "比例 0.1" -> "10%"
	- "比例: 10%" -> "10%"
	- "10%" -> "10%"
	- "强度: 0-10%" -> "Intensity: 0-10%"
	- "强度: 0-比例 0.1" -> "Intensity: 0-10%"
	"""
	import re
	
	# Lin18 specific label mappings (must be done first before general translations)
	lin18_mappings = {
		# Fig10 shuffle mappings
		"反转顶点": "Reverse顶点顺序",
		"打乱顶点": "Shuffle顶点顺序",
		"反转对象": "Reverse对象顺序",
		"打乱对象": "Shuffle对象顺序",
		# Fig11 compound attack mappings
		"删点10%": "A1 Vertex Deletion 10%",
		"增点强度1比例50%": "A2 Vertex Addition Intensity1 5000%%",
		"删对象50%": "A3 Object Deletion 50%",
		"噪声强度0.8比例50%": "A4 Noise攻击 Intensity0.8 5000%%",
		"裁剪左10%": "A5 Y-axisCenterCrop50%",
		"平移右1%": "A6 Translation X20,Y40",
		"缩放0.9": "A7 Scale 90%",
		"旋转45度": "A8 顺时针Rotation180°",
		"X镜像": "A9 Flip Y轴镜像Flip",
		"顺序: 反转顶点→反转对象": "A10 顺序: Reverse顶点→Reverse对象",
		# Fig12 sequential compound mapping
		"全部攻击顺序执行": "复合(A1→A10)",
	}
	
	for lin18_label, standard_label in lin18_mappings.items():
		if label == lin18_label:
			label = standard_label
			break
	
	# Handle "比例: X%" pattern -> "X%"
	label = re.sub(r'比例[:：]\s*', '', label)
	
	# Handle "比例 0.X" pattern -> "X0%"
	match = re.search(r'比例\s*([0-9.]+)', label)
	if match:
		ratio_val = float(match.group(1))
		percentage = f"{int(ratio_val * 100)}%"
		# Replace the matched pattern
		label = re.sub(r'比例\s*[0-9.]+', percentage, label)

	# Harmonize specific crop/random patterns across datasets
	# e.g., "随机裁剪40%对象" or "随机裁剪40%" -> "RandomCrop40%"
	label = re.sub(r'随机\s*裁剪\s*([0-9]+%)对象', r'RandomCrop\1', label)
	label = re.sub(r'随机\s*裁剪\s*([0-9]+%)', r'RandomCrop\1', label)
	
	# Harmonize flip patterns
	label = re.sub(r'同时\s*X\s*、?\s*Y\s*轴\s*镜像\s*翻转', 'SimultaneousX, Y轴镜像Flip', label)
	
	# Translation dictionary for Chinese to English
	translations = {
		# Attack types
		"强度": "Intensity",
		"Intensity: ": "",  # Remove "Intensity: " prefix for brevity
		"顶点删除": "Vertex Deletion",
		"顶点增加": "Vertex Addition",
		"删除对象": "Object Deletion",
		"对象删除": "Object Deletion",
		"噪声": "Noise",
		"裁剪": "Crop",
		"平移": "Translation",
		"缩放": "Scale",
		"旋转": "Rotation",
		"翻转": "Flip",
		"打乱": "Shuffle",
		"组合攻击": "Compound",
		"顺序组合": "Sequential",
		# Crop directions
		"裁剪左上角": "Upper-Left",
		"裁剪中心": "Center",
		"裁剪右下角": "Lower-Right",
		"随机裁剪": "Random",
		"左上角": "Upper-Left",
		"中心": "Center",
		"右下角": "Lower-Right",
		"随机": "Random",
		# Translation directions
		"沿X轴平移": "X-axis",
		"沿Y轴平移": "Y-axis",
		"沿X轴": "X-axis",
		"沿Y轴": "Y-axis",
		"分别": "Separate",
		# Flip/Rotation
		"镜像翻转": "Mirror",
		"旋转同时": "Rot+Flip",
		"同时": "Simultaneous",
		# Shuffle
		"反转顺序打乱": "Reverse+Shuffle",
		"反转": "Reverse",
		"顺序打乱": "Shuffle",
		"烧烤": "BBQ",
		"打乱对象": "Shuffle Obj",
		# Compound attacks - simplify
		"Fig": "A",
		"比例": "",
		# Remove Chinese punctuation
		"，": ", ",
		"、": ", ",
		# Common words
		"方向": "",
		"方式": "",
	}
	
	# Apply translations
	for chinese, english in translations.items():
		label = label.replace(chinese, english)
	
	return label


def parse_fig_averages_from_csv(
	csv_path: Path,
	has_nested_structure: bool = False
) -> Tuple[List[str], List[float]]:
	"""Parse Fig CSV to extract x-labels and average NC values.
	
	Args:
		csv_path: Path to the CSV file
		has_nested_structure: If True, handles header->subheader->average structure
	
	Returns:
		Tuple of (x_labels, averages)
	"""
	x_labels: List[str] = []
	averages: List[float] = []
	current_header: str | None = None
	current_subheader: str | None = None

	with csv_path.open("r", encoding="utf-8") as f:
		reader = csv.reader(f)
		for row in reader:
			if not row:
				continue
			c0 = row[0].strip() if len(row) >= 1 else ""
			c1 = row[1].strip() if len(row) >= 2 else ""
			c2 = row[2].strip().lower() if len(row) >= 3 and row[2] else ""

			# Skip column header lines
			if c0 in (
				"删除比例", "删点比例", "增加顶点", "添加顶点", "删除对象比例",
				"噪声", "裁剪", "平移方式", "缩放", "旋转", "翻转",
				"打乱", "组合攻击", "顺序组合攻击",
				"Ratio", "ratio", "强度", "Intensity"
			):
				continue

			if c2 == "header":
				current_header = normalize_label(c0)
				if not has_nested_structure:
					current_subheader = None
				continue

			if c2 == "subheader":
				current_subheader = normalize_label(c0)
				continue

			if c2 == "average":
				try:
					avg = float(c1)
				except ValueError:
					continue
				
				# Build combined label
				if has_nested_structure:
					if current_header and current_subheader:
						label = f"{current_header}-{current_subheader}"
					elif current_subheader:
						label = current_subheader
					elif current_header:
						label = current_header
					else:
						label = normalize_label(c0)
				else:
					label = current_header if current_header else normalize_label(c0)
				
				x_labels.append(label)
				averages.append(avg)
				
				if not has_nested_structure:
					current_header = None


	return x_labels, averages


def parse_fig_series(
	fig_dir: Path,
	fig_name: str,
	has_nested_structure: bool = False
) -> Tuple[List[str], List[float]]:
	"""Load Fig series from CSV (preferred) or XLSX fallback."""
	csv_path = fig_dir / f"{fig_name}.csv"
	if csv_path.exists():
		return parse_fig_averages_from_csv(csv_path, has_nested_structure)

	xlsx_path = fig_dir / f"{fig_name}.xlsx"
	if xlsx_path.exists():
		try:
			import pandas as pd  # type: ignore
		except Exception as exc:
			raise RuntimeError(
				f"pandas required for Excel but not available: {xlsx_path}"
			) from exc

		df = pd.read_excel(xlsx_path, header=None, dtype=str)
		x_labels: List[str] = []
		averages: List[float] = []
		current_header: str | None = None
		current_subheader: str | None = None

		for _, row in df.iterrows():
			c0 = str(row[0]).strip() if 0 in row and pd.notna(row[0]) else ""
			c1 = str(row[1]).strip() if 1 in row and pd.notna(row[1]) else ""
			c2 = str(row[2]).strip().lower() if 2 in row and pd.notna(row[2]) else ""

			if c0 in (
				"删除比例", "删点比例", "增加顶点", "添加顶点", "删除对象比例",
				"噪声", "裁剪", "平移方式", "缩放", "旋转", "翻转",
				"打乱", "组合攻击", "顺序组合攻击",
				"Ratio", "ratio", "强度", "Intensity"
			):
				continue
			
			if c2 == "header":
				current_header = normalize_label(c0)
				if not has_nested_structure:
					current_subheader = None
				continue
			
			if c2 == "subheader":
				current_subheader = normalize_label(c0)
				continue
			
			if c2 == "average":
				try:
					avg = float(c1)
				except ValueError:
					continue
				
				if has_nested_structure:
					if current_header and current_subheader:
						label = f"{current_header}-{current_subheader}"
					elif current_subheader:
						label = current_subheader
					elif current_header:
						label = current_header
					else:
						label = normalize_label(c0)
				else:
					label = current_header if current_header else normalize_label(c0)
				
				x_labels.append(label)
				averages.append(avg)
				
				if not has_nested_structure:
					current_header = None

		return x_labels, averages

	raise FileNotFoundError(f"CSV/XLSX not found in {fig_dir}")


def configure_matplotlib_for_sci() -> None:
	"""Configure matplotlib with SCI publication style."""
	mpl.rcParams.update({
		"font.family": "sans-serif",
		"font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
		"axes.labelsize": AXIS_LABEL_FONT_SIZE,
		"axes.titlesize": AXIS_LABEL_FONT_SIZE,
		"xtick.labelsize": XTICK_FONT_SIZE,
		"ytick.labelsize": YTICK_FONT_SIZE,
		"legend.fontsize": LEGEND_FONT_SIZE,
		"axes.spines.top": True,
		"axes.spines.right": True,
		"axes.linewidth": 1.0,
		"figure.dpi": 150,
		"axes.unicode_minus": False,
	})


def plot_comparison(
	series_map: Dict[str, Tuple[List[str], List[float]]],
	save_dir: Path,
	fig_name: str,
	xlabel: str,
	figsize: Tuple[float, float] = (6.5, 4.5),
	x_rotation: int = 0,
	x_ha: str = "center",
	x_filter_step: int = 1,
	markersize: float = DEFAULT_MARKERSIZE,
	xtick_fontsize: int | None = None,
	xtick_labels: List[str] | None = None,
	plot_labels: List[str] | None = None,
	xtick_label_map: Dict[str, str] | None = None,
	legend_shift_right_col_down: bool = True,
	series_style_override: Dict[str, Dict[str, float]] | None = None,
) -> None:
	"""Generate comparison plot for multiple series.
	
	Args:
		series_map: Dict mapping experiment name to (labels, values)
		save_dir: Directory to save outputs
		fig_name: Base name for output files
		xlabel: X-axis label
		figsize: Figure size (width, height) - 默认高度增加到4.5以容纳图例
		x_rotation: X-tick label rotation angle
		x_ha: X-tick label horizontal alignment
		x_filter_step: Show every Nth x-tick label (e.g., 2 = every other label)
	"""
	configure_matplotlib_for_sci()
	fig, ax = plt.subplots(figsize=figsize)

	for side in ("top", "right", "left", "bottom"):
		ax.spines[side].set_visible(True)
		ax.spines[side].set_linewidth(1.0)

	# Use first series as master x-axis (full labels)
	any_key = next(iter(series_map))
	x_labels = series_map[any_key][0]
	series_order = [
		name for name in METHOD_PLOT_ORDER if name in series_map
	]
	series_order.extend(
		name for name in series_map if name not in series_order
	)

	# If plotting a subset for equal spacing, pre-compute filtered positions
	if plot_labels is not None:
		filtered_indices = [i for i, lbl in enumerate(x_labels) if lbl in plot_labels]
		plot_x_vals = list(range(len(filtered_indices)))
		plot_x_labels = [x_labels[i] for i in filtered_indices]
	else:
		filtered_indices = None
		plot_x_vals = list(range(len(x_labels)))
		plot_x_labels = list(x_labels)

	min_y = float("inf")
	max_y = float("-inf")

	series_names: List[str] = list(series_order)
	aligned_values_map: Dict[str, List[float]] = {}
	fallback_colors = ["#d62728", "#9467bd", "#ff7f0e", "#2ca02c", "#1f77b4", "#17becf"]
	fallback_markers = ["o", "v", "^", "s", "D", "*"]

	for idx, name in enumerate(series_names):
		labels, values = series_map[name]
		# Align with master x-axis
		if labels != x_labels:
			label_to_val = {lbl: val for lbl, val in zip(labels, values)}
			values = [label_to_val.get(lbl, float("nan")) for lbl in x_labels]

		# Use subset if requested for equal spacing and reduced markers
		if filtered_indices is not None:
			values_to_plot = [values[i] for i in filtered_indices]
		else:
			values_to_plot = list(values)

		aligned_values_map[name] = list(values)
		display_name = METHOD_DISPLAY_NAME_MAP.get(name, name)
		color = METHOD_COLOR_MAP.get(name, fallback_colors[idx % len(fallback_colors)])
		marker = METHOD_MARKER_MAP.get(name, fallback_markers[idx % len(fallback_markers)])
		z = 5 if name == "Proposed" else 3

		# 统一所有折线的线宽和端点大小
		ax.plot(
			plot_x_vals,
			values_to_plot,
			label=display_name,
			color=color,
			marker=marker,
			linewidth=DEFAULT_LINEWIDTH,  # 统一线宽1.2
			markersize=DEFAULT_MARKERSIZE,  # 统一端点大小4.5
			markeredgewidth=0.5,
			zorder=z,
			clip_on=False,  # 允许符号溢出坐标框，不被裁剪
		)
		valid_vals = [v for v in values_to_plot if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)]
		if valid_vals:
			min_y = min(min_y, *valid_vals)
			max_y = max(max_y, *valid_vals)

	ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE)
	ax.set_ylabel("NC Value", fontsize=AXIS_LABEL_FONT_SIZE)
	ax.tick_params(axis="x", labelsize=xtick_fontsize if xtick_fontsize is not None else XTICK_FONT_SIZE)
	ax.tick_params(axis="y", labelsize=YTICK_FONT_SIZE)

	# 统一Y轴设置：从0开始，刻度为0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0
	ax.set_ylim(0.0, 1.05)  # Y轴从0开始，上限留出空间
	y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
	ax.set_yticks(y_ticks)
	# Y轴刻度标签：去掉0刻度的数值显示
	y_tick_labels = ["", "0.2", "0.4", "0.6", "0.8", "0.9", "1.0"]
	ax.set_yticklabels(y_tick_labels)
	
	# 只在指定刻度位置显示虚线网格
	ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
	ax.xaxis.grid(False)  # X轴不显示网格线
	
	# X轴范围：第一个刻度从最左边开始，最后一个刻度到最右边，充分利用X轴空间
	# 符号通过clip_on=False允许溢出坐标框，不会被截断
	if len(plot_x_vals) > 1:
		ax.set_xlim(plot_x_vals[0], plot_x_vals[-1])

	labels_fontsize = xtick_fontsize if xtick_fontsize is not None else XTICK_FONT_SIZE
	if plot_labels is not None:
		ax.set_xticks(plot_x_vals)
		_display_labels = [
			(xtick_label_map.get(lbl, lbl) if xtick_label_map else lbl)
			for lbl in plot_x_labels
		]
		ax.set_xticklabels(_display_labels, rotation=x_rotation, ha=x_ha, fontsize=labels_fontsize)
	elif xtick_labels is not None:
		filtered_positions = [i for i, lbl in enumerate(x_labels) if lbl in xtick_labels]
		filtered_labels = [x_labels[i] for i in filtered_positions]
		ax.set_xticks(filtered_positions)
		_display_labels = [
			(xtick_label_map.get(lbl, lbl) if xtick_label_map else lbl)
			for lbl in filtered_labels
		]
		ax.set_xticklabels(_display_labels, rotation=x_rotation, ha=x_ha, fontsize=labels_fontsize)
	elif x_filter_step > 1:
		filtered_positions = [i for i in range(0, len(plot_x_vals), x_filter_step)]
		filtered_labels = [plot_x_labels[i] for i in filtered_positions]
		ax.set_xticks(filtered_positions)
		_display_labels = [
			(xtick_label_map.get(lbl, lbl) if xtick_label_map else lbl)
			for lbl in filtered_labels
		]
		ax.set_xticklabels(_display_labels, rotation=x_rotation, ha=x_ha, fontsize=labels_fontsize)
	else:
		ax.set_xticks(plot_x_vals)
		_display_labels = [
			(xtick_label_map.get(lbl, lbl) if xtick_label_map else lbl)
			for lbl in plot_x_labels
		]
		ax.set_xticklabels(_display_labels, rotation=x_rotation, ha=x_ha, fontsize=labels_fontsize)

	handles, legend_labels = ax.get_legend_handles_labels()
	label_to_handle = {lbl: h for h, lbl in zip(legend_labels, handles)}
	ordered_handles: List[object] = []
	ordered_labels_filtered: List[str] = []
	if legend_shift_right_col_down:
		for display_label in LEGEND_DISPLAY_ORDER:
			handle = label_to_handle.get(display_label)
			if handle is not None:
				ordered_handles.append(handle)
				ordered_labels_filtered.append(display_label)
	legend_handles = ordered_handles if ordered_handles else handles
	legend_labels_final = ordered_labels_filtered if ordered_labels_filtered else legend_labels
	legend = ax.legend(
		handles=legend_handles,
		labels=legend_labels_final,
		frameon=True,
		loc="lower center",  # 图例位于坐标框内部底部居中
		bbox_to_anchor=(0.5, 0.02),  # 在坐标框内，X轴正上方
		borderpad=0.4,  # 图例边框与内容的间距
		labelspacing=0.4,  # 图例项之间的垂直间距
		handlelength=1.5,  # 图例线条长度缩短
		handletextpad=0.4,  # 图例线条与文字间距缩小
		ncol=3,
		columnspacing=0.8,  # 列间距缩小，避免溢出
		fontsize=LEGEND_FONT_SIZE,
	)
	frame = legend.get_frame()
	frame.set_facecolor((1.0, 1.0, 1.0, 0.85))  # 半透明白色背景，避免遮挡数据线
	frame.set_edgecolor("#888888")
	frame.set_linewidth(1.0)

	plt.tight_layout()
	save_dir.mkdir(parents=True, exist_ok=True)
	png_dir = save_dir / "PNG"
	csv_dir = save_dir / "CSV"
	manuscript_dir = save_dir.parent / "zzManuscript" / "AcademicGraphs"
	for d in (png_dir, csv_dir, manuscript_dir):
		d.mkdir(parents=True, exist_ok=True)
	fig_path_png = png_dir / f"{fig_name}_comparison.png"
	plt.savefig(fig_path_png.as_posix(), dpi=300, bbox_inches="tight")
	plt.close(fig)
	try:
		shutil.copy2(str(fig_path_png), str(manuscript_dir / fig_path_png.name))
	except Exception:
		pass

	csv_out = csv_dir / f"{fig_name}_comparison.csv"
	with csv_out.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["Label", *series_names])
		for i, label in enumerate(x_labels):
			row = [label]
			for name in series_names:
				vals = aligned_values_map.get(name, [])
				row.append(vals[i] if i < len(vals) else "")
			writer.writerow(row)

	print(f"Saved: {fig_path_png}")
	print(f"Saved: {csv_out}")
def plot_comparison_bar(
	series_map: Dict[str, Tuple[List[str], List[float]]],
	save_dir: Path,
	fig_name: str,
	xlabel: str,
	figsize: Tuple[float, float] = (10.0, 5.0),
	xtick_label_map: Dict[str, str] | None = None,
	bar_width: float = 0.15,
	show_method_labels: bool = False,
	show_values: bool = False,
	bar_spacing_multiplier: float = 1.5,
) -> None:
	"""Generate bar chart comparison plot for multiple series.
	
	Args:
		series_map: Dict mapping experiment name to (labels, values)
		save_dir: Directory to save outputs
		fig_name: Base name for output files
		xlabel: X-axis label
		figsize: Figure size (width, height)
		xtick_label_map: Optional mapping to override x-tick label display text
		bar_width: Width of each bar (default 0.15 for better spacing)
		show_method_labels: If True, show method names under each bar
		show_values: If True, show NC values on top of each bar
		bar_spacing_multiplier: Multiplier for bar spacing (default 1.5)
	"""
	configure_matplotlib_for_sci()
	fig, ax = plt.subplots(figsize=figsize)

	for side in ("top", "right", "left", "bottom"):
		ax.spines[side].set_visible(True)
		ax.spines[side].set_linewidth(1.0)

	# Use first series as master x-axis
	any_key = next(iter(series_map))
	x_labels = series_map[any_key][0]
	
	# Apply label mapping if provided
	if xtick_label_map:
		display_labels = [xtick_label_map.get(lbl, lbl) for lbl in x_labels]
	else:
		# Shorten labels for better display
		display_labels = []
		for label in x_labels:
			# Extract key parts from compound attack labels
			if "Fig" in label:
				parts = label.split()
				if len(parts) >= 2:
					display_labels.append(f"A{parts[0].replace('Fig', '')}")
				else:
					display_labels.append(label[:10])
			else:
				display_labels.append(label[:15])
	
	series_order = [
		name for name in METHOD_PLOT_ORDER if name in series_map
	]
	series_order.extend(
		name for name in series_map if name not in series_order
	)
	aligned_values_map: Dict[str, List[float]] = {}
	series_names: List[str] = list(series_order)

	for name in series_order:
		labels, values = series_map[name]
		if labels != x_labels:
			label_to_val = {lbl: val for lbl, val in zip(labels, values)}
			values = [label_to_val.get(lbl, float("nan")) for lbl in x_labels]
		aligned_values_map[name] = list(values)

	# Bar chart configuration
	x_pos = list(range(len(x_labels)))
	colors = METHOD_COLOR_MAP

	# Add spacing between bars by using a multiplier larger than bar_width
	bar_spacing = bar_width * bar_spacing_multiplier
	
	series_count = len(series_order)
	for idx, name in enumerate(series_order):
		values = aligned_values_map[name]
		offset = (idx - series_count / 2) * bar_spacing + bar_spacing / 2
		positions = [x + offset for x in x_pos]

		# Special handling for Fig12: use different colors for each bar
		if fig_name == "Fig12_avg_nc" and len(series_order) == 1:
			# For Fig12 single series, assign different color to each bar
			method_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf"]
			bar_colors = []
			valid_positions = []
			for i, val in enumerate(values):
				if not (isinstance(val, float) and val != val):  # Not NaN
					bar_colors.append(method_colors[i % len(method_colors)])
					valid_positions.append(positions[i])
			if bar_colors:
				bars = ax.bar(
					valid_positions,
					[val for val in values if not (isinstance(val, float) and val != val)],
					width=bar_width,
					color=bar_colors,
					edgecolor='none',
					linewidth=0,
					zorder=5,
				)
		else:
			bars = ax.bar(
				positions,
				values,
				width=bar_width,
				label=METHOD_DISPLAY_NAME_MAP.get(name, name),
				color=colors.get(name, "#888888"),
				edgecolor='none',  # Remove black border
				linewidth=0,
				zorder=5 if name == "Proposed" else 3,
			)
		
		# Add value labels on top of bars if requested
		if show_values:
			for bar, val in zip(bars, values):
				if not (val != val):  # Check if not NaN
					height = bar.get_height()
					ax.text(
						bar.get_x() + bar.get_width() / 2,
						height,
						f'{val:.2f}',
						ha='center',
						va='bottom',
						fontsize=11,  # Same as Y-axis tick labels
						rotation=0,
					)

	ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE)
	ax.set_ylabel("NC Value", fontsize=AXIS_LABEL_FONT_SIZE)
	ax.tick_params(axis="x", labelsize=XTICK_FONT_SIZE)
	ax.tick_params(axis="y", labelsize=YTICK_FONT_SIZE)
	
	# Set X-axis ticks and labels
	if show_method_labels:
		# Show method name under each bar
		all_bar_positions = []
		all_bar_labels = []
		for idx, name in enumerate(series_order):
			offset = (idx - series_count / 2) * bar_spacing + bar_spacing / 2
			for x in x_pos:
				all_bar_positions.append(x + offset)
				all_bar_labels.append(METHOD_DISPLAY_NAME_MAP.get(name, name))
		ax.set_xticks(all_bar_positions)
		ax.set_xticklabels(all_bar_labels, rotation=0, ha="center", fontsize=XTICK_FONT_SIZE)
	else:
		ax.set_xticks(x_pos)
		ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=XTICK_FONT_SIZE)
	
	ax.set_ylim(0, 1.05)  # Upper limit with headroom
	
	# Set y-ticks with 0.9 included
	ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
	
	ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

	# Fig12柱状图不显示图例（图例信息已在X轴标签中体现）
	
	plt.tight_layout()
	
	# 保存图片和CSV
	save_dir.mkdir(parents=True, exist_ok=True)
	png_dir = save_dir / "PNG"
	csv_dir = save_dir / "CSV"
	manuscript_dir = save_dir.parent / "zzManuscript" / "AcademicGraphs"
	for d in (png_dir, csv_dir, manuscript_dir):
		d.mkdir(parents=True, exist_ok=True)
	fig_path_png = png_dir / f"{fig_name}_comparison.png"
	plt.savefig(fig_path_png.as_posix(), dpi=300, bbox_inches="tight")
	plt.close(fig)
	try:
		shutil.copy2(str(fig_path_png), str(manuscript_dir / fig_path_png.name))
	except Exception:
		pass

	csv_out = csv_dir / f"{fig_name}_comparison.csv"
	with csv_out.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["Label", *series_names])
		for i, label in enumerate(x_labels):
			row = [label]
			for name in series_names:
				vals = aligned_values_map.get(name, [])
				row.append(vals[i] if i < len(vals) else "")
			writer.writerow(row)

	print(f"Saved: {fig_path_png}")
	print(f"Saved: {csv_out}")


def load_all_experiments(
	root: Path,
	fig_number: int,
	fig_name: str,
	has_nested_structure: bool = False
) -> Dict[str, Tuple[List[str], List[float]]]:
	"""Load data from all five experiments for a given figure.
	
	Args:
		root: Project root directory
		fig_number: Figure number (1-12)
		fig_name: Base CSV filename (e.g., "fig1_delete_vertices_nc")
		has_nested_structure: Whether CSV has nested header/subheader structure
	
	Returns:
		Dict mapping experiment name to (labels, values)
	"""
	series_map: Dict[str, Tuple[List[str], List[float]]] = {}

	# Load in desired display order: Proposed, Xi25, Wu25, Xi24, Tan24, Lin18
	
	# Load Proposed (our method)
	ours_dir = root / f"zNC-Test/NC-Results/Fig{fig_number}"
	labels, averages = parse_fig_series(ours_dir, fig_name, has_nested_structure)
	series_map["Proposed"] = (labels, averages)

	# Lin18 uses different filenames for some figures
	lin18_filename_map = {
		"fig10_shuffle_nc": "fig10_reorder_nc",
		"fig11_compound_nc": "fig11_combined_nc",
		"fig12_compound_seq_nc": "fig12_compound_nc",
	}
	
	# Load contrast experiments in chronological order
	for exp_name in ["Xi25", "Wu25", "Xi24", "Tan24", "Lin18"]:
		fig_dir = root / f"zzContrastExperiment/{exp_name}/NC-Results/Fig{fig_number}"
		
		# Use special filename for Lin18 if needed
		if exp_name == "Lin18" and fig_name in lin18_filename_map:
			current_fig_name = lin18_filename_map[fig_name]
		else:
			current_fig_name = fig_name
		
		try:
			labels, values = parse_fig_series(fig_dir, current_fig_name, has_nested_structure)
			series_map[exp_name] = (labels, values)
		except FileNotFoundError:
			print(f"Warning: {exp_name} Fig{fig_number} not found, skipping...")
			continue

	return series_map

