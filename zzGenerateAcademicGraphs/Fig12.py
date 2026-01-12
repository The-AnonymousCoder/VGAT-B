from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison_bar


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent

	series_map = load_all_experiments(
		root=root,
		fig_number=12,
		fig_name="fig12_compound_seq_nc",
		has_nested_structure=False,
	)

	# 重新组织数据：X轴显示方法名，每个方法一个bar
	method_order = ["Proposed", "Tan24", "Wu25", "Xi24", "Xi25", "Lin18"]
	method_display_names = ["VGAT-B", "Tan et al.[17]", "Wu et al.[24]", "Xi et al.[22]", "Xi et al.[23]", "Lin et al.[25]"]

	# 创建数据：X轴标签和对应的NC值
	x_labels = method_display_names  # X轴标签
	values = []
	for method in method_order:
		if method in series_map:
			labels, method_values = series_map[method]
			if method_values:
				values.append(method_values[0])  # NC值
			else:
				values.append(0.0)
		else:
			values.append(0.0)

	# 使用一个虚拟的系列名，这样只有一个系列，每个位置一个bar
	new_series_map = {"Methods": (x_labels, values)}

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison_bar(
		series_map=new_series_map,
		save_dir=output_dir,
		fig_name="Fig12_avg_nc",
		xlabel="",
		figsize=(8.0, 5.0),
		bar_width=0.6,  # 增大bar宽度
		show_method_labels=False,  # 不显示方法标签，直接用X轴标签
		show_values=True,  # 显示数值
		bar_spacing_multiplier=1.0,  # 减小间距
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
