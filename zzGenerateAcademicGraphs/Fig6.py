from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=6,
		fig_name="fig6_translate_nc",
		has_nested_structure=False,
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig6_avg_nc",
		xlabel="Translation settings",
		x_rotation=0,
		x_ha="center",
		xtick_label_map={
			"X-axisTranslation20": "X 20 units",
			"Y-axisTranslation20": "Y 20 units",
			"沿X, Y轴SeparateTranslation20": "X&Y 20 units",
			"X-axisTranslation20, Y-axisTranslation40": "X 20,Y 40 units",
			"X-axisTranslation30, Y-axisTranslation10": "X 30,Y 10 units",
		},
		legend_shift_right_col_down=True,
		series_style_override={
			"Proposed": {"markersize": 2.5, "linewidth": 0.8},
		},
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
