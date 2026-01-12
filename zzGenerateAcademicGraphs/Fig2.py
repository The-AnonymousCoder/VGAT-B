from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=2,
		fig_name="fig2_add_vertices_nc",
		has_nested_structure=True,
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig2_avg_nc",
		xlabel="Vertex addition intensity-ratio",
		x_rotation=0,
		x_ha="center",
		x_filter_step=1,
		markersize=3.0,
		# Restore to default xtick font (None uses rcParams)
		xtick_fontsize=None,
		# Plot only these labels with equal spacing and markers only on them
		plot_labels=[
			"0-10%", "0-50%", "0-90%",
			"1-10%", "1-50%", "1-90%",
			"2-10%", "2-50%", "2-90%",
		],
		legend_shift_right_col_down=True,
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
