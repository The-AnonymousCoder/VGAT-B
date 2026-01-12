from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=9,
		fig_name="fig9_flip_nc",
		has_nested_structure=False,
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig9_avg_nc",
		xlabel="Flip settings",
		x_rotation=0,
		x_ha="center",
		xtick_label_map={
			"X轴镜像Flip": "X-axis Mirror",
			"Y轴镜像Flip": "Y-axis Mirror",
			"SimultaneousX, Y轴镜像Flip": "X&Y Simultaneous",
		},
		series_style_override={
			"Proposed": {"markersize": 2.5, "linewidth": 0.8},
		},
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
