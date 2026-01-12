from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=10,
		fig_name="fig10_shuffle_nc",
		has_nested_structure=False,
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig10_avg_nc",
		xlabel="Shuffle settings",
		x_rotation=0,
		x_ha="center",
		xtick_label_map={
			"Reverse顶点顺序": "Reverse Vertices",
			"Shuffle顶点顺序": "Shuffle Vertices",
			"Reverse对象顺序": "Reverse Objects",
			"Shuffle对象顺序": "Shuffle Objects",
		},
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
