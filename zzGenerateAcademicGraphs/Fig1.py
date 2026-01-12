from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=1,
		fig_name="fig1_delete_vertices_nc",
		has_nested_structure=True,  # Xi24/Wu25/Xi25 use subheader format
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig1_avg_nc",
		xlabel="Vertex deletion ratio (%)",
		x_rotation=0,
		x_ha="center",
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
