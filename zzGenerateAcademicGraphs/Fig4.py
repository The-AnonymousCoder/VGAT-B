from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=4,
		fig_name="fig4_noise_nc",
		has_nested_structure=True,  # Xi24/Wu25/Xi25 use subheader format
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig4_avg_nc",
		xlabel="Noise intensity-ratio",
		x_rotation=45,
		x_ha="right",
		x_filter_step=1,
		markersize=3.0,
		xtick_fontsize=None,
		plot_labels=[
			"0.4-10%", "0.4-50%", "0.4-90%",
			"0.6-10%", "0.6-50%", "0.6-90%",
			"0.8-10%", "0.8-50%", "0.8-90%",
		],
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
