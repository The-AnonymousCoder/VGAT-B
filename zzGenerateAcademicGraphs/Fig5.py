from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=5,
		fig_name="fig5_crop_nc",
		has_nested_structure=False,
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig5_avg_nc",
		xlabel="Crop settings",
		x_rotation=0,
		x_ha="center",
		plot_labels=[
			"X-axisCenterCrop50%",
			"Y-axisCenterCrop50%",
			"CropUpper-Left",
			"CropLower-Right",
			"RandomCrop40%",
		],
		xtick_label_map={
			"X-axisCenterCrop50%": "Center 50% (X)",
			"Y-axisCenterCrop50%": "Center 50% (Y)",
			"CropUpper-Left": "Upper-Left",
			"CropLower-Right": "Lower-Right",
			"RandomCrop40%": "Random 40%",
		},
		legend_shift_right_col_down=True,
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
