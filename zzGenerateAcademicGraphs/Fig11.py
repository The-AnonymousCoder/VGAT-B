from __future__ import annotations

import sys
from pathlib import Path

from common import load_all_experiments, plot_comparison


def main() -> int:
	# 自动获取项目根目录（zzGenerateAcademicGraphs的父目录）
	root = Path(__file__).parent.parent
	
	series_map = load_all_experiments(
		root=root,
		fig_number=11,
		fig_name="fig11_compound_nc",
		has_nested_structure=False,
	)

	output_dir = root / "zzGenerateAcademicGraphs"
	plot_comparison(
		series_map=series_map,
		save_dir=output_dir,
		fig_name="Fig11_avg_nc",
		xlabel="Attack types",
		figsize=(10.0, 5.5),
		x_rotation=0,
		x_ha="center",
		xtick_label_map={
			"A1 Vertex Deletion 10%": "A1",
			"A2 Vertex Addition Intensity1 5000%%": "A2",
			"A3 Object Deletion 50%": "A3",
			"A4 Noise攻击 Intensity0.8 5000%%": "A4",
			"A5 Y-axisCenterCrop50%": "A5",
			"A6 Translation X20,Y40": "A6",
			"A7 Scale 90%": "A7",
			"A8 顺时针Rotation180°": "A8",
			"A9 Flip Y轴镜像Flip": "A9",
			"A10 顺序: Reverse顶点→Reverse对象": "A10",
		},
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
