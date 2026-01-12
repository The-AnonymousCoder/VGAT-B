# -*- coding: utf-8 -*-
"""
Xi25/Fig3.py - 对象删除攻击的NC评估（提取使用 Xi25；攻击与输出模仿 Wu25）
- 攻击：删除对象比例 0.1~0.9（步长0.1）
- 数据：`embed/Cat32_*.shp`
- 输出：`NC-Results/Fig3/` CSV、XLSX、PNG
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import geopandas as gpd
from pyproj import CRS as _CRS  # type: ignore

import sys
try:
	from . import extract as xi25_extract  # type: ignore
except Exception:
	import os as _os
	BASE = _os.path.dirname(_os.path.abspath(__file__))
	if BASE not in sys.path:
		sys.path.insert(0, BASE)
	import extract as xi25_extract  # type: ignore

# 字体
try:
	preferred_fonts = [
		'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC',
		'Heiti SC', 'Hiragino Sans GB', 'Source Han Sans CN', 'STHeiti',
		'Arial Unicode MS'
	]
	available = {f.name for f in font_manager.fontManager.ttflist}
	chosen = None
	for fname in preferred_fonts:
		if fname in available:
			chosen = fname
			break
	if chosen:
		rcParams = rcParams
		rcParams['font.sans-serif'] = [chosen]
	rcParams['axes.unicode_minus'] = False
except Exception:
	pass

SCRIPT_DIR = Path(__file__).resolve().parent
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig3'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
import os
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

DELETE_RATIOS = [i / 100 for i in range(10, 100, 10)]


def generate_delete_objects_variant(src_path: Path, pct: int) -> Path:
	base = src_path.stem
	subdir = (SCRIPT_DIR / 'attacked' / 'delete_objects' / 'Fig3_delete_objects' / base)
	subdir.mkdir(parents=True, exist_ok=True)
	gdf = gpd.read_file(str(src_path))
	attacked = gdf.copy()
	n_total = len(gdf)
	num_to_delete = int(n_total * pct / 100)
	if num_to_delete > 0 and n_total > 0:
		import random
		random.seed(42)
		indices = list(range(n_total))
		to_del = random.sample(indices, min(num_to_delete, len(indices)))
		attacked = attacked.drop(to_del).reset_index(drop=True)
	# CRS 保障：沿用源数据，或 .prj，或 WGS84
	try:
		if getattr(gdf, 'crs', None) is not None:
			attacked.set_crs(gdf.crs, allow_override=True, inplace=True)  # type: ignore
		else:
			prj_path = Path(src_path).with_suffix('.prj')
			if prj_path.exists():
				try:
					crs_obj = _CRS.from_wkt(prj_path.read_text(encoding='utf-8', errors='ignore'))
					attacked.set_crs(crs_obj, allow_override=True, inplace=True)  # type: ignore
				except Exception:
					attacked.set_crs('EPSG:4326', allow_override=True, inplace=True)  # type: ignore
			else:
				attacked.set_crs('EPSG:4326', allow_override=True, inplace=True)  # type: ignore
	except Exception:
		pass
	out_path = subdir / f'delete_{pct}pct_objects.shp'
	attacked.to_file(str(out_path), driver='ESRI Shapefile')
	return out_path


def _xi25_extract_nc(attacked_path: Path, watermark_path: Path) -> float:
	out_dir = SCRIPT_DIR / 'extract' / 'watermark'
	out_dir.mkdir(parents=True, exist_ok=True)
	out_img = out_dir / f'{attacked_path.stem}.png'
	sys.argv = [
		'extract',
		'--in-shp', str(attacked_path),
		'--orig-watermark', str(watermark_path),
		'--out-img', str(out_img),
	]
	xi25_extract.main()
	try:
		from .metrics import nc as metric_nc  # type: ignore
	except Exception:
		from metrics import nc as metric_nc  # type: ignore
	from PIL import Image
	wm = (np.array(Image.open(str(watermark_path)).convert('L')) > 127).astype(np.uint8)
	ex = (np.array(Image.open(str(out_img)).convert('L')) > 127).astype(np.uint8)
	return float(metric_nc(ex, wm))


def run_evaluation():
	print('=== Xi25 Fig3：对象删除攻击 NC 评估 ===')
	if KEEP_EXISTING and any(DIR_RESULTS.glob('*.csv')):
		print(f'⚠️ KEEP_EXISTING=True，且 Fig3 结果已存在于 {DIR_RESULTS}，跳过')
		return
	if not (SCRIPT_DIR / WATERMARK).exists():
		print('缺少必要文件: Cat32.png')
		return
	if len(VECTOR_FILES) == 0:
		print('未发现嵌入矢量：期待 embed/ 下 Cat32_*.shp')
		return

	rows = []
	# 仅选择非点数据
	selected: List[tuple[Path, str]] = []
	for vector_file, file_name in zip(VECTOR_FILES, FILE_NAMES):
		try:
			gdf_types = set(gpd.read_file(str(vector_file)).geom_type.astype(str).unique())
			if any('Point' in t for t in gdf_types):
				print(f'跳过点数据: {file_name}')
				continue
			selected.append((vector_file, file_name))
		except Exception:
			selected.append((vector_file, file_name))
	if len(selected) == 0:
		print('无可用于评估的线/面数据。')
		return
	USE_FILES = [p for p, _ in selected]
	USE_NAMES = [n for _, n in selected]
	nc_matrix = np.zeros((len(USE_NAMES), len(DELETE_RATIOS)))

	for file_idx, (vector_file, file_name) in enumerate(zip(USE_FILES, USE_NAMES)):
		abs_vector = str(vector_file)
		print(f'处理: {file_name}')
		# 已预过滤点数据
		for ratio_idx, delete_ratio in enumerate(DELETE_RATIOS):
			try:
				pct = int(round(delete_ratio * 100))
				attacked_path = generate_delete_objects_variant(Path(abs_vector), pct)
				nc_value = _xi25_extract_nc(attacked_path, SCRIPT_DIR / WATERMARK)
				rows.append({
					'file': file_name,
					'delete_ratio': delete_ratio,
					'nc': nc_value,
				})
				nc_matrix[file_idx, ratio_idx] = nc_value
				print(f'  ratio={delete_ratio:.1f} -> NC={nc_value:.4f}')
			except Exception as exc:
				print('  失败:', file_name, delete_ratio, exc)
				nc_matrix[file_idx, ratio_idx] = 0.0

	df = pd.DataFrame(rows, columns=['file', 'delete_ratio', 'nc'])

	# 层次结构输出
	hierarchical_rows: List[dict] = []
	for ratio in DELETE_RATIOS:
		pct = int(round(ratio * 100))
		hierarchical_rows.append({'对象删除': f'比例: {pct}%', 'VGAT': '', '类型': 'subheader'})
		for file_name in FILE_NAMES:
			sub = df[(df['file'] == file_name) & (df['delete_ratio'] == ratio)]
			nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
			hierarchical_rows.append({'对象删除': f'  {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
		avg_nc = float(df[df['delete_ratio'] == ratio]['nc'].mean()) if not df.empty else 0.0
		hierarchical_rows.append({'对象删除': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

	hierarchical_df = pd.DataFrame(hierarchical_rows)

	csv_path = DIR_RESULTS / 'fig3_delete_objects_nc.csv'
	xlsx_path = DIR_RESULTS / 'fig3_delete_objects_nc.xlsx'
	hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
	try:
		with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
			hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
	except Exception as e:
		print('Excel保存警告:', e)

	# 绘图
	plt.figure(figsize=(10, 6))
	for file_idx, file_name in enumerate(FILE_NAMES):
		plt.plot(DELETE_RATIOS, nc_matrix[file_idx, :], '-o', label=file_name)
	avg_curve = np.mean(nc_matrix, axis=0)
	plt.plot(DELETE_RATIOS, avg_curve, 'k-o', linewidth=2.5, label='平均')
	plt.grid(True, alpha=0.3)
	plt.xlabel('删除比例')
	plt.ylabel('NC')
	plt.title('对象删除攻击的NC鲁棒性（Xi25/Fig3）')
	plt.ylim(0, 1.05)
	plt.legend(loc='best', fontsize=8, ncol=2)
	plt.tight_layout()
	fig_path = DIR_RESULTS / 'fig3_delete_objects_nc.png'
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close()

	print('结果表保存:', csv_path)
	print('Excel保存:', xlsx_path)
	print('曲线图保存:', fig_path)
	print('=== 完成 ===')


if __name__ == '__main__':
	run_evaluation()

