# -*- coding: utf-8 -*-
"""
Xi24/Fig7.py - 缩放攻击的NC评估（提取使用 Xi24；攻击与输出模仿 Tan24/zNC-Test）
- 攻击：scale 0.1, 0.5, 0.9, 1.3, 1.7, 2.1（绕质心等比缩放）
- 数据：`embed/` 下 `Cat32_*.shp`
- 输出：`NC-Results/Fig7/` CSV、XLSX、PNG
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.affinity import scale as shp_scale  # type: ignore
from pyproj import CRS as _CRS  # type: ignore

import sys
try:
	from . import watermark_extract  # type: ignore
except Exception:
	BASE = os.path.dirname(os.path.abspath(__file__))
	if BASE not in sys.path:
		sys.path.insert(0, BASE)
	import watermark_extract  # type: ignore


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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig7'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'scale' / 'Fig7_scale'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
import os
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

SCALE_FACTORS = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]


def _ensure_crs(attacked: gpd.GeoDataFrame, src_path: Path, src_gdf: gpd.GeoDataFrame) -> None:
	try:
		if getattr(src_gdf, 'crs', None) is not None:
			attacked.set_crs(src_gdf.crs, allow_override=True, inplace=True)  # type: ignore
			return
		prj_path = src_path.with_suffix('.prj')
		if prj_path.exists():
			try:
				crs_obj = _CRS.from_wkt(prj_path.read_text(encoding='utf-8', errors='ignore'))
				attacked.set_crs(crs_obj, allow_override=True, inplace=True)  # type: ignore
				return
			except Exception:
				pass
		attacked.set_crs('EPSG:4326', allow_override=True, inplace=True)  # type: ignore
	except Exception:
		pass


def generate_scale_variants(src_path: Path) -> Dict[float, Path]:
	base = src_path.stem
	subdir = DIR_ATTACKED / base
	subdir.mkdir(parents=True, exist_ok=True)

	gdf = gpd.read_file(str(src_path))
	base_crs = gdf.crs if getattr(gdf, 'crs', None) is not None else 'EPSG:4326'
	bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
	global_center_x = (bounds[0] + bounds[2]) / 2
	global_center_y = (bounds[1] + bounds[3]) / 2
	global_center = (global_center_x, global_center_y)

	attacked_paths: Dict[float, Path] = {}
	for s in SCALE_FACTORS:
		try:
			attacked = gdf.copy()
			attacked['geometry'] = attacked['geometry'].apply(lambda geom: shp_scale(geom, s, s, origin=global_center))
			_ensure_crs(attacked, src_path, gdf)
			out_path = subdir / f'scale_{int(round(s*100))}pct.shp'
			if KEEP_EXISTING and out_path.exists():
				print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
				attacked_paths[s] = out_path
			else:
				attacked.to_file(str(out_path), driver='ESRI Shapefile')
				attacked_paths[s] = out_path
		except Exception:
			pass

	return attacked_paths


def _xi24_extract_nc(attacked_path: Path, watermark_path: Path) -> float:
	"""通过 Xi24 提取主程序计算 NC 值。"""
	# 读取原始水印
	from PIL import Image
	w_original = Image.open(str(watermark_path)).convert('L')
	w_original = np.array(w_original)
	w_original = (w_original > 128).astype(int)
	
	# 提取水印
	w_extracted = watermark_extract.extract_watermark(
		str(attacked_path),
		w_original.shape,
		THRatio=0.3,
		mfactor=1e6,
		Q=10
	)
	
	# 计算NC
	nc = watermark_extract.calculate_nc(w_extracted, w_original)
	
	# 保存提取的水印图片
	out_dir = SCRIPT_DIR / 'extract' / 'watermark'
	out_dir.mkdir(parents=True, exist_ok=True)
	out_img = out_dir / f'{attacked_path.stem}.png'
	extracted_img = Image.fromarray((w_extracted * 255).astype(np.uint8))
	extracted_img.save(str(out_img))
	
	return float(nc)


def run_evaluation():
	print('=== Xi24 Fig7：缩放攻击 NC 评估 ===')
	if KEEP_EXISTING:
		csv_path_check = DIR_RESULTS / 'fig7_scale_nc.csv'
		if csv_path_check.exists():
			print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig7: {csv_path_check}')
			return
	missing = []
	if not (SCRIPT_DIR / WATERMARK).exists():
		missing.append(WATERMARK)
	if len(VECTOR_FILES) == 0:
		print('未发现嵌入矢量：期待 embed/ 下 Cat32_*.shp')
		return
	if missing:
		print(f'缺少必要文件: {missing}')
		return

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
	order_scales = list(SCALE_FACTORS)

	# 记录每文件×每攻击的 NC
	nc_matrix = np.zeros((len(USE_NAMES), len(order_scales)))
	rows: List[Dict] = []

	for file_idx, (vector_file, file_name) in enumerate(zip(USE_FILES, USE_NAMES)):
		print(f'处理: {file_name}')
		attacked_map = generate_scale_variants(Path(str(vector_file)))
		for s_idx, s in enumerate(order_scales):
			try:
				shp_path = attacked_map.get(s)
				if shp_path is None:
					nc_value = 0.0
				else:
					nc_value = _xi24_extract_nc(shp_path, SCRIPT_DIR / WATERMARK)
				nc_matrix[file_idx, s_idx] = nc_value
				rows.append({'file': file_name, 'scale': s, 'nc': nc_value})
				print(f'  scale={s:.2f} -> NC={nc_value:.4f}')
			except Exception as exc:
				print('  失败:', file_name, s, exc)
				nc_matrix[file_idx, s_idx] = 0.0

	df = pd.DataFrame(rows, columns=['file', 'scale', 'nc'])

	# 层次表：缩放比例 -> 各文件 -> 平均
	hierarchical_rows: List[dict] = []
	for s in order_scales:
		label = f'{int(round(s*100))}%'
		hierarchical_rows.append({'缩放比例': label, 'VGAT': '', '类型': 'header'})
		for file_name in FILE_NAMES:
			sub = df[(df['file'] == file_name) & (df['scale'] == s)]
			nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
			hierarchical_rows.append({'缩放比例': f'  {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
		avg_nc = float(df[df['scale'] == s]['nc'].mean()) if not df.empty else 0.0
		hierarchical_rows.append({'缩放比例': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

	hierarchical_df = pd.DataFrame(hierarchical_rows)

	# 保存 CSV/XLSX
	csv_path = DIR_RESULTS / 'fig7_scale_nc.csv'
	xlsx_path = DIR_RESULTS / 'fig7_scale_nc.xlsx'
	hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
	try:
		with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
			hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
	except Exception as e:
		print('Excel保存警告:', e)

	# 绘图
	plt.figure(figsize=(10, 6))
	x = np.arange(len(order_scales))
	labels = [f'{int(round(s*100))}%' for s in order_scales]
	for file_idx, file_name in enumerate(USE_NAMES):
		plt.plot(x, nc_matrix[file_idx, :], '-o', alpha=0.8, label=file_name)
	avg_vals = np.mean(nc_matrix, axis=0)
	plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
	plt.grid(True, alpha=0.3)
	plt.xticks(x, labels)
	plt.xlabel('缩放比例')
	plt.ylabel('NC')
	plt.title('缩放攻击 的NC鲁棒性（Xi24/Fig7）')
	plt.legend(loc='best', fontsize=8, ncol=2)
	plt.tight_layout()
	fig_path = DIR_RESULTS / 'fig7_scale_nc.png'
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close()

	print('结果表保存:', csv_path)
	print('Excel保存:', xlsx_path)
	print('曲线图保存:', fig_path)
	print('=== 完成 ===')


if __name__ == '__main__':
	run_evaluation()



