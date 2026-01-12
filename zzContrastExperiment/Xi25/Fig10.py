# -*- coding: utf-8 -*-
"""
Xi25/Fig10.py - 顺序打乱攻击的NC评估（提取使用 Xi25；攻击与输出模仿 Tan24/zNC-Test）
- 攻击：reverse_vertices / shuffle_vertices / reverse_objects / shuffle_objects
- 数据：`embed/` 下 `Cat32_*.shp`
- 输出：`NC-Results/Fig10/` CSV、XLSX、PNG
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend for servers/CI
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.geometry import LineString, Polygon  # type: ignore
from pyproj import CRS as _CRS  # type: ignore

import sys
try:
	from . import extract as xi25_extract  # type: ignore
except Exception:
	BASE = os.path.dirname(os.path.abspath(__file__))
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig10'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'shuffle' / 'Fig10_shuffle'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
import os
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

ORDER_ATTACKS: List[Tuple[str, str]] = [
	('reverse_vertices', '反转顶点顺序'),
	('shuffle_vertices', '打乱顶点顺序'),
	('reverse_objects', '反转对象顺序'),
	('shuffle_objects', '打乱对象顺序'),
]


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


def reverse_vertices_geom(geom):
	try:
		if geom.geom_type == 'LineString':
			return LineString(list(geom.coords)[::-1])
		elif geom.geom_type == 'Polygon':
			ext = list(geom.exterior.coords)
			ext = ext[:-1][::-1] + [ext[0]]
			holes = []
			for ring in geom.interiors:
				rc = list(ring.coords)
				rc = rc[:-1][::-1] + [rc[0]]
				holes.append(rc)
			return Polygon(ext, holes=holes if holes else None)
	except Exception:
		pass
	return geom


def shuffle_vertices_geom(geom):
	import random
	try:
		if geom.geom_type == 'LineString':
			coords = list(geom.coords)
			if len(coords) <= 2:
				return geom
			core = coords[1:-1]
			random.shuffle(core)
			return LineString([coords[0]] + core + [coords[-1]])
		elif geom.geom_type == 'Polygon':
			ext = list(geom.exterior.coords)
			if len(ext) <= 4:
				return geom
			core = ext[1:-2]
			random.shuffle(core)
			new_ext = [ext[0]] + core + [ext[-2], ext[-1]]
			holes = []
			for ring in geom.interiors:
				rc = list(ring.coords)
				if len(rc) > 4:
					core_r = rc[1:-2]
					random.shuffle(core_r)
					holes.append([rc[0]] + core_r + [rc[-2], rc[-1]])
				else:
					holes.append(rc)
			return Polygon(new_ext, holes=holes if holes else None)
	except Exception:
		pass
	return geom


def generate_shuffle_variants(src_path: Path) -> Dict[str, Path]:
	base = src_path.stem
	subdir = DIR_ATTACKED / base
	subdir.mkdir(parents=True, exist_ok=True)

	gdf = gpd.read_file(str(src_path))
	attacked_paths: Dict[str, Path] = {}
	for key, _label in ORDER_ATTACKS:
		attacked = gdf.copy()
		if key == 'reverse_vertices':
			attacked['geometry'] = attacked['geometry'].apply(reverse_vertices_geom)
		elif key == 'shuffle_vertices':
			attacked['geometry'] = attacked['geometry'].apply(shuffle_vertices_geom)
		elif key == 'reverse_objects':
			attacked = attacked.iloc[::-1].reset_index(drop=True)
		elif key == 'shuffle_objects':
			attacked = attacked.sample(frac=1).reset_index(drop=True)
		_ensure_crs(attacked, src_path, gdf)
		out_path = subdir / f'{key}.shp'
		attacked.to_file(str(out_path), driver='ESRI Shapefile')
		attacked_paths[key] = out_path

	return attacked_paths


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
	print('=== Xi25 Fig10：顺序打乱攻击 NC 评估 ===')
	if KEEP_EXISTING and any(DIR_RESULTS.glob('*.csv')):
		print(f'⚠️ KEEP_EXISTING=True，且 Fig10 结果已存在于 {DIR_RESULTS}，跳过')
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
	order_keys = [k for k, _ in ORDER_ATTACKS]
	key_to_label = {k: lbl for k, lbl in ORDER_ATTACKS}

	# 记录每文件×每攻击的 NC
	nc_matrix = np.zeros((len(USE_NAMES), len(order_keys)))
	rows: List[Dict] = []

	for file_idx, (vector_file, file_name) in enumerate(zip(USE_FILES, USE_NAMES)):
		print(f'处理: {file_name}')
		attacked_map = generate_shuffle_variants(Path(str(vector_file)))
		for k_idx, key in enumerate(order_keys):
			try:
				shp_path = attacked_map.get(key)
				if shp_path is None:
					nc_value = 0.0
				else:
					nc_value = _xi25_extract_nc(shp_path, SCRIPT_DIR / WATERMARK)
				nc_matrix[file_idx, k_idx] = nc_value
				rows.append({'file': file_name, 'key': key, 'nc': nc_value})
				print(f'  {key} -> NC={nc_value:.4f}')
			except Exception as exc:
				print('  失败:', file_name, key, exc)
				nc_matrix[file_idx, k_idx] = 0.0

	df = pd.DataFrame(rows, columns=['file', 'key', 'nc'])

	# 层次表：顺序策略 -> 各文件 -> 平均
	hierarchical_rows: List[dict] = []
	for key in order_keys:
		label = key_to_label.get(key, key)
		hierarchical_rows.append({'打乱顺序方式': label, 'VGAT': '', '类型': 'header'})
		for file_name in FILE_NAMES:
			sub = df[(df['file'] == file_name) & (df['key'] == key)]
			nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
			hierarchical_rows.append({'打乱顺序方式': f'  {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
		avg_nc = float(df[df['key'] == key]['nc'].mean()) if not df.empty else 0.0
		hierarchical_rows.append({'打乱顺序方式': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

	hierarchical_df = pd.DataFrame(hierarchical_rows)

	# 保存 CSV/XLSX
	csv_path = DIR_RESULTS / 'fig10_shuffle_nc.csv'
	xlsx_path = DIR_RESULTS / 'fig10_shuffle_nc.xlsx'
	hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
	try:
		with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
			hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
	except Exception as e:
		print('Excel保存警告:', e)

	# 绘图
	plt.figure(figsize=(10, 6))
	x = np.arange(len(order_keys))
	labels = [key_to_label.get(k, k) for k in order_keys]
	for file_idx, file_name in enumerate(USE_NAMES):
		plt.plot(x, nc_matrix[file_idx, :], '-o', alpha=0.8, label=file_name)
	avg_vals = np.mean(nc_matrix, axis=0)
	plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='平均')
	plt.grid(True, alpha=0.3)
	plt.xticks(x, labels)
	plt.xlabel('打乱顺序方式')
	plt.ylabel('NC')
	plt.title('打乱顺序攻击 的NC鲁棒性（Xi25/Fig10）')
	plt.legend(loc='best', fontsize=8, ncol=2)
	plt.tight_layout()
	fig_path = DIR_RESULTS / 'fig10_shuffle_nc.png'
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close()

	print('结果表保存:', csv_path)
	print('Excel保存:', xlsx_path)
	print('曲线图保存:', fig_path)
	print('=== 完成 ===')


if __name__ == '__main__':
	run_evaluation()



