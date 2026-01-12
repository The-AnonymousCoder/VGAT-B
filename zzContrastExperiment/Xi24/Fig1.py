# -*- coding: utf-8 -*-
"""
Xi24/Fig1.py - 顶点删除攻击的NC评估（对比试验：提取使用 Xi24；攻击与输出结构模仿 Wu25）
- 攻击：删点比例 0.1~0.9（步长0.1）
- 数据：使用 `embed/` 下 以 `Cat32_*.shp` 命名的含水印矢量文件
- 结果：输出到 `NC-Results/Fig1/` 下的 CSV、XLSX、PNG
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from pyproj import CRS as _CRS  # type: ignore
from shapely.geometry.polygon import orient

# 使用 Xi24 的提取主程序
import sys
try:
	from . import watermark_extract  # type: ignore
except Exception:
	BASE = os.path.dirname(os.path.abspath(__file__))
	if BASE not in sys.path:
		sys.path.insert(0, BASE)
	import watermark_extract  # type: ignore


# 中文字体设置
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
		rcParams = rcParams  # alias
		rcParams['font.sans-serif'] = [chosen]
	rcParams['axes.unicode_minus'] = False
except Exception:
	pass

SCRIPT_DIR = Path(__file__).resolve().parent
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig1'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'delete' / 'Fig1_delete_vertices'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
import os
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'

# 自动发现 embed 目录下 Cat32_*.shp
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

DELETE_FACTORS = [i / 100 for i in range(10, 100, 10)]


def delete_vertices_from_geom(geom, pct: int):
	if geom is None:
		return geom
	try:
		if isinstance(geom, LineString):
			coords = list(geom.coords)
			if len(coords) <= 2:
				return geom
			n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
			if n_to_delete >= len(coords) - 2:
				return geom
			core_idx = list(range(1, len(coords) - 1))
			import random
			random.seed(42)
			to_del = set(random.sample(core_idx, n_to_delete))
			new_coords = [coords[0]] + [coords[i] for i in range(1, len(coords) - 1) if i not in to_del] + [coords[-1]]
			return LineString(new_coords)
		elif isinstance(geom, Polygon):
			ext = list(geom.exterior.coords)
			if len(ext) <= 4:
				return geom
			n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
			if n_to_delete >= len(ext) - 4:
				return geom
			core_idx = list(range(1, len(ext) - 2))
			import random
			random.seed(42)
			to_del = set(random.sample(core_idx, n_to_delete))
			new_ext = [ext[0]] + [ext[i] for i in range(1, len(ext) - 2) if i not in to_del] + [ext[-2], ext[-1]]
			holes = []
			for ring in geom.interiors:
				ring_coords = list(ring.coords)
				if len(ring_coords) > 4:
					n_h = max(1, int((len(ring_coords) - 4) * pct / 100))
					if n_h < len(ring_coords) - 4:
						core_idx_h = list(range(1, len(ring_coords) - 2))
						import random
						random.seed(42)
						to_del_h = set(random.sample(core_idx_h, n_h))
						new_ring = [ring_coords[0]] + [ring_coords[i] for i in range(1, len(ring_coords) - 2) if i not in to_del_h] + [ring_coords[-2], ring_coords[-1]]
						holes.append(new_ring)
					else:
						holes.append(ring_coords)
				else:
					holes.append(ring_coords)
			poly = Polygon(new_ext, holes=holes if holes else None)
			try:
				poly = orient(poly, sign=1.0)
			except Exception:
				pass
			return poly
		elif isinstance(geom, MultiPolygon):
			new_polys = []
			for pg in geom.geoms:
				new_polys.append(delete_vertices_from_geom(pg, pct))
			return MultiPolygon(new_polys)
	except Exception:
		return geom
	return geom


def generate_attacked_variant(src_path: Path, pct: int) -> Path:
	base = src_path.stem
	subdir = DIR_ATTACKED / base
	subdir.mkdir(parents=True, exist_ok=True)
	gdf = gpd.read_file(str(src_path))
	attacked = gdf.copy()
	attacked['geometry'] = attacked['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, pct))
	# 确保写出时具备 CRS，优先沿用源数据；否则尝试读取 .prj；再不行用 WGS84
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
	out_path = subdir / f'delete_{pct}pct_vertices.shp'
	if KEEP_EXISTING and out_path.exists():
		print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
		return out_path
	attacked.to_file(str(out_path), driver='ESRI Shapefile')
	return out_path


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
	print('=== Xi24 Fig1：顶点删除攻击 NC 评估 ===')
	# 跳过机制：若设置 KEEP_EXISTING 且结果 CSV 已存在，则直接跳过整个 Fig 的运行
	if KEEP_EXISTING:
		csv_path_check = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
		if csv_path_check.exists():
			print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig1: {csv_path_check}')
			return
	missing = []
	if not (SCRIPT_DIR / WATERMARK).exists():
		missing.append(WATERMARK)
	if len(VECTOR_FILES) == 0:
		print('未发现嵌入矢量：请先运行 Xi24/batch_test 或 Xi24/embed 以生成 embed/Cat32_*.shp')
		return
	if missing:
		print(f'缺少必要文件: {missing}')
		return

	rows = []
	# 仅选择非点数据用于处理与平均
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
	nc_matrix = np.zeros((len(USE_NAMES), len(DELETE_FACTORS)))

	for file_idx, (vector_file, file_name) in enumerate(zip(USE_FILES, USE_NAMES)):
		abs_vector = str(vector_file)
		print(f'处理: {file_name}')
		# 此处已事先过滤点数据
		for factor_idx, delete_factor in enumerate(DELETE_FACTORS):
			try:
				pct = int(round(delete_factor * 100))
				attacked_path = generate_attacked_variant(Path(abs_vector), pct)
				nc_value = _xi24_extract_nc(attacked_path, SCRIPT_DIR / WATERMARK)
				rows.append({
					'file': file_name,
					'delete_factor': delete_factor,
					'nc': nc_value,
				})
				nc_matrix[file_idx, factor_idx] = nc_value
				print(f'  ratio={delete_factor:.1f} -> NC={nc_value:.4f}')
			except Exception as exc:
				print('  失败:', file_name, delete_factor, exc)
				nc_matrix[file_idx, factor_idx] = 0.0

	# 确保空结果时也具备列，避免 KeyError
	df = pd.DataFrame(rows, columns=['file', 'delete_factor', 'nc'])

	# 层次结构输出（模仿 Wu25）
	hierarchical_rows: List[dict] = []
	for factor in DELETE_FACTORS:
		pct = int(round(factor * 100))
		hierarchical_rows.append({'删点比例': f'{pct}%', 'VGAT': '', '类型': 'subheader'})
		for file_name in FILE_NAMES:
			if df.empty:
				nc_value = 0.0
			else:
				sub = df[(df['file'] == file_name) & (df['delete_factor'] == factor)]
				nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
			hierarchical_rows.append({'删点比例': f'  {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
		avg_nc = float(df[df['delete_factor'] == factor]['nc'].mean()) if not df.empty else 0.0
		hierarchical_rows.append({'删点比例': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

	hierarchical_df = pd.DataFrame(hierarchical_rows)

	# 保存 CSV/XLSX
	csv_path = DIR_RESULTS / 'fig1_delete_vertices_nc.csv'
	xlsx_path = DIR_RESULTS / 'fig1_delete_vertices_nc.xlsx'
	hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
	try:
		with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
			hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
	except Exception as e:
		print('Excel保存警告:', e)

	# 绘图
	plt.figure(figsize=(10, 6))
	for file_idx, file_name in enumerate(FILE_NAMES):
		plt.plot(DELETE_FACTORS, nc_matrix[file_idx, :], '-o', label=file_name)
	avg_curve = np.mean(nc_matrix, axis=0)
	plt.plot(DELETE_FACTORS, avg_curve, 'k-o', linewidth=2.5, label='平均')
	plt.grid(True, alpha=0.3)
	plt.xlabel('删点比例')
	plt.ylabel('NC')
	plt.title('顶点删除攻击的NC鲁棒性（Xi24/Fig1）')
	plt.ylim(0, 1.05)
	plt.legend(loc='best', fontsize=8, ncol=2)
	plt.tight_layout()
	fig_path = DIR_RESULTS / 'fig1_delete_vertices_nc.png'
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close()

	print('结果表保存:', csv_path)
	print('Excel保存:', xlsx_path)
	print('曲线图保存:', fig_path)
	print('=== 完成 ===')


if __name__ == '__main__':
	run_evaluation()

