# -*- coding: utf-8 -*-
"""
Xi24/Fig2.py - 增加顶点攻击的NC评估（对比试验：提取用 Xi24；攻击与输出模仿 Wu25）
- 攻击：addRatio 0.1~0.9 × strength 0/1/2
- 数据：`embed/` 下 以 `Cat32_*.shp` 命名
- 结果：`NC-Results/Fig2/` 输出 CSV、XLSX、PNG
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from pyproj import CRS as _CRS  # type: ignore

import sys
try:
	from . import watermark_extract  # type: ignore
except Exception:
	import os as _os
	BASE = _os.path.dirname(_os.path.abspath(__file__))
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig2'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
import os
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

STRENGTHS: List[int] = [0, 1, 2]
ADD_RATIOS: List[float] = [i / 100 for i in range(10, 100, 10)]


def add_vertices_to_geom(geom, strength: int, pct: int):
	if geom is None:
		return geom
	try:
		if isinstance(geom, LineString):
			coords = list(geom.coords)
			if len(coords) < 2:
				return geom
			n_to_add = min(3, max(1, int((len(coords) - 1) * pct / 100)))
			new_coords = [coords[0]]
			for i in range(len(coords) - 1):
				p1, p2 = coords[i], coords[i + 1]
				new_coords.append(p1)
				for j in range(n_to_add):
					t = (j + 1) / (n_to_add + 1)
					mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
					if strength == 1:
						noise = np.random.normal(0, 0.01, 2)
						mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
					elif strength == 2:
						noise = np.random.normal(0, 0.05, 2)
						mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
					new_coords.append(mid)
			new_coords.append(coords[-1])
			return LineString(new_coords)
		elif isinstance(geom, Polygon):
			ext = list(geom.exterior.coords)
			if len(ext) < 4:
				return geom
			n_to_add = min(3, max(1, int((len(ext) - 1) * pct / 100)))
			new_ext = [ext[0]]
			for i in range(len(ext) - 1):
				p1, p2 = ext[i], ext[i + 1]
				new_ext.append(p1)
				for j in range(n_to_add):
					t = (j + 1) / (n_to_add + 1)
					mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
					if strength == 1:
						noise = np.random.normal(0, 0.01, 2)
						mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
					elif strength == 2:
						noise = np.random.normal(0, 0.05, 2)
						mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
					new_ext.append(mid)
			new_ext.append(ext[-1])
			holes = []
			for ring in geom.interiors:
				ring_coords = list(ring.coords)
				if len(ring_coords) >= 4:
					new_ring = [ring_coords[0]]
					for i in range(len(ring_coords) - 1):
						p1, p2 = ring_coords[i], ring_coords[i + 1]
						new_ring.append(p1)
						for j in range(n_to_add):
							t = (j + 1) / (n_to_add + 1)
							mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
							if strength == 1:
								noise = np.random.normal(0, 0.01, 2)
								mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
							elif strength == 2:
								noise = np.random.normal(0, 0.05, 2)
								mid = (mid[0] + float(noise[0]), mid[1] + float(noise[1]))
							new_ring.append(mid)
					new_ring.append(ring_coords[-1])
					holes.append(new_ring)
				else:
					holes.append(ring_coords)
			return Polygon(new_ext, holes=holes if holes else None)
		elif isinstance(geom, (MultiLineString, MultiPolygon)):
			geoms = []
			for g in geom.geoms:
				geoms.append(add_vertices_to_geom(g, strength, pct))
			if isinstance(geom, MultiLineString):
				return MultiLineString(geoms)
			else:
				return MultiPolygon(geoms)
	except Exception:
		return geom
	return geom


def generate_add_variant(src_path: Path, strength: int, pct: int) -> Path:
	base = src_path.stem
	subdir = (SCRIPT_DIR / 'attacked' / 'add' / 'Fig2_add_vertices' / base)
	subdir.mkdir(parents=True, exist_ok=True)
	gdf = gpd.read_file(str(src_path))
	attacked = gdf.copy()
	attacked['geometry'] = attacked['geometry'].apply(lambda geom: add_vertices_to_geom(geom, strength, pct))
	# CRS 保障：优先源数据；否则 .prj；最后 WGS84
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
	out_path = subdir / f'add_strength{strength}_{pct}pct_vertices.shp'
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
	print('=== Xi24 Fig2：增加顶点攻击 NC 评估 ===')
	if KEEP_EXISTING:
		csv_path_check = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
		if csv_path_check.exists():
			print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig2: {csv_path_check}')
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
	nc_tensor = np.zeros((len(USE_NAMES), len(STRENGTHS), len(ADD_RATIOS)))

	for file_idx, (vector_file, file_name) in enumerate(zip(USE_FILES, USE_NAMES)):
		abs_vector = str(vector_file)
		print(f'处理: {file_name}')
		# 已预过滤点数据
		for s_idx, strength in enumerate(STRENGTHS):
			for r_idx, add_ratio in enumerate(ADD_RATIOS):
				try:
					pct = int(round(add_ratio * 100))
					attacked_path = generate_add_variant(Path(abs_vector), strength, pct)
					nc_value = _xi24_extract_nc(attacked_path, SCRIPT_DIR / WATERMARK)
					rows.append({
						'file': file_name,
						'strength': strength,
						'add_ratio': add_ratio,
						'nc': nc_value,
					})
					nc_tensor[file_idx, s_idx, r_idx] = nc_value
					print(f'  s={strength}, r={add_ratio:.1f} -> NC={nc_value:.4f}')
				except Exception as exc:
					print('  失败:', file_name, strength, add_ratio, exc)
					nc_tensor[file_idx, s_idx, r_idx] = 0.0

	df = pd.DataFrame(rows, columns=['file', 'strength', 'add_ratio', 'nc'])

	# 层次结构输出
	hierarchical_rows: List[dict] = []
	for strength in STRENGTHS:
		hierarchical_rows.append({'增加顶点': f'强度: {strength}', 'VGAT': '', '类型': 'header'})
		for add_ratio in ADD_RATIOS:
			hierarchical_rows.append({'增加顶点': f'  比例 {add_ratio:.1f}', 'VGAT': '', '类型': 'subheader'})
			for file_name in FILE_NAMES:
				sub = df[(df['file'] == file_name) & (df['strength'] == strength) & (df['add_ratio'] == add_ratio)]
				nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
				hierarchical_rows.append({'增加顶点': f'    {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
			avg_nc = float(df[(df['strength'] == strength) & (df['add_ratio'] == add_ratio)]['nc'].mean()) if not df.empty else 0.0
			hierarchical_rows.append({'增加顶点': '    Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

	hierarchical_df = pd.DataFrame(hierarchical_rows)

	csv_path = DIR_RESULTS / 'fig2_add_vertices_nc.csv'
	xlsx_path = DIR_RESULTS / 'fig2_add_vertices_nc.xlsx'
	hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
	try:
		with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
			hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
	except Exception as e:
		print('Excel保存警告:', e)

	# 绘图
	plt.figure(figsize=(12, 8))
	for i, strength in enumerate(STRENGTHS):
		plt.subplot(2, 2, i + 1)
		for file_idx, file_name in enumerate(FILE_NAMES):
			plt.plot(ADD_RATIOS, nc_tensor[file_idx, i, :], '-o', label=file_name, markersize=4)
		avg_curve = np.mean(nc_tensor[:, i, :], axis=0)
		plt.plot(ADD_RATIOS, avg_curve, 'k-o', linewidth=2.5, label='平均', markersize=6)
		plt.grid(True, alpha=0.3)
		plt.xlabel('增加比例')
		plt.ylabel('NC')
		plt.title(f'强度 {strength}：增加顶点攻击的NC鲁棒性')
		plt.legend(loc='best', fontsize=6, ncol=2)
		plt.ylim(0, 1.1)

	plt.subplot(2, 2, 4)
	for i, strength in enumerate(STRENGTHS):
		avg_curve = np.mean(nc_tensor[:, i, :], axis=0)
		plt.plot(ADD_RATIOS, avg_curve, '-o', linewidth=2, label=f'强度 {strength}', markersize=6)
	plt.grid(True, alpha=0.3)
	plt.xlabel('增加比例')
	plt.ylabel('NC')
	plt.title('综合对比：不同强度的增加顶点攻击')
	plt.legend(loc='best')
	plt.ylim(0, 1.1)

	plt.tight_layout()
	fig_path = DIR_RESULTS / 'fig2_add_vertices_nc.png'
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close()

	print('结果表保存:', csv_path)
	print('Excel保存:', xlsx_path)
	print('曲线图保存:', fig_path)
	print('=== 完成 ===')


if __name__ == '__main__':
	run_evaluation()

