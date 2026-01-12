# -*- coding: utf-8 -*-
"""
Xi25/Fig4.py - 噪声攻击的NC评估（提取使用 Xi25；攻击与输出模仿 Wu25）
- 攻击：ratio 0.1/0.3/0.5/0.7/0.9 × strength 0.4/0.6/0.8
- 数据：`embed/Cat32_*.shp`
- 输出：`NC-Results/Fig4/` CSV、XLSX、PNG
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig4'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
import os
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]

STRENGTHS = [0.4, 0.6, 0.8]
RATIOS = [i / 100 for i in [10, 30, 50, 70, 90]]


def jitter_vertices_geom(geom, pct: int, strength: float):
	if geom is None:
		return geom
	import random
	try:
		if isinstance(geom, LineString):
			coords = list(geom.coords)
			n = len(coords)
			k = max(1, int(n * pct / 100))
			idx = list(range(n))
			chosen = set(random.sample(idx, min(k, len(idx))))
			new_coords = []
			for i, (x, y) in enumerate(coords):
				if i in chosen:
					new_coords.append((x + random.uniform(-strength, strength), y + random.uniform(-strength, strength)))
				else:
					new_coords.append((x, y))
			return LineString(new_coords)
		elif isinstance(geom, Polygon):
			ext = list(geom.exterior.coords)
			n = len(ext)
			k = max(1, int(n * pct / 100))
			idx = list(range(n))
			chosen = set(random.sample(idx, min(k, len(idx))))
			new_ext = []
			for i, (x, y) in enumerate(ext):
				if i in chosen:
					new_ext.append((x + random.uniform(-strength, strength), y + random.uniform(-strength, strength)))
				else:
					new_ext.append((x, y))
			holes = []
			for ring in geom.interiors:
				ring_coords = list(ring.coords)
				nr = len(ring_coords)
				kr = max(1, int(nr * pct / 100))
				idxr = list(range(nr))
				chosen_r = set(random.sample(idxr, min(kr, len(idxr))))
				new_ring = []
				for i, (x, y) in enumerate(ring_coords):
					if i in chosen_r:
						new_ring.append((x + random.uniform(-strength, strength), y + random.uniform(-strength, strength)))
					else:
						new_ring.append((x, y))
				holes.append(new_ring)
			return Polygon(new_ext, holes=holes if holes else None)
		elif isinstance(geom, (MultiLineString, MultiPolygon)):
			geoms = []
			for g in geom.geoms:
				geoms.append(jitter_vertices_geom(g, pct, strength))
			if isinstance(geom, MultiLineString):
				return MultiLineString(geoms)
			else:
				return MultiPolygon(geoms)
	except Exception:
		return geom
	return geom


def generate_noise_variant(src_path: Path, pct: int, strength: float) -> Path:
	base = src_path.stem
	subdir = (SCRIPT_DIR / 'attacked' / 'noise' / 'Fig4_noise' / base)
	subdir.mkdir(parents=True, exist_ok=True)
	gdf = gpd.read_file(str(src_path))
	attacked = gdf.copy()
	attacked['geometry'] = attacked['geometry'].apply(lambda geom: jitter_vertices_geom(geom, pct, strength))
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
	s_str = str(strength).replace('.', '_')
	out_path = subdir / f'noise_{pct}pct_strength_{s_str}.shp'
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
	print('=== Xi25 Fig4：噪声攻击 NC 评估 ===')
	if KEEP_EXISTING and any(DIR_RESULTS.glob('*.csv')):
		print(f'⚠️ KEEP_EXISTING=True，且 Fig4 结果已存在于 {DIR_RESULTS}，跳过')
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
	nc_tensor = np.zeros((len(USE_NAMES), len(STRENGTHS), len(RATIOS)))

	for file_idx, (vector_file, file_name) in enumerate(zip(USE_FILES, USE_NAMES)):
		abs_vector = str(vector_file)
		print(f'处理: {file_name}')
		# 已预过滤点数据
		for s_idx, strength in enumerate(STRENGTHS):
			for r_idx, ratio in enumerate(RATIOS):
				try:
					pct = int(round(ratio * 100))
					attacked_path = generate_noise_variant(Path(abs_vector), pct, float(strength))
					nc_value = _xi25_extract_nc(attacked_path, SCRIPT_DIR / WATERMARK)
					rows.append({
						'file': file_name,
						'strength': strength,
						'ratio': ratio,
						'nc': nc_value,
					})
					nc_tensor[file_idx, s_idx, r_idx] = nc_value
					print(f'  s={strength:.1f}, r={ratio:.1f} -> NC={nc_value:.4f}')
				except Exception as exc:
					print('  失败:', file_name, strength, ratio, exc)
					nc_tensor[file_idx, s_idx, r_idx] = 0.0

	df = pd.DataFrame(rows, columns=['file', 'strength', 'ratio', 'nc'])

	# 层次结构输出
	hierarchical_rows: List[dict] = []
	for strength in STRENGTHS:
		hierarchical_rows.append({'噪声': f'强度: {strength}', 'VGAT': '', '类型': 'header'})
		for ratio in RATIOS:
			hierarchical_rows.append({'噪声': f'  比例 {ratio:.1f}', 'VGAT': '', '类型': 'subheader'})
			for file_name in FILE_NAMES:
				sub = df[(df['file'] == file_name) & (df['strength'] == strength) & (df['ratio'] == ratio)]
				nc_value = float(sub['nc'].iloc[0]) if len(sub) > 0 else 0.0
				hierarchical_rows.append({'噪声': f'    {file_name}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
			avg_nc = float(df[(df['strength'] == strength) & (df['ratio'] == ratio)]['nc'].mean()) if not df.empty else 0.0
			hierarchical_rows.append({'噪声': '    Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

	hierarchical_df = pd.DataFrame(hierarchical_rows)

	csv_path = DIR_RESULTS / 'fig4_noise_nc.csv'
	xlsx_path = DIR_RESULTS / 'fig4_noise_nc.xlsx'
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
			plt.plot(RATIOS, nc_tensor[file_idx, i, :], '-o', label=file_name, markersize=4)
		avg_curve = np.mean(nc_tensor[:, i, :], axis=0)
		plt.plot(RATIOS, avg_curve, 'k-o', linewidth=2.5, label='平均', markersize=6)
		plt.grid(True, alpha=0.3)
		plt.xlabel('扰动顶点比例（%）')
		plt.ylabel('NC')
		plt.title(f'强度 {strength}：噪声攻击的NC鲁棒性')
		plt.legend(loc='best', fontsize=6, ncol=2)
		plt.ylim(0, 1.1)

	plt.subplot(2, 2, 4)
	for i, strength in enumerate(STRENGTHS):
		avg_curve = np.mean(nc_tensor[:, i, :], axis=0)
		plt.plot(RATIOS, avg_curve, '-o', linewidth=2, label=f'强度 {strength}', markersize=6)
	plt.grid(True, alpha=0.3)
	plt.xlabel('扰动顶点比例（%）')
	plt.ylabel('NC')
	plt.title('综合对比：不同强度的噪声攻击')
	plt.legend(loc='best')
	plt.ylim(0, 1.1)

	plt.tight_layout()
	fig_path = DIR_RESULTS / 'fig4_noise_nc.png'
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close()

	print('结果表保存:', csv_path)
	print('Excel保存:', xlsx_path)
	print('曲线图保存:', fig_path)
	print('=== 完成 ===')


if __name__ == '__main__':
	run_evaluation()

