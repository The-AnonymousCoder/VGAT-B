# -*- coding: utf-8 -*-
"""
Xi24/Fig12.py - 复合攻击（一次性按 Fig1→Fig10 顺序依次执行）的NC评估
- 步骤与 Tan24(Fig12) 一致，但提取采用 Xi24 提取器
- 数据：`embed/` 下 `Cat32_*.shp`
- 输出：`NC-Results/Fig12/` CSV、XLSX、PNG
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend for servers/CI
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import geopandas as gpd
from shapely.geometry import LineString, Polygon  # type: ignore
from shapely.affinity import translate as shp_translate, scale as shp_scale, rotate as shp_rotate  # type: ignore
from pyproj import CRS as _CRS  # type: ignore

import sys
	from . import watermark_extract  # type: ignore
except Exception:
	BASE = os.path.dirname(os.path.abspath(__file__))
	if BASE not in sys.path:
		sys.path.insert(0, BASE)
	import watermark_extract  # type: ignore


# 字体
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
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig12'
DIR_ATTACKED = SCRIPT_DIR / 'attacked' / 'compound_seq' / 'Fig12_compound_seq'
DIR_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATTACKED.mkdir(parents=True, exist_ok=True)
import os
KEEP_EXISTING = True

WATERMARK = 'Cat32.png'
EMBED_DIR = SCRIPT_DIR / 'embed'
VECTOR_FILES = sorted([p for p in EMBED_DIR.glob('Cat32_*.shp')])
FILE_NAMES = [p.stem.replace('Cat32_', '') for p in VECTOR_FILES]


def delete_vertices_from_geom(geom, pct):
	import random
		if geom.geom_type == 'LineString':
			coords = list(geom.coords)
			if len(coords) <= 2:
				return geom
			n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
			if n_to_delete >= len(coords) - 2:
				return geom
			indices = list(range(1, len(coords) - 1))
			to_delete = set(random.sample(indices, min(n_to_delete, len(indices))))
			new_coords = [coords[0]] + [coords[i] for i in range(1, len(coords) - 1) if i not in to_delete] + [coords[-1]]
			return LineString(new_coords)
		elif geom.geom_type == 'Polygon':
			ext = list(geom.exterior.coords)
			if len(ext) <= 4:
				return geom
			n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
			if n_to_delete >= len(ext) - 4:
				return geom
			indices = list(range(1, len(ext) - 2))
			to_delete = set(random.sample(indices, min(n_to_delete, len(indices))))
			new_ext = [ext[0]] + [ext[i] for i in range(1, len(ext) - 2) if i not in to_delete] + [ext[-2], ext[-1]]
			holes = []
			for ring in geom.interiors:
				rc = list(ring.coords)
				if len(rc) > 4:
					n_h = max(1, int((len(rc) - 4) * pct / 100))
					if n_h < len(rc) - 4:
						idx_h = list(range(1, len(rc) - 2))
						del_h = set(random.sample(idx_h, min(n_h, len(idx_h))))
						holes.append([rc[0]] + [rc[i] for i in range(1, len(rc) - 2) if i not in del_h] + [rc[-2], rc[-1]])
					else:
						holes.append(rc)
				else:
					holes.append(rc)
			return Polygon(new_ext, holes=holes if holes else None)
	except Exception:
		pass
	return geom


def add_vertices_to_geom(geom, pct, strength_level=1):
	from numpy.random import normal
	noise_sigma = 0.01 if strength_level == 1 else 0.0
		if geom.geom_type == 'LineString':
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
					if noise_sigma > 0:
						mid = (mid[0] + float(normal(0, noise_sigma)), mid[1] + float(normal(0, noise_sigma)))
					new_coords.append(mid)
			new_coords.append(coords[-1])
			return LineString(new_coords)
		elif geom.geom_type == 'Polygon':
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
					if noise_sigma > 0:
						mid = (mid[0] + float(normal(0, noise_sigma)), mid[1] + float(normal(0, noise_sigma)))
					new_ext.append(mid)
			new_ext.append(ext[-1])
			holes = []
			for ring in geom.interiors:
				rc = list(ring.coords)
				if len(rc) >= 4:
					new_rc = [rc[0]]
					for i in range(len(rc) - 1):
						p1, p2 = rc[i], rc[i + 1]
						new_rc.append(p1)
						for j in range(n_to_add):
							t = (j + 1) / (n_to_add + 1)
							mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
							if noise_sigma > 0:
								mid = (mid[0] + float(normal(0, noise_sigma)), mid[1] + float(normal(0, noise_sigma)))
							new_rc.append(mid)
					new_rc.append(rc[-1])
					holes.append(new_rc)
				else:
					holes.append(rc)
			return Polygon(new_ext, holes=holes if holes else None)
	except Exception:
		pass
	return geom


def jitter_vertices(geom, pct, strength):
	import random
		if geom.geom_type == 'LineString':
			coords = list(geom.coords)
			n = len(coords)
			k = max(1, int(n * pct / 100))
			idx = list(range(n))
			chosen = set(random.sample(idx, min(k, len(idx))))
			new_coords = []
			for i, c in enumerate(coords):
				if i in chosen:
					new_coords.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
				else:
					new_coords.append(c)
			return LineString(new_coords)
		elif geom.geom_type == 'Polygon':
			ext = list(geom.exterior.coords)
			n = len(ext)
			k = max(1, int(n * pct / 100))
			idx = list(range(n))
			chosen = set(random.sample(idx, min(k, len(idx))))
			new_ext = []
			for i, c in enumerate(ext):
				if i in chosen:
					new_ext.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
				else:
					new_ext.append(c)
			holes = []
			for ring in geom.interiors:
				rc = list(ring.coords)
				n2 = len(rc); k2 = max(1, int(n2 * pct / 100))
				idx2 = list(range(n2)); chosen2 = set(random.sample(idx2, min(k2, len(idx2))))
				new_rc = []
				for i, c in enumerate(rc):
					if i in chosen2:
						new_rc.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
					else:
						new_rc.append(c)
				holes.append(new_rc)
			return Polygon(new_ext, holes=holes if holes else None)
	except Exception:
		pass
	return geom


def reverse_vertices_geom(geom):
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


def _ensure_crs(attacked: gpd.GeoDataFrame, src_path: Path, src_gdf: gpd.GeoDataFrame) -> None:
		if getattr(src_gdf, 'crs', None) is not None:
			attacked.set_crs(src_gdf.crs, allow_override=True, inplace=True)  # type: ignore
			return
		prj_path = src_path.with_suffix('.prj')
		if prj_path.exists():
				crs_obj = _CRS.from_wkt(prj_path.read_text(encoding='utf-8', errors='ignore'))
				attacked.set_crs(crs_obj, allow_override=True, inplace=True)  # type: ignore
				return
			except Exception:
				pass
		attacked.set_crs('EPSG:4326', allow_override=True, inplace=True)  # type: ignore
	except Exception:
		pass


def generate_compound_seq(src_path: Path) -> Path:
	import random
	import gc

	base = src_path.stem
	subdir = DIR_ATTACKED / base
	subdir.mkdir(parents=True, exist_ok=True)

		gdf = gpd.read_file(str(src_path))
		original_gdf = gpd.read_file(str(src_path))  # 用于 CRS
	except Exception as e:
		print(f"  ❌ 读取源文件失败 {base}: {e}")
		# 如果读取失败，返回一个不存在的路径，让后续处理失败但不崩溃
		return subdir / 'failed.shp'

	# 1 顶点删除10%
	gdf['geometry'] = gdf['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, 10))
	gc.collect()

	# 2 顶点增加 强度1 比例50%
	gdf['geometry'] = gdf['geometry'].apply(lambda geom: add_vertices_to_geom(geom, 50, strength_level=1))
	gc.collect()

	# 3 对象删除50%
	n_total = len(gdf)
	n_del = int(n_total * 0.5)
	if n_del > 0 and n_total > 0:
		idx = random.sample(range(n_total), min(n_del, n_total))
		gdf = gdf.drop(idx).reset_index(drop=True)
	gc.collect()

	# 4 噪声扰动 强度0.8 比例50%
	gdf['geometry'] = gdf['geometry'].apply(lambda geom: jitter_vertices(geom, 50, 0.8))
	gc.collect()

	# 5 沿Y轴中心裁剪50%
	bounds = gdf.total_bounds
	mid_y = (bounds[1] + bounds[3]) / 2
	bdf = gdf.geometry.bounds
	gdf = gdf[bdf['miny'] < mid_y].reset_index(drop=True)
	del bdf
	gc.collect()
	
	# 6 平移 X=20, Y=40
	gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_translate(geom, 20, 40))

	# ✅ 使用全局中心作为变换原点，与zNC-Test/Fig12策略一致
	global_center = ((bounds[0] + bounds[2]) / 2, mid_y)
	del bounds, mid_y
	gc.collect()

	# 7 缩放90%
	gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 0.9, 0.9, origin=global_center))
	# 8 旋转180°
	gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_rotate(geom, 180, origin=global_center))
	# 9 Y轴镜像翻转
	gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 1.0, -1.0, origin=global_center))
	gc.collect()
	
	# 10 反转顶点顺序 → 反转对象顺序
	gdf['geometry'] = gdf['geometry'].apply(reverse_vertices_geom)
	gdf = gdf.iloc[::-1].reset_index(drop=True)

		_ensure_crs(gdf, src_path, original_gdf)
		out_path = subdir / 'compound_seq_all.shp'
		if KEEP_EXISTING and out_path.exists():
			print(f'  ⏭️  跳过已存在攻击文件: {out_path.name}')
		else:
			gdf.to_file(str(out_path), driver='ESRI Shapefile')
	except Exception as save_error:
		print(f"  ❌ 保存复合攻击文件失败 {base}: {save_error}")
		# 返回失败路径
		return subdir / 'failed.shp'
	
	# 清理内存
	del gdf, original_gdf
	gc.collect()
	
	return out_path


def _xi24_extract_nc(attacked_path: Path, watermark_path: Path, file_name: str) -> float:
	"""通过 Xi24 提取主程序计算 NC 值。"""
	import gc
	from PIL import Image

	# 检查数据集大小，对于大数据集使用不同的参数
	gdf_check = gpd.read_file(str(attacked_path))
	total_vertices = sum(len(geom.coords) if hasattr(geom, 'coords') else
						(len(geom.exterior.coords) + sum(len(interior.coords) for interior in geom.interiors))
						if hasattr(geom, 'exterior') else 0
						for geom in gdf_check.geometry if geom is not None)
	del gdf_check
	gc.collect()

	# 读取原始水印
	w_original = Image.open(str(watermark_path)).convert('L')
	w_original = np.array(w_original)
	w_original = (w_original > 128).astype(int)

	# 根据数据集大小调整参数
	if total_vertices > 50000:  # 对于大数据集
		print(f"  📊 大数据集检测 ({total_vertices} 顶点)，使用简化参数")
		extract_params = {
			'THRatio': 0.5,  # 更高的阈值，减少特征点
			'mfactor': 1e5,  # 降低倍数因子
			'Q': 20          # 更大的量化步长
		}
	else:  # 标准参数
		extract_params = {
			'THRatio': 0.3,
			'mfactor': 1e6,
			'Q': 10
		}

	# 直接尝试提取，不进行复杂的几何修复
	attacked_path_for_extract = attacked_path

			print(f"  ❌ 文件读取失败 {file_name}: {read_error}")
			# 尝试使用更宽容的读取方式
				import fiona
				with fiona.open(str(attacked_path)) as src:
					features = list(src)
				# 如果能读取fiona，尝试重建GeoDataFrame
				if features:
					gdf_attacked = gpd.GeoDataFrame.from_features(features)
					print(f"  ⚠️ 使用fiona重新读取成功，{len(gdf_attacked)} 个几何对象")
				else:
					print(f"  ❌ fiona也无法读取 {file_name}")
					return 0.0
			except Exception as fiona_error:
				print(f"  ❌ fiona读取也失败 {file_name}: {fiona_error}")
				return 0.0

		valid_geoms = []

		for i, geom in enumerate(gdf_attacked.geometry):
				if geom is not None and not geom.is_empty:
					# 多种几何修复策略
					fixed_geom = None

					# 策略1：直接验证几何是否可以访问坐标
						if hasattr(geom, 'coords'):
							list(geom.coords)
							fixed_geom = geom
						elif hasattr(geom, 'exterior'):
							list(geom.exterior.coords)
							fixed_geom = geom
					except:
						pass

					# 策略2：如果上述失败，尝试修复几何
					if fixed_geom is None:
							if not geom.is_valid:
								# 多种缓冲修复策略
								for buffer_dist in [0, 0.001, -0.001, 0.01, -0.01]:
										fixed_geom = geom.buffer(buffer_dist)
										if fixed_geom is not None and not fixed_geom.is_empty and fixed_geom.is_valid:
											fixed_geom = fixed_geom.simplify(0.001)  # 轻微简化
											break
									except:
										continue

								# 如果缓冲修复失败，尝试convex_hull
								if fixed_geom is None or not fixed_geom.is_valid:
										fixed_geom = geom.convex_hull
									except:
										pass
							else:
								fixed_geom = geom
						except:
							pass

					# 策略3：处理Multi-part geometries
					if fixed_geom is None and hasattr(geom, 'geoms'):
							valid_parts = []
							for part in geom.geoms:
									if hasattr(part, 'coords'):
										list(part.coords)
										valid_parts.append(part)
									elif hasattr(part, 'exterior'):
										list(part.exterior.coords)
										valid_parts.append(part)
								except:
									continue

							if valid_parts:
								if len(valid_parts) == 1:
									fixed_geom = valid_parts[0]
								else:
									from shapely.geometry import MultiPolygon, GeometryCollection
										fixed_geom = MultiPolygon(valid_parts)
									except:
										fixed_geom = GeometryCollection(valid_parts)
						except:
							pass

					# 策略4：最后的尝试 - 创建简化的边界框
					if fixed_geom is None:
							# 使用几何的边界框创建简单的矩形
							minx, miny, maxx, maxy = geom.bounds
							if minx is not None and miny is not None and maxx is not None and maxy is not None:
								from shapely.geometry import Polygon
								fixed_geom = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
						except:
							pass

					# 如果修复成功，添加到有效几何列表
					if fixed_geom is not None and not fixed_geom.is_empty:
						valid_geoms.append(fixed_geom)

			except Exception:
				continue  # 跳过无法处理的几何

		if len(valid_geoms) == 0:
			print(f"  ⚠️ 无有效几何 {file_name}，尝试直接提取")
			# 如果没有有效几何，尝试直接用原始复合攻击文件提取
			attacked_path_for_extract = attacked_path
		elif len(valid_geoms) < len(gdf_attacked):
			print(f"  ⚠️ 几何修复: {len(gdf_attacked)} → {len(valid_geoms)} 个有效几何")
			# 创建修复后的临时文件
			import tempfile
			import os
			gdf_fixed = gpd.GeoDataFrame(geometry=valid_geoms, crs=gdf_attacked.crs)
			with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as tmp:
				tmp_path = tmp.name
				gdf_fixed.to_file(tmp_path.replace('.shp', ''), driver='ESRI Shapefile')
				attacked_path_for_extract = tmp_path.replace('.shp', '.shp')
			except Exception as save_error:
				print(f"  ❌ 保存临时文件失败 {file_name}: {save_error}")
				# 如果保存失败，尝试直接提取
				attacked_path_for_extract = attacked_path
		else:
			attacked_path_for_extract = attacked_path

			# 提取水印
			w_extracted = watermark_extract.extract_watermark(
				str(attacked_path_for_extract),
				w_original.shape,
				**extract_params
			)
		except MemoryError:
			print(f"  ❌ 内存不足，跳过大数据集 {file_name}")
			w_extracted = None
		except Exception as e:
			print(f"  ❌ 标准提取失败 {file_name}: {e}")
			# 尝试备用提取方法：使用更宽松的参数
				print(f"  🔄 尝试备用提取方法...")
				backup_params = {
					'THRatio': 0.8,  # 更宽松的阈值
					'mfactor': 1e4,  # 更小的倍数因子
					'Q': 50          # 更大的量化步长
				}
				w_extracted = watermark_extract.extract_watermark(
					str(attacked_path_for_extract),
					w_original.shape,
					**backup_params
				)
				print(f"  ✅ 备用提取成功")
			except Exception as e2:
				print(f"  ❌ 备用提取也失败 {file_name}: {e2}")
				w_extracted = None
		finally:
			# 清理临时文件
			if 'tmp_path' in locals():
					base = tmp_path.replace('.shp', '')
					for ext in ['.shp', '.dbf', '.shx', '.prj']:
						path = base + ext
						if os.path.exists(path):
							os.unlink(path)
				except:
					pass

		if w_extracted is None:
			return 0.0

	except Exception as e:
		print(f"  ❌ 几何处理失败 {file_name}: {e}")
		return 0.0
	
	# 计算NC
	nc = watermark_extract.calculate_nc(w_extracted, w_original)
	
	# 保存提取的水印图片（使用 file_name 区分不同矢量地图）
	out_dir = SCRIPT_DIR / 'extract' / 'watermark' / 'Fig12'
	out_dir.mkdir(parents=True, exist_ok=True)
	out_img = out_dir / f'Cat32_{file_name}_extracted.png'
	extracted_img = Image.fromarray((w_extracted * 255).astype(np.uint8))
	extracted_img.save(str(out_img))
	
	# 清理内存
	del w_original, w_extracted, extracted_img
	gc.collect()
	
	return float(nc)


def run_evaluation():
	print('=== Xi24 Fig12：复合(顺序) 攻击 NC 评估 ===')
	if KEEP_EXISTING:
		csv_path_check = DIR_RESULTS / 'fig12_compound_seq_nc.csv'
		if csv_path_check.exists():
			print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig12: {csv_path_check}')
			return
		if csv_path_check.exists():
			print(f'⚠️ KEEP_EXISTING=True，且结果已存在，跳过 Fig12: {csv_path_check}')
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

	# 记录每文件的 NC
	vals: List[float] = []
	rows: List[Dict] = []

	# 批量处理：逐个处理矢量地图以避免内存问题
	import gc
	for idx, (vector_file, file_name) in enumerate(zip(USE_FILES, USE_NAMES)):
		print(f'[{idx+1}/{len(USE_FILES)}] 处理: {file_name}')
			# 生成攻击数据
			out_path = generate_compound_seq(Path(str(vector_file)))

			# 检查攻击文件是否生成成功
			if 'failed' in str(out_path):
				print(f'  ❌ 复合攻击生成失败 {file_name}')
				nc_value = 0.0  # 无法生成攻击文件，返回0
			else:
				# 提取水印并计算 NC
				nc_value = _xi24_extract_nc(out_path, SCRIPT_DIR / WATERMARK, file_name)

			vals.append(nc_value)
			rows.append({'file': file_name, 'nc': nc_value})
			if nc_value > 0.01:
				print(f'  ✓ NC={nc_value:.4f}')
			else:
				print(f'  ⚠️ NC={nc_value:.4f} (复合攻击后提取困难)')

			# 显式释放内存
			del out_path
			gc.collect()

		except Exception as exc:
			print(f'  ✗ 失败: {exc}')
			# 提取失败，返回0
			nc_value = 0.0
			vals.append(nc_value)
			rows.append({'file': file_name, 'nc': nc_value})
			gc.collect()

	df = pd.DataFrame(rows, columns=['file', 'nc']).sort_values('file')

	# 层次表：单一分组
	hierarchical_rows: List[dict] = []
	hierarchical_rows.append({'复合攻击(顺序)': '复合(Fig1→Fig10)', 'VGAT': '', '类型': 'header'})
	for _, row in df.iterrows():
		hierarchical_rows.append({'复合攻击(顺序)': f"  {row['file']}", 'VGAT': f"{row['nc']:.6f}", '类型': 'data'})
	hierarchical_rows.append({'复合攻击(顺序)': '  Average', 'VGAT': f"{df['nc'].mean():.6f}", '类型': 'average'})

	hierarchical_df = pd.DataFrame(hierarchical_rows)

	# 保存 CSV/XLSX
	csv_path = DIR_RESULTS / 'fig12_compound_seq_nc.csv'
	xlsx_path = DIR_RESULTS / 'fig12_compound_seq_nc.xlsx'
	hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
		with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
			hierarchical_df.to_excel(writer, sheet_name='NC结果', index=False)
	except Exception as e:
		print('Excel保存警告:', e)

	# 柱状图
	plt.figure(figsize=(10, 6))
	x = np.arange(len(df))
	plt.bar(x, list(df['nc']), alpha=0.8)
	plt.axhline(df['nc'].mean(), color='k', linestyle='--', label='平均')
	plt.xticks(x, list(df['file']), rotation=20)
	plt.ylabel('NC')
	plt.title('复合攻击(一次性顺序 Fig1→Fig10) 的NC鲁棒性（Xi24/Fig12）')
	plt.legend()
	fig_path = DIR_RESULTS / 'fig12_compound_seq_nc.png'
	plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()

	print('结果表保存:', csv_path)
	print('Excel保存:', xlsx_path)
	print('柱状图保存:', fig_path)
	print('=== 完成 ===')


if __name__ == '__main__':
	run_evaluation()




