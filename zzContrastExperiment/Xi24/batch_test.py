# -*- coding: utf-8 -*-
"""
Xi24 Batch Test - 使用Xi24水印算法的批量测试
基于DWT-SVD-QIM的矢量数据水印嵌入和提取测试
"""

from __future__ import annotations

import os
import sys
import io
import glob
import warnings
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
from PIL import Image
import numpy as np

# 允许同目录导入
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
	sys.path.insert(0, BASE)
KEEP_EXISTING = os.environ.get('KEEP_EXISTING', '0') in ['1', 'true', 'True']

try:
	from . import watermark_embed  # type: ignore
	from . import watermark_extract  # type: ignore
except Exception:
	import watermark_embed  # type: ignore
	import watermark_extract  # type: ignore


def _ensure_dirs():
	Path(os.path.join(BASE, 'embed')).mkdir(parents=True, exist_ok=True)
	Path(os.path.join(BASE, 'extract', 'watermark')).mkdir(parents=True, exist_ok=True)


def _run_xi24_embed(src_shp: str, watermark_file: str) -> str:
	"""调用 Xi24 的嵌入函数，输出到 embed/Cat32_*.shp，返回输出路径。"""
	_ensure_dirs()
	stem = Path(src_shp).stem
	out_shp = str(Path(BASE) / 'embed' / f'Cat32_{stem}.shp')
	
	# 调用Xi24的嵌入函数
	_, _, _, _ = watermark_embed.embed_watermark(
		src_shp,
		watermark_file,
		out_shp,
		THRatio=0.3,
		mfactor=1e6,
		Q=10
	)
	return out_shp


def _run_xi24_extract_get_metrics(in_shp: str, watermark_file: str) -> Dict[str, float]:
	"""调用 Xi24 的提取函数并从输出图片计算 NC/BER。"""
	_ensure_dirs()
	stem = Path(in_shp).stem
	
	# 读取原始水印
	w_original = Image.open(watermark_file).convert('L')
	w_original = np.array(w_original)
	w_original = (w_original > 128).astype(int)
	
	# 提取水印
	w_extracted = watermark_extract.extract_watermark(
		in_shp,
		w_original.shape,
		THRatio=0.3,
		mfactor=1e6,
		Q=10
	)
	
	# 计算NC
	nc = watermark_extract.calculate_nc(w_extracted, w_original)
	
	# 计算BER
	ber = 1.0 - nc
	
	# 保存提取的水印图片
	out_img = str(Path(BASE) / 'extract' / 'watermark' / f'{stem}.png')
	extracted_img = Image.fromarray((w_extracted * 255).astype(np.uint8))
	extracted_img.save(out_img)
	
	return {
		'NC': float(nc),
		'BER': float(ber),
	}


def test_single_file(shp_file: str, watermark_file: str):
	print(f"\n{'='*60}")
	print(f"正在测试文件: {os.path.basename(shp_file)}")
	print(f"{'='*60}")
	try:
		original_gdf = gpd.read_file(shp_file)
		print("原始数据统计:")
		print(f"  - 总要素数: {len(original_gdf)}")
		print(f"  - 几何类型: {original_gdf.geom_type.value_counts().to_dict()}")
		print(f"  - 坐标系: {original_gdf.crs}")

		# 守卫：不支持 Point/MultiPoint
		try:
			geom_types = set(original_gdf.geom_type.astype(str).unique())
			if any('Point' in t for t in geom_types):
				print("⚠️ 含 Point/MultiPoint 类型，当前算法不支持，跳过")
				return {
					'file': os.path.basename(shp_file),
					'status': '跳过: 含点要素',
					'original_features': len(original_gdf),
					'original_geom_types': str(original_gdf.geom_type.value_counts().to_dict()),
					'embedded_features': 0,
					'extracted_features': 0,
					'nc_value': 0.0,
					'ber_value': 1.0,
					'feature_preserved': False,
				}
		except Exception:
			pass

		print("\n开始嵌入水印...")
		embedded_file = _run_xi24_embed(shp_file, watermark_file)
		print(f"✅ 水印嵌入完成: {embedded_file}")

		embedded_gdf = gpd.read_file(embedded_file)
		print("嵌入后数据统计:")
		print(f"  - 总要素数: {len(embedded_gdf)}")
		print(f"  - 几何类型: {embedded_gdf.geom_type.value_counts().to_dict()}")

		print("\n开始提取水印...")
		eva_factor = _run_xi24_extract_get_metrics(embedded_file, watermark_file)
		print("✅ 水印提取完成")

		print("\n算法性能评估:")
		print(f"  - NC值: {eva_factor['NC']:.6f}")
		print(f"  - BER值: {eva_factor['BER']:.6f}")

		feature_preserved = len(original_gdf) == len(embedded_gdf)
		print(f"  - 要素完整性: {'✅ 保持' if feature_preserved else '❌ 丢失'}")

		result = {
			'file': os.path.basename(shp_file),
			'original_features': len(original_gdf),
			'original_geom_types': str(original_gdf.geom_type.value_counts().to_dict()),
			'embedded_features': len(embedded_gdf),
			'extracted_features': len(embedded_gdf),
			'nc_value': eva_factor['NC'],
			'ber_value': eva_factor['BER'],
			'feature_preserved': feature_preserved,
			'status': '成功'
		}
		return result
	except Exception as e:
		print(f"❌ 测试失败: {str(e)}")
		import traceback
		traceback.print_exc()
		return {
			'file': os.path.basename(shp_file),
			'status': f'失败: {str(e)}',
			'original_features': 0,
			'original_geom_types': '',
			'embedded_features': 0,
			'extracted_features': 0,
			'nc_value': 0.0,
			'ber_value': 1.0,
			'feature_preserved': False,
		}


def main():
	# 强制使用UTF-8编码，避免控制台乱码
	try:
		sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
		sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
	except Exception:
		pass
	# 忽略 pyogrio 的环方向告警
	try:
		warnings.filterwarnings(
			"ignore",
			message=r".*contains polygon\(s\) with rings with invalid winding order.*",
			category=RuntimeWarning,
			module=r"pyogrio\.raw"
		)
	except Exception:
		pass

	print("开始批量测试pso_data文件夹中的所有矢量地图")
	print("测试水印算法的嵌入和提取性能（Xi24）")

	# 优先使用环境变量 PSO_DATA_DIR（便于集中管理），否则默认使用项目中的 zNC-Test/vector-data
	project_root = Path(__file__).resolve().parents[2]
	data_folder = os.environ.get('PSO_DATA_DIR', str(project_root / 'zNC-Test' / 'vector-data'))
	watermark_file = os.path.join(BASE, 'Cat32.png')

	# 如果设置了 KEEP_EXISTING 并且 embed/extract 目录已有输出，则跳过全部运行
	if KEEP_EXISTING:
		embed_dir = Path(BASE) / 'embed'
		extract_dir = Path(BASE) / 'extract' / 'watermark'
		if embed_dir.exists() and any(embed_dir.glob('*.shp')) and extract_dir.exists() and any(extract_dir.glob('*.png')):
			print('⚠️ KEEP_EXISTING=True，且 embed/extract 目录已有输出，跳过批量测试')
			return pd.DataFrame([])

	shp_files = glob.glob(os.path.join(data_folder, '*.shp'))
	print(f"\n发现 {len(shp_files)} 个矢量文件:")
	for i, shp_file in enumerate(shp_files, 1):
		print(f"  {i}. {os.path.basename(shp_file)}")

	results = []
	for shp_file in shp_files:
		results.append(test_single_file(shp_file, watermark_file))

	print(f"\n{'='*80}")
	print("批量测试汇总报告")
	print(f"{'='*80}")

	results_df = pd.DataFrame(results)
	print("详细结果:")
	print(results_df.to_string(index=False))

	success_count = len(results_df[results_df['status'] == '成功'])
	total_count = len(results_df)
	success_rate = (success_count / total_count * 100) if total_count else 0.0
	print(f"\n性能统计:")
	print(f"  - 测试文件总数: {total_count}")
	print(f"  - 成功测试数: {success_count}")
	print(f"  - 成功率: {success_rate:.1f}%")
	if success_count > 0:
		successful_results = results_df[results_df['status'] == '成功']
		avg_nc = successful_results['nc_value'].mean()
		avg_ber = successful_results['ber_value'].mean()
		total_features = successful_results['original_features'].sum()
		print(f"  - 平均NC值: {avg_nc:.6f}")
		print(f"  - 平均BER值: {avg_ber:.6f}")
		print(f"  - 总处理要素数: {total_features}")
		print(f"  - 要素完整性: {successful_results['feature_preserved'].all()}")

	print("\n测试完成! 嵌入输出到 embed/，提取图输出到 extract/watermark/")
	return results_df


if __name__ == '__main__':
	main()
