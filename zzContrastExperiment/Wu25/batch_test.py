# -*- coding: utf-8 -*-
"""
批量测试脚本 - 测试所有pso_data中的矢量地图
"""
import os
import sys
import io
import glob
import geopandas as gpd
import warnings
import pandas as pd
from embed import embed
from extract import extract
from pathlib import Path
KEEP_EXISTING = os.environ.get('KEEP_EXISTING', '0') in ['1', 'true', 'True']

def test_single_file(shp_file, watermark_file):
    """测试单个矢量文件的嵌入和提取过程"""
    print(f"\n{'='*60}")
    print(f"正在测试文件: {os.path.basename(shp_file)}")
    print(f"{'='*60}")
    
    try:
        # 读取原始数据信息
        original_gdf = gpd.read_file(shp_file)
        print(f"原始数据统计:")
        print(f"  - 总要素数: {len(original_gdf)}")
        print(f"  - 几何类型: {original_gdf.geom_type.value_counts().to_dict()}")
        print(f"  - 坐标系: {original_gdf.crs}")
        
        # 嵌入水印
        print(f"\n开始嵌入水印...")
        embedded_file = embed(shp_file, watermark_file)
        print(f"✅ 水印嵌入完成: {embedded_file}")
        
        # 检查嵌入后的数据
        embedded_gdf = gpd.read_file(embedded_file)
        print(f"嵌入后数据统计:")
        print(f"  - 总要素数: {len(embedded_gdf)}")
        print(f"  - 几何类型: {embedded_gdf.geom_type.value_counts().to_dict()}")
        
        # 提取水印
        print(f"\n开始提取水印...")
        extracted_file, eva_factor = extract(embedded_file, watermark_file)
        print(f"✅ 水印提取完成: {extracted_file}")
        
        # 检查提取后的数据
        extracted_gdf = gpd.read_file(extracted_file)
        print(f"提取后数据统计:")
        print(f"  - 总要素数: {len(extracted_gdf)}")
        print(f"  - 几何类型: {extracted_gdf.geom_type.value_counts().to_dict()}")
        
        # 算法性能评估
        print(f"\n算法性能评估:")
        print(f"  - NC值: {eva_factor['NC']:.6f}")
        print(f"  - BER值: {eva_factor['BER']:.6f}")
        
        # 数据完整性检查
        feature_preserved = len(original_gdf) == len(extracted_gdf)
        print(f"  - 要素完整性: {'✅ 保持' if feature_preserved else '❌ 丢失'}")
        
        result = {
            'file': os.path.basename(shp_file),
            'original_features': len(original_gdf),
            'original_geom_types': str(original_gdf.geom_type.value_counts().to_dict()),
            'embedded_features': len(embedded_gdf),
            'extracted_features': len(extracted_gdf),
            'nc_value': eva_factor['NC'],
            'ber_value': eva_factor['BER'],
            'feature_preserved': feature_preserved,
            'status': '成功'
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        result = {
            'file': os.path.basename(shp_file),
            'status': f'失败: {str(e)}',
            'original_features': 0,
            'original_geom_types': '',
            'embedded_features': 0,
            'extracted_features': 0,
            'nc_value': 0,
            'ber_value': 1,
            'feature_preserved': False
        }
        return result

def main():
    # 强制使用UTF-8编码，避免控制台乱码
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
    """主测试函数"""
    # 忽略pyogrio读取时关于多边形环方向的提示（输入数据会被自动校正，不影响后续处理）
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
    print("测试水印算法的嵌入和提取性能")
    
    # 设置文件路径：优先使用环境变量 PSO_DATA_DIR，否则默认项目内 zNC-Test/vector-data
    project_root = Path(__file__).resolve().parents[2]
    data_folder = os.environ.get('PSO_DATA_DIR', str(project_root / 'zNC-Test' / 'vector-data'))
    watermark_file = os.path.join(os.path.dirname(__file__), "Cat32.png")
    # 如果设置了 KEEP_EXISTING 并且 embed/extract 已有输出，则跳过全部运行
    if KEEP_EXISTING:
        embed_dir = Path(__file__).resolve().parent / 'embed'
        extract_dir = Path(__file__).resolve().parent / 'extract'
        if embed_dir.exists() and any(embed_dir.glob('*.shp')) and extract_dir.exists() and any(extract_dir.glob('*.png')):
            print('⚠️ KEEP_EXISTING=True，且 embed/extract 目录已有输出，跳过批量测试')
            return pd.DataFrame([])
    
    # 获取所有shp文件
    shp_files = glob.glob(os.path.join(data_folder, "*.shp"))
    print(f"\n发现 {len(shp_files)} 个矢量文件:")
    for i, shp_file in enumerate(shp_files, 1):
        print(f"  {i}. {os.path.basename(shp_file)}")
    
    # 测试每个文件
    results = []
    for shp_file in shp_files:
        result = test_single_file(shp_file, watermark_file)
        results.append(result)
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("批量测试汇总报告")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    
    print("详细结果:")
    print(results_df.to_string(index=False))
    
    # 统计成功率
    success_count = len(results_df[results_df['status'] == '成功'])
    total_count = len(results_df)
    success_rate = success_count / total_count * 100
    
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
    
    print(f"\n测试完成! 结果已保存到embed/和extract/文件夹中")
    
    return results_df

if __name__ == "__main__":
    results = main()
