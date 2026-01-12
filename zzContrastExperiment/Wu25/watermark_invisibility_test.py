# -*- coding: utf-8 -*-
"""
水印嵌入前后的不可见性测试
对比原始文件与嵌入水印后文件的几何差异
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon


def calculate_geometric_error(original_file, watermarked_file):
    """
    计算原始文件和水印文件之间的几何误差
    模拟MATLAB中的SuperError函数
    
    参数:
        original_file: str - 原始shapefile路径
        watermarked_file: str - 嵌入水印后的shapefile路径
    
    返回:
        dict - 包含各种误差指标的字典
    """
    try:
        # 读取shapefile
        original_gdf = gpd.read_file(original_file)
        watermarked_gdf = gpd.read_file(watermarked_file)
        
        print(f"  原始文件要素数: {len(original_gdf)}")
        print(f"  水印文件要素数: {len(watermarked_gdf)}")
        
        # 初始化误差计算变量
        rmse_sum = 0
        n_total = 0
        max_error_list = []
        mean_error_sum = 0
        
        # 确保两个文件有相同数量的要素
        min_features = min(len(original_gdf), len(watermarked_gdf))
        
        for i in range(min_features):
            # 获取第i个要素的几何
            orig_geom = original_gdf.iloc[i].geometry
            water_geom = watermarked_gdf.iloc[i].geometry
            
            if orig_geom is None or water_geom is None:
                continue
            if orig_geom.is_empty or water_geom.is_empty:
                continue
            
            # 提取坐标数组
            orig_coords = extract_all_coordinates(orig_geom)
            water_coords = extract_all_coordinates(water_geom)
            
            if len(orig_coords) == 0 or len(water_coords) == 0:
                continue
            
            # 确保坐标数量一致（取最小值）
            min_coords = min(len(orig_coords), len(water_coords))
            if min_coords == 0:
                continue
            
            orig_coords = orig_coords[:min_coords]
            water_coords = water_coords[:min_coords]
            
            # 转换为numpy数组
            orig_array = np.array(orig_coords)
            water_array = np.array(water_coords)
            
            # 计算坐标差值
            dx = orig_array[:, 0] - water_array[:, 0]
            dy = orig_array[:, 1] - water_array[:, 1]
            
            # 计算距离误差的平方
            distance_squared = dx**2 + dy**2
            distance_error = np.sqrt(distance_squared)
            
            if len(distance_error) > 0:
                # 记录最大误差
                max_error_single = np.max(distance_error)
                max_error_list.append(max_error_single)
                
                # 累计计算
                rmse_sum += np.sum(distance_squared)
                mean_error_sum += np.sum(distance_error)
                n_total += len(distance_error)
        
        # 计算最终误差指标
        if n_total > 0 and len(max_error_list) > 0:
            max_error = np.max(max_error_list)
            mean_error = mean_error_sum / n_total
            mse = rmse_sum / n_total
            rmse = np.sqrt(rmse_sum / n_total)
        else:
            max_error = mean_error = mse = rmse = 0
        
        return {
            'max_error': max_error,
            'mean_error': mean_error,
            'mse': mse,
            'rmse': rmse,
            'total_points': n_total,
            'total_features': min_features
        }
        
    except Exception as e:
        print(f"  ❌ 计算误差失败: {str(e)}")
        return {
            'max_error': -1,
            'mean_error': -1,
            'mse': -1,
            'rmse': -1,
            'total_points': 0,
            'total_features': 0
        }


def extract_all_coordinates(geometry):
    """
    从geometry中提取所有坐标点
    """
    coords = []
    
    if geometry.geom_type == 'Point':
        coords.append((geometry.x, geometry.y))
        
    elif geometry.geom_type == 'LineString':
        coords.extend(list(geometry.coords))
        
    elif geometry.geom_type == 'Polygon':
        # 外环
        coords.extend(list(geometry.exterior.coords))
        # 内环
        for interior in geometry.interiors:
            coords.extend(list(interior.coords))
            
    elif geometry.geom_type == 'MultiPoint':
        for point in geometry.geoms:
            coords.append((point.x, point.y))
            
    elif geometry.geom_type == 'MultiLineString':
        for line in geometry.geoms:
            coords.extend(list(line.coords))
            
    elif geometry.geom_type == 'MultiPolygon':
        for polygon in geometry.geoms:
            coords.extend(extract_all_coordinates(polygon))
    
    return coords


def test_watermark_invisibility():
    """
    测试水印嵌入的不可见性
    """
    print("开始水印嵌入不可见性测试")
    print("=" * 60)
    
    # 定义文件对：原始文件(pso_data) vs 水印文件(embed)
    file_pairs = [
        {
            'name': 'Boundary',
            'original': 'pso_data/Boundary.shp',      # 原始文件
            'watermarked': 'embed/M_Boundary.shp'     # 嵌入水印后的文件
        },
        {
            'name': 'Road',
            'original': 'pso_data/Road.shp',
            'watermarked': 'embed/M_Road.shp'
        },
        {
            'name': 'Landuse', 
            'original': 'pso_data/Landuse.shp',
            'watermarked': 'embed/M_Landuse.shp'
        },
        {
            'name': 'Railways',
            'original': 'pso_data/Railways.shp',
            'watermarked': 'embed/M_Railways.shp'
        },
        {
            'name': 'Building',
            'original': 'pso_data/Building.shp',
            'watermarked': 'embed/M_Building.shp'
        },
        {
            'name': 'Lake',
            'original': 'pso_data/gis_osm_railways_free_1.shp',
            'watermarked': 'embed/M_gis_osm_railways_free_1.shp'
        }
    ]
    
    # 存储结果
    results = []
    
    for pair in file_pairs:
        print(f"\n正在分析: {pair['name']}")
        print("-" * 40)
        
        # 检查文件是否存在
        if not os.path.exists(pair['original']):
            print(f"  ❌ 原始文件不存在: {pair['original']}")
            continue
            
        if not os.path.exists(pair['watermarked']):
            print(f"  ❌ 水印文件不存在: {pair['watermarked']}")
            continue
        
        # 计算误差
        error_metrics = calculate_geometric_error(pair['original'], pair['watermarked'])
        
        # 显示结果
        if error_metrics['max_error'] >= 0:
            print(f"  ✅ 分析完成:")
            print(f"    最大误差: {error_metrics['max_error']:.8f}")
            print(f"    平均误差: {error_metrics['mean_error']:.8f}")
            print(f"    均方误差(MSE): {error_metrics['mse']:.8f}")
            print(f"    均方根误差(RMSE): {error_metrics['rmse']:.8f}")
            print(f"    总坐标点数: {error_metrics['total_points']}")
            print(f"    总要素数: {error_metrics['total_features']}")
            
            # 评估不可见性
            invisibility_level = evaluate_invisibility(error_metrics)
            print(f"    不可见性评级: {invisibility_level}")
        
        # 存储结果
        results.append({
            'file_name': pair['name'],
            'original_file': pair['original'],
            'watermarked_file': pair['watermarked'],
            **error_metrics
        })
    
    # 生成报告
    generate_invisibility_report(results)
    
    return results


def evaluate_invisibility(error_metrics):
    """
    评估不可见性等级
    """
    max_error = error_metrics['max_error']
    rmse = error_metrics['rmse']
    
    if max_error < 0:
        return "❌ 计算失败"
    elif max_error == 0 and rmse == 0:
        return "🔵 完全一致 (可能是同一文件)"
    elif max_error < 1e-10 and rmse < 1e-10:
        return "🟢 极佳 (几乎不可见)"
    elif max_error < 1e-6 and rmse < 1e-6:
        return "🟢 优秀 (不可见)"
    elif max_error < 1e-3 and rmse < 1e-3:
        return "🟡 良好 (基本不可见)"
    elif max_error < 1 and rmse < 1:
        return "🟠 一般 (可能可见)"
    else:
        return "🔴 较差 (明显可见)"


def generate_invisibility_report(results):
    """
    生成不可见性分析报告
    """
    print("\n" + "=" * 80)
    print("水印不可见性分析报告")
    print("=" * 80)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 过滤有效结果
    valid_results = df[df['max_error'] >= 0]
    
    if len(valid_results) == 0:
        print("❌ 没有有效的分析结果")
        return
    
    print(f"\n📊 总体统计:")
    print(f"  分析文件数: {len(valid_results)}")
    print(f"  失败文件数: {len(df) - len(valid_results)}")
    
    if len(valid_results) > 0:
        print(f"\n📈 误差统计 (所有文件):")
        print(f"  平均最大误差: {valid_results['max_error'].mean():.8f}")
        print(f"  最大误差范围: [{valid_results['max_error'].min():.8f}, {valid_results['max_error'].max():.8f}]")
        print(f"  平均RMSE: {valid_results['rmse'].mean():.8f}")
        print(f"  RMSE范围: [{valid_results['rmse'].min():.8f}, {valid_results['rmse'].max():.8f}]")
        print(f"  平均MSE: {valid_results['mse'].mean():.8f}")
        print(f"  平均均值误差: {valid_results['mean_error'].mean():.8f}")
        
        print(f"\n📋 各文件不可见性评估:")
        for _, row in valid_results.iterrows():
            invisibility = evaluate_invisibility(row.to_dict())
            print(f"  {row['file_name']:10s}: {invisibility}")
            print(f"    {'':12s}Max={row['max_error']:.8f}, RMSE={row['rmse']:.8f}")
    
    # 保存结果
    df.to_csv('watermark_invisibility_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\n📁 详细结果已保存到: watermark_invisibility_analysis.csv")
    
    # 绘制图表
    if len(valid_results) > 0:
        plot_invisibility_metrics(valid_results)


def plot_invisibility_metrics(results_df):
    """
    绘制不可见性指标图表
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    file_names = results_df['file_name']
    
    # 最大误差 (对数尺度)
    ax1.bar(file_names, results_df['max_error'], color='red', alpha=0.7)
    ax1.set_title('最大几何误差', fontsize=14, fontweight='bold')
    ax1.set_ylabel('最大误差')
    ax1.set_yscale('log')
    ax1.tick_params(axis='x', rotation=45)
    
    # RMSE (对数尺度)
    ax2.bar(file_names, results_df['rmse'], color='blue', alpha=0.7)
    ax2.set_title('均方根误差 (RMSE)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    # MSE (对数尺度)
    ax3.bar(file_names, results_df['mse'], color='green', alpha=0.7)
    ax3.set_title('均方误差 (MSE)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MSE')
    ax3.set_yscale('log')
    ax3.tick_params(axis='x', rotation=45)
    
    # 平均误差 (对数尺度)
    ax4.bar(file_names, results_df['mean_error'], color='orange', alpha=0.7)
    ax4.set_title('平均几何误差', fontsize=14, fontweight='bold')
    ax4.set_ylabel('平均误差')
    ax4.set_yscale('log')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle('水印嵌入几何误差分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('watermark_invisibility_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('watermark_invisibility_metrics.pdf', bbox_inches='tight')
    
    print(f"📊 不可见性指标图表已保存到: watermark_invisibility_metrics.png")
    plt.show()


def main():
    """
    主函数
    """
    print("🔍 水印嵌入不可见性测试工具")
    print("=" * 60)
    print("本工具分析水印嵌入前后文件的几何差异，评估水印的不可见性")
    print("=" * 60)
    
    print("\n📁 文件配置:")
    print("  原始文件目录: pso_data/")
    print("  水印文件目录: embed/")
    print("  对比文件对数: 6对")
    print("\n⚠️  注意事项:")
    print("1. 误差单位与坐标系统的单位一致")
    print("2. 分析包括最大误差、平均误差、MSE、RMSE等指标")
    print("3. 生成详细报告和可视化图表")
    
    # 执行测试
    results = test_watermark_invisibility()
    
    print("\n🎉 水印不可见性测试完成!")
    
    return results


if __name__ == "__main__":
    main()
