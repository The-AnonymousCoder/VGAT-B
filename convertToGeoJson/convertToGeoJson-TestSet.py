#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描并转换 SourceData/TestSet 下所有矢量数据文件为 GeoJson/TestSet 的脚本（Append模式）。

用法：
    在本目录下运行：
        python convertToGeoJson-TestSet.py

说明：
    逻辑与 convertToGeoJson.py 保持一致，但只处理 TestSet 数据集下的所有矢量数据文件（GDB和SHP），
    使用 append 模式，不会删除对应文件夹下的现有文件。
"""

import sys
import logging
from pathlib import Path

try:
    import geopandas as gpd
except ImportError as e:
    print("错误：缺少必要的地理数据处理库")
    print("请安装以下库：pip install geopandas fiona")
    print(f"详细错误：{e}")
    sys.exit(1)

from fiona.drvsupport import supported_drivers

from convertToGeoJson import VectorToGeoJsonConverter, logger as base_logger


logger = logging.getLogger(__name__)


def main():
    """主函数：扫描并转换 TestSet 数据集下的所有矢量数据文件（Append模式）"""
    try:
        # 确保支持 GDB 读取
        supported_drivers["OpenFileGDB"] = "r"

        # 创建转换器
        converter = VectorToGeoJsonConverter()

        logger.info("开始扫描并转换 TestSet 数据集下的所有矢量数据文件（Append模式）...")
        logger.info("⚠️  注意：本次运行不会删除现有文件，采用追加模式")

        converted_count = 0
        failed_count = 0

        # 转换GDB文件（扫描所有GDB文件，Append模式）
        logger.info("--- 转换TestSet中的GDB文件（Append模式）---")
        gdb_files = converter.discover_gdb_files("TestSet")

        for gdb_path, folder_name in gdb_files:
            logger.info(f"处理TestSet中的GDB文件: {gdb_path}")
            # 获取图层并逐个检查是否存在
            layers = converter.get_gdb_layers(gdb_path)
                
                for layer in layers:
                    # 生成输出文件名
                output_filename = f"{folder_name}-{layer}.geojson"
                    output_path = converter.test_output_dir / output_filename
                    
                    # Append模式：如果文件已存在，跳过
                    if output_path.exists():
                        logger.info(f"文件已存在，跳过: {output_filename} (Append模式)")
                        continue
                    
                    try:
                        # 读取图层数据
                    gdf = gpd.read_file(gdb_path, layer=layer)
                        
                        if gdf.empty:
                            logger.warning(f"图层 {layer} 为空，跳过")
                            continue
                        
                        # 转换为WGS84坐标系（GeoJSON标准）
                        if gdf.crs and gdf.crs != 'EPSG:4326':
                            gdf = gdf.to_crs('EPSG:4326')
                        
                        # 保存为GeoJSON
                        gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
                        
                        logger.info(f"成功转换: {layer} -> {output_filename} ({len(gdf)} 个要素)")
                        converted_count += 1
                        
                    except Exception as e:
                        logger.error(f"转换GDB图层 {layer} 失败: {e}")
                        failed_count += 1

        # 转换SHP文件（扫描所有SHP文件，Append模式）
        logger.info("--- 转换TestSet中的SHP文件（Append模式）---")
        shp_files = converter.discover_shp_files("TestSet")

        for shp_path, folder_name, file_name in shp_files:
            # 生成输出文件名
            output_filename = f"{folder_name}-{file_name}.geojson"
                    output_path = converter.test_output_dir / output_filename
                    
                    # Append模式：如果文件已存在，跳过
                    if output_path.exists():
                        logger.info(f"文件已存在，跳过: {output_filename} (Append模式)")
                        continue
                    
            logger.info(f"处理TestSet中的SHP文件: {shp_path}")
            if converter.convert_shp_to_geojson(shp_path, folder_name, file_name, converter.test_output_dir):
                            converted_count += 1
            else:
                failed_count += 1

        logger.info("TestSet转换完成（Append模式）")
        logger.info(f"成功转换: {converted_count} 个文件")
        logger.info(f"转换失败: {failed_count} 个文件")
        logger.info(f"输出目录: {converter.test_output_dir}")
        logger.info("✅ 采用追加模式，未删除现有文件")

        # 列出生成的文件
        if converter.test_output_dir.exists():
            geojson_files = list(converter.test_output_dir.glob("*.geojson"))
            logger.info(f"TestSet GeoJSON文件总数: {len(geojson_files)}")
            for file in sorted(geojson_files):
                logger.info(f"  - {file.name}")

    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()
