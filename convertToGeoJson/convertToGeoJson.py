#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矢量地图数据转换为GeoJSON格式的脚本
功能：将SourceData目录下的GDB和SHP文件转换为GeoJSON格式
输出：转换后的文件保存在GeoJson目录下，命名格式为"文件夹名-图层名"
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import json

try:
    import geopandas as gpd
    import fiona
    from fiona.drvsupport import supported_drivers
except ImportError as e:
    print("错误：缺少必要的地理数据处理库")
    print("请安装以下库：pip install geopandas fiona")
    print(f"详细错误：{e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VectorToGeoJsonConverter:
    """矢量数据到GeoJSON转换器"""
    
    def __init__(self, source_dir: str = "SourceData", output_dir: str = "GeoJson"):
        """
        初始化转换器
        
        Args:
            source_dir: 源数据目录
            output_dir: 输出目录
        """
        self.script_dir = Path(__file__).parent
        self.source_dir = self.script_dir / source_dir
        self.output_dir = self.script_dir / output_dir
        
        # 确保输出目录存在
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建TrainingSet和TestSet输出目录
        self.training_output_dir = self.output_dir / "TrainingSet"
        self.test_output_dir = self.output_dir / "TestSet"
        self.training_output_dir.mkdir(exist_ok=True)
        self.test_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"源数据目录: {self.source_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"训练集输出目录: {self.training_output_dir}")
        logger.info(f"测试集输出目录: {self.test_output_dir}")
    
    def discover_gdb_files(self, dataset: str) -> List[Tuple[str, str]]:
        """
        发现指定数据集中的所有GDB文件
        
        Args:
            dataset: 数据集名称 ("TrainingSet" 或 "TestSet")
            
        Returns:
            List[Tuple[str, str]]: (gdb文件路径, 文件夹名称) 的列表
        """
        gdb_files = []
        gdb_dir = self.source_dir / dataset / "GDB"
        
        if not gdb_dir.exists():
            logger.warning(f"GDB目录不存在: {gdb_dir}")
            return gdb_files
        
        for item in gdb_dir.iterdir():
            if item.is_dir() and item.suffix.lower() == '.gdb':
                folder_name = item.stem  # 去掉.gdb后缀
                gdb_files.append((str(item), folder_name))
                logger.info(f"发现{dataset}中的GDB文件: {item}")
        
        return gdb_files
    
    def discover_shp_files(self, dataset: str) -> List[Tuple[str, str, str]]:
        """
        发现指定数据集中的所有SHP文件
        
        Args:
            dataset: 数据集名称 ("TrainingSet" 或 "TestSet")
            
        Returns:
            List[Tuple[str, str, str]]: (shp文件路径, 文件夹名称, 文件名) 的列表
        """
        shp_files = []
        shp_dir = self.source_dir / dataset / "SHP"
        
        if not shp_dir.exists():
            logger.warning(f"SHP目录不存在: {shp_dir}")
            return shp_files
        
        # 递归查找所有.shp文件，但排除以.shp结尾的文件夹
        for shp_file in shp_dir.rglob("*.shp"):
            # 检查是否为真正的文件（不是文件夹）
            if not shp_file.is_file():
                logger.warning(f"跳过以.shp结尾的文件夹: {shp_file}")
                continue
                
            # 获取相对于SHP目录的父文件夹名称
            relative_path = shp_file.relative_to(shp_dir)
            folder_name = relative_path.parts[0] if len(relative_path.parts) > 1 else shp_dir.name
            file_name = shp_file.stem  # 去掉.shp后缀
            
            shp_files.append((str(shp_file), folder_name, file_name))
            logger.info(f"发现{dataset}中的SHP文件: {shp_file} (文件夹: {folder_name})")
        
        return shp_files
    
    def get_gdb_layers(self, gdb_path: str) -> List[str]:
        """
        获取GDB文件中的所有图层
        
        Args:
            gdb_path: GDB文件路径
            
        Returns:
            List[str]: 图层名称列表
        """
        try:
            layers = fiona.listlayers(gdb_path)
            logger.info(f"GDB {gdb_path} 包含 {len(layers)} 个图层: {layers}")
            return layers
        except Exception as e:
            logger.error(f"无法读取GDB文件 {gdb_path} 的图层: {e}")
            return []
    
    def convert_gdb_to_geojson(self, gdb_path: str, folder_name: str, output_dir: Path) -> int:
        """
        将GDB文件转换为GeoJSON
        
        Args:
            gdb_path: GDB文件路径
            folder_name: 文件夹名称（用于命名）
            output_dir: 输出目录路径
            
        Returns:
            int: 成功转换的图层数量
        """
        success_count = 0
        layers = self.get_gdb_layers(gdb_path)
        
        for layer in layers:
            try:
                # 读取图层数据
                gdf = gpd.read_file(gdb_path, layer=layer)
                
                if gdf.empty:
                    logger.warning(f"图层 {layer} 为空，跳过")
                    continue
                
                # 转换为WGS84坐标系（GeoJSON标准）
                if gdf.crs and gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                
                # 生成输出文件名：文件夹名-图层名
                output_filename = f"{folder_name}-{layer}.geojson"
                output_path = output_dir / output_filename
                
                # 保存为GeoJSON
                gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
                
                logger.info(f"成功转换: {layer} -> {output_filename} ({len(gdf)} 个要素)")
                success_count += 1
                
            except Exception as e:
                logger.error(f"转换GDB图层 {layer} 失败: {e}")
        
        return success_count
    
    def convert_shp_to_geojson(self, shp_path: str, folder_name: str, file_name: str, output_dir: Path) -> bool:
        """
        将SHP文件转换为GeoJSON
        
        Args:
            shp_path: SHP文件路径
            folder_name: 文件夹名称
            file_name: 文件名（不包含扩展名）
            output_dir: 输出目录路径
            
        Returns:
            bool: 转换是否成功
        """
        # 尝试多种编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # 读取SHP文件
                gdf = gpd.read_file(shp_path, encoding=encoding)
                
                if gdf.empty:
                    logger.warning(f"SHP文件 {shp_path} 为空，跳过")
                    return False
                
                # 转换为WGS84坐标系（GeoJSON标准）
                if gdf.crs and gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                
                # 生成输出文件名：文件夹名-文件名
                output_filename = f"{folder_name}-{file_name}.geojson"
                output_path = output_dir / output_filename
                
                # 保存为GeoJSON（始终使用UTF-8编码）
                gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
                
                logger.info(f"成功转换: {shp_path} -> {output_filename} ({len(gdf)} 个要素) [编码: {encoding}]")
                return True
                
            except UnicodeDecodeError as e:
                # 编码错误，尝试下一种编码
                logger.debug(f"编码 {encoding} 读取 {shp_path} 失败: {e}")
                continue
            except Exception as e:
                # 其他错误，也尝试下一种编码
                logger.debug(f"编码 {encoding} 处理 {shp_path} 时出错: {e}")
                continue
        
        # 所有编码都失败了
        logger.error(f"转换SHP文件 {shp_path} 失败: 尝试了所有编码格式 {encodings} 都无法读取文件")
        return False
    
    def clean_output_directory(self, output_dir: Path, dataset: str):
        """
        清理输出目录中的旧文件
        
        Args:
            output_dir: 输出目录路径
            dataset: 数据集名称
        """
        if output_dir.exists():
            old_files = list(output_dir.glob("*.geojson"))
            if old_files:
                logger.info(f"清理{dataset}中的 {len(old_files)} 个旧GeoJSON文件")
                for old_file in old_files:
                    try:
                        old_file.unlink()
                        logger.debug(f"删除旧文件: {old_file.name}")
                    except Exception as e:
                        logger.warning(f"删除旧文件 {old_file.name} 失败: {e}")
            else:
                logger.info(f"{dataset}输出目录为空，无需清理")
        else:
            logger.info(f"{dataset}输出目录不存在，创建新目录")
            output_dir.mkdir(parents=True, exist_ok=True)

    def convert_dataset(self, dataset: str, output_dir: Path):
        """
        转换指定数据集的所有矢量数据文件
        
        Args:
            dataset: 数据集名称 ("TrainingSet" 或 "TestSet")
            output_dir: 输出目录
        """
        logger.info(f"=== 开始转换{dataset}数据集 ===")
        
        # 首先清理旧文件
        self.clean_output_directory(output_dir, dataset)
        
        converted_count = 0
        failed_count = 0
        
        # 转换GDB文件
        logger.info(f"--- 转换{dataset}中的GDB文件 ---")
        gdb_files = self.discover_gdb_files(dataset)
        
        for gdb_path, folder_name in gdb_files:
            logger.info(f"处理{dataset}中的GDB文件: {gdb_path}")
            converted_layers = self.convert_gdb_to_geojson(gdb_path, folder_name, output_dir)
            converted_count += converted_layers
        
        # 转换SHP文件
        logger.info(f"--- 转换{dataset}中的SHP文件 ---")
        shp_files = self.discover_shp_files(dataset)
        
        for shp_path, folder_name, file_name in shp_files:
            logger.info(f"处理{dataset}中的SHP文件: {shp_path}")
            if self.convert_shp_to_geojson(shp_path, folder_name, file_name, output_dir):
                converted_count += 1
            else:
                failed_count += 1
        
        logger.info(f"{dataset}转换完成: 成功转换 {converted_count} 个文件, 失败 {failed_count} 个文件")
        return converted_count, failed_count
    
    def convert_all(self):
        """
        转换所有支持的矢量数据文件
        """
        logger.info("开始矢量数据转换...")
        logger.info("本次运行将完全覆盖之前生成的所有GeoJSON文件")
        
        # 清理旧的日志文件（可选）
        log_file = Path("conversion.log")
        if log_file.exists():
            try:
                import time
                # 备份之前的日志，使用当前时间戳
                timestamp = int(time.time())
                backup_log = Path(f"conversion_backup_{timestamp}.log")
                if not backup_log.exists():
                    import shutil
                    shutil.copy2(log_file, backup_log)
                    logger.info(f"旧日志已备份为: {backup_log.name}")
            except Exception as e:
                logger.warning(f"备份日志文件失败: {e}")
        
        total_converted = 0
        total_failed = 0
        
        # 转换TrainingSet
        training_converted, training_failed = self.convert_dataset("TrainingSet", self.training_output_dir)
        total_converted += training_converted
        total_failed += training_failed
        
        # 转换TestSet
        test_converted, test_failed = self.convert_dataset("TestSet", self.test_output_dir)
        total_converted += test_converted
        total_failed += test_failed
        
        # 输出统计信息
        logger.info("=== 转换完成 ===")
        logger.info(f"总计成功转换: {total_converted} 个文件")
        logger.info(f"总计转换失败: {total_failed} 个文件")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("✅ 所有生成的GeoJSON文件已完全替换之前的对应文件")
        logger.info("📁 每次运行都会自动清理并重新生成所有输出文件")
        
        # 列出生成的文件
        self._list_generated_files("TrainingSet", self.training_output_dir)
        self._list_generated_files("TestSet", self.test_output_dir)
    
    def _list_generated_files(self, dataset: str, output_dir: Path):
        """
        列出指定数据集生成的文件
        
        Args:
            dataset: 数据集名称
            output_dir: 输出目录
        """
        if output_dir.exists():
            geojson_files = list(output_dir.glob("*.geojson"))
            logger.info(f"{dataset}生成的GeoJSON文件数量: {len(geojson_files)}")
            for file in sorted(geojson_files):
                logger.info(f"  - {file.name}")
        else:
            logger.warning(f"{dataset}输出目录不存在: {output_dir}")


def main():
    """主函数"""
    try:
        # 检查驱动支持
        supported_drivers['OpenFileGDB'] = 'r'  # 确保支持GDB读取
        
        # 创建转换器并执行转换
        converter = VectorToGeoJsonConverter()
        converter.convert_all()
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()
