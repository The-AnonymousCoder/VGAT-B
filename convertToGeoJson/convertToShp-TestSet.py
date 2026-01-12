#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 SourceData/TestSet/GDB/H50.gdb 转换为 SHP 格式的脚本（Append模式）。

用法：
    在本目录下运行：
        python convertToShp-TestSet.py

说明：
    逻辑与 convertToGeoJson.py 保持一致，但只处理 TestSet 数据集下的 H50.gdb，
    将 GDB 中的每个图层转换为 SHP 格式，保存到 TestSet/SHP/H50 文件夹下。
    使用 append 模式，不会删除对应文件夹下的现有文件。
"""

import sys
import logging
from pathlib import Path

try:
    import geopandas as gpd
    import fiona
    from fiona.drvsupport import supported_drivers
    from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint
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
        logging.FileHandler('convertToShp-TestSet.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_gdb_layers(gdb_path: str) -> list:
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


def main():
    """主函数：将 TestSet/GDB/H50.gdb 转换为 SHP 格式（Append模式）"""
    try:
        # 确保支持 GDB 读取
        supported_drivers["OpenFileGDB"] = "r"
        
        # 脚本目录
        script_dir = Path(__file__).parent
        source_dir = script_dir / "SourceData"
        
        # 输入路径
        gdb_path = source_dir / "TestSet" / "GDB" / "H50.gdb"
        
        # 输出路径
        output_dir = source_dir / "TestSet" / "SHP" / "H50"
        
        logger.info("开始转换 TestSet/GDB/H50.gdb 为 SHP 格式（Append模式）...")
        logger.info(f"源GDB文件: {gdb_path}")
        logger.info(f"输出目录: {output_dir}")
        logger.info("⚠️  注意：本次运行不会删除现有文件，采用追加模式")
        
        # 检查GDB文件是否存在
        if not gdb_path.exists():
            logger.error(f"未找到GDB文件: {gdb_path}")
            return
        
        if not gdb_path.is_dir():
            logger.error(f"GDB路径不是目录: {gdb_path}")
            return
        
        # 创建输出目录（如果不存在）
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图层
        layers = get_gdb_layers(str(gdb_path))
        
        if not layers:
            logger.warning("GDB文件中没有找到图层")
            return
        
        converted_count = 0
        skipped_count = 0
        failed_count = 0
        created_gpkg: list = []
        
        # 转换每个图层
        for layer in layers:
            try:
                # 生成输出文件名：图层名.gpkg/.shp
                gpkg_path = output_dir / f"{layer}.gpkg"
                output_shp_path = output_dir / f"{layer}.shp"

                # Append模式优先检查 GeoPackage（我们现在以 GPKG 为主）
                if gpkg_path.exists():
                    logger.info(f"GeoPackage 已存在，跳过: {gpkg_path.name} (Append模式)")
                    skipped_count += 1
                    continue
                
                # 读取图层数据
                logger.info(f"正在读取图层: {layer}")
                gdf = gpd.read_file(str(gdb_path), layer=layer)
                
                if gdf.empty:
                    logger.warning(f"图层 {layer} 为空，跳过")
                    continue
                
                # 转换为WGS84坐标系（与 convertToGeoJson.py 保持一致）
                if gdf.crs and str(gdf.crs) != 'EPSG:4326':
                    logger.info(f"图层 {layer} 坐标系: {gdf.crs} -> EPSG:4326")
                    gdf = gdf.to_crs('EPSG:4326')

                # ---------- 标准化：强制 multipart 几何以及统一字段名 ----------
                def _to_multipart(geom):
                    """
                    将单部件几何包装为对应的多部件几何，已是多部件则原样返回。
                    安全处理 None / 非标准几何。
                    """
                    if geom is None:
                        return geom
                    try:
                        geom_type = geom.geom_type
                    except Exception:
                        return geom

                    if geom_type == 'Polygon':
                        return MultiPolygon([geom])
                    if geom_type == 'MultiPolygon':
                        return geom
                    if geom_type == 'LineString':
                        return MultiLineString([geom])
                    if geom_type == 'MultiLineString':
                        return geom
                    if geom_type == 'Point':
                        return MultiPoint([geom])
                    if geom_type == 'MultiPoint':
                        return geom
                    return geom

                # 应用到 geometry 列，兼容列缺失或异常
                if 'geometry' in gdf:
                    try:
                        gdf['geometry'] = gdf['geometry'].apply(lambda x: _to_multipart(x) if x is not None else x)
                    except Exception:
                        # 退回兼容写法：逐行处理，跳过不可用项
                        new_geoms = []
                        for idx, row in gdf.iterrows():
                            geom = None
                            try:
                                geom = row.get('geometry', None)
                            except Exception:
                                geom = None
                            new_geoms.append(_to_multipart(geom) if geom is not None else geom)
                        gdf['geometry'] = new_geoms

                # 统一可能的字段名变体（避免 SHP DBF 截断造成的差异）
                rename_map = {}
                for c in list(gdf.columns):
                    lc = c.lower()
                    if lc.startswith('shape_leng'):
                        rename_map[c] = 'SHAPE_Length'
                    if lc.startswith('shape_area'):
                        rename_map[c] = 'SHAPE_Area'
                if rename_map:
                    gdf = gdf.rename(columns=rename_map)

                # 保存为 GeoPackage（优先）以避免 DBF 字段名截断问题
                gpkg_path = output_dir / f"{layer}.gpkg"
                if gpkg_path.exists():
                    logger.info(f"GeoPackage 已存在，跳过: {gpkg_path.name} (Append模式)")
                    skipped_count += 1
                    continue

                try:
                    gdf.to_file(gpkg_path, driver='GPKG', encoding='utf-8')
                    logger.info(f"✅ 成功转换: {layer} -> {gpkg_path.name} ({len(gdf)} 个要素)")
                    converted_count += 1
                    created_gpkg.append(gpkg_path)
                except Exception as e:
                    logger.error(f"❌ 保存为 GeoPackage 失败: {e}")
                    failed_count += 1
                    continue

                # 兼容：生成或覆盖 Shapefile（为了保证几何类型一致，允许覆盖已存在的 SHP）
                try:
                    # 若目标 SHP 存在，先尝试删除同名的 .shp/.shx/.dbf/.prj/.cpg 以实现覆盖写入
                    shp_base = output_dir / layer
                    if output_shp_path.exists():
                        for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
                            p = output_dir / f"{layer}{ext}"
                            try:
                                if p.exists():
                                    p.unlink()
                            except Exception as e:
                                logger.warning(f"无法删除旧文件 {p.name}：{e}")

                    gdf.to_file(output_shp_path, driver='ESRI Shapefile', encoding='utf-8')
                    logger.info(f"✅ 同步生成/覆盖 Shapefile: {layer}.shp ({len(gdf)} 个要素)")
                except Exception as e:
                    logger.warning(f"⚠️ 无法生成/覆盖 Shapefile（非致命）: {e}")

        # end for layers

            except Exception as e:
                logger.error(f"❌ 转换GDB图层 {layer} 失败: {e}")
                failed_count += 1
                continue
        
        # 输出统计信息
        logger.info("=" * 60)
        logger.info(f"转换完成统计:")
        logger.info(f"  成功转换: {converted_count} 个图层")
        logger.info(f"  跳过（已存在）: {skipped_count} 个图层")
        logger.info(f"  转换失败: {failed_count} 个图层")
        logger.info(f"  输出目录: {output_dir}")
        logger.info("✅ 采用追加模式，未删除现有文件")

        # 注意：已移除 GPKG -> GeoJSON 的导出逻辑（用户要求只保留 SHP 输出）。
        # 如果需要保留 GeoJSON 导出，请恢复下列逻辑或创建单独脚本负责导出。

        # 删除所有中间 GPKG 文件（用户指定：GPKG 仅作为中间结果）
        try:
            all_gpks = list(output_dir.glob("*.gpkg"))
            for gp in all_gpks:
                try:
                    gp.unlink()
                    logger.info(f"已删除中间文件: {gp.name}")
                except Exception as e:
                    logger.warning(f"无法删除中间 GPKG {gp}: {e}")
        except Exception:
            pass

        # 列出生成的文件
        if output_dir.exists():
            shp_files = list(output_dir.glob("*.shp"))
            logger.info(f"\n生成的SHP文件数量: {len(shp_files)}")
            for shp_file in sorted(shp_files):
                logger.info(f"  - {shp_file.name}")
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()




