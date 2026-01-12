#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 zNC-Test/vector-data-geojson 目录下的 GeoJSON 文件批量转换为 ESRI Shapefile，
并将结果保存到 zNC-Test/vector-data（保留相对目录结构）。
采用 Append 模式：如果目标 .shp 已存在则跳过，不会覆盖或删除已有文件。
"""

import sys
import logging
from pathlib import Path

try:
    import geopandas as gpd
except ImportError as e:
    print("错误：缺少 geopandas（或其依赖）。请在合适的环境中运行（例如 conda activate WSL_py39）")
    print(f"详细错误：{e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("geojson_to_shp.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    script_dir = Path(__file__).parent
    src_root = script_dir / "vector-data-geojson"
    dst_root = script_dir / "vector-data"

    logger.info("开始将 GeoJSON 转换为 SHP（Append 模式）")
    logger.info(f"源目录: {src_root}")
    logger.info(f"目标目录: {dst_root}")

    if not src_root.exists():
        logger.error(f"未找到源目录: {src_root}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    # 查找 .geojson 和 .json 文件
    geojson_files = list(src_root.rglob("*.geojson")) + list(src_root.rglob("*.json"))
    if not geojson_files:
        logger.info("未找到任何 GeoJSON 文件，退出")
        return

    converted = 0
    skipped = 0
    failed = 0

    for src in sorted(geojson_files):
        try:
            rel = src.relative_to(src_root)
            out_subdir = dst_root / rel.parent
            out_subdir.mkdir(parents=True, exist_ok=True)

            out_shp = out_subdir / f"{src.stem}.shp"

            # Append 模式：存在则跳过
            if out_shp.exists():
                logger.info(f"目标已存在，跳过: {out_shp.relative_to(script_dir)}")
                skipped += 1
                continue

            logger.info(f"读取: {src.relative_to(script_dir)}")
            gdf = gpd.read_file(str(src))

            if gdf is None or gdf.empty:
                logger.warning(f"文件为空或无法读取，跳过: {src.name}")
                skipped += 1
                continue

            # 若非 WGS84 则转换为 EPSG:4326（跟其它转换脚本保持一致）
            if getattr(gdf, "crs", None) and str(gdf.crs) != "EPSG:4326":
                logger.info(f"转换坐标系: {src.name} ({gdf.crs}) -> EPSG:4326")
                gdf = gdf.to_crs("EPSG:4326")

            # 保存为 Shapefile（ESRI Shapefile driver）
            # 注意：Shapefile 字段名长度限制，可能触发用户警告，这里保持默认行为
            gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
            logger.info(f"成功转换: {src.name} -> {out_shp.relative_to(script_dir)} ({len(gdf)} 要素)")
            converted += 1

        except Exception as e:
            logger.error(f"转换失败: {src} -> {e}")
            failed += 1
            continue

    logger.info("=" * 50)
    logger.info(f"转换完成：成功 {converted}，已跳过 {skipped}，失败 {failed}")
    logger.info(f"输出目录: {dst_root}")


if __name__ == "__main__":
    main()




































