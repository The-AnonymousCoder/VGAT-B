#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lin18 通用工具函数
"""

from pathlib import Path
from typing import List
import geopandas as gpd


def discover_valid_shapefiles(pso_dir: Path, max_count: int = 8) -> List[Path]:
    """
    扫描 pso_data 目录，返回有效的 shapefile（排除点矢量数据）
    
    Args:
        pso_dir: pso_data 目录路径
        max_count: 最大返回数量
    
    Returns:
        有效的 shapefile 路径列表
    """
    if not pso_dir.exists():
        return []
    
    shp_files = [p for p in sorted(pso_dir.glob('*.shp')) if not p.name.startswith('._')]
    valid = []
    
    for p in shp_files:
        # 检查必需的辅助文件
        if not (p.with_suffix('.dbf').exists() and p.with_suffix('.shx').exists()):
            continue
        
        # 过滤点矢量数据（Lin18 不支持 Point/MultiPoint）
        try:
            gdf = gpd.read_file(str(p))
            geom_types = gdf.geometry.geom_type.unique()
            if any(gt in ['Point', 'MultiPoint'] for gt in geom_types):
                print(f'  跳过点矢量: {p.stem} (类型: {list(geom_types)})')
                continue
            valid.append(p)
        except Exception as e:
            print(f'  读取失败: {p.stem} - {e}')
            continue
    
    return valid[:max_count]


def is_point_vector(shapefile_path: Path) -> bool:
    """
    检查 shapefile 是否为点矢量数据
    
    Args:
        shapefile_path: shapefile 路径
    
    Returns:
        True 如果是点矢量，否则 False
    """
    try:
        gdf = gpd.read_file(str(shapefile_path))
        geom_types = gdf.geometry.geom_type.unique()
        return any(gt in ['Point', 'MultiPoint'] for gt in geom_types)
    except Exception:
        return False
