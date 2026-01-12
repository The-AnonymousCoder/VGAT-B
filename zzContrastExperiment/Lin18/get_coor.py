# -*- coding: utf-8 -*-
"""
get_coor_fixed.py - 重构版坐标提取模块
修复 Multi* 几何类型处理的问题
"""
import numpy as np
import sys


def sort_multi_geometry_parts(geometry):
    """
    对 Multi* 几何的各部分按左下角坐标排序，确保顺序稳定
    
    :param geometry: shapely 几何对象
    :return: 排序后的几何对象
    """
    if 'Multi' not in geometry.geom_type:
        return geometry
    
    # 获取每个部分的边界框
    parts_with_bounds = []
    for part in geometry.geoms:
        bounds = part.bounds  # (minx, miny, maxx, maxy)
        # 使用 (minx, miny) 作为排序键，确保稳定的顺序
        parts_with_bounds.append((bounds[0], bounds[1], part))
    
    # 按 (x, y) 排序
    parts_with_bounds.sort(key=lambda x: (x[0], x[1]))
    
    # 重构 Multi* 几何
    sorted_parts = [p[2] for p in parts_with_bounds]
    
    from shapely.geometry import MultiPolygon, MultiLineString, MultiPoint
    if geometry.geom_type == 'MultiPolygon':
        return MultiPolygon(sorted_parts)
    elif geometry.geom_type == 'MultiLineString':
        return MultiLineString(sorted_parts)
    elif geometry.geom_type == 'MultiPoint':
        return MultiPoint(sorted_parts)
    else:
        return geometry


def get_coor_nested(shpfile):
    """
    获取坐标，将每个几何对象作为一个数组，并将所有数组组合成一个大数组
    
    重要修复：
    1. 不再在遇到 Multi* 时清空并重新遍历
    2. 对所有几何类型使用统一的处理逻辑
    3. 保持坐标顺序的一致性
    4. 对 Multi* 几何的各部分排序，确保嵌入和提取时顺序一致
    
    :param shpfile: GeoDataFrame 对象
    :return: 坐标嵌套数组和几何类型列表
    """
    x_coords = []
    y_coords = []
    feature_type = []
    
    for geom in shpfile.geometry:
        if geom is None or geom.is_empty:
            continue
        
        # 关键优化：对 Multi* 几何排序
        geom = sort_multi_geometry_parts(geom)
        
        geom_type = geom.geom_type
        feature_type.append(geom_type)
        
        # 统一使用 split_coordinates_by_geometry_type 处理
        x, y = split_coordinates_by_geometry_type(geom)
        x_coords.append(x)
        y_coords.append(y)
    
    coor_nested = np.array([x_coords, y_coords], dtype=object)
    return coor_nested, feature_type


def get_coor_array(coor_nested, shp_type):
    """
    将嵌套坐标数组在 x 和 y 方向分别合并成一维数组
    
    :param coor_nested: shp数据顶点的嵌套数组
    :param shp_type: 要素类型的数组
    :return: 返回合并的坐标数组 [2 x N]，N为总顶点数
    """
    x_array = []
    y_array = []
    
    # 遍历数组中的每个要素
    for i in range(coor_nested.shape[1]):
        try:
            x_part = coor_nested[0, i]
            y_part = coor_nested[1, i]
            
            # 现在所有类型都返回一维数组（包括 Multi*）
            if np.size(x_part) > 0:
                x_array = np.hstack((x_array, x_part))
                y_array = np.hstack((y_array, y_part))
                    
        except (IndexError, ValueError) as e:
            print(f"警告: 处理特征 #{i} ({shp_type[i]}) 时出错: {e}", file=sys.stderr, flush=True)
            continue
    
    coor_array = np.vstack((x_array, y_array))
    return coor_array


def split_coordinates_by_geometry_type(geometry):
    """
    根据几何类型拆分坐标
    
    重要改进：
    1. 不再在遇到 Multi* 时清空并重新遍历整个文件
    2. 对 Multi* 类型 concatenate 各部分坐标（与原始代码兼容）
    3. 保持坐标顺序的确定性
    
    :param geometry: shapely 几何对象
    :return: (x_coords, y_coords) 一维numpy数组
    """
    if geometry is None or geometry.is_empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    
    geom_type = geometry.geom_type
    
    if geom_type == "Point":
        return np.array([geometry.x]), np.array([geometry.y])
    
    elif geom_type == "LineString":
        x_coords, y_coords = geometry.xy
        return np.array(x_coords), np.array(y_coords)
    
    elif geom_type == "Polygon":
        # 处理外环和内环（完整Polygon）
        x_coords = []
        y_coords = []
        # 外环
        exterior_x, exterior_y = geometry.exterior.xy
        x_coords.append(np.array(exterior_x))
        y_coords.append(np.array(exterior_y))
        
        # 内环（holes）- 关键优化：按左下角坐标排序
        if len(geometry.interiors) > 0:
            interiors_with_bounds = []
            for interior in geometry.interiors:
                bounds = interior.bounds  # (minx, miny, maxx, maxy)
                interiors_with_bounds.append((bounds[0], bounds[1], interior))
            
            # 排序内环，确保顺序稳定
            interiors_with_bounds.sort(key=lambda x: (x[0], x[1]))
            
            for _, _, interior in interiors_with_bounds:
                interior_x, interior_y = interior.xy
                x_coords.append(np.array(interior_x))
                y_coords.append(np.array(interior_y))
        
        # Concatenate
        x_coords = np.concatenate(x_coords) if len(x_coords) > 0 else np.array([], dtype=float)
        y_coords = np.concatenate(y_coords) if len(y_coords) > 0 else np.array([], dtype=float)
        return x_coords, y_coords
    
    elif geom_type == "MultiPoint":
        x_coords = [point.x for point in geometry.geoms]
        y_coords = [point.y for point in geometry.geoms]
        return np.array(x_coords), np.array(y_coords)
    
    elif geom_type == "MultiLineString":
        # Concatenate 所有 LineString 的坐标（与原始代码兼容）
        x_coords = []
        y_coords = []
        for line in geometry.geoms:
            line_x, line_y = line.xy
            x_coords.append(np.array(line_x))
            y_coords.append(np.array(line_y))
        # 关键修复：concatenate 而不是返回嵌套列表
        x_coords = np.concatenate(x_coords) if x_coords else np.array([], dtype=float)
        y_coords = np.concatenate(y_coords) if y_coords else np.array([], dtype=float)
        return x_coords, y_coords
    
    elif geom_type == "MultiPolygon":
        # Concatenate 所有 Polygon 的坐标（包括内环）
        # 注意：geometry.geoms 已经在 sort_multi_geometry_parts 中排序过
        x_coords = []
        y_coords = []
        for polygon in geometry.geoms:
            # 外环
            poly_x, poly_y = polygon.exterior.xy
            x_coords.append(np.array(poly_x))
            y_coords.append(np.array(poly_y))
            
            # 内环（holes）- 关键优化：按左下角坐标排序
            if len(polygon.interiors) > 0:
                interiors_with_bounds = []
                for interior in polygon.interiors:
                    bounds = interior.bounds
                    interiors_with_bounds.append((bounds[0], bounds[1], interior))
                
                # 排序内环
                interiors_with_bounds.sort(key=lambda x: (x[0], x[1]))
                
                for _, _, interior in interiors_with_bounds:
                    interior_x, interior_y = interior.xy
                    x_coords.append(np.array(interior_x))
                    y_coords.append(np.array(interior_y))
        
        # Concatenate
        x_coords = np.concatenate(x_coords) if x_coords else np.array([], dtype=float)
        y_coords = np.concatenate(y_coords) if y_coords else np.array([], dtype=float)
        return x_coords, y_coords
    
    else:
        print(f"警告: 不支持的几何类型 {geom_type}", file=sys.stderr, flush=True)
        return np.array([], dtype=float), np.array([], dtype=float)

