import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Polygon, LineString


def geometry_to_xy(geom):
    """将几何对象转换为带NaN分隔符的坐标序列"""
    x, y = [], []
    if geom is None or geom.is_empty:
        return x, y
    geom_type = geom.geom_type

    if geom_type == 'Polygon':
        # 处理外环
        exterior = list(geom.exterior.coords)
        if exterior:
            x.extend([coord[0] for coord in exterior])
            y.extend([coord[1] for coord in exterior])
            x.append(np.nan)
            y.append(np.nan)
        # 处理内环
        for interior in geom.interiors:
            coords = list(interior.coords)
            x.extend([coord[0] for coord in coords])
            y.extend([coord[1] for coord in coords])
            x.append(np.nan)
            y.append(np.nan)
    elif geom_type == 'LineString':
        coords = list(geom.coords)
        x.extend([coord[0] for coord in coords])
        y.extend([coord[1] for coord in coords])
        x.append(np.nan)
        y.append(np.nan)
    elif geom_type in ['MultiPolygon', 'MultiLineString']:
        for part in geom.geoms:
            part_x, part_y = geometry_to_xy(part)
            x.extend(part_x)
            y.extend(part_y)
    elif geom_type == 'Point':
        x.append(geom.x)
        y.append(geom.y)

    # 移除末尾的NaN
    if x and np.isnan(x[-1]):
        x.pop()
        y.pop()
    return x, y


def reverse_xy_coordinates(x, y):
    """反转坐标序列并保持NaN分隔"""
    segments = []
    current_x, current_y = [], []

    for xi, yi in zip(x, y):
        if np.isnan(xi) or np.isnan(yi):
            if current_x:
                segments.append((current_x, current_y))
                current_x, current_y = [], []
        else:
            current_x.append(xi)
            current_y.append(yi)
    if current_x:
        segments.append((current_x, current_y))

    # 反转每个段
    reversed_segments = [(sx[::-1], sy[::-1]) for sx, sy in reversed(segments)]

    # 重新组装坐标
    new_x, new_y = [], []
    for seg_x, seg_y in reversed_segments:
        new_x.extend(seg_x)
        new_y.extend(seg_y)
        new_x.append(np.nan)
        new_y.append(np.nan)

    # 移除末尾的NaN
    if new_x and np.isnan(new_x[-1]):
        new_x.pop()
        new_y.pop()
    return new_x, new_y


def xy_to_geometry(new_x, new_y, original_geom):
    """将坐标序列转换回几何对象"""
    geom_type = original_geom.geom_type
    rings = []
    current_ring = []

    for xi, yi in zip(new_x, new_y):
        if np.isnan(xi) or np.isnan(yi):
            if current_ring:
                rings.append(current_ring)
                current_ring = []
        else:
            current_ring.append((xi, yi))
    if current_ring:
        rings.append(current_ring)

    try:
        if geom_type == 'Polygon':
            if not rings:
                return Polygon()
            exterior = rings[0]
            interiors = rings[1:] if len(rings) > 1 else []
            return Polygon(exterior, interiors)
        elif geom_type == 'LineString':
            return LineString(rings[0]) if rings else LineString()
        elif geom_type in ['MultiPolygon', 'MultiLineString']:
            # 简化为单部件处理
            return original_geom.__class__(rings[0]) if rings else original_geom
        else:
            return original_geom
    except:
        return original_geom


def vertex_reorganization_attack(shp_file, output_path):
    """通用顶点重组攻击函数"""
    try:
        gdf = gpd.read_file(shp_file)
    except Exception as e:
        raise ValueError(f"文件读取失败: {e}")

    new_gdf = gdf.copy()

    for idx in new_gdf.index:
        geom = new_gdf.at[idx, 'geometry']
        if geom is None or geom.is_empty:
            continue

        x, y = geometry_to_xy(geom)
        new_x, new_y = reverse_xy_coordinates(x, y)
        new_geom = xy_to_geometry(new_x, new_y, geom)
        new_gdf.at[idx, 'geometry'] = new_geom

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_gdf.to_file(output_path)
    print(f"处理完成，结果保存至: {output_path}")
    return output_path


def attacks6_1_vertex_reorganization(shp_file, out_shpfile):
    """对应MATLAB的vertex_reorganization攻击"""
    output_dir = os.path.join('attacked', 'object_reorganized')
    output_path = os.path.join(output_dir, out_shpfile)
    return vertex_reorganization_attack(shp_file, output_path)


def attacks6_2_object_reorganization(shp_file, out_shpfile):
    """对应MATLAB的object_reorganization攻击"""
    output_dir = os.path.join('attacked', 'vertex_reorganized')
    output_filename = f"vertex_reorganized_{out_shpfile}"
    output_path = os.path.join(output_dir, output_filename)
    return vertex_reorganization_attack(shp_file, output_path)


# 示例用法
if __name__ == "__main__":
    # 测试attacks6_1
    attacks6_1_vertex_reorganization("input.shp", "output_6_1.shp")

    # 测试attacks6_2
    attacks6_2_object_reorganization("input.shp", "output_6_2.shp")