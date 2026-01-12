from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
import geopandas as gpd
import numpy as np
import os


def attacks5_object_crop(shpFile, outshpfile, axis='X'):
    """
    attacks5_object_crop - 沿X轴或Y轴裁剪矢量地图（保留一半）

    Parameters:
    shpFile (str): 输入shapefile文件路径
    outshpfile (str): 输出shapefile文件名
    axis (str): 裁剪轴向，'X'表示沿X轴裁剪（保留X坐标较小的一半），'Y'表示沿Y轴裁剪

    Returns:
    str: 保存后的输出shapefile完整路径
    """
    # 读取shapefile
    try:
        gdf = gpd.read_file(shpFile)
    except Exception as e:
        raise ValueError(f"无法读取文件，错误: {e}")

    # 收集所有坐标以计算边界
    all_coords = []
    for geom in gdf.geometry:
        if geom is not None and not geom.is_empty:
            if geom.geom_type == 'Point':
                all_coords.append((geom.x, geom.y))
            elif geom.geom_type in ['LineString', 'LinearRing']:
                all_coords.extend(list(geom.coords))
            elif geom.geom_type == 'Polygon':
                all_coords.extend(list(geom.exterior.coords))
                for interior in geom.interiors:
                    all_coords.extend(list(interior.coords))
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    all_coords.extend(list(line.coords))
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    all_coords.extend(list(poly.exterior.coords))
                    for interior in poly.interiors:
                        all_coords.extend(list(interior.coords))

    if not all_coords:
        raise ValueError("未找到有效的几何坐标")

    # 计算裁剪边界
    if axis.upper() == 'X':
        # 沿X轴裁剪，保留X坐标较小的一半
        all_x = [coord[0] for coord in all_coords]
        min_x, max_x = min(all_x), max(all_x)
        crop_boundary = min_x + (max_x - min_x) / 2
        print(f"沿X轴裁剪: X范围[{min_x:.6f}, {max_x:.6f}], 裁剪边界={crop_boundary:.6f}")
    elif axis.upper() == 'Y':
        # 沿Y轴裁剪，保留Y坐标较小的一半
        all_y = [coord[1] for coord in all_coords]
        min_y, max_y = min(all_y), max(all_y)
        crop_boundary = min_y + (max_y - min_y) / 2
        print(f"沿Y轴裁剪: Y范围[{min_y:.6f}, {max_y:.6f}], 裁剪边界={crop_boundary:.6f}")
    else:
        raise ValueError('axis 参数无效，请使用 "X" 或 "Y"')

    def crop_coords(coords_list, axis, boundary):
        """根据轴向和边界裁剪坐标列表"""
        if axis.upper() == 'X':
            return [(x, y) for x, y in coords_list if x <= boundary]
        else:  # Y轴
            return [(x, y) for x, y in coords_list if y <= boundary]

    # 裁剪几何体
    cropped_geometries = []
    valid_count = 0
    
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            cropped_geometries.append(None)
            continue

        try:
            if geom.geom_type == 'Point':
                if axis.upper() == 'X':
                    keep = geom.x <= crop_boundary
                else:
                    keep = geom.y <= crop_boundary
                
                if keep:
                    cropped_geometries.append(geom)
                    valid_count += 1
                else:
                    cropped_geometries.append(None)

            elif geom.geom_type == 'LineString':
                coords = list(geom.coords)
                cropped_coords = crop_coords(coords, axis, crop_boundary)
                
                if len(cropped_coords) >= 2:
                    cropped_geometries.append(LineString(cropped_coords))
                    valid_count += 1
                else:
                    cropped_geometries.append(None)

            elif geom.geom_type == 'Polygon':
                # 处理外环
                exterior_coords = list(geom.exterior.coords)
                cropped_exterior = crop_coords(exterior_coords, axis, crop_boundary)
                
                if len(cropped_exterior) >= 4:  # 至少需要4个点形成闭合多边形
                    # 确保多边形闭合
                    if cropped_exterior[0] != cropped_exterior[-1]:
                        cropped_exterior.append(cropped_exterior[0])
                    
                    # 处理内环（如果有）
                    cropped_interiors = []
                    for interior in geom.interiors:
                        interior_coords = list(interior.coords)
                        cropped_interior = crop_coords(interior_coords, axis, crop_boundary)
                        if len(cropped_interior) >= 4:
                            if cropped_interior[0] != cropped_interior[-1]:
                                cropped_interior.append(cropped_interior[0])
                            cropped_interiors.append(cropped_interior)
                    
                    try:
                        new_polygon = Polygon(cropped_exterior, cropped_interiors)
                        if new_polygon.is_valid:
                            cropped_geometries.append(new_polygon)
                            valid_count += 1
                        else:
                            cropped_geometries.append(None)
                    except:
                        cropped_geometries.append(None)
                else:
                    cropped_geometries.append(None)

            elif geom.geom_type == 'MultiLineString':
                cropped_lines = []
                for line in geom.geoms:
                    coords = list(line.coords)
                    cropped_coords = crop_coords(coords, axis, crop_boundary)
                    if len(cropped_coords) >= 2:
                        cropped_lines.append(LineString(cropped_coords))
                
                if cropped_lines:
                    if len(cropped_lines) == 1:
                        cropped_geometries.append(cropped_lines[0])
                    else:
                        cropped_geometries.append(MultiLineString(cropped_lines))
                    valid_count += 1
                else:
                    cropped_geometries.append(None)

            elif geom.geom_type == 'MultiPolygon':
                cropped_polygons = []
                for poly in geom.geoms:
                    # 处理外环
                    exterior_coords = list(poly.exterior.coords)
                    cropped_exterior = crop_coords(exterior_coords, axis, crop_boundary)
                    
                    if len(cropped_exterior) >= 4:
                        if cropped_exterior[0] != cropped_exterior[-1]:
                            cropped_exterior.append(cropped_exterior[0])
                        
                        # 处理内环
                        cropped_interiors = []
                        for interior in poly.interiors:
                            interior_coords = list(interior.coords)
                            cropped_interior = crop_coords(interior_coords, axis, crop_boundary)
                            if len(cropped_interior) >= 4:
                                if cropped_interior[0] != cropped_interior[-1]:
                                    cropped_interior.append(cropped_interior[0])
                                cropped_interiors.append(cropped_interior)
                        
                        try:
                            new_polygon = Polygon(cropped_exterior, cropped_interiors)
                            if new_polygon.is_valid:
                                cropped_polygons.append(new_polygon)
                        except:
                            pass
                
                if cropped_polygons:
                    if len(cropped_polygons) == 1:
                        cropped_geometries.append(cropped_polygons[0])
                    else:
                        cropped_geometries.append(MultiPolygon(cropped_polygons))
                    valid_count += 1
                else:
                    cropped_geometries.append(None)

            else:
                # 对于其他几何类型，保持原样
                cropped_geometries.append(geom)
                valid_count += 1

        except Exception as e:
            print(f"处理几何体时出错: {e}")
            cropped_geometries.append(None)

    # 创建新的GeoDataFrame
    cropped_gdf = gdf.copy()
    cropped_gdf.geometry = cropped_geometries
    
    # 移除空几何体的行
    cropped_gdf = cropped_gdf[cropped_gdf.geometry.notna()]
    cropped_gdf = cropped_gdf.reset_index(drop=True)

    # 创建输出目录
    output_dir = os.path.join('attacked', 'crop')
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名
    output_name = f'crop_{axis.lower()}_{outshpfile}'
    output_path = os.path.join(output_dir, output_name)

    # 保存文件
    cropped_gdf.to_file(output_path)

    print(f'裁剪攻击完成: 原始要素数={len(gdf)}, 保留要素数={len(cropped_gdf)}, 沿{axis.upper()}轴裁剪')
    
    return output_path
