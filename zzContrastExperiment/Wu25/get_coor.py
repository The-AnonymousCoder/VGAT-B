# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 19:33
# @Author  :Fivem
# @File    : get_coor.py
# @Software: PyCharm
# @last modified:2023/11/8 19:33
import numpy as np


def get_coor_nested(shpfile):
    """
    Getting coordinates makes each object an array and combines all arrays into one large array
    :param shpfile:
    :return: a list that has two-dimensional of coordinates and shp type
    """
    # 获取不同几何对象的坐标
    x_coords = []
    y_coords = []
    feature_type = []
    for geom in shpfile.geometry:
        # 对于空的Point，LineString来说 保存为文件之后 再读取得到的geom就是None
        if geom != None:
            if geom.geom_type == 'Point':
                x_coords.append(np.array([geom.x]))
                y_coords.append(np.array([geom.y]))

            elif geom.geom_type == 'LineString':
                # 处理线几何对象的坐标
                x_coords.append(np.array(geom.xy[0]))
                y_coords.append(np.array(geom.xy[1]))

            elif geom.geom_type == 'MultiLineString':
                x_coords = []
                y_coords = []
                feature_type = []

                for geom in shpfile.geometry:
                    x, y = split_coordinates_by_geometry_type(geom)
                    x_coords.append(x)
                    y_coords.append(y)
                    feature_type.append(geom.geom_type if geom else None)

                coor_nested = np.array([x_coords, y_coords], dtype=object)
                return coor_nested, feature_type
                #
                # for line in geometry.geoms:
                #     line_x, line_y = line.xy
                #     x_coords.append(line_x)
                #     y_coords.append(line_y)
                x_mult = []
                y_mult = []
                for line in geom.geoms:
                    coords = np.array(line.xy)
                    x_mult.append(coords[0])
                    y_mult.append(coords[1])
                x_coords.append(x_mult)
                y_coords.append(y_mult)

            elif geom.geom_type == 'Polygon':
                # 处理多边形几何对象的坐标
                # 处理多边形几何对象的外部环坐标
                coords = geom.exterior.coords
                x_coords.append(np.array(coords).T[0, :])
                y_coords.append(np.array(coords).T[1, :])
                # # 处理多边形几何对象的内部环坐标
                # for interior in geom.interiors:
                #     print('是')
                #     interior_coords = np.array(interior.coords)
                #     x_coords.append(interior_coords[:, 0])
                #     y_coords.append(interior_coords[:, 1])

            elif geom.geom_type == 'MultiPolygon':

                x_coords = []
                y_coords = []
                feature_type = []

                for geom in shpfile.geometry:
                    x, y = split_coordinates_by_geometry_type(geom)
                    x_coords.append(x)
                    y_coords.append(y)
                    feature_type.append(geom.geom_type if geom else None)

                coor_nested = np.array([x_coords, y_coords], dtype=object)
                return coor_nested, feature_type


                x_mult = []
                y_mult = []
                for polygon in geom.geoms:
                    coords = polygon.exterior.coords
                    x_mult.append(np.array(coords).T[0, :])
                    y_mult.append(np.array(coords).T[1, :])
                    #     interior_coords = np.array(interior.coords)
                    #     np.append(x_mult, interior_coords[:, 0])
                    #     np.append(y_mult, interior_coords[:, 1])
                x_coords.append(x_mult)
                y_coords.append(y_mult)
            else:
                if geom.geom_type not in feature_type:
                    # 对于空的Point，LineString来说，未写入文件之前 geom.geom_type为geometrycollection
                    print(f"\n存在未解析类型{geom.geom_type}")
            feature_type.append(geom.geom_type)
        coor_nested = np.array([x_coords, y_coords], dtype=object)
    return coor_nested, feature_type


def get_coor_array(coor_nested, shp_type):
    """
    将得到的嵌套数组，分别在下x和y方向进行合并
    :param coor_nested: shp数据顶点的嵌套数组
    :param shp_type: 要素类型的数组
    :return: 返回合并的数组
    """
    x_array = []
    y_array = []
    # 遍历数组中的每个要素
    for i in range(coor_nested.shape[1]):
        try:
            if isinstance(coor_nested[:, i][0][0], np.ndarray):
                if len(coor_nested[:, i][0]) == 0:
                    continue
                else:
                    for j in range(len(coor_nested[:, i][0])):
                        coor_group = np.vstack((coor_nested[:, i][0][j], coor_nested[:, i][1][j]))
                        x_array = np.hstack((x_array, coor_group[0, :]))
                        y_array = np.hstack((y_array, coor_group[1, :]))
            else:
                x_array = np.hstack((x_array, coor_nested[0, i]))
                y_array = np.hstack((y_array, coor_nested[1, i]))
        except IndexError as e:
            print(e)
            if shp_type[i] in ['MultiPolygon', 'MultiLineString']:
                if len(coor_nested[:, i][0]) == 0:
                    continue
                else:
                    for j in range(len(coor_nested[:, i][0])):
                        coor_group = np.vstack((coor_nested[:, i][0][j], coor_nested[:, i][1][j]))
                        x_array = np.hstack((x_array, coor_group[0, :]))
                        y_array = np.hstack((y_array, coor_group[1, :]))
            else:
                x_array = np.hstack((x_array, coor_nested[0, i]))
                y_array = np.hstack((y_array, coor_nested[1, i]))
    coor_array = np.vstack((x_array, y_array))
    return coor_array


def split_coordinates_by_geometry_type(geometry):
    """
    根据几何类型拆分坐标，确保 vstack 前数组一致
    :param geometry: shapely 几何对象
    :return: 坐标数组 [x_coords, y_coords]
    """
    x_coords = []
    y_coords = []

    if geometry is None or geometry.is_empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    geom_type = geometry.geom_type

    if geom_type == "Point":
        x_coords = [geometry.x]
        y_coords = [geometry.y]

    elif geom_type == "LineString":
        x_coords, y_coords = geometry.xy

    elif geom_type == "MultiLineString":
        for line in geometry.geoms:
            line_x, line_y = line.xy
            x_coords.append(line_x)
            y_coords.append(line_y)

    elif geom_type == "Polygon":
        exterior_x, exterior_y = geometry.exterior.xy
        x_coords.append(exterior_x)
        y_coords.append(exterior_y)
        for interior in geometry.interiors:
            interior_x, interior_y = interior.xy
            x_coords.append(interior_x)
            y_coords.append(interior_y)

    elif geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            exterior_x, exterior_y = polygon.exterior.xy
            x_coords.append(exterior_x)
            y_coords.append(exterior_y)
            for interior in polygon.interiors:
                interior_x, interior_y = interior.xy
                x_coords.append(interior_x)
                y_coords.append(interior_y)

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    # 处理单个数值和数组混合的情况
    def ensure_array(data):
        if isinstance(data, (int, float)):
            return np.array([data])  # 转换为数组
        return np.array(data)       # 保留原数组

    # 转换为数组并过滤空数组
    x_coords = [ensure_array(x) for x in x_coords if isinstance(x, (list, np.ndarray)) or np.size(x) > 0]
    y_coords = [ensure_array(y) for y in y_coords if isinstance(y, (list, np.ndarray)) or np.size(y) > 0]

    x_coords = np.concatenate(x_coords) if x_coords else np.array([], dtype=float)
    y_coords = np.concatenate(y_coords) if y_coords else np.array([], dtype=float)

    return x_coords, y_coords




# def split_coordinates_by_geometry_type(geometry):
#     """
#     根据几何类型拆分坐标，确保 vstack 前数组一致
#     :param geometry: shapely 几何对象
#     :return: 坐标数组 [x_coords, y_coords]
#     """
#     x_coords = []
#     y_coords = []
#
#     if geometry is None or geometry.is_empty:
#         return np.array([], dtype=float), np.array([], dtype=float)
#
#     geom_type = geometry.geom_type
#
#     if geom_type == "Point":
#         # Point 类型
#         x_coords = [geometry.x]
#         y_coords = [geometry.y]
#
#     elif geom_type == "LineString":
#         # LineString 类型
#         x_coords, y_coords = geometry.xy
#
#     elif geom_type == "MultiLineString":
#         # MultiLineString 类型
#         for line in geometry.geoms:
#             line_x, line_y = line.xy
#             x_coords.append(line_x)
#             y_coords.append(line_y)
#
#     elif geom_type == "Polygon":
#         # Polygon 类型
#         # 外环
#         exterior_x, exterior_y = geometry.exterior.xy
#         x_coords.append(exterior_x)
#         y_coords.append(exterior_y)
#         # 内环
#         for interior in geometry.interiors:
#             interior_x, interior_y = interior.xy
#             x_coords.append(interior_x)
#             y_coords.append(interior_y)
#
#     elif geom_type == "MultiPolygon":
#         # MultiPolygon 类型
#         for polygon in geometry.geoms:
#             # 外环
#             exterior_x, exterior_y = polygon.exterior.xy
#             x_coords.append(exterior_x)
#             y_coords.append(exterior_y)
#             # 内环
#             for interior in polygon.interiors:
#                 interior_x, interior_y = interior.xy
#                 x_coords.append(interior_x)
#                 y_coords.append(interior_y)
#
#     else:
#         raise ValueError(f"Unsupported geometry type: {geom_type}")
#
#     # 将嵌套列表展平，并保证为 numpy 数组
#     x_coords = np.concatenate([np.array(x) for x in x_coords]) if x_coords else np.array([], dtype=float)
#     y_coords = np.concatenate([np.array(y) for y in y_coords]) if y_coords else np.array([], dtype=float)
#
#     return x_coords, y_coords
