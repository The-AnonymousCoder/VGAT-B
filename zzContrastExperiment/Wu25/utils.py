# -*- coding: utf-8 -*-
# @Time    : 2024/8/9 上午9:51
# @Author  :Fivem
# @File    : utils.py
# @Software: PyCharm
# @last modified:2024/8/9 上午9:51
import math
from decimal import Decimal

import numpy as np


def calculate_projection_point(point, point_a, point_b, point_c):
    """
    已知三角形三个点a,b,c 将 p 投影到通过点p和点c 并且平行于 ab 的直线上
    :param point: 形如np.array([x,y])
    :param point_a:三角形一边的顶点a
    :param point_b:三角形一边的另外一个顶点b
    :param point_c:三角形剩余的顶点c
    :return:投影后的顶点
    """
    # 计算方向向量
    v = point_b - point_a

    # 计算单位方向向量
    unit_v = v / np.linalg.norm(v)

    # 计算从点 p 到点 c 的向量
    pc = point - point_c

    # 将 pc 投影到 v 的方向上，得到投影向量
    projection_length = np.dot(pc, unit_v)
    projection_vector = projection_length * unit_v

    # 计算投影点的坐标
    projection_point = point_c + projection_vector
    return projection_point


def calculate_angle(point1, point2):
    """
    计算向量|p1p2|与x轴的夹角
    :param point1: 起点
    :param point2: 终点
    :return: 夹角（弧度制）
    """
    x1, y1 = point1
    x2, y2 = point2
    # 向量的横向分量和纵向分量
    dx = x2 - x1
    dy = y2 - y1

    # 计算与x轴的夹角，atan2返回值在[-pi, pi]之间
    angle_radians = Decimal(math.atan2(dy, dx))

    # # 将角度转换为度数
    # angle_degrees = math.degrees(angle_radians)
    # print(math.cos(angle_degrees))
    return angle_radians


def point_to_line_Distance(point, point_a, point_b):
    """
    计算point到点 a 和点 b 所在直线的垂直距离
    :param point: 要计算距离的点
    :param point_a: 直线的一端点
    :param point_b: 直线的另一端点
    :return: 垂直距离
    """
    if np.array_equal(point_a, point_b):  # 若直线两端点为同一点,则范围点与直线其中一端点的距离
        distance = (math.sqrt((point_a[0] - point[0]) ** 2 + (point_a[1] - point[1]) ** 2))
    elif point_a[0] == point_b[0]:  # 若直线为垂线
        distance = abs(point[0] - point_a[0])
    else:
        slope = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
        intercept = point_a[1] - slope * point_a[0]
        distance = abs(slope * point[0] - point[1] + intercept) / Decimal(math.sqrt(1 + slope ** 2))
    return distance


def calculate_area(point_a, point_b, point_c):
    """
    计算由三个顶点组成的三角形的面积
    :param point_a:
    :param point_b:
    :param point_c:
    :return:
    """
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c

    # 计算三角形面积 using the determinant method
    area = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / Decimal(2.0))

    return area


def get_coordinates_compatible(gdf):
    """
    兼容不同版本的geopandas的get_coordinates方法
    """
    import pandas as pd
    
    # 尝试使用新版本的get_coordinates方法
    if hasattr(gdf, 'get_coordinates'):
        return gdf.get_coordinates()
    
    # 对于旧版本，手动提取坐标
    coords_list = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Point':
            coords_list.append([idx, geom.x, geom.y])
        elif geom.geom_type == 'LineString':
            for coord in geom.coords:
                coords_list.append([idx, coord[0], coord[1]])
        elif geom.geom_type == 'Polygon':
            for coord in geom.exterior.coords:
                coords_list.append([idx, coord[0], coord[1]])
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                for coord in line.coords:
                    coords_list.append([idx, coord[0], coord[1]])
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                for coord in poly.exterior.coords:
                    coords_list.append([idx, coord[0], coord[1]])
    
    # 转换为DataFrame
    coords_df = pd.DataFrame(coords_list, columns=['index', 'x', 'y'])
    coords_df = coords_df.set_index('index')
    return coords_df


def reconstruct_geometry_from_coords(coord_group, geom_type):
    """
    从坐标数组重构几何对象
    """
    from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
    from decimal import Decimal
    
    if geom_type == 'Point':
        if len(coord_group) > 0:
            return Point([(Decimal(str(x)), Decimal(str(y))) for x, y in coord_group[:1]])
        else:
            return Point()
    
    elif geom_type == 'LineString':
        if len(coord_group) > 1:
            return LineString([(Decimal(str(x)), Decimal(str(y))) for x, y in coord_group])
        else:
            return LineString()
    
    elif geom_type == 'Polygon':
        if len(coord_group) > 2:
            return Polygon([(Decimal(str(x)), Decimal(str(y))) for x, y in coord_group])
        else:
            return Polygon()
    
    elif geom_type in ['MultiLineString', 'MultiPolygon']:
        # 对于Multi类型，需要特殊处理
        # 这里简化处理，将所有坐标作为一个几何对象
        if geom_type == 'MultiLineString':
            if len(coord_group) > 1:
                return MultiLineString([LineString([(Decimal(str(x)), Decimal(str(y))) for x, y in coord_group])])
            else:
                return MultiLineString()
        else:  # MultiPolygon
            if len(coord_group) > 2:
                return MultiPolygon([Polygon([(Decimal(str(x)), Decimal(str(y))) for x, y in coord_group])])
            else:
                return MultiPolygon()
    
    return None


def calculate_circumradius(point_a, point_b, point_c):
    """
    计算由三个顶点组成的三角形的外切园半径
    :param point_a:
    :param point_b:
    :param point_c:
    :return:
    """
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c

    # 计算三角形的边长
    a = Decimal(math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2))  # BC
    b = Decimal(math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2))  # AC
    c = Decimal(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))  # AB

    area = calculate_area(point_a, point_b, point_c)

    # 计算外接圆半径 using the formula R = (abc) / (4 * area)
    if area == 0:
        circumradius = float('inf')  # 特殊情况：如果三个点共线，则面积为零，外接圆半径无穷大
    else:
        circumradius = (a * b * c) / (Decimal(4.0) * area)

    return circumradius
