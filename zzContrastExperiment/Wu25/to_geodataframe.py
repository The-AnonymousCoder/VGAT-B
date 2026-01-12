# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 10:52
# @Author  :Fivem
# @File    : to_geodataframe.py
# @Software: PyCharm
# @last modified:2023/11/15 10:52
from decimal import Decimal

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, MultiLineString


def to_geodataframe(dataframe, index, coor_group, shpfile_type):
    """
    writes the updated coordinates pair that embed watermark to geodataframe
    :param coor_group: coordinates pair that embed watermark
    :return: a geodataframe
    """
    if shpfile_type == 'Point':
        if len(coor_group) > 0:
            # dataframe['geometry'][index] = Point(list(zip(coor_group[0], coor_group[1])))
            dataframe.loc[index, "geometry"] = Point(
                [(Decimal(x), Decimal(y)) for x, y in zip(coor_group[:, 0], coor_group[:, 1])])
        else:
            dataframe.loc[index, "geometry"] = Point()

    elif shpfile_type == 'LineString':
        if len(coor_group) > 1:
            dataframe.loc[index, "geometry"] = LineString(
                [(Decimal(str(x)), Decimal(str(y))) for x, y in zip(coor_group[:, 0], coor_group[:, 1])])
        else:
            dataframe.loc[index, "geometry"] = LineString()

    elif shpfile_type == 'MultiLineString':
        lines = []
        mult_coor_group = coor_group
        for j in range(mult_coor_group.shape[1]):
            coor_group = np.vstack((np.vstack(mult_coor_group[:, j])))
            if coor_group.shape[1] > 1:
                # lines.append(LineString(list(zip(coor_group[0], coor_group[1]))))
                lines.append(LineString([(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])]))
            else:
                lines.append(LineString())
        # 过滤掉空的 LineString
        non_empty_lines = [line for line in lines if not line.is_empty]
        if non_empty_lines:
            dataframe.loc[index, "geometry"] = MultiLineString(non_empty_lines)
        else:
            dataframe.loc[index, "geometry"] = MultiLineString()

    elif shpfile_type == 'Polygon':
        if len(coor_group) > 2:
            # dataframe['geometry'][index] = Polygon(list(zip(coor_group[0], coor_group[1])))
            dataframe.loc[index, "geometry"] = Polygon(
                [(Decimal(str(x)), Decimal(str(y))) for x, y in zip(coor_group[:, 0], coor_group[:, 1])])
        else:
            dataframe.loc[index, "geometry"] = Polygon()

    elif shpfile_type == 'MultiPolygon':
        polygons = []
        mult_coor_group = coor_group
        for j in range(mult_coor_group.shape[1]):
            coor_group = np.vstack((np.vstack(mult_coor_group[:, j])))
            if coor_group.shape[1] > 2:
                # polygons.append(Polygon(list(zip(coor_group[0], coor_group[1]))))
                polygons.append(Polygon([(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])]))
            else:
                polygons.append(Polygon())
        dataframe.loc[index, "geometry"] = MultiPolygon(polygons)
    else:
        print(f"存在未写入的数组{shpfile_type}")
    return dataframe


if __name__ == "__main__":
    gdf = gpd.GeoDataFrame()
    # coorArray = [()]
    # line = Point(coorArray)
    gdf = gdf.append({'geometry': Point()}, ignore_index=True)
    coorArray = [(73.85844152801786, 15.940917473114041)]
    line = Point(coorArray)
    gdf = gdf.append({'geometry': line}, ignore_index=True)
    print(gdf)
    gdf.plot()
    plt.show()
