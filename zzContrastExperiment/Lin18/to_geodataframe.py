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
    Writes the updated coordinates pair that embed watermark to geodataframe
    :param dataframe: GeoDataFrame to be updated
    :param index: Row index to be updated
    :param coor_group: Coordinates pair that embed watermark
    :param shpfile_type: The type of shapefile geometry (Point, LineString, etc.)
    :return: Updated GeoDataFrame
    """
    # Ensure the GeoDataFrame has a 'geometry' column
    if 'geometry' not in dataframe.columns:
        dataframe = gpd.GeoDataFrame(dataframe, geometry='geometry')

    if shpfile_type == 'Point':
        if len(coor_group[1]) > 0:
            dataframe.loc[index, 'geometry'] = Point(
                [(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])]
            )
        else:
            dataframe.loc[index, 'geometry'] = Point()

    elif shpfile_type == 'LineString':
        if len(coor_group[1]) > 1:
            dataframe.loc[index, 'geometry'] = LineString(
                [(Decimal(str(x)), Decimal(str(y))) for x, y in zip(coor_group[0], coor_group[1])]
            )
        else:
            dataframe.loc[index, 'geometry'] = LineString()

    elif shpfile_type == 'MultiLineString':
        lines = []
        mult_coor_group = coor_group
        for j in range(mult_coor_group.shape[1]):
            coor_group = np.vstack((np.vstack(mult_coor_group[:, j])))
            if coor_group.shape[1] > 1:
                lines.append(LineString([(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])]))
            else:
                lines.append(LineString())
        non_empty_lines = [line for line in lines if not line.is_empty]
        dataframe.loc[index, 'geometry'] = MultiLineString(non_empty_lines) if non_empty_lines else MultiLineString()

    elif shpfile_type == 'Polygon':
        if len(coor_group[1]) > 2:
            dataframe.loc[index, 'geometry'] = Polygon(
                [(Decimal(str(x)), Decimal(str(y))) for x, y in zip(coor_group[0], coor_group[1])]
            )
        else:
            dataframe.loc[index, 'geometry'] = Polygon()

    elif shpfile_type == 'MultiPolygon':
        polygons = []
        mult_coor_group = coor_group
        for j in range(mult_coor_group.shape[1]):
            coor_group = np.vstack((np.vstack(mult_coor_group[:, j])))
            if coor_group.shape[1] > 2:
                polygons.append(Polygon([(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])]))
            else:
                polygons.append(Polygon())
        dataframe.loc[index, 'geometry'] = MultiPolygon(polygons)
    else:
        print("存在未写入的数组")
    return dataframe


# def to_geodataframe(dataframe, index, coor_group, shpfile_type):
#
#     if 'geometry' not in dataframe.columns:
#         dataframe = gpd.GeoDataFrame(dataframe, geometry='geometry')
#
#     """
#     writes the updated coordinates pair that embed watermark to geodataframe
#     :param coor_group: coordinates pair that embed watermark
#     :return: a geodataframe
#     """
#     if shpfile_type == 'Point':
#         if len(coor_group[1]) > 0:
#             # dataframe['geometry'][index] = Point(list(zip(coor_group[0], coor_group[1])))
#             dataframe.loc['geometry'][index] = Point(
#                 [(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])])
#         else:
#             dataframe.loc['geometry'][index] = Point()
#
#     elif shpfile_type == 'LineString':
#         if len(coor_group[1]) > 1:
#             # dataframe['geometry'][index] = LineString(list(zip(coor_group[0], coor_group[1])))
#             dataframe.loc['geometry'][index] = LineString([(Decimal(str(x)), Decimal(str(y))) for x, y in zip(coor_group[0], coor_group[1])])
#         else:
#             dataframe.loc['geometry'][index] = LineString()
#
#     elif shpfile_type == 'MultiLineString':
#         lines = []
#         mult_coor_group = coor_group
#         for j in range(mult_coor_group.shape[1]):
#             coor_group = np.vstack((np.vstack(mult_coor_group[:, j])))
#             if coor_group.shape[1] > 1:
#                 # lines.append(LineString(list(zip(coor_group[0], coor_group[1]))))
#                 lines.append(LineString([(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])]))
#             else:
#                 lines.append(LineString())
#         # 过滤掉空的 LineString
#         non_empty_lines = [line for line in lines if not line.is_empty]
#         if non_empty_lines:
#             dataframe.loc['geometry'][index] = MultiLineString(non_empty_lines)
#         else:
#             dataframe.loc['geometry'][index] = MultiLineString()
#
#     elif shpfile_type == 'Polygon':
#         if len(coor_group[1]) > 2:
#             # dataframe['geometry'][index] = Polygon(list(zip(coor_group[0], coor_group[1])))
#             dataframe.loc['geometry'][index] = Polygon([(Decimal(str(x)), Decimal(str(y))) for x, y in zip(coor_group[0], coor_group[1])])
#         else:
#             dataframe.loc['geometry'][index] = Polygon()
#
#     elif shpfile_type == 'MultiPolygon':
#         polygons = []
#         mult_coor_group = coor_group
#         for j in range(mult_coor_group.shape[1]):
#             coor_group = np.vstack((np.vstack(mult_coor_group[:, j])))
#             if coor_group.shape[1] > 2:
#                 # polygons.append(Polygon(list(zip(coor_group[0], coor_group[1]))))
#                 polygons.append(Polygon([(Decimal(x), Decimal(y)) for x, y in zip(coor_group[0], coor_group[1])]))
#             else:
#                 polygons.append(Polygon())
#         dataframe.loc['geometry'][index] = MultiPolygon(polygons)
#     else:
#         print("存在未写入的数组")
#     return dataframe


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
