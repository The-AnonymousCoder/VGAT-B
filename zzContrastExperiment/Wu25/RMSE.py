# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 19:32
# @Author  :Fivem
# @File    : RMSE.py
# @Software: PyCharm
# @last modified:2023/1/15 19:32
import glob
import math
import os
import sys

import geopandas as gpd
import numpy as np

script = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dirname:上一级路径
path = os.path.join(script, 'vector_process')
sys.path.append(path)
from get_coor import get_coor_nested, get_coor_array
from select_file import select_file, select_folder


def calculation_error(first_coor_array, second_coor_array):
    """

    Args:
        first_coor_array:
        second_coor_array:

    Returns:

    """
    array = (first_coor_array - second_coor_array) ** 2
    distances = array[:, 0] + array[:, 1]
    MSE = sum(distances) / len(distances)  # 均方误差
    RMSE = math.sqrt(MSE)  # 均方根误差
    distances = distances.astype('float64')
    Max_Error = max(np.sqrt(distances))  # 最大误差
    error = {'均方误差MSE': MSE, '均方根误差RMSE': RMSE, '最大误差Max_Error': Max_Error}
    return error


if __name__ == '__main__':
    """
    get coordinates array of original shpfile
    """
    # get path of shpfile
    original_shpfile_path = select_file('select original shpfile', [("shpfile", '*.shp')])
    # read shpfile
    original_shpfile = gpd.read_file(original_shpfile_path)
    # get nested coordinate array
    original_coor_nested, feature_type = get_coor_nested(original_shpfile)
    # Merge all arrays for each row into one array
    original_coor_array = get_coor_array(original_coor_nested, feature_type)
    ## -----------------------单个文件对比—-----------------------------
    # """
    # get coordinates array of embed shpfile
    # """
    # embed_shpfile_path = select_file('select embed shpfile', [("shpfile", '*.shp')])
    # embed_shpfile = gpd.read_file(embed_shpfile_path)
    # embed_coor_nested, feature_type = get_coor_nested(embed_shpfile)
    # embed_coor_array = get_coor_array(embed_coor_nested, feature_type)
    # # for i in range(np.shape(embed_coor_array)[1]):
    # #     print(original_coor_array[:,i],embed_coor_array[:,i])
    # error = calculation_error(original_coor_array, embed_coor_array)
    # print(error)

    # ------------------对文件夹中所有文件与原始文件对比----------------------------
    folder_path = select_folder()
    # 获取文件夹所有后缀为shp的文件路径 保存为数组
    shapefiles = glob.glob(os.path.join(folder_path, '*.shp'))
    for shpfile_path in shapefiles:
        print('------------------------------------')
        print(f'当前处理的数据是{os.path.basename(shpfile_path)}')
        embed_shpfile = gpd.read_file(shpfile_path)
        embed_coor_nested, feature_type = get_coor_nested(embed_shpfile)
        embed_coor_array = get_coor_array(embed_coor_nested, feature_type)
        error = calculation_error(original_coor_array, embed_coor_array)
        print(error)
