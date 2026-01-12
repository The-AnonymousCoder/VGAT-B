# -*- coding: utf-8 -*-
# @Time    : 2024/2/24 20:34
# @Author  :Fivem
# @File    : embed.py
# @Software: PyCharm
# @last modified:2024/2/24 20:34

import math
import os
import sys

import geopandas as gpd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from get_coor import get_coor_nested, get_coor_array,split_coordinates_by_geometry_type
from select_file import select_file
from to_geodataframe import to_geodataframe
from decimal import Decimal
from shapely.geometry import Polygon

def watermark_embed(tran_coor, w, coor_l, R, n, r):
    """
    对坐标嵌入水印
    :param tran_coor: 变换后的坐标
    :param w: 水印
    :param coor_l: 区域左下角顶点的坐标
    :return:返回嵌入水印的坐标
    """
    x, y = tran_coor
    xl, yl = coor_l
    Rw = R * math.sqrt(w + 1) / 2 ** (n / 2)
    Rw_ = R * math.sqrt(w) / 2 ** (n / 2)
    Tw = R * math.sqrt(w + 1) * (math.sqrt(w + 1) - math.sqrt(w))
    delta_w = R * (math.sqrt(w + 1) - math.sqrt(w)) / 2 ** (n / 2)
    if y - yl >= R - Tw:
        embed_x = Rw * ((x - xl) / R * r + (1 - r) / 2) + xl
        embed_y = delta_w * ((y - yl - R + Tw) / Tw * r + (1 - r) / 2) + yl + Rw_
    else:
        embed_x = delta_w * ((x - xl) / R * r + (1 - r) / 2) + xl + Rw_
        embed_y = Rw_ * ((y - yl) / (R - Tw) * r + (1 - r) / 2) + yl

    return np.vstack([embed_x, embed_y])


def coor_process(coor, vr1, vr2, W, dis, R, n, r, i, t):
    """
    对坐标进行处理
    :param coor: 需要处理的坐标
    :return: 嵌入水印的坐标
    """
    x, y = coor
    x_r1, y_r1 = vr1
    x_r2, y_r2 = vr2
    w = int(''.join(map(str, W[t])), 2)
    # print(w)
    coor = ([x - (x_r1 + x_r2) / 2, y - (y_r1 + y_r2) / 2] @ np.vstack([
        [(x_r2 - x_r1) / dis, -(y_r2 - y_r1) / dis], [(y_r2 - y_r1) / dis, (x_r2 - x_r1) / dis]]))
    coor_l = np.floor(np.array(coor) / R) * R
    # print(coor_l)
    embed_coor = watermark_embed(coor, w, coor_l, R, n, r)
    # print('-------')
    # print(coor,embed_coor)
    # print(coor_l,np.floor(embed_coor[:, 0] / R) * R)
    embed_coor = (embed_coor.T @ np.vstack(
        [[(x_r2 - x_r1) / dis, (y_r2 - y_r1) / dis], [-(y_r2 - y_r1) / dis, (x_r2 - x_r1) / dis]]) + [
                      (x_r1 + x_r2) / 2, (y_r1 + y_r2) / 2]).reshape((2, 1))
    return embed_coor


def coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i, t):
    """
    对坐标组进行处理
    :param coor_group:需要处理的坐标组
    :return: 返回嵌入水印的坐标组
    """
    embed_coor_group = np.array([[], []])
    for coor in coor_group.T:
        if i in indexes:
            embed_coor = coor[:, np.newaxis]
        else:
            embed_coor = coor_process(coor, vr1, vr2, W, dis, R, n, r, i, t)
            t += 1
        embed_coor_group = np.concatenate((embed_coor_group, embed_coor), axis=1)
        i += 1
    return embed_coor_group, i, t


def traversal_nested_coor_group(coor_nested, feature_type, vr1, vr2, W, dis, R, n, r, indexes, i, t):
    """
    对于多线、多面等情况，执行此函数
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param feature_type: 要素的类型
    :return: 返回该要素更新后的嵌套坐标组
    """
    processed_x_nested = []
    processed_y_nested = []
    # 遍历要素中的每个坐标组
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])
        # 对坐标进行平移
        processed_coor_group, i, t = coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i, t)
        # 如果要素为多面，则需要满足首位顶点的坐标相同
        if (feature_type == 'MultiPolygon'
                and np.size(processed_coor_group) not in [0, 2]
                and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
            processed_coor_group[:, -1] = processed_coor_group[:, 0]
        processed_x_nested.append(processed_coor_group[0, :])
        processed_y_nested.append(processed_coor_group[1, :])
    return np.array([processed_x_nested, processed_y_nested], dtype=object), i, t



def traversal_coor_group(coor_nested, shp_type, processed_shpfile, vr1, vr2, W, dis, R, n, r, indexes, i, t):
    """
    对所有要素进行遍历
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param shp_type: 每个要素类型组成的数组
    :param processed_shpfile: 处理后的shp文件
    :return: processed_shpfile
    """
    # 遍历每个几何要素
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])
        feature_type = shp_type[feature_index]
        # 判断是否是多线、多面等的情况
        if isinstance(coor_group[0, 0], np.ndarray):
            processed_coor_group, i, t = traversal_nested_coor_group(coor_group, feature_type, vr1, vr2, W, dis, R, n,
                                                                     r, indexes, i, t)
        else:
            processed_coor_group, i, t = coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i, t)
            # 如果要素为面，则需要满足首尾顶点的坐标相同
            if (feature_type == 'Polygon'
                    and np.size(processed_coor_group) not in [0, 2]
                    and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
                processed_coor_group[:, -1] = processed_coor_group[:, 0]
        # 将改变的要素坐标组更新到geodataframe
        processed_shpfile = to_geodataframe(processed_shpfile, feature_index, processed_coor_group,
                                            shp_type[feature_index])
    return processed_shpfile


def embed(shpfile_path, watermark_path):
    # -------------------------预定义--------------------------------
    n = 4  # 嵌入强度
    tau = 10 ** (-6)  # 精度容差
    r = 0.999  # 约束因子 (0<r<=1)
    K = 1  # 参考顶点选择的密钥
    i = 0  # 用来记录当前索引号 包括参考顶点
    t = 0  # 用来记录当前索引号

    # -------------------------数据读取--------------------------------
    original_shpfile = gpd.read_file(shpfile_path)
    coor_nested, feature_type = get_coor_nested(original_shpfile)
    watermark = np.loadtxt(watermark_path, dtype='int')

    # -------------------------数据预处理--------------------------------
    # 对一维数组进行分组，每n个为一组
    W = np.array_split(watermark, len(watermark) // n)
    coor_array = get_coor_array(coor_nested, feature_type)  # 将嵌套坐标数组合并成一个数组

    # 基于K，获得两个参考顶点
    indexes = (K, coor_array.shape[1] - K - 1) if K != coor_array.shape[1] - K else (K - 1, K)
    # 第一种取法
    vr1 = coor_array[:, indexes[0]]  # 第一个参考顶点
    vr2 = coor_array[:, indexes[1]]  # 第二个参考顶点
    # 相关参数计算
    dis = np.linalg.norm(vr1 - vr2)  # 计算两个参考顶点的距离
    c = math.sqrt(1 + (2 ** n - 1) ** (-1))
    Maxd = math.sqrt(1 / (c ** 2) * (1 + 1 / (1 + c) ** 2))
    D = math.ceil(dis * Maxd / tau)  # 分割系数
    R = dis / D  # 原始分块的边长

    # -------------------------水印嵌入--------------------------------
    # Go through each object
    watermarked_shpfile = original_shpfile.copy()
    watermarked_shpfile = traversal_coor_group(coor_nested, feature_type, watermarked_shpfile, vr1, vr2, W, dis, R, n,
                                               r, indexes, i, t)


    embedded_coor_array = get_coor_array(get_coor_nested(watermarked_shpfile)[0],
                                         get_coor_nested(watermarked_shpfile)[1])
    # 计算误差
    errors = calculation_error(coor_array, embedded_coor_array)
    max_error = np.max(errors)  # 最大误差
    mean_error = np.mean(errors)  # 平均误差
    print(f"最大误差: {max_error}", flush=True)
    print(f"平均误差: {mean_error}", flush=True)

    # # 计算误差
    # error = calculation_error(coor_array, get_coor_array(get_coor_nested(watermarked_shpfile)[0],
    #                                                      get_coor_nested(watermarked_shpfile)[1]))
    # print(error)

    # -------------------------数据输出--------------------------------
    # watermarked_shpfile 已经在 traversal_coor_group() 中更新完成，无需重新创建
    # 确保输出目录存在（使用绝对路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embed_dir = os.path.join(script_dir, 'embed')
    os.makedirs(embed_dir, exist_ok=True)
    output_shapefile_path = os.path.join(embed_dir, f'{os.path.splitext(os.path.basename(watermark_path))[0]}.shp')
    # Save GeoDataFrame as Shapefile
    watermarked_shpfile.to_file(output_shapefile_path)
    print("Shapefile创建完成，已保存为", output_shapefile_path, flush=True)

    # # -------------------------对比绘制--------------------------------
    # # 创建图形和坐标轴
    # fig, ax = plt.subplots(figsize=(10, 10))
    # # 绘图
    # original_shpfile.plot(ax=ax, color='red', linewidth=3)
    # watermarked_shpfile.plot(ax=ax, color='blue', linewidth=1.5)
    # # 添加图例
    # ax.legend(['Origin', 'Watermarking'])
    # # 显示图形
    # plt.show()

    return output_shapefile_path

def calculation_error(original_array, embedded_array):
    """
    计算坐标嵌入前后的误差。
    :param original_array: 原始坐标数组
    :param embedded_array: 嵌入水印后的坐标数组
    :return: 误差数组
    """
    # 检查数组是否为空
    if original_array.size == 0 or embedded_array.size == 0:
        print("Error: One of the coordinate arrays is empty.")
        return np.array([])  # 返回空数组表示错误

    # 检查数组形状是否匹配
    if original_array.shape != embedded_array.shape:
        print(f"Warning: Arrays have different shapes. Aligning arrays...")
        # 对齐数组形状
        min_length = min(original_array.shape[1], embedded_array.shape[1])
        original_array = original_array[:, :min_length]
        embedded_array = embedded_array[:, :min_length]

    # 计算误差
    return np.linalg.norm(original_array - embedded_array, axis=0)



if __name__ == "__main__":
    # -------------------------数据对比--------------------------------
    # shpfile_path = select_file("select shpfile", [("shpfile", "*.shp")])
    # shpfile_path = r'pso_data/Boundary.shp'  Boundary0 Building00 Lake1 Landuse01 Railways1 Road0
    shpfile_path = r'pso_data/gis_osm_railways_free_1.shp'
    print("当前处理的矢量数据为：", os.path.basename(shpfile_path))
    # watermark_path = select_file('select the watermark', [('watermark file', '*.png *.jpg')])
    watermark_path = r'Cat32.png'

    original_shpfile = gpd.read_file(shpfile_path)
    coor_nested, feature_type = get_coor_nested(original_shpfile)
    coor_array = get_coor_array(coor_nested, feature_type)

    # 使用二值化水印 (0/1)，与 extract.py 的 NC 计算保持一致
    wm_img = Image.open(watermark_path).convert('L').resize((32, 32))
    watermark = (np.array(wm_img) > 127).astype(int).flatten()

    repeat_time = (coor_array.shape[1] - 2) * 4 // len(watermark)
    watermark = np.tile(watermark, repeat_time)
    watermark = np.hstack((watermark, watermark[:(coor_array.shape[1] - 2) * 4 % len(watermark)]))
    # 使用绝对路径创建 watermark 目录
    watermark_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'watermark')
    if not os.path.exists(watermark_dir):
        os.makedirs(watermark_dir)
    watermark_path = os.path.join(watermark_dir, f'{os.path.splitext(os.path.basename(watermark_path))[0]+os.path.splitext(os.path.basename(shpfile_path))[0]}.txt')
    np.savetxt(watermark_path, watermark, delimiter='', fmt='%d')
    embed(shpfile_path, watermark_path)

    # # -------------------------对比绘制--------------------------------
    # # 创建图形和坐标轴
    # fig, ax = plt.subplots(figsize=(10, 10))
    # # 绘图
    # original_shpfile.plot(ax=ax, color='red', linewidth=3)
    # watermarked_shpfile.plot(ax=ax, color='blue', linewidth=1.5)
    # # 添加图例
    # ax.legend(['Origin', 'Watermarking'])
    # # 显示图形
    # plt.show()
