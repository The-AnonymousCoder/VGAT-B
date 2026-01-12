# -*- coding: utf-8 -*-
# @Time    : 2024/8/7 上午10:22
# @Author  :Fivem
# @File    : extract.py
# @Software: PyCharm
# @last modified:2024/8/7 上午10:22


import math
import os
import random
from collections import Counter
from decimal import Decimal

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from BER import BER
from NC import NC, image_to_array
from pathlib import Path

from utils import point_to_line_Distance, calculate_projection_point, get_coordinates_compatible
from to_geodataframe import to_geodataframe
from shapely.geometry.polygon import orient


def watermark_extract(coord, static_arg, dynamic_arg):
    """
    对坐标嵌入水印
    :param coord: 变换后的坐标
    :param static_arg: 静态参数字典
    :param dynamic_arg: 动态参数字典
    :return:返回嵌入水印的坐标
    """
    n, R = static_arg.values()
    W, r, index, projection_point = dynamic_arg.values()

    w = r % R // (R / 2 ** n)
    gamma = r // R
    original_r = (r - gamma * R) * 2 ** n + gamma * R - w * R

    W[index].append(int(w))
    dynamic_arg['W'] = W
    return original_r, dynamic_arg


def coord_process(coord, static_arg, dynamic_arg):
    """
    对坐标进行处理
    :param coord: 需要处理的坐标
    :param static_arg: 静态参数字典
    :param dynamic_arg: 动态参数字典
    :return: 嵌入水印的坐标
    """
    original_coord, dynamic_arg = watermark_extract(coord, static_arg, dynamic_arg)
    return original_coord, dynamic_arg


def coord_df_process(coord_df, static_arg, dynamic_arg):
    """
    对坐标组进行处理
    :param coord_df:需要处理的坐标组
    :param static_arg: 静态参数字典
    :param dynamic_arg: 动态参数字典
    :return: 返回嵌入水印的坐标组
    """
    n, R = static_arg.values()
    W = dynamic_arg["W"]
    processed_coord_group = coord_df.copy().values
    for i in range(2, len(processed_coord_group), 2):
        coord = np.array([Decimal(coord) for coord in processed_coord_group[i - 1, :2]])
        x, y = coord
        coord1 = np.array([Decimal(coord) for coord in processed_coord_group[i - 2, :2]])
        x1, y1 = coord1
        coord2 = np.array([Decimal(coord) for coord in processed_coord_group[i, :2]])
        x2, y2 = coord2
        if np.array_equal(coord1, coord2):
            continue

        mid_coord = (coord1 + coord2) / 2
        projection_point = calculate_projection_point(mid_coord, coord1, coord2, coord)

        h1 = np.linalg.norm(coord1 - coord2)
        h2 = np.linalg.norm(coord - projection_point)
        r = h2 / h1
        dynamic_arg['r'] = r

        random.seed(int(r * Decimal(1e8)))  # 以ratio作为种子生成随机数
        # index = random.randint(0, len(W) - 1)  # 计算索引
        index = int(int(r * Decimal(1e8)) % len(W))

        dynamic_arg["index"] = index
        dynamic_arg["projection_point"] = projection_point

        original_r, dynamic_arg = coord_process(coord, static_arg, dynamic_arg)
        original_h2 = original_r * h1
        direction = (coord - projection_point) / h2
        original_coord = projection_point + direction * original_h2

        processed_coord_group[i - 1, :2] = original_coord

    # print(coor_group)
    # print(original_coord_group)
    processed_coord_df = pd.DataFrame(processed_coord_group, columns=coord_df.columns, index=coord_df.index)
    # processed_coord_df["new_index"] = processed_coord_df["new_index"].astype(int)
    return processed_coord_df, dynamic_arg


def traversal_feature(processed_coord_df, processed_shpfile):
    """
    遍历相同行索引的坐标作为要素，还原到shpfile中
    :param processed_coord_df: 含水印的坐标dataframe
    :param processed_shpfile: shp文件
    :return: processed_shpfile
    """
    processed_coord_df = processed_coord_df.groupby(level=0)
    for idx, group in processed_coord_df:
        processed_coord_group = np.array(group.iloc[:, :2])
        geom_type = processed_shpfile.loc[[idx], :].geom_type[idx]
        # 如果要素为面，则需要满足首尾顶点的坐标相同
        if (geom_type == 'Polygon'
                and np.size(processed_coord_group) not in [0, 2]
                and not np.array_equal(processed_coord_group[0, :], processed_coord_group[-1, 0])):
            processed_coord_group[-1, :] = processed_coord_group[0, :]
        # 将改变的要素坐标组更新到geodataframe
        processed_shpfile = to_geodataframe(processed_shpfile, idx, processed_coord_group,
                                            geom_type)
    return processed_shpfile


def extract(shpfile_path, watermark_path):
    # -------------------------预定义--------------------------------
    n = 4  # 嵌入强度
    # tau = 10 ** (-6)  # 精度容差
    side_length = 32  # 水印图像边长
    R = Decimal(1e-8)
    l = math.ceil((side_length ** 2) / n)  # 水印长度
    W = [[] for _ in range(l)]

    # -------------------------数据读取--------------------------------
    shpfile = gpd.read_file(shpfile_path)

    crs = shpfile.crs  # 保存原始坐标系信息
    if crs is not None and getattr(crs, "is_projected", False):
        R *= Decimal(1e-2)

    coord_df = get_coordinates_compatible(shpfile)
    coord_array = np.array(coord_df)
    watermark = image_to_array(watermark_path)

    # -------------------------数据预处理--------------------------------
    # 矢量数据处理
    # shpfile = shpfile.to_crs(epsg=4326)  # 将坐标系转换为WGS 84 (EPSG:4326)
    processed_shpfile = shpfile.copy()

    # 坐标dataframe处理
    # 使用 groupby 和 cumcount 创建新列
    processed_coord_df = get_coordinates_compatible(processed_shpfile).copy()
    # processed_coord_df['new_index'] = processed_coord_df.groupby(level=0).cumcount()
    # processed_coord_df = processed_coord_df.sort_values(by='x')

    # 参数字典处理
    static_arg = {'n': n, 'R': R}
    dynamic_arg = {'W': W}

    # -------------------------水印提取--------------------------------
    processed_coord_df, dynamic_arg = coord_df_process(processed_coord_df, static_arg, dynamic_arg)
    # 坐标dataframe重排序
    # processed_coord_df = processed_coord_df.sort_index().groupby(level=0).apply(
    #     lambda x: x.sort_values('new_index')).reset_index(level=0,
    #                                                       drop=True)
    processed_shpfile = traversal_feature(processed_coord_df, processed_shpfile)

    # # 将坐标系转换回原始坐标系
    # processed_shpfile = processed_shpfile.to_crs(crs)

    # -------------------------水印处理--------------------------------
    W = dynamic_arg["W"]
    # 计算每个数组出现的次数
    empty_array_flag = 0  # 标志变量，用于跟踪是否存在空数组
    for i in range(len(W)):
        # 使用Counter统计每个数字的出现次数
        counter = Counter(W[i])
        if counter:
            w = counter.most_common(1)[0][0]
            w = 0 if w < 0 else w
            W[i] = [int(digit) for digit in format(w, f"0{n}b")]  # 转换n位二进制
        else:
            empty_array_flag += 1
            W[i] = [0] * n
    # 在循环结束后检查标志变量并打印消息
    if empty_array_flag:
        print(f'存在{empty_array_flag}个空数组，补零')

    W = [item for w in W for item in w]
    W = W[:side_length ** 2]  # 截断多余的长度
    processed_watermark = np.array(W).reshape((side_length, side_length))

    # -------------------------计算误差--------------------------------
    # processed_coor_array = np.array(processed_shpfile.get_coordinates())
    # error = calculation_error(original_coor_array, processed_coor_array)
    # print(error)
    # 评估NC值
    nc = NC(watermark, processed_watermark)
    ber = BER(watermark, processed_watermark)
    print(f'NC值为{nc},BER值为{ber}')
    eva_factor = {'NC': nc, 'BER': ber}

    # -------------------------数据输出--------------------------------
    # 输出到模块目录下的 extract 文件夹，避免依赖当前工作目录
    module_dir = Path(__file__).resolve().parent
    folder_name = module_dir / 'extract'
    (folder_name / 'shpfile').mkdir(parents=True, exist_ok=True)
    (folder_name / 'watermark').mkdir(parents=True, exist_ok=True)

    output_shapefile_path = str(folder_name / 'shpfile' / os.path.basename(shpfile_path))
    # 在写出前，规范多边形环方向，避免潜在的环方向问题
    try:
        processed_shpfile["geometry"] = processed_shpfile.geometry.apply(
            lambda g: orient(g, sign=1.0) if (g is not None and not g.is_empty and g.geom_type in ["Polygon", "MultiPolygon"]) else g
        )
    except Exception:
        pass

    # 若原数据存在坐标系则沿用；若不存在且坐标值像经纬度，则默认设置为WGS84
    crs_to_write = crs
    if crs_to_write is None:
        try:
            minx, miny, maxx, maxy = processed_shpfile.geometry.total_bounds
            if np.isfinite([minx, miny, maxx, maxy]).all():
                if max(abs(minx), abs(maxx)) <= 180 and max(abs(miny), abs(maxy)) <= 90:
                    crs_to_write = "EPSG:4326"
        except Exception:
            pass
    if crs_to_write is not None:
        processed_shpfile = processed_shpfile.set_crs(crs_to_write, allow_override=True)

    # 将GeoDataFrame保存为shp文件
    processed_shpfile.to_file(output_shapefile_path, index=False)
    print("Shapefile创建完成，已保存为", output_shapefile_path)

    output_watermark_path = str(folder_name / 'watermark' / f'{os.path.splitext(os.path.basename(shpfile_path))[0]}.png')
    Image.fromarray(processed_watermark.astype(bool)).save(output_watermark_path)
    print("水印创建完成，已保存为", output_watermark_path)

    # # -------------------------对比绘制--------------------------------
    # plt.subplot(1, 2, 2)
    # plt.imshow(processed_watermark, cmap='gray')
    # plt.title('extract')
    # plt.subplot(1, 2, 1)
    # plt.imshow(watermark, cmap='gray')
    # plt.title('original')
    #
    # # 创建图形和坐标轴
    # fig, ax = plt.subplots(figsize=(10, 10))
    # # 绘图
    # shpfile.plot(ax=ax, color='red', linewidth=3)
    # processed_shpfile.plot(ax=ax, color='blue', linewidth=1.5)
    # # 添加图例
    # ax.legend(['Origin', 'Watermarking'])
    # # 显示图形
    # plt.show()

    return output_shapefile_path, eva_factor


if __name__ == '__main__':
    # 使用指定的数据路径，需要先运行embed.py生成水印数据
    watermarked_shpfile_path = "embed/M_Boundary.shp"  # 嵌入水印后的文件
    original_watermark_path = "Cat32.png"  # 原始水印图片

    print("当前处理的矢量数据为：", os.path.basename(watermarked_shpfile_path))
    print("当前使用的原始水印图片为：", os.path.basename(original_watermark_path))

    extract(watermarked_shpfile_path, original_watermark_path)
