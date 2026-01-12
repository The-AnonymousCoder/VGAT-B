# -*- coding: utf-8 -*-
# @Time    : 2024/8/7 下午4:52
# @Author  :Fivem
# @File    : embed.py
# @Software: PyCharm
# @last modified:2024/8/7 下午4:52
import os
import random
from decimal import Decimal
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from RMSE import calculation_error

from extract import extract
from utils import calculate_projection_point, get_coordinates_compatible
from to_geodataframe import to_geodataframe
from shapely.geometry.polygon import orient

"""
该文件进行了三种改动
1、高度作为种子 效果最好的一个 唯一缺陷是 存在一个数据的nc为0.7 原因：高度乘的系数太小 
2、面积作为种子 不适用投影坐标系的数据 投影坐标系数据nc值很低
3、投影变换到wgs84 第二个问题解决 但是投影坐标系的数据精度很差

优势：

缺陷：1、对比值改动时，遇见投影坐标系数据会放大误差
     2、如何确保geopandas写入，能够多保留几位小数

改进： 1、对每个要素归一化 抵抗缩放攻击
      2、投影到无坐标系统 抵抗投影变换
      3、所有要素转成点  迎合点地图
"""


def watermark_embed(coord, static_arg, dynamic_arg):
    """
    对坐标嵌入水印
    :param coord: 变换后的坐标
    :param static_arg:
    :param dynamic_arg:
    :return:返回嵌入水印的坐标
    """
    n, R, W = static_arg.values()
    r, w, projection_point = dynamic_arg.values()

    gamma = r // R
    d = r - gamma * R
    embed_r = gamma * R + (w * R + d) / 2 ** n
    return embed_r


def coord_process(coord, static_arg, dynamic_arg):
    """
    对坐标进行处理
    :param coord: 需要处理的坐标
    :param static_arg:
    :param dynamic_arg:
    :return: 嵌入水印的坐标
    """
    embed_coor = watermark_embed(coord, static_arg, dynamic_arg)
    return embed_coor


def coord_df_process(coord_df, static_arg, dynamic_arg):
    """
    对坐标dataframe进行处理
    :param coord_df:需要处理的坐标组
    :param static_arg: 静态参数字典
    :param dynamic_arg: 动态参数字典
    :return: 返回嵌入水印的坐标组
    """
    n, R, W = static_arg.values()
    processed_coord_group = coord_df.copy().values
    for i in range(2, len(processed_coord_group), 2):
        if i == 20:
            pass
        coord = np.array([Decimal(coord) for coord in processed_coord_group[i - 1, :2]])  # 点V(2n+1)
        x, y = coord
        coord1 = np.array([Decimal(coord) for coord in processed_coord_group[i - 2, :2]])  # 点V(2n)
        x1, y1 = coord1
        coord2 = np.array([Decimal(coord) for coord in processed_coord_group[i, :2]])  # 点V(2n+2)
        x2, y2 = coord2
        if np.array_equal(coord1, coord2):
            continue

        mid_coord = (coord1 + coord2) / 2
        projection_point = calculate_projection_point(mid_coord, coord1, coord2, coord)  # 求虚拟顶点

        h1 = np.linalg.norm(coord1 - coord2)  # 点V(2n)和V(2n+2)组成的向量的模长
        h2 = np.linalg.norm(coord - projection_point)  # 点V(2n+1)和V(m)组成的向量的模长
        r = h2 / h1  # 模长的比值
        dynamic_arg['r'] = r

        random.seed(int(r * Decimal(1e8)))  # 以ratio作为种子生成随机数
        # index = random.randint(0, len(W) - 1)  # 计算索引
        index = int(int(r * Decimal(1e8)) % len(W))
        w = int(''.join(map(str, W[index])), 2)  # 将指定索引的水印转换成十进制

        dynamic_arg["w"] = w
        dynamic_arg["projection_point"] = projection_point

        embed_r = coord_process(coord, static_arg, dynamic_arg)
        embed_h2 = embed_r * h1
        direction = (coord - projection_point) / h2
        embed_coord = projection_point + direction * embed_h2

        processed_coord_group[i - 1, :2] = embed_coord

        # # todo:调试
        # error = calculation_error(coord.reshape(1, 2), embed_coord.reshape(1, 2))
        # if error['均方根误差RMSE'] > 3.37e-6:
        #     print(i - 1, error['均方根误差RMSE'])

    processed_coord_df = pd.DataFrame(processed_coord_group, columns=coord_df.columns, index=coord_df.index)
    # processed_coord_df["new_index"] = processed_coord_df["new_index"].astype(int)
    return processed_coord_df


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


def embed(shpfile_path, watermark_path):
    # -------------------------预定义--------------------------------
    n = 4  # 嵌入强度
    # tau = 10 ** (-6)  # 精度容差
    R = Decimal(1e-8)
    # R = 1e-7

    # -------------------------数据读取--------------------------------
    shpfile = gpd.read_file(shpfile_path)

    # 避免投影坐标系的坐标值过大会放大系数，所以给投影坐标系数据再乘以1e-2
    crs = shpfile.crs  # 保存原始坐标系信息
    if crs is not None and getattr(crs, "is_projected", False):
        R *= Decimal(1e-2)

    original_coord_df = get_coordinates_compatible(shpfile)
    original_coord_array = np.array(original_coord_df)
    watermark = np.array(Image.open(watermark_path)).astype(int)

    # -------------------------数据预处理--------------------------------
    # 水印处理
    watermark = list(watermark.flatten())  # 将水印图像矩阵转为一维数组
    watermark += [0] * ((n - len(watermark) % n) % n)  # 对长度不足n的子数组补0
    # 对一维数组进行分组，每n个为一组
    W = np.array_split(watermark, len(watermark) // n)

    # # 矢量数据处理
    # shpfile = shpfile.to_crs(epsg=4326)  # 将坐标系转换为WGS 84 (EPSG:4326)
    processed_shpfile = shpfile.copy()

    # 坐标dataframe处理
    # 使用 groupby 和 cumcount 创建新列
    coord_df = get_coordinates_compatible(processed_shpfile).copy()
    # coord_df['new_index'] = coord_df.groupby(level=0).cumcount()
    # todo:不排序误差小；排序并嵌入水印后，其顺序会打乱
    # coord_df = coord_df.sort_values(by='x')

    # 参数字典处理
    static_arg = {'n': n, 'R': R, 'W': W}
    dynamic_arg = {}

    # -------------------------水印嵌入--------------------------------
    # Go through each object
    processed_coord_df = coord_df_process(coord_df, static_arg, dynamic_arg)
    # 坐标dataframe重排序
    # processed_coord_df = processed_coord_df.sort_index().groupby(level=0).apply(
    #     lambda x: x.sort_values('new_index')).reset_index(level=0, drop=True)
    processed_shpfile = traversal_feature(processed_coord_df, processed_shpfile)

    # # 将坐标系转换回原始坐标系
    # processed_shpfile = processed_shpfile.to_crs(crs)

    # # -------------------------计算误差--------------------------------
    # processed_coord_array = np.array(processed_shpfile.get_coordinates())
    # error = calculation_error(original_coord_array, processed_coord_array)
    # print(error)

    # -------------------------数据输出--------------------------------
    # 输出到模块目录下的 embed 文件夹，避免依赖当前工作目录
    module_dir = Path(__file__).resolve().parent
    embed_dir = module_dir / 'embed'
    embed_dir.mkdir(parents=True, exist_ok=True)
    output_shapefile_path = str(embed_dir / f'{os.path.splitext(os.path.basename(watermark_path))[0]}_{os.path.basename(shpfile_path)}')

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

    # Save GeoDataFrame as Shapefile
    processed_shpfile.to_file(output_shapefile_path, index=False)
    print("Shapefile创建完成，已保存为", output_shapefile_path)

    # # -------------------------对比绘制--------------------------------
    # # 创建图形和坐标轴
    # fig, ax = plt.subplots(figsize=(10, 10))
    # # 绘图
    # shpfile.plot(ax=ax, color='red', linewidth=3)
    # processed_shpfile.plot(ax=ax, color='blue', linewidth=1.5)
    # # 添加图例
    # ax.legend(['Origin', 'Watermarking'])
    # # 显示图形
    # plt.show()

    return output_shapefile_path


if __name__ == "__main__":
    # 使用指定的数据路径
    original_shpfile_path = "pso_data/Boundary.shp"
    original_watermark_path = "Cat32.png"

    print("当前处理的矢量数据为：", os.path.basename(original_shpfile_path))
    print("当前使用的水印图片为：", os.path.basename(original_watermark_path))
    
    watermarked_shapefile_path = embed(original_shpfile_path, original_watermark_path)
    print(f"嵌入水印完成，文件已保存到：{watermarked_shapefile_path}")
