# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 15:40
# @Author  :Fivem
# @File    : NC.py
# @Software: PyCharm
# @last modified:2023/11/8 15:40
import os
import sys

import numpy as np
from PIL import Image

script = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dirname:上一级路径
path = os.path.join(script, 'vector_process')
sys.path.append(path)
# from ..vector_process.select_file import select_file


def image_to_array(path):
    image = Image.open(path)  # 替换为你的PNG格式二值图像文件路径
    # 转换为NumPy数组
    image_array = np.array(image).astype(int)
    return image_array


def NC(original_watermark, extract_watermark):
    """
    calculate normalized correlation(NC)
    按照NC.m的逻辑实现：NC = fenzi / sqrt(fenmu1 * fenmu2)
    
    :param original_watermark: 原始水印图像数组 (对应NC.m中的mark_prime)
    :param extract_watermark: 提取水印图像数组 (对应NC.m中的mark_get)
    :return: NC值
    """
    # 检查输入尺寸是否相同 (与NC.m逻辑一致)
    if original_watermark.shape != extract_watermark.shape:
        raise ValueError('Input vectors must be the same size!')
    
    # 获取图像尺寸
    m, n = original_watermark.shape
    
    # 按照NC.m的逻辑计算NC值
    fenzi = 0   # 分子
    fenmu1 = 0  # 分母1：mark_get的平方和
    fenmu2 = 0  # 分母2：mark_prime的平方和
    
    # 遍历所有像素点计算 (与NC.m的双重循环逻辑一致)
    for i in range(m):
        for j in range(n):
            fenzi += extract_watermark[i, j] * original_watermark[i, j]  # mark_get * mark_prime
            fenmu1 += extract_watermark[i, j] ** 2                       # mark_get^2
            fenmu2 += original_watermark[i, j] ** 2                      # mark_prime^2
    
    # 计算NC值：NC = fenzi / sqrt(fenmu1 * fenmu2)
    N = fenzi / np.sqrt(fenmu1 * fenmu2)
    
    return N


if __name__ == "__main__":
    # original_watermark_path = select_file('select the original watermark',
    #                                       [("watermark file", "*.png *.jpg")])  # 读取PNG图像文件
    #
    original_watermark = image_to_array(r'D:\00study\paper\watermark\写作\自适应算数编码\数据\水印\1.png')
    # extract_watermark_path = select_file('select the extract watermark', [("watermark file", "*.png *.jpg")])
    extract_watermark = image_to_array(r"D:\00study\paper\watermark\写作\自适应算数编码\实验\Wang\extract\watermark\1building_vertex_delete_ratio_80.png")
    # original_watermark = np.array([[0,1],[0,1]])
    # extract_watermark = np.array([[0,1],[0,0]])
    print(NC(original_watermark, extract_watermark))
