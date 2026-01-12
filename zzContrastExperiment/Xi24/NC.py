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
    Calculate Normalized Correlation (NC) for binary images
    按照NC.m的逻辑实现：NC = (1 - XOR) / (m * n)
    用于二值图像的相似度计算
    
    :param original_watermark: 原始水印图像数组 (对应NC.m中的mark_prime)
    :param extract_watermark: 提取水印图像数组 (对应NC.m中的mark_get)
    :return: NC值 (范围在0到1之间，1表示完全相同)
    """
    # 检查输入尺寸是否相同 (与NC.m逻辑一致)
    if original_watermark.shape != extract_watermark.shape:
        raise ValueError('Input vectors must be the same size!')
    
    # 转换为numpy数组
    original_watermark = np.array(original_watermark, dtype=int)
    extract_watermark = np.array(extract_watermark, dtype=int)
    
    # 获取图像尺寸
    m, n = original_watermark.shape
    
    # 按照NC.m的逻辑计算NC值
    # fenzi = 1 - xor(mark_get, mark_prime)
    # fenzi = sum(sum(fenzi))
    # N = fenzi / (m * n)
    fenzi = 1 - np.logical_xor(extract_watermark, original_watermark)
    fenzi = np.sum(fenzi)
    N = fenzi / (m * n)
    
    return float(N)


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
