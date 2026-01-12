# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 15:40
# @Author  :Fivem
# @File    : NC.py
# @Software: PyCharm
# @last modified:2023/11/8 15:40

import numpy as np
from PIL import Image


def image_to_array(path):
    """加载水印图像，规范为 32x32 且二值(0/1)。"""
    image = Image.open(path).convert('L').resize((32, 32))
    arr = np.array(image)
    # 二值化为 0/1
    bin_arr = (arr > 127).astype(int)
    return bin_arr


def NC(original_watermark, extract_watermark):
    """
    calculate normalized correlation(NC)
    :param original_watermark:
    :param extract_watermark:
    :return:
    """
    # 检查图像形状
    if original_watermark.shape != extract_watermark.shape:
        exit('Input watermark must be the same size!')
    # 检查图像是否为二进制图像（若不是则尝试阈值化）
    elif (~np.all((original_watermark == 0) | (original_watermark == 1)) or
          ~np.all((extract_watermark == 0) | (extract_watermark == 1))):
        original_watermark = (original_watermark > 0).astype(int)
        extract_watermark = (extract_watermark > 0).astype(int)

    result = np.sum(original_watermark * extract_watermark) / (
            np.sqrt(np.sum(original_watermark ** 2)) * np.sqrt(np.sum(extract_watermark ** 2)))
    return result


if __name__ == "__main__":
    original_watermark = np.array([[0, 1], [0, 1]])
    extract_watermark = np.array([[0, 1], [0, 0]])
    extract_watermark = image_to_array(r"../extract/watermark/delete_MRailways_factor_0.1.png")
    original_watermark = image_to_array(r'Cat32.png')
    print(NC(original_watermark, extract_watermark))
