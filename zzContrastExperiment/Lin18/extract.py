# -*- coding: utf-8 -*-
# @Time    : 2024/2/24 21:19
# @Author  :Fivem
# @File    : extract.py
# @Software: PyCharm
# @last modified:2024/2/24 21:19
import math
import os
import sys
from collections import Counter

import geopandas as gpd
import numpy as np
from PIL import Image
try:
    from get_coor import get_coor_nested, get_coor_array
except Exception:
    from .get_coor import get_coor_nested, get_coor_array

try:
    from select_file import select_file
except Exception:
    from .select_file import select_file

try:
    from to_geodataframe import to_geodataframe
except Exception:
    from .to_geodataframe import to_geodataframe
try:
    from BER import BER
    from NC import NC, image_to_array
except Exception:
    # 兼容性回退：在环境缺少 BER/NC 模块时提供最小实现以便调试
    print("[extract] 警告：无法导入 BER/NC 模块，启用本地回退实现进行调试", flush=True)
    def image_to_array(path):
        try:
            im = Image.open(path).convert('L')
            arr = np.array(im) > 127
            return arr.astype(int)
        except Exception:
            return np.zeros((32, 32), dtype=int)

    def NC(original, watermark):
        # 归一化互相关（与项目中其他实现兼容的近似）
        va = original.flatten().astype(float)
        vb = watermark.flatten().astype(float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.sum(va * vb) / (na * nb))

    def BER(original, watermark):
        try:
            a = np.asarray(original).flatten().astype(int)
            b = np.asarray(watermark).flatten().astype(int)
            L = min(a.size, b.size)
            if L == 0:
                return 1.0
            return float(np.sum(a[:L] != b[:L]) / L)
        except Exception:
            return 1.0


def calculate_layer(coordinate, coordinate_l, R, n):
    """
    calculate the serial number of the horizontal and vertical gaps of the non-reference vertex.
    Returns:

    """
    for w in range(2 ** n):
        if R * math.sqrt(w) / 2 ** (n / 2) <= coordinate - coordinate_l < R * math.sqrt(w + 1) / 2 ** (n / 2):
            return int(w)


def watermark_extract(tran_coor, coor_l, R, n, r):
    """
    对坐标嵌入水印
    :param tran_coor: 变换后的坐标
    :param coor_l: 区域左下角顶点的坐标
    :return:返回嵌入水印的坐标
    """
    tran_x, tran_y = tran_coor
    xl, yl = coor_l
    layer_x = calculate_layer(tran_x, xl, R, n)
    layer_y = calculate_layer(tran_y, yl, R, n)
    if layer_x is None:
        w = layer_y
        print(layer_x, layer_y)
    elif layer_y is None:
        w = layer_x
        print(layer_x, layer_y)
    else:
        w = max(layer_x, layer_y)
        # print(w)
    Rw = R * math.sqrt(w + 1) / 2 ** (n / 2)
    Rw_ = R * math.sqrt(w) / 2 ** (n / 2)
    Tw = R * math.sqrt(w + 1) * (math.sqrt(w + 1) - math.sqrt(w))
    delta_w = R * (math.sqrt(w + 1) - math.sqrt(w)) / 2 ** (n / 2)
    if layer_y == w:
        original_x = R / r * ((tran_x - xl) / Rw - (1 - r) / 2) + xl
        original_y = Tw / r * ((tran_y - yl - Rw_) / delta_w - (1 - r) / 2) + R - Tw + yl
    else:
        original_x = R / r * ((tran_x - xl - Rw_) / delta_w - (1 - r) / 2) + xl
        original_y = (R - Tw) / r * ((tran_y - yl) / Rw_ - (1 - r) / 2) + yl

    return np.vstack([original_x, original_y]), w


def coor_process(coor, vr1, vr2, W, dis, R, n, r, i):
    """
    对坐标进行处理
    :param coor: 需要处理的坐标
    :return: 嵌入水印的坐标
    """
    x, y = coor
    x_r1, y_r1 = vr1
    x_r2, y_r2 = vr2

    coor = ([x - (x_r1 + x_r2) / 2, y - (y_r1 + y_r2) / 2] @ np.vstack([
        [(x_r2 - x_r1) / dis, -(y_r2 - y_r1) / dis], [(y_r2 - y_r1) / dis, (x_r2 - x_r1) / dis]]))
    coor_l = np.floor(coor / R) * R
    original_coor, w = watermark_extract(coor, coor_l, R, n, r)
    original_coor = (original_coor.T @ np.vstack(
        [[(x_r2 - x_r1) / dis, (y_r2 - y_r1) / dis], [-(y_r2 - y_r1) / dis, (x_r2 - x_r1) / dis]]) + [
                         (x_r1 + x_r2) / 2, (y_r1 + y_r2) / 2]).reshape((2, 1))
    W.append(w)
    return original_coor, W


def coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i):
    """
    对坐标组进行处理
    :param coor_group:需要处理的坐标组
    :return: 返回嵌入水印的坐标组
    """

    extract_coor_group = np.array([[], []])
    for coor in coor_group.T:
        if i in indexes:
            extract_coor = coor[:, np.newaxis]
        else:
            extract_coor, W = coor_process(coor, vr1, vr2, W, dis, R, n, r, i)
        extract_coor_group = np.concatenate((extract_coor_group, extract_coor), axis=1)
        i += 1

    # 将nan值替换成原值
    if extract_coor_group.dtype != np.float64:
        extract_coor_group = extract_coor_group.astype(np.float64)
    extract_coor_group[:, np.where(np.isnan(extract_coor_group))[1]] = coor_group[:,
                                                                       np.where(np.isnan(extract_coor_group))[1]]

    # extract_coor_group[:, np.where(np.isinf(extract_coor_group))[1]] = coor_group[:,
    #                                                                    np.where(np.isinf(extract_coor_group))[1]]
    return extract_coor_group, W, i


def traversal_nested_coor_group(coor_nested, feature_type, vr1, vr2, W, dis, R, n, r, indexes, i):
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
        processed_coor_group, W, i = coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i)
        # 如果要素为多面，则需要满足首位顶点的坐标相同
        if (feature_type in ['MultiPolygon', 'Polygon']
                and np.size(processed_coor_group) not in [0, 2]
                and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
            processed_coor_group[:, -1] = processed_coor_group[:, 0]
        processed_x_nested.append(processed_coor_group[0, :])
        processed_y_nested.append(processed_coor_group[1, :])
    return np.array([processed_x_nested, processed_y_nested], dtype=object), W, i


def traversal_coor_group(coor_nested, shp_type, processed_shpfile, vr1, vr2, dis, R, n, r, indexes):
    """
    对所有要素进行遍历
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param shp_type: 每个要素类型组成的数组
    :param processed_shpfile: 处理后的shp文件
    :return: processed_shpfile
    """
    # ----------------定义局部变量----------------------
    i = 0
    W = []
    # 遍历每个几何要素
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])

        feature_type = shp_type[feature_index]
        # 判断是否是多线、多面等的情况
        if isinstance(coor_group[0, 0], np.ndarray):
            processed_coor_group, W, i = traversal_nested_coor_group(coor_group, feature_type, vr1, vr2, W, dis, R, n,
                                                                     r,
                                                                     indexes, i)
        # todo:1
        elif isinstance(coor_nested[:, feature_index][0], list):
            coor_group = np.empty((2, len(coor_nested[:, feature_index][0])), dtype=object)
            # 将列表填充到数组中
            coor_group[0, :] = coor_nested[:, feature_index][0]
            coor_group[1, :] = coor_nested[:, feature_index][1]
            processed_coor_group, W, i = traversal_nested_coor_group(coor_group, feature_type, vr1, vr2, W, dis, R, n,
                                                                     r, indexes, i)
        else:
            processed_coor_group, W, i = coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i)
            # 如果要素为面，则需要满足首尾顶点的坐标相同
            if (feature_type == 'Polygon'
                    and np.size(processed_coor_group) not in [0, 2]
                    and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
                processed_coor_group[:, -1] = processed_coor_group[:, 0]
        # 将改变的要素坐标组更新到geodataframe
        processed_shpfile = to_geodataframe(processed_shpfile, feature_index, processed_coor_group,
                                            shp_type[feature_index])
    return processed_shpfile, W




def extract(watermarked_shpfile_path, original_watermark_path):
    # -------------------------预定义--------------------------------
    n = 4  # 嵌入强度
    tau = 10 ** (-6)  # 精度容差
    r = 0.999  # 约束因子 (0<r<=1)
    side_length = 32

    # 不再使用固定 num 映射；改为自适应重构

    # -------------------------数据读取--------------------------------
    watermarked_shpfile = gpd.read_file(watermarked_shpfile_path)
    watermarked_coor_nested, feature_type = get_coor_nested(watermarked_shpfile)

    # -------------------------数据预处理--------------------------------
    coor_array = get_coor_array(watermarked_coor_nested, feature_type)  # 将嵌套坐标数组合并成一个数组
    # 尝试多种参考顶点 K，以提高对不同几何/顶点分布的鲁棒性
    original_shpfile = watermarked_shpfile.copy()
    W = []
    chosen_K = None
    traversal_exc = None

    # 根据坐标数量动态构建候选 K（至少尝试 1，最多尝试到 5 或者半数）
    max_possible_k = max(1, coor_array.shape[1] // 2)
    max_try_k = min(5, max_possible_k)
    k_candidates = list(range(1, max_try_k + 1))

    for K_try in k_candidates:
        try:
            # 基于 K_try 计算参考顶点索引
            if K_try != coor_array.shape[1] - K_try:
                indexes = (K_try, coor_array.shape[1] - K_try - 1)
            else:
                # 避免相同索引
                indexes = (K_try - 1, K_try)

            vr1 = coor_array[:, indexes[0]]
            vr2 = coor_array[:, indexes[1]]

            # 相关参数计算（防御异常：dis=0 或类型异常时回退下一候选 K）
            dis = float(np.linalg.norm(vr1 - vr2))
            if dis <= 0:
                # 无效距离，尝试下一个 K
                continue
            c = math.sqrt(1 + (2 ** n - 1) ** (-1))
            Maxd = math.sqrt(1 / (c ** 2) * (1 + 1 / (1 + c) ** 2))
            D = max(1, int(math.ceil(dis * Maxd / tau)))  # 分割系数
            R = dis / D  # 原始分块的边长

            try:
                # 调用 traversal，期望返回 (processed_shpfile, W_list)
                processed_shp, W_try = traversal_coor_group(watermarked_coor_nested, feature_type, original_shpfile.copy(),
                                                            vr1, vr2, dis, R, n, r, indexes)
            except Exception as e:
                traversal_exc = e
                W_try = []
            # 诊断输出：记录每个 K_try 的结果长度和是否有异常
            print(f"[extract] K_try={K_try}, len(W_try={len(W_try) if W_try is not None else 'None'}), traversal_exc={repr(traversal_exc)}", flush=True)
            if W_try and len(W_try) > 0:
                W = W_try
                original_shpfile = processed_shp
                chosen_K = K_try
                print(f"[extract] 成功使用 K={K_try} 重构到 W (len={len(W)})", flush=True)
                break
        except Exception as e:
            traversal_exc = e
            continue

    if chosen_K is None:
        # 记录诊断信息，方便定位为何没有成功重构
        print(f"[extract] 未能基于 K 候选 {k_candidates} 重构到有效 W，coor_array.shape={coor_array.shape}", flush=True)
        if traversal_exc is not None:
            print(f"[extract] 最后一次异常: {repr(traversal_exc)}", flush=True)

    # 计算每个数组出现的次数
    watermark = []
    # 若 W 为空则直接回退为零矩阵，避免 ufunc sqrt 错误
    if len(W) == 0:
        watermark = np.zeros((side_length, side_length), dtype=int)
        original_watermark = image_to_array(original_watermark_path)
        # 诊断：输出水印数组摘要
        try:
            ow = original_watermark
            ww = watermark
            print(f"[extract] original_watermark unique={np.unique(ow)}, watermark unique={np.unique(ww)}", flush=True)
            dp = float(np.sum(ow.flatten().astype(float) * ww.flatten().astype(float)))
            na = float(np.linalg.norm(ow.flatten().astype(float)))
            nb = float(np.linalg.norm(ww.flatten().astype(float)))
            print(f"[extract] dot={dp}, na={na}, nb={nb}", flush=True)
        except Exception:
            pass
        nc = NC(original_watermark, watermark)
        ber = BER(original_watermark, watermark)
        # 使用脚本所在目录而非上层目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_name = os.path.join(script_dir, 'extract')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if not os.path.exists(f'{folder_name}/watermark'):
            os.makedirs(f'{folder_name}/watermark')
        output_watermark_path = f'{folder_name}/watermark/{os.path.splitext(os.path.basename(watermarked_shpfile_path))[0]}.png'
        Image.fromarray(watermark.astype(bool)).save(output_watermark_path)
        return watermarked_shpfile_path, {'NC': nc, 'BER': ber}, nc

    # 展开为比特流，并按实际长度自适应投票重构 32x32
    bit_stream = []
    bit_stream.extend([int(digit) for digit in format(w, f"0{n}b")] for w in W)
    bit_stream = [b for chunk in bit_stream for b in chunk]
    bit_stream = np.array(bit_stream, dtype=int)
    L = int(bit_stream.size)
    if L == 0:
        watermark = np.zeros((side_length, side_length), dtype=int)
    else:
        repeat_time = max(1, L // (side_length * side_length))
        voted = []
        for i in range(side_length * side_length):
            indices = [j * side_length * side_length + i for j in range(repeat_time)
                       if (j * side_length * side_length + i) < L]
            if not indices:
                voted.append(0)
                continue
            values = bit_stream[indices]
            counter = Counter(values.tolist())
            voted.append(counter.most_common(1)[0][0])
        watermark = np.array(voted).reshape(side_length, side_length)

    # 评估NC值
    original_watermark = image_to_array(original_watermark_path)
    # 诊断：输出重构水印摘要用于定位 NC 问题
    try:
        ow = original_watermark
        ww = watermark
        print(f"[extract] original_watermark unique={np.unique(ow)}, watermark unique={np.unique(ww)}", flush=True)
        dp = float(np.sum(ow.flatten().astype(float) * ww.flatten().astype(float)))
        na = float(np.linalg.norm(ow.flatten().astype(float)))
        nb = float(np.linalg.norm(ww.flatten().astype(float)))
        print(f"[extract] dot={dp}, na={na}, nb={nb}", flush=True)
        # 诊断：尝试与反向位进行比较，判断是否存在位翻转情况
        try:
            dp_inv = float(np.sum(ow.flatten().astype(float) * (1 - ww).flatten().astype(float)))
            print(f"[extract] dot_inv={dp_inv}", flush=True)
        except Exception:
            dp_inv = None
    except Exception:
        pass
    # 如果反向位更匹配原始水印，则将 watermark 取反后用于评估
    try:
        if dp is not None and dp_inv is not None and dp_inv > dp:
            print("[extract] 检测到 watermark 可能是反向编码，已对 watermark 取反后评估", flush=True)
            watermark = (1 - watermark).astype(int)
    except Exception:
        pass
    nc = NC(original_watermark, watermark)
    ber = BER(original_watermark, watermark)
    # print(f'NC值为{nc},BER值为{ber}')
    error = {'NC': nc, 'BER': ber}

    # -------------------------数据输出--------------------------------
    # 创建文件夹（使用绝对路径，输出到当前脚本目录下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, 'extract')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists(f'{folder_name}/shpfile'):
        os.makedirs(f'{folder_name}/shpfile')

    if not os.path.exists(f'{folder_name}/watermark'):
        os.makedirs(f'{folder_name}/watermark')

    output_shapefile_path = f'{folder_name}/shpfile/{os.path.basename(watermarked_shpfile_path)}'
    # 继承 CRS 后保存，以避免写出时无CRS告警
    try:
        if getattr(watermarked_shpfile, 'crs', None) is not None:
            original_shpfile.set_crs(watermarked_shpfile.crs, allow_override=True, inplace=True)  # type: ignore
    except Exception:
        pass
    original_shpfile.to_file(output_shapefile_path)
    print("Shapefile创建完成，已保存为", output_shapefile_path, flush=True)

    output_watermark_path = f'{folder_name}/watermark/{os.path.splitext(os.path.basename(watermarked_shpfile_path))[0]}.png'
    Image.fromarray(watermark.astype(bool)).save(output_watermark_path)
    print("水印创建完成，已保存为", output_watermark_path, flush=True)

    return output_shapefile_path, error,nc


if __name__ == '__main__':
    # 配置参数 MRailways MBuilding 0  MLanduse MBoundary MRoad MLake1
    # watermarked_shpfile_path = select_file('select the watermarked shpfile', [('shpfile', '*.shp')])
    # watermarked_shpfile_path = r"embed/猫爪32gis_osm_railways_free_1.shp" Boundary00 Building00 Lake1 Landuse01 Railways1 Road01
    # gis_osm_waterways_free_1.shp gis_osm_landuse_a_free_1.shp gis_osm_natural_free_1.shp Boundary.shp BRGA.shp  gis_osm_railways_free_1.shp
    # watermarked_shpfile_path = r"attacked/delete/delete_MRailways_factor_0.5.shp"
    watermarked_shpfile_path = r"attacked/cropped/half_cropped_Mgis_osm_railways_free_1.shp"
    # watermarked_shpfile_path = r"embed/Mgis_osm_railways_free_1.shp"
    print("当前处理的矢量数据为：", os.path.basename(watermarked_shpfile_path))
    # original_watermark_path = select_file('select the watermarked text', [('text', '*.png')])
    original_watermark_path = r'Cat32.png'
    extract(watermarked_shpfile_path, original_watermark_path)
