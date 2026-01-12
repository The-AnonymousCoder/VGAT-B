import os
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

def attacks2_vertex_add(watermarkedshp, outshpfile, addRatio, strength, tolerance):
    """
    实现随机增点攻击
    参数:
        watermarkedshp: 含水印的矢量地图文件路径
        outshpfile: 输出文件名
        addRatio: 添加顶点的比例 (0-1)
        strength: 偏移强度
        tolerance: 容差
    返回:
        保存后的文件路径
    """
    # 使用 numpy 随机种子保证可重复性
    np.random.seed(2)  # 替换 random.seed(2)

    # 读取shapefile
    gdf = gpd.read_file(watermarkedshp)

    new_geoms = []
    for geom in gdf.geometry:
        # 检查几何体是否为 None
        if geom is None:
            continue  # 跳过 None 类型的几何体

        if geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        elif geom.geom_type == 'Polygon':
            # 对于Polygon类型，处理其外环和内环
            exterior_coords = list(geom.exterior.coords)
            interiors_coords = [list(ring.coords) for ring in geom.interiors]
            lines = [LineString(exterior_coords)] + [LineString(interior) for interior in interiors_coords]
        else:
            lines = [geom]

        new_lines = []
        for line in lines:
            # 获取坐标点并转换为列表
            coords = list(line.coords)
            if len(coords) < 2:
                new_lines.append(line)
                continue

            # 初始化新坐标列表
            new_coords = [coords[0]]

            # 遍历顶点对
            for j in range(1, len(coords)):
                prev = coords[j-1]
                curr = coords[j]

                # 决定是否添加新点
                if np.random.random() < addRatio:  # 替换 random.random()
                    # 计算新点位置
                    rd2 = np.random.random()  # 替换 random.random()
                    new_x = prev[0] + rd2 * (curr[0] - prev[0]) + strength * tolerance
                    new_y = prev[1] + rd2 * (curr[1] - prev[1]) - strength * tolerance
                    new_coords.append((new_x, new_y))

                # 添加当前点
                new_coords.append(curr)

            # 创建新的LineString
            new_lines.append(LineString(new_coords))

        # 处理MultiLineString
        new_geoms.append(MultiLineString(new_lines) if len(new_lines) > 1 else new_lines[0])

    # 创建新的GeoDataFrame
    new_gdf = gpd.GeoDataFrame(geometry=new_geoms, crs=gdf.crs)

    # 创建输出目录
    output_dir = os.path.join('attacked', 'add')
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名
    output_name = f'add_s{strength}_ratio{addRatio}_{outshpfile}'
    output_path = os.path.join(output_dir, output_name)

    # 保存文件
    new_gdf.to_file(output_path)
    return output_path
