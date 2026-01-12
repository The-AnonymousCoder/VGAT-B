import os
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon

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

    def add_vertices_to_coords(coords):
        """为坐标列表添加顶点"""
        if len(coords) < 2:
            return coords
            
        new_coords = [coords[0]]
        
        # 遍历顶点对
        for j in range(1, len(coords)):
            prev = coords[j-1]
            curr = coords[j]

            # 决定是否添加新点
            if np.random.random() < addRatio:
                # 计算新点位置
                rd2 = np.random.random()
                new_x = prev[0] + rd2 * (curr[0] - prev[0]) + strength * tolerance
                new_y = prev[1] + rd2 * (curr[1] - prev[1]) - strength * tolerance
                new_coords.append((new_x, new_y))

            # 添加当前点
            new_coords.append(curr)
            
        return new_coords

    new_geoms = []
    for geom in gdf.geometry:
        # 检查几何体是否为 None
        if geom is None or geom.is_empty:
            new_geoms.append(geom)
            continue

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            new_coords = add_vertices_to_coords(coords)
            new_geoms.append(LineString(new_coords))
            
        elif geom.geom_type == 'MultiLineString':
            new_lines = []
            for line in geom.geoms:
                coords = list(line.coords)
                new_coords = add_vertices_to_coords(coords)
                new_lines.append(LineString(new_coords))
            new_geoms.append(MultiLineString(new_lines))
            
        elif geom.geom_type == 'Polygon':
            # 处理外环
            exterior_coords = list(geom.exterior.coords)
            new_exterior_coords = add_vertices_to_coords(exterior_coords)
            
            # 处理内环
            new_interiors = []
            for interior in geom.interiors:
                interior_coords = list(interior.coords)
                new_interior_coords = add_vertices_to_coords(interior_coords)
                new_interiors.append(new_interior_coords)
                
            new_geoms.append(Polygon(new_exterior_coords, new_interiors))
            
        elif geom.geom_type == 'MultiPolygon':
            new_polygons = []
            for poly in geom.geoms:
                # 处理外环
                exterior_coords = list(poly.exterior.coords)
                new_exterior_coords = add_vertices_to_coords(exterior_coords)
                
                # 处理内环
                new_interiors = []
                for interior in poly.interiors:
                    interior_coords = list(interior.coords)
                    new_interior_coords = add_vertices_to_coords(interior_coords)
                    new_interiors.append(new_interior_coords)
                    
                new_polygons.append(Polygon(new_exterior_coords, new_interiors))
            new_geoms.append(MultiPolygon(new_polygons))
            
        else:
            # 对于其他几何类型（如Point），保持不变
            new_geoms.append(geom)

    # 保留所有原始属性并更新几何
    new_gdf = gdf.copy()
    new_gdf.geometry = new_geoms

    # 生成输出路径：若传入绝对/带目录，则尊重之；否则写入默认目录并加前缀
    if os.path.isabs(outshpfile) or os.path.dirname(outshpfile):
        output_path = outshpfile
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        output_dir = os.path.join('attacked', 'add')
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(outshpfile)
        base = f'add_s{strength}_ratio{addRatio}_{base}'
        output_path = os.path.join(output_dir, base)

    # 保存文件
    new_gdf.to_file(output_path)
    return output_path
