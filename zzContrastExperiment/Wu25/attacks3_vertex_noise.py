import os
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon

def attacks3_vertex_noise(originshpfile, outshpfile, ratio, strength):
    """
    实现随机噪声攻击
    参数:
        originshpfile: 原始矢量地图文件路径
        outshpfile: 输出文件名
        ratio: 添加噪声的比例 (0-1)
        strength: 噪声强度
    返回:
        保存后的文件路径
    """
    # 设置随机种子保证可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 读取shapefile
    gdf = gpd.read_file(originshpfile)

    def add_noise_to_coords(coords):
        """为坐标列表添加噪声"""
        if len(coords) < 1:
            return coords
            
        new_coords = []
        
        for x, y in coords:
            if random.random() < ratio:
                # 添加随机噪声
                noise_x = (-strength + 2 * strength * random.random()) * 1e-6
                noise_y = (-strength + 2 * strength * random.random()) * 1e-6
                new_x = x + noise_x
                new_y = y + noise_y
                new_coords.append((new_x, new_y))
            else:
                new_coords.append((x, y))
                
        return new_coords

    new_geoms = []
    for geom in gdf.geometry:
        # 检查几何体是否为 None
        if geom is None or geom.is_empty:
            new_geoms.append(geom)
            continue

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            new_coords = add_noise_to_coords(coords)
            new_geoms.append(LineString(new_coords))
            
        elif geom.geom_type == 'MultiLineString':
            new_lines = []
            for line in geom.geoms:
                coords = list(line.coords)
                new_coords = add_noise_to_coords(coords)
                new_lines.append(LineString(new_coords))
            new_geoms.append(MultiLineString(new_lines))
            
        elif geom.geom_type == 'Polygon':
            # 处理外环
            exterior_coords = list(geom.exterior.coords)
            new_exterior_coords = add_noise_to_coords(exterior_coords)
            
            # 处理内环
            new_interiors = []
            for interior in geom.interiors:
                interior_coords = list(interior.coords)
                new_interior_coords = add_noise_to_coords(interior_coords)
                new_interiors.append(new_interior_coords)
                
            new_geoms.append(Polygon(new_exterior_coords, new_interiors))
            
        elif geom.geom_type == 'MultiPolygon':
            new_polygons = []
            for poly in geom.geoms:
                # 处理外环
                exterior_coords = list(poly.exterior.coords)
                new_exterior_coords = add_noise_to_coords(exterior_coords)
                
                # 处理内环
                new_interiors = []
                for interior in poly.interiors:
                    interior_coords = list(interior.coords)
                    new_interior_coords = add_noise_to_coords(interior_coords)
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
        output_dir = os.path.join('attacked', 'noise')
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(outshpfile)
        base = f'noise_s{strength}_r{ratio}_{base}'
        output_path = os.path.join(output_dir, base)

    # 保存文件
    new_gdf.to_file(output_path)
    return output_path