import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
from shapely.affinity import rotate, scale, translate

def attacks7_geometric(shp_file, outshp_file, angle, scale_factor, x_shift, y_shift):
    """
    attacks7_geometric - 对矢量数据执行几何变换攻击（旋转、缩放、平移）

    Parameters:
    shp_file (str): 输入shapefile文件路径
    outshp_file (str): 输出shapefile文件名
    angle (float): 旋转角度（度）
    scale_factor (float): 缩放因子（>1放大，<1缩小）
    x_shift (float): X轴平移距离
    y_shift (float): Y轴平移距离

    Returns:
    str: 保存的shapefile文件路径
    """

    try:
        # 读取shapefile
        gdf = gpd.read_file(shp_file)
    except Exception as e:
        raise ValueError(f"无法读取文件，请检查文件路径: {e}")

    print(f"几何变换攻击: 旋转{angle}°, 缩放{scale_factor}, 平移({x_shift}, {y_shift})")

    def transform_geometry(geom):
        """对单个几何体进行变换"""
        if geom is None or geom.is_empty:
            return geom
        
        try:
            # 计算几何体的质心作为旋转中心
            centroid = geom.centroid
            
            # 应用变换：先旋转，再缩放，最后平移
            # 1. 绕质心旋转
            if angle != 0:
                geom = rotate(geom, angle, origin=centroid, use_radians=False)
            
            # 2. 缩放（以质心为中心）
            if scale_factor != 1.0:
                geom = scale(geom, xfact=scale_factor, yfact=scale_factor, origin=centroid)
            
            # 3. 平移
            if x_shift != 0 or y_shift != 0:
                geom = translate(geom, xoff=x_shift, yoff=y_shift)
            
            return geom
            
        except Exception as e:
            print(f"变换几何体时出错: {e}")
            return geom

    # 对所有几何体进行变换
    transformed_geometries = []
    for geom in gdf.geometry:
        transformed_geom = transform_geometry(geom)
        transformed_geometries.append(transformed_geom)

    # 创建新的GeoDataFrame
    transformed_gdf = gdf.copy()
    transformed_gdf.geometry = transformed_geometries

    # 创建输出目录
    output_dir = os.path.join('attacked', 'geometric')
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名
    output_name = f"geo_r{angle}_s{scale_factor}_t{x_shift}_{y_shift}_{outshp_file}"
    output_path = os.path.join(output_dir, output_name)

    # 保存变换后的shapefile
    transformed_gdf.to_file(output_path)

    print(f"几何变换攻击完成: 文件已保存到 {output_path}")

    return output_path
