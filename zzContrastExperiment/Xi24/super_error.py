# -*- coding: utf-8 -*-
"""
SuperError - Calculate position errors between original and watermarked vector maps
计算原始矢量地图和嵌入水印后矢量地图的位置误差
"""

import numpy as np
import geopandas as gpd


def SuperError(original_gdf, watermarked_gdf):
    """
    Calculate position errors between original and watermarked vector maps
    计算原始矢量地图和嵌入水印后矢量地图的位置误差
    
    Args:
        original_gdf: original GeoDataFrame
        watermarked_gdf: watermarked GeoDataFrame
        
    Returns:
        max_error: maximum position error
        mean_error: mean position error
        rmse: root mean square error
    """
    rmse = 0
    n = 0
    max_error_list = []
    mean_error = 0
    
    # Process each feature
    for i in range(len(original_gdf)):
        # Get geometries
        orig_geom = original_gdf.geometry.iloc[i]
        water_geom = watermarked_gdf.geometry.iloc[i]
        
        # Extract coordinates based on geometry type
        if orig_geom.geom_type == 'LineString':
            orig_coords = np.array(orig_geom.coords)
        elif orig_geom.geom_type == 'Polygon':
            orig_coords = np.array(orig_geom.exterior.coords)
        elif orig_geom.geom_type == 'Point':
            orig_coords = np.array([[orig_geom.x, orig_geom.y]])
        else:
            # Skip unsupported geometry types
            continue
        
        if water_geom.geom_type == 'LineString':
            water_coords = np.array(water_geom.coords)
        elif water_geom.geom_type == 'Polygon':
            water_coords = np.array(water_geom.exterior.coords)
        elif water_geom.geom_type == 'Point':
            water_coords = np.array([[water_geom.x, water_geom.y]])
        else:
            continue
        
        # Remove NaN values
        orig_mask = ~np.isnan(orig_coords).any(axis=1)
        water_mask = ~np.isnan(water_coords).any(axis=1)
        
        orig_coords = orig_coords[orig_mask]
        water_coords = water_coords[water_mask]
        
        # Check if coordinate counts match
        if len(orig_coords) != len(water_coords):
            continue
        
        if len(orig_coords) == 0:
            continue
        
        # Calculate coordinate differences
        dxarray = orig_coords[:, 0] - water_coords[:, 0]
        dyarray = orig_coords[:, 1] - water_coords[:, 1]
        
        # Calculate squared errors
        absolute_error_2 = dxarray**2 + dyarray**2
        
        # Calculate error metrics
        sqrt_errors = np.sqrt(absolute_error_2)
        max_err = np.max(sqrt_errors)
        
        # Skip if error is too large (likely an error in processing)
        # This matches the MATLAB logic: if( max(sqrt(absolute_error_2))>10 )
        if max_err > 10:
            continue
        
        # Accumulate statistics
        max_error_list.append(max_err)
        rmse += np.sum(absolute_error_2)
        mean_error += np.sum(sqrt_errors)
        n += len(water_coords)
    
    # Calculate final statistics
    if n > 0 and len(max_error_list) > 0:
        max_error = np.max(max_error_list)
        mean_error = mean_error / n
        rmse = np.sqrt(rmse / n)
    else:
        max_error = 0.0
        mean_error = 0.0
        rmse = 0.0
    
    return max_error, mean_error, rmse


def getSuperError(original_shp_file, watermarked_shp_file):
    """
    Calculate SuperError by reading shapefiles
    通过读取shapefile计算SuperError
    
    Args:
        original_shp_file: path to original shapefile
        watermarked_shp_file: path to watermarked shapefile
        
    Returns:
        max_error: maximum position error
        mean_error: mean position error
        rmse: root mean square error
    """
    # Read shapefiles
    original_gdf = gpd.read_file(original_shp_file)
    watermarked_gdf = gpd.read_file(watermarked_shp_file)
    
    # Calculate errors
    max_error, mean_error, rmse = SuperError(original_gdf, watermarked_gdf)
    
    return max_error, mean_error, rmse


if __name__ == "__main__":
    # Test SuperError calculation
    import sys
    
    if len(sys.argv) >= 3:
        original_shp = sys.argv[1]
        watermarked_shp = sys.argv[2]
        
        max_err, mean_err, rmse = getSuperError(original_shp, watermarked_shp)
        
        print(f"SuperError Results:")
        print(f"  Max Error:  {max_err:.6f}")
        print(f"  Mean Error: {mean_err:.6f}")
        print(f"  RMSE:       {rmse:.6f}")
    else:
        print("Usage: python super_error.py <original_shp> <watermarked_shp>")
        print("\nExample:")
        print("  python super_error.py pso_data/BOUL.shp Embed/embeded_BOUL.shp")

