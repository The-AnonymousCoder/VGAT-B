# -*- coding: utf-8 -*-
"""
Watermark Embedding Algorithm
基于DWT-SVD-QIM的矢量数据水印嵌入算法
"""

import numpy as np
import geopandas as gpd
from PIL import Image
from douglas import Douglas
from arnold import arnold_make
from dwt import qcybaseddwt_forward, iqcybaseddwt


def embed_watermark(shp_file, watermark_image, output_file=None, 
                    THRatio=0.3, mfactor=1e6, Q=10):
    """
    Embed watermark into vector map
    将水印嵌入矢量地图
    
    Args:
        shp_file: path to input shapefile
        watermark_image: path to watermark image or numpy array
        output_file: path to output shapefile (optional)
        THRatio: threshold ratio for Douglas-Peucker algorithm (default: 0.3)
        mfactor: multiplication factor (default: 1e6)
        Q: quantization step (default: 10)
        
    Returns:
        watermarked_gdf: GeoDataFrame with embedded watermark
        max_error: maximum position error
        mean_error: mean position error
        rmse: root mean square error
    """
    # Read shapefile
    original_gdf = gpd.read_file(shp_file)
    
    # Read and process watermark image
    if isinstance(watermark_image, str):
        w = Image.open(watermark_image).convert('L')
        w = np.array(w)
        # Convert to binary (threshold at 128)
        w = (w > 128).astype(int)
    else:
        w = watermark_image
    
    # Arnold scrambling
    w = arnold_make(w)
    
    size_w = w.shape
    M = size_w[0] * size_w[1]
    
    # Flatten watermark for indexing
    w_flat = w.flatten()
    
    # Create a copy for watermarked data
    watermarked_gdf = original_gdf.copy()
    
    # Process each feature
    for i in range(len(original_gdf)):
        geom = original_gdf.geometry.iloc[i]
        
        # Extract coordinates based on geometry type
        if geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
        elif geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
        elif geom.geom_type == 'MultiLineString' or geom.geom_type == 'MultiPolygon':
            # Skip multi-geometries for now
            continue
        else:
            continue
        
        # Remove NaN values
        mask = ~np.isnan(coords).any(axis=1)
        coords = coords[mask]
        
        # For Polygon, remove the last point (duplicate of first point for closed ring)
        if geom.geom_type == 'Polygon' and len(coords) > 0:
            if np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
        
        if len(coords) < 4:
            continue
        
        xarray = coords[:, 0]
        yarray = coords[:, 1]
        
        # Douglas-Peucker algorithm to extract feature points
        xys = coords
        vertV = np.array([xys[-1, 1] - xys[0, 1], -xys[-1, 0] + xys[0, 0]])
        
        if np.all(vertV == 0):
            continue
        
        baseL = np.abs(np.sum((xys - xys[0]) * vertV / np.linalg.norm(vertV), axis=1))
        TH = np.max(baseL) * THRatio
        
        ps, ids = Douglas(xys, TH)
        
        # Convert to 0-based indexing
        ids = ids - 1
        
        if len(ids) < 4:
            continue
        
        # Convert to complex numbers
        Feature_xys = xarray[ids] + 1j * yarray[ids]
        
        # First level DWT
        L, H = qcybaseddwt_forward(Feature_xys, i)
        
        # Second level DWT on H
        L2, H2 = qcybaseddwt_forward(H, i)
        
        # SVD decomposition
        u_l2, s_l2, vh_l2 = np.linalg.svd(L2.reshape(-1, 1), full_matrices=False)
        u_h2, s_h2, vh_h2 = np.linalg.svd(H2.reshape(-1, 1), full_matrices=False)
        
        # Calculate k
        k = s_l2[0] / s_h2[0]
        km = k * mfactor
        
        # QIM embedding
        p = int(np.floor(km / 100)) % M
        
        if w_flat[p] == 0 and (km % Q) > Q/2:
            km = km - Q/2
        elif w_flat[p] == 1 and (km % Q) < Q/2:
            km = km + Q/2
        
        # Modified k
        k_mod = km / mfactor
        s_l2_mod = s_h2[0] * k_mod
        
        # Reconstruct L2
        L2_mod = u_l2 * s_l2_mod * vh_l2
        L2_mod = L2_mod.flatten()
        
        # Inverse DWT
        H_mod = iqcybaseddwt(L2_mod[:len(H2)], H2)
        Feature_xys_mod = iqcybaseddwt(L, H_mod[:len(L)])
        
        # Extract modified coordinates
        x_mod = np.real(Feature_xys_mod)
        y_mod = np.imag(Feature_xys_mod)
        
        # Update coordinates
        new_coords = coords.copy()
        new_coords[ids, 0] = x_mod[:len(ids)]
        new_coords[ids, 1] = y_mod[:len(ids)]
        
        # Update geometry (using .loc to avoid FutureWarning)
        if geom.geom_type == 'LineString':
            from shapely.geometry import LineString
            watermarked_gdf.loc[i, 'geometry'] = LineString(new_coords)
        elif geom.geom_type == 'Polygon':
            from shapely.geometry import Polygon
            watermarked_gdf.loc[i, 'geometry'] = Polygon(new_coords)
    
    # Calculate errors
    max_error, mean_error, rmse = calculate_errors(original_gdf, watermarked_gdf)
    
    # Save to file if output path is provided
    if output_file:
        # Preserve CRS if it exists, otherwise try to read from .prj file or use WGS84
        if original_gdf.crs is not None:
            watermarked_gdf.crs = original_gdf.crs
        else:
            # Try to read CRS from .prj file
            from pathlib import Path
            from pyproj import CRS as _CRS
            prj_path = Path(shp_file).with_suffix('.prj')
            if prj_path.exists():
                try:
                    crs_obj = _CRS.from_wkt(prj_path.read_text(encoding='utf-8', errors='ignore'))
                    watermarked_gdf.set_crs(crs_obj, allow_override=True, inplace=True)
                except Exception:
                    # Use WGS84 as fallback
                    watermarked_gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)
            else:
                # Use WGS84 as default
                watermarked_gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)
        watermarked_gdf.to_file(output_file)
    
    return watermarked_gdf, max_error, mean_error, rmse


def calculate_errors(original_gdf, watermarked_gdf):
    """
    Calculate position errors between original and watermarked maps
    计算原始地图和嵌入水印后地图的位置误差
    """
    rmse = 0
    n = 0
    max_error_list = []
    mean_error = 0
    
    for i in range(len(original_gdf)):
        orig_geom = original_gdf.geometry.iloc[i]
        water_geom = watermarked_gdf.geometry.iloc[i]
        
        # Extract coordinates
        if orig_geom.geom_type == 'LineString':
            orig_coords = np.array(orig_geom.coords)
            water_coords = np.array(water_geom.coords)
        elif orig_geom.geom_type == 'Polygon':
            orig_coords = np.array(orig_geom.exterior.coords)
            water_coords = np.array(water_geom.exterior.coords)
        else:
            continue
        
        # Remove NaN
        orig_mask = ~np.isnan(orig_coords).any(axis=1)
        water_mask = ~np.isnan(water_coords).any(axis=1)
        
        orig_coords = orig_coords[orig_mask]
        water_coords = water_coords[water_mask]
        
        if len(orig_coords) != len(water_coords):
            continue
        
        # Calculate errors
        dxarray = orig_coords[:, 0] - water_coords[:, 0]
        dyarray = orig_coords[:, 1] - water_coords[:, 1]
        absolute_error_2 = dxarray**2 + dyarray**2
        
        max_err = np.max(np.sqrt(absolute_error_2))
        
        # Skip if error is too large (likely an error in processing)
        if max_err > 10:
            continue
        
        max_error_list.append(max_err)
        rmse += np.sum(absolute_error_2)
        mean_error += np.sum(np.sqrt(absolute_error_2))
        n += len(water_coords)
    
    if n > 0 and len(max_error_list) > 0:
        max_error = np.max(max_error_list)
        mean_error = mean_error / n
        rmse = np.sqrt(rmse / n)
    else:
        max_error = 0
        mean_error = 0
        rmse = 0
    
    return max_error, mean_error, rmse


if __name__ == "__main__":
    # Test watermark embedding
    shp_file = "pso_data/BOUL.shp"
    watermark_file = "Cat32.png"
    output_file = "Embed/embeded_BOUL.shp"
    
    watermarked_gdf, max_err, mean_err, rmse = embed_watermark(
        shp_file, watermark_file, output_file
    )
    
    print(f"Watermark embedded successfully!")
    print(f"Max error: {max_err}")
    print(f"Mean error: {mean_err}")
    print(f"RMSE: {rmse}")

