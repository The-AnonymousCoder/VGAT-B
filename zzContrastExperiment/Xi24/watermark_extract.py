# -*- coding: utf-8 -*-
"""
Watermark Extraction Algorithm
基于DWT-SVD-QIM的矢量数据水印提取算法
"""

import numpy as np
import geopandas as gpd
from PIL import Image
from douglas import Douglas
from arnold import arnold_get
from dwt import qcybaseddwt_forward


def extract_watermark(shp_file, watermark_size, THRatio=0.3, mfactor=1e6, Q=10, use_prior_for_uncovered=False):
    """
    Extract watermark from watermarked vector map
    从嵌入水印的矢量地图中提取水印
    
    Args:
        shp_file: path to watermarked shapefile
        watermark_size: tuple (height, width) of original watermark
        THRatio: threshold ratio for Douglas-Peucker algorithm (default: 0.3)
        mfactor: multiplication factor (default: 1e6)
        Q: quantization step (default: 10)
        
    Returns:
        extracted_watermark: extracted watermark image (numpy array)
    """
    # Read shapefile
    gdf = gpd.read_file(shp_file)
    
    # Get watermark dimensions
    if isinstance(watermark_size, tuple):
        size_h, size_w = watermark_size
    else:
        size_h = size_w = watermark_size
    
    M = size_h * size_w
    
    # Initialize watermark accumulator for each pixel
    ww = {i: [] for i in range(M)}
    
    # Process each feature
    for i in range(len(gdf)):
        geom = gdf.geometry.iloc[i]
        
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
        
        # Extract watermark bit
        p = int(np.floor(km / 100)) % M
        
        if (km % Q) <= Q/2:
            ww[p].append(0)
        else:
            ww[p].append(1)
    
    # Vote for each watermark pixel
    w_E = np.zeros(M, dtype=int)
    
    # Prior probability (watermarks typically have ~45% ones)
    prior_prob_one = 0.45
    
    for i in range(M):
        if len(ww[i]) > 0:
            v = np.sum(ww[i]) / len(ww[i])
            if v < 0.5:
                w_E[i] = 0
            else:
                w_E[i] = 1
        else:
            # For uncovered positions
            if use_prior_for_uncovered:
                # Use prior probability instead of defaulting to 0
                w_E[i] = 1 if np.random.random() < prior_prob_one else 0
            else:
                w_E[i] = 0  # Default behavior
    
    # Reshape to image
    w_E = w_E.reshape(size_h, size_w)
    
    # Inverse Arnold transform
    w_E = arnold_get(w_E)
    
    return w_E


def calculate_nc(extracted_watermark, original_watermark):
    """
    Calculate Normalized Correlation (NC) between extracted and original watermark
    计算提取水印和原始水印的归一化相关系数
    
    This version matches the MATLAB NC.m implementation using XOR logic
    """
    if extracted_watermark.shape != original_watermark.shape:
        raise ValueError('Input vectors must be the same size!')
    
    m, n = extracted_watermark.shape
    
    # Calculate using XOR logic (matching MATLAB implementation)
    # fenzi = 1 - xor(mark_get, mark_prime)
    fenzi = 1 - np.logical_xor(extracted_watermark, original_watermark)
    fenzi = np.sum(fenzi)
    
    # NC = fenzi / (m * n)
    nc = fenzi / (m * n)
    
    return nc


if __name__ == "__main__":
    # Test watermark extraction
    watermarked_shp = "Embed/embeded_BOUL.shp"
    original_watermark_path = "Cat32.png"
    
    # Read original watermark
    w_original = Image.open(original_watermark_path).convert('L')
    w_original = np.array(w_original)
    w_original = (w_original > 128).astype(int)
    
    # Extract watermark
    w_extracted = extract_watermark(watermarked_shp, w_original.shape)
    
    # Calculate NC
    nc = calculate_nc(w_extracted, w_original)
    
    print(f"NC value: {nc}")
    
    # Save extracted watermark
    extracted_img = Image.fromarray((w_extracted * 255).astype(np.uint8))
    extracted_img.save("extracted_watermark.png")
    print("Extracted watermark saved as 'extracted_watermark.png'")

