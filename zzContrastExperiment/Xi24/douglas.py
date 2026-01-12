# -*- coding: utf-8 -*-
"""
Douglas-Peucker Algorithm Implementation
道格拉斯-普克算法：用于曲线简化，提取特征点
"""

import numpy as np


def Douglas(points, tolerance):
    """
    Douglas-Peucker algorithm for line simplification
    
    Args:
        points: numpy array of shape (n, 2), [x, y] coordinates
        tolerance: distance threshold for simplification
        
    Returns:
        ps: simplified points
        ix: indices of retained vertices (1-based to match MATLAB)
    """
    if len(points) == 0:
        return points, np.array([1])
    
    if len(points) == 1:
        return points, np.array([1])
    
    if len(points) == 2:
        # Calculate distance between two points
        d = np.sqrt(np.sum((points[0] - points[1])**2))
        if d <= tolerance:
            ps = np.mean(points, axis=0, keepdims=True)
            ix = np.array([1])
        else:
            ps = points
            ix = np.array([1, 2])
        return ps, ix
    
    # Initialize retention flags
    n_vertices = len(points)
    retained = np.ones(n_vertices, dtype=bool)
    
    def simplify_rec(ixs, ixe):
        """Recursive simplification function"""
        # Check if start and end points are the same
        same_se = np.allclose(points[ixs], points[ixe], atol=1e-10)
        
        if same_se:
            # Calculate distance to start point only
            if ixe - ixs > 1:
                pt = points[ixs+1:ixe]
                d = np.sqrt(np.sum((pt - points[ixs])**2, axis=1))
            else:
                return
        else:
            # Calculate perpendicular distance to line from ixs to ixe
            if ixe - ixs <= 1:
                return
            
            pt = points[ixs+1:ixe] - points[ixs]
            a = points[ixe] - points[ixs]
            
            # Project points onto line direction
            a_norm_sq = np.dot(a, a)
            if a_norm_sq < 1e-10:
                d = np.sqrt(np.sum(pt**2, axis=1))
            else:
                beta = np.dot(pt, a) / a_norm_sq
                b = pt - np.outer(beta, a)
                d = np.sqrt(np.sum(b**2, axis=1))
        
        # Find maximum distance
        if len(d) == 0:
            return
        
        dmax_idx = np.argmax(d)
        dmax = d[dmax_idx]
        ixc = ixs + 1 + dmax_idx
        
        # If maximum distance is less than tolerance, remove intermediate points
        if dmax <= tolerance:
            if ixs != ixe - 1:
                retained[ixs+1:ixe] = False
        else:
            # Recursively simplify segments
            simplify_rec(ixs, ixc)
            simplify_rec(ixc, ixe)
    
    # Start recursive simplification
    simplify_rec(0, n_vertices - 1)
    
    # Get simplified points and indices (1-based to match MATLAB)
    ps = points[retained]
    ix = np.where(retained)[0] + 1  # Convert to 1-based indexing
    
    return ps, ix

