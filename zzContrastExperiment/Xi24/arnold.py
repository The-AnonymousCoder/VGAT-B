# -*- coding: utf-8 -*-
"""
Arnold Transform and Inverse Arnold Transform
Arnold置乱变换：用于水印图像的预处理，增强安全性
"""

import numpy as np


def arnold_make(img, n=20, a=1, b=1):
    """
    Arnold transformation (scrambling)
    对水印图像进行Arnold置乱
    
    Args:
        img: input image (binary image, numpy array)
        n: number of iterations (default: 20)
        a: Arnold parameter (default: 1)
        b: Arnold parameter (default: 1)
        
    Returns:
        arnolded_img: scrambled image
    """
    h, w = img.shape
    
    # Ensure image is square
    if h != w:
        size = min(h, w)
        img = img[:size, :size]
        print(f"Image resized to {size}x{size} to make it square")
    
    h, w = img.shape
    N = h
    
    result = img.copy()
    
    # Perform Arnold transformation n times
    for iteration in range(n):
        temp = np.zeros((h, w), dtype=img.dtype)
        for y in range(h):
            for x in range(w):
                # Arnold transformation formula
                xx = (x + b * y) % N
                yy = (a * x + (a * b + 1) * y) % N
                temp[yy, xx] = result[y, x]
        result = temp
    
    return result


def arnold_get(img, n=20, a=1, b=1):
    """
    Inverse Arnold transformation (de-scrambling)
    对水印图像进行Arnold逆置乱，恢复原始图像
    
    Args:
        img: scrambled image (numpy array)
        n: number of iterations (default: 20, must match arnold_make)
        a: Arnold parameter (default: 1)
        b: Arnold parameter (default: 1)
        
    Returns:
        recovered_img: recovered original image
    """
    h, w = img.shape
    N = h
    
    result = img.copy()
    
    # Perform inverse Arnold transformation n times
    for iteration in range(n):
        temp = np.zeros((h, w), dtype=img.dtype)
        for y in range(h):
            for x in range(w):
                # Inverse Arnold transformation formula
                xx = ((a * b + 1) * x - b * y) % N
                yy = (-a * x + y) % N
                temp[yy, xx] = result[y, x]
        result = temp
    
    return result


if __name__ == "__main__":
    # Test Arnold transformation
    from PIL import Image
    
    # Create a simple test pattern
    test_img = np.array([[0, 1, 0, 1],
                         [1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [1, 0, 1, 0]], dtype=np.uint8)
    
    print("Original image:")
    print(test_img)
    
    # Scramble
    scrambled = arnold_make(test_img)
    print("\nScrambled image:")
    print(scrambled)
    
    # Recover
    recovered = arnold_get(scrambled)
    print("\nRecovered image:")
    print(recovered)
    
    # Verify
    print("\nImages match:", np.array_equal(test_img, recovered))

