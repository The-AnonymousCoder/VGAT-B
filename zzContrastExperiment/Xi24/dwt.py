# -*- coding: utf-8 -*-
"""
Discrete Wavelet Transform (DWT) Implementation
离散小波变换：用于矢量坐标的多分辨率分解
"""

import numpy as np


def qcybaseddwt_forward(original_signal, index=None):
    """
    Forward discrete wavelet transform
    离散小波变换（正向）
    
    Args:
        original_signal: input signal (1D array or complex array)
        index: index for debugging (optional)
        
    Returns:
        L: low frequency coefficients (approximation)
        H: high frequency coefficients (detail)
    """
    signal = np.array(original_signal, dtype=complex)
    
    # If signal length is odd, pad with last element
    if len(signal) % 2 == 1:
        signal = np.append(signal, signal[-1])
    
    N = len(signal)
    
    # Construct wavelet transformation matrix
    Wn = np.zeros((N, N), dtype=float)
    
    for i in range(N // 2):
        Wn[i, 2*i] = 1
        Wn[i, 2*i + 1] = 1
        j = N // 2 + i
        Wn[j, 2*i] = -1
        Wn[j, 2*i + 1] = 1
    
    # Normalize
    Wn = Wn * np.sqrt(0.5)
    
    # Apply transformation
    try:
        temp_value = Wn @ signal
    except Exception as e:
        if index is not None:
            print(f"Error at index {index}")
        print(f"Signal: {signal}")
        raise e
    
    # Split into low and high frequency components
    L = temp_value[:N // 2]
    H = temp_value[N // 2:]
    
    return L, H


def qcybaseddwt_inverse(L, H):
    """
    Inverse discrete wavelet transform
    离散小波逆变换
    
    Args:
        L: low frequency coefficients
        H: high frequency coefficients
        
    Returns:
        signal: reconstructed signal
    """
    L = np.array(L, dtype=complex)
    H = np.array(H, dtype=complex)
    
    N = len(L) * 2
    
    # Construct wavelet transformation matrix
    Wn = np.zeros((N, N), dtype=float)
    
    for i in range(N // 2):
        Wn[i, 2*i] = 1
        Wn[i, 2*i + 1] = 1
        j = N // 2 + i
        Wn[j, 2*i] = -1
        Wn[j, 2*i + 1] = 1
    
    # Normalize
    Wn = Wn * np.sqrt(0.5)
    
    # Apply inverse transformation
    combined = np.concatenate([L, H])
    signal = np.linalg.inv(Wn) @ combined
    
    return signal


# Alias functions to match MATLAB naming
def qcybaseddwt(arg1, arg2=None):
    """
    Wrapper function that handles both forward and inverse transform
    根据参数数量自动选择正向或逆向变换
    
    If called with one argument: forward transform
    If called with two arguments: inverse transform
    """
    if arg2 is None:
        # This is the inverse transform case: qcybaseddwt(L, H)
        # But we need two arguments, so this shouldn't happen
        raise ValueError("Need two arguments for inverse transform")
    else:
        # Could be forward (signal, index) or inverse (L, H)
        # Check if arg2 is a scalar (index) or array
        if np.isscalar(arg2) or (isinstance(arg2, (int, float))):
            # Forward transform
            return qcybaseddwt_forward(arg1, arg2)
        else:
            # Inverse transform
            return qcybaseddwt_inverse(arg1, arg2)


def iqcybaseddwt(L, H):
    """
    Alias for inverse transform
    逆离散小波变换（与MATLAB命名一致）
    """
    return qcybaseddwt_inverse(L, H)


if __name__ == "__main__":
    # Test DWT
    print("Testing DWT...")
    
    # Test with real signal
    test_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    print(f"Original signal: {test_signal}")
    
    L, H = qcybaseddwt_forward(test_signal)
    print(f"Low freq (L): {L}")
    print(f"High freq (H): {H}")
    
    reconstructed = qcybaseddwt_inverse(L, H)
    print(f"Reconstructed: {reconstructed}")
    print(f"Error: {np.max(np.abs(test_signal - reconstructed.real))}")
    
    # Test with complex signal
    test_complex = np.array([1+2j, 3+4j, 5+6j, 7+8j])
    print(f"\nOriginal complex signal: {test_complex}")
    
    L, H = qcybaseddwt_forward(test_complex)
    print(f"Low freq (L): {L}")
    print(f"High freq (H): {H}")
    
    reconstructed = qcybaseddwt_inverse(L, H)
    print(f"Reconstructed: {reconstructed}")
    print(f"Error: {np.max(np.abs(test_complex - reconstructed))}")

