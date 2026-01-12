from __future__ import annotations

from typing import Tuple
import numpy as np
import pywt


def _spread_with_virtual_vertices(coordinates: np.ndarray) -> np.ndarray:
	"""
	Create sequence v of length 2n where odd indices (0-based even) are the
	original coordinates and even indices (0-based odd) are virtual vertices
	equal to the average of adjacent real vertices, with wrap-around for the
	last virtual vertex.
	Mirrors MATLAB sdwt.m pre-processing.
	"""
	coordinates = np.asarray(coordinates, dtype=float).ravel()
	n: int = coordinates.shape[0]
	v = np.zeros(2 * n, dtype=float)
	v[0::2] = coordinates
	# Fill virtual vertices (between real vertices)
	for i in range(1, 2 * n - 1, 2):
		v[i] = 0.5 * (v[i - 1] + v[i + 1])
	# Last virtual vertex wraps around
	v[-1] = 0.5 * (v[0] + v[-2])
	return v


def sdwt(coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Semi-discrete wavelet transform used in the paper/Matlab code.
	Returns (L, H) as in MATLAB: [L, H] = dwt(v, 'haar') with v constructed
	by inserting virtual vertices between each pair of coordinates.
	"""
	v = _spread_with_virtual_vertices(coordinates)
	L, H = pywt.dwt(v, 'haar')
	return np.asarray(L, dtype=float), np.asarray(H, dtype=float)


def isdwt(L: np.ndarray, H: np.ndarray) -> np.ndarray:
	"""
	Inverse of sdwt: perform idwt(L, H, 'haar') to reconstruct the doubled
	sequence, then take every other sample (original real vertices positions).
	Mirrors MATLAB isdwt.m behavior.
	"""
	L = np.asarray(L, dtype=float).ravel()
	H = np.asarray(H, dtype=float).ravel()
	scoord = pywt.idwt(L, H, 'haar')
	# Extract original coordinates (odd indices in MATLAB → 0-based even here)
	n = L.shape[0]
	coords = np.asarray(scoord, dtype=float)[0 : 2 * n : 2]
	return coords
