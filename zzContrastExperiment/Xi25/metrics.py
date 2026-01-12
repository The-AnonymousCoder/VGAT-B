from __future__ import annotations

import numpy as np


def nc(mark_get: np.ndarray, mark_prime: np.ndarray) -> float:
	"""Normalized correlation, matching MATLAB NC.m"""
	a = np.asarray(mark_get, dtype=float)
	b = np.asarray(mark_prime, dtype=float)
	if a.shape != b.shape:
		raise ValueError("Input arrays must have the same shape")
	num = float(np.sum(a * b))
	den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
	return num / den if den != 0 else 0.0


def ber(mark_get: np.ndarray, mark_prime: np.ndarray) -> float:
	"""Bit error rate (in percentage) for binary images, matching MATLAB BER.m"""
	a = np.asarray(mark_get).astype(bool)
	b = np.asarray(mark_prime).astype(bool)
	if a.shape != b.shape:
		raise ValueError("Input arrays must have the same shape")
	diff = np.logical_xor(a, b)
	return 100.0 * float(np.sum(diff)) / float(a.size)
