from __future__ import annotations

import numpy as np


def _logistic_sequence(length: int, init: float) -> np.ndarray:
	"""
	Generate the logistic-like sequence used in MATLAB code:
	    l[0] = init
	    l[i] = 1 - 2 * l[i-1]^2
	Returns array of shape (length,).
	"""
	l = np.zeros(length, dtype=float)
	l[0] = float(init)
	for i in range(1, length):
		l[i] = 1.0 - 2.0 * (l[i - 1] * l[i - 1])
	return l


def logistic_encrypt(arr: np.ndarray, init: float) -> np.ndarray:
	"""
	Permutation encryption matching MATLAB logisticE.m.
	Sort the logistic sequence and use the sort indices to permute arr.
	"""
	original_shape = arr.shape
	flat = np.asarray(arr).ravel()
	l = _logistic_sequence(flat.size, init)
	lindex = np.argsort(l, kind='mergesort')  # stable like MATLAB
	encrypted = flat[lindex]
	return encrypted.reshape(original_shape)


def logistic_decrypt(arr: np.ndarray, init: float) -> np.ndarray:
	"""
	Inverse permutation matching MATLAB logisticD.m.
	"""
	original_shape = arr.shape
	flat = np.asarray(arr).ravel()
	l = _logistic_sequence(flat.size, init)
	lindex = np.argsort(l, kind='mergesort')
	decrypted = np.empty_like(flat)
	decrypted[lindex] = flat
	return decrypted.reshape(original_shape)
