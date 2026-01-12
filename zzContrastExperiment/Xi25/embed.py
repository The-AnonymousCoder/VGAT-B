from __future__ import annotations

import argparse
from typing import List
import numpy as np
from PIL import Image
import os

try:
	from .sdwt import sdwt, isdwt
	from .logistic import logistic_encrypt
	from .shapefile_io import read_shapefile, write_shapefile, ShapeRecord
except Exception:
	# fallback to absolute imports when script is run as top-level module
	from sdwt import sdwt, isdwt
	from logistic import logistic_encrypt
	from shapefile_io import read_shapefile, write_shapefile, ShapeRecord


def _watermark_qim_embed(X: np.ndarray, Y: np.ndarray, w_bits: np.ndarray, R: float, Mcof: float) -> np.ndarray:
	LX, HX = sdwt(X)
	_, HY = sdwt(Y)
	HX_mod = np.zeros_like(HX, dtype=float)
	watermark_len = int(w_bits.size)
	for n in range(HX.size):
		if HY[n] == 0:
			HX_mod[n] = HX[n] * Mcof
			continue
		ratio = (HX[n] / HY[n]) * Mcof
		p = int(np.mod(np.floor(Y[n] * 100000.0), watermark_len))
		# 使用 ratio 的绝对值判断余数位置，避免符号导致的判断反转
		mod_abs = np.mod(np.abs(ratio), R)
		if w_bits[p] == 0 and mod_abs > (R / 2.0):
			ratio -= (R / 2.0)
		elif w_bits[p] == 1 and mod_abs < (R / 2.0):
			ratio += (R / 2.0)
		HX_mod[n] = ratio * HY[n]
	HX_mod = HX_mod / Mcof
	X_embedded = isdwt(LX, HX_mod)
	return X_embedded


def _watermark_correction(xs: np.ndarray, ys: np.ndarray, w_bits: np.ndarray, R: float, Mcof: float, repeat: int = 3) -> np.ndarray:
	X_corr = xs.copy()
	for _ in range(repeat):
		LXc, HXc = sdwt(X_corr)
		_, HYc = sdwt(ys)
		HXc_mod = np.zeros_like(HXc, dtype=float)
		watermark_len = int(w_bits.size)
		for n in range(HXc.size):
			if HYc[n] == 0:
				HXc_mod[n] = HXc[n] * Mcof
				continue
			ratio = (HXc[n] / HYc[n]) * Mcof
			p = int(np.mod(np.floor(ys[n] * 100000.0), watermark_len))
			mod_abs = np.mod(np.abs(ratio), R)
			if w_bits[p] == 0 and mod_abs > (R / 2.0):
				ratio -= (R / 2.0)
			elif w_bits[p] == 1 and mod_abs < (R / 2.0):
				ratio += (R / 2.0)
			HXc_mod[n] = ratio * HYc[n]
		HXc_mod = HXc_mod / Mcof
		X_corr = isdwt(LXc, HXc_mod)
	return X_corr


def main():
	parser = argparse.ArgumentParser(description='Vector map watermark embedding (Python).')
	parser.add_argument('--in-shp', default='../虚拟顶点算法_Matlab/data/hyda3857.shp')
	parser.add_argument('--watermark', default='../虚拟顶点算法_Matlab/cat32.png')
	parser.add_argument('--out-prefix', default='../虚拟顶点算法_Matlab/Embed/check_')
	parser.add_argument('--R', type=float, default=3e-2)
	parser.add_argument('--Mcof', type=float, default=1e6)
	parser.add_argument('--logistic-init', type=float, default=0.98)
	args = parser.parse_args()

	records = read_shapefile(args.in_shp)
	wm_img = Image.open(args.watermark).convert('L')
	wm_bin = (np.array(wm_img) > 127).astype(np.uint8)
	wm_enc = logistic_encrypt(wm_bin, args.logistic_init).astype(np.uint8)
	watermark_len = int(wm_enc.size)
	w_bits = wm_enc.reshape(-1)

	watermarked: List[ShapeRecord] = []
	for rec in records:
		x = rec.X
		y = rec.Y
		valid = ~(np.isnan(x) | np.isnan(y))
		x_arr = x[valid]
		y_arr = y[valid]
		if x_arr.size == 0:
			watermarked.append(rec)
			continue
		X_emb = _watermark_qim_embed(x_arr, y_arr, w_bits, args.R, args.Mcof)
		X_emb = _watermark_correction(X_emb, y_arr, w_bits, args.R, args.Mcof, repeat=3)
		# rebuild with NaNs in original positions
		x_out = x.copy()
		x_out[valid] = X_emb
		watermarked.append(ShapeRecord(X=x_out, Y=rec.Y.copy(), Geometry=rec.Geometry, record=rec.record))

	stem = os.path.splitext(os.path.basename(args.in_shp))[0]
	out_shp = write_shapefile(watermarked, args.out_prefix + stem, template_path=args.in_shp)
	print('Wrote watermarked shapefile:', out_shp)


if __name__ == '__main__':
	main()