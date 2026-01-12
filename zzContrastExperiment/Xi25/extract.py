from __future__ import annotations

import argparse
from typing import List
import numpy as np
from PIL import Image

# 同时兼容包内/脚本两种导入方式
try:
	from .sdwt import sdwt
	from .logistic import logistic_decrypt
	from .metrics import nc, ber
	from .shapefile_io import read_shapefile
except Exception:
	from sdwt import sdwt  # type: ignore
	from logistic import logistic_decrypt  # type: ignore
	from metrics import nc, ber  # type: ignore
	from shapefile_io import read_shapefile  # type: ignore


def main():
	parser = argparse.ArgumentParser(description='Vector map watermark extraction (Python).')
	parser.add_argument('--in-shp', required=True)
	parser.add_argument('--orig-watermark', required=True)
	parser.add_argument('--R', type=float, default=3e-2)
	parser.add_argument('--Mcof', type=float, default=1e6)
	parser.add_argument('--logistic-init', type=float, default=0.98)
	parser.add_argument('--out-img', default='extracted.png')
	args = parser.parse_args()

	records = read_shapefile(args.in_shp)
	w_img = Image.open(args.orig_watermark).convert('L')
	w_bin = (np.array(w_img) > 127).astype(np.uint8)
	M = w_bin.shape
	watermark_len = int(w_bin.size)
	ww: list[list[int]] = [[] for _ in range(watermark_len)]

	for rec in records:
		x = rec.X
		y = rec.Y
		valid = ~(np.isnan(x) | np.isnan(y))
		x_arr = x[valid]
		y_arr = y[valid]
		if x_arr.size == 0:
			continue
		LX, HX = sdwt(x_arr)
		LY, HY = sdwt(y_arr)
		for n in range(HX.size):
			if HY[n] == 0:
				continue
			ratio = (HX[n] / HY[n]) * args.Mcof
			p = int(np.mod(np.floor(y_arr[n] * 100000.0), watermark_len))
			# ✅ 修复：对ratio取绝对值，避免X轴翻转时HX变号导致判断反转
			bit = 0 if (np.mod(np.abs(ratio), args.R) <= (args.R / 2.0)) else 1
			ww[p].append(bit)

	w2 = np.zeros(watermark_len, dtype=np.uint8)
	for i in range(watermark_len):
		if len(ww[i]) == 0:
			w2[i] = 0
			continue
		v = np.sum(ww[i]) / float(len(ww[i]))
		w2[i] = 0 if v < 0.5 else 1
	w2 = w2.reshape(M[0], M[1])
	w2 = logistic_decrypt(w2, args.logistic_init)

	Image.fromarray((w_bin * 255).astype(np.uint8)).save('orig_watermark_bin.png')
	Image.fromarray((w2 * 255).astype(np.uint8)).save(args.out_img)

	print('NC =', nc(w2, w_bin))
	print('BER =', ber(w2, w_bin))


if __name__ == '__main__':
	main()
