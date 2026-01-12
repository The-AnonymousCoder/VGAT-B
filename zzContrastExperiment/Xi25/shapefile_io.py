from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import shapefile  # pyshp
import numpy as np


@dataclass
class ShapeRecord:
	X: np.ndarray
	Y: np.ndarray
	Geometry: str
	record: list


def _open_reader_with_fallback(path: str) -> shapefile.Reader:
	# Try common encodings for Chinese DBF files
	encodings = [None, 'utf-8', 'gbk', 'cp936', 'gb2312', 'latin-1']
	last_exc: Exception | None = None
	for enc in encodings:
		try:
			sr = shapefile.Reader(path, encoding=enc) if enc else shapefile.Reader(path)
			# Attempt to read records to trigger decoding
			try:
				_ = sr.records()
			except UnicodeDecodeError:
				# try to relax errors if supported
				try:
					sr.encodingErrors = 'ignore'  # type: ignore[attr-defined]
					_ = sr.records()
				except Exception as e2:
					last_exc = e2
					raise
			return sr
		except Exception as e:
			last_exc = e
			continue
	# Final fallback (ignore errors)
	sr = shapefile.Reader(path, encoding='latin-1')
	try:
		sr.encodingErrors = 'ignore'  # type: ignore[attr-defined]
	except Exception:
		pass
	return sr


def read_shapefile(path: str) -> List[ShapeRecord]:
	sr = _open_reader_with_fallback(path)
	shapes = sr.shapes()
	recs = sr.records()
	result: List[ShapeRecord] = []
	for shp, rec in zip(shapes, recs):
		# Polyline/Polygon assumed: shp.points + shp.parts
		pts = np.asarray(shp.points, dtype=float)
		# Insert NaN between parts to mirror MATLAB behavior
		parts = list(shp.parts) + [len(pts)]
		xs: list = []
		ys: list = []
		for i in range(len(parts) - 1):
			seg = pts[parts[i] : parts[i + 1]]
			if seg.size == 0:
				continue
			xs.extend(seg[:, 0].tolist())
			ys.extend(seg[:, 1].tolist())
			if i < len(parts) - 2:
				xs.append(np.nan)
				ys.append(np.nan)
		geom = shp.shapeTypeName
		result.append(ShapeRecord(X=np.asarray(xs), Y=np.asarray(ys), Geometry=geom, record=list(rec)))
	return result


def write_shapefile(shapes: List[ShapeRecord], out_prefix: str, template_path: str | None = None) -> str:
	# Determine geometry from first record
	geom_name = shapes[0].Geometry if shapes else 'POLYLINE'
	if 'POLY' in geom_name.upper():
		shape_type = shapefile.POLYLINE
	else:
		shape_type = shapefile.POLYLINE
	w = shapefile.Writer(out_prefix, shapeType=shape_type)
	# copy fields if template provided
	if template_path:
		sr = _open_reader_with_fallback(template_path)
		for f in sr.fields[1:]:
			w.field(*f)
	else:
		w.field('ID', 'N', 10, 0)
	for idx, s in enumerate(shapes):
		# Reconstruct parts by splitting on NaN into list of parts
		x = s.X
		y = s.Y
		parts_pts: List[List[Tuple[float, float]]] = []
		current: List[Tuple[float, float]] = []
		for xi, yi in zip(x, y):
			if np.isnan(xi) or np.isnan(yi):
				if current:
					parts_pts.append(current)
					current = []
				continue
			current.append((float(xi), float(yi)))
		if current:
			parts_pts.append(current)
		if not parts_pts:
			# write minimal degenerate line
			parts_pts = [[(0.0, 0.0), (0.0, 0.0)]]
		w.line(parts_pts)
		if template_path:
			w.record(*s.record)
		else:
			w.record(ID=idx + 1)
	w.close()
	return f"{out_prefix}.shp"
