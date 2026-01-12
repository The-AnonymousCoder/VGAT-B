#!/usr/bin/env python3
"""Batch generate all comparison figures (Fig1-Fig12)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
	script_dir = Path(__file__).parent

	# Discover plotting scripts in this directory (exclude helpers)
	script_paths = sorted(
		p for p in script_dir.glob("*.py")
		if p.name not in ("run_all.py", "common.py", "__init__.py")
	)

	# Prepare output folders
	png_dir = script_dir / "PNG"
	csv_dir = script_dir / "CSV"
	manuscript_png_dir = script_dir.parent / "zzManuscript" / "AcademicGraphs"
	for d in (png_dir, csv_dir, manuscript_png_dir):
		d.mkdir(parents=True, exist_ok=True)

	failed = []
	for script_path in script_paths:
		print(f"\n{'='*60}")
		print(f"Running {script_path.name}...")
		print('='*60)
		try:
			result = subprocess.run([sys.executable, script_path.as_posix()], capture_output=False)
		except Exception as exc:
			print(f"❌ Failed to run {script_path.name}: {exc}")
			failed.append(script_path.name)
			continue

		if result.returncode != 0:
			print(f"❌ {script_path.name} failed with exit code {result.returncode}")
			failed.append(script_path.name)
		else:
			print(f"✅ {script_path.name} completed successfully")

		# After running, collect PNG and CSV outputs produced under this script_dir
		# Move PNGs to PNG dir and copy to manuscript dir; move CSVs to CSV dir.
		for f in script_dir.rglob("*.png"):
			# Skip source PNGs that belong to manuscript or other known input dirs
			try:
				# Only consider files inside script_dir tree
				if script_dir in f.resolve().parents or f.resolve().parent == script_dir:
					dest = png_dir / f.name
					shutil.move(str(f), str(dest))
					# copy to manuscript folder
					shutil.copy2(str(dest), str(manuscript_png_dir / dest.name))
			except Exception:
				continue

		for f in script_dir.rglob("*.csv"):
			try:
				if script_dir in f.resolve().parents or f.resolve().parent == script_dir:
					dest = csv_dir / f.name
					shutil.move(str(f), str(dest))
			except Exception:
				continue

	# Summary
	print(f"\n{'='*60}")
	print("Summary:")
	print('='*60)
	print(f"Total scripts discovered: {len(script_paths)}")
	print(f"Failed: {len(failed)}")
	if failed:
		print("\nFailed scripts:")
		for s in failed:
			print(f"  - {s}")
		return 1
	# Move any remaining PNG/CSV files left in script_dir root into the collected folders
	for f in script_dir.glob("*.png"):
		try:
			dest = png_dir / f.name
			if not dest.exists():
				shutil.move(str(f), str(dest))
				shutil.copy2(str(dest), str(manuscript_png_dir / dest.name))
		except Exception:
			continue
	for f in script_dir.glob("*.csv"):
		try:
			dest = csv_dir / f.name
			if not dest.exists():
				shutil.move(str(f), str(dest))
		except Exception:
			continue

	print(f"\n✅ All discovered scripts executed; PNGs collected in: {png_dir}, CSVs in: {csv_dir}")
	print(f"✅ Copied PNGs to manuscript folder: {manuscript_png_dir}")
	return 0


if __name__ == "__main__":
	sys.exit(main())

