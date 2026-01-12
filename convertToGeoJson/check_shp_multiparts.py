#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 SourceData/TestSet/SHP/H50 下的 SHP 文件几何类型，确保只包含 Multi* 类型。
"""
from pathlib import Path
import sys
try:
    import geopandas as gpd
except Exception as e:
    print("ERROR: geopandas import failed:", e)
    sys.exit(2)

def main():
    p = Path(__file__).parent / "SourceData" / "TestSet" / "SHP" / "H50"
    if not p.exists():
        print("SHP output directory not found:", p)
        return 1
    shps = sorted(p.glob("*.shp"))
    any_single = []
    for s in shps:
        try:
            g = gpd.read_file(str(s))
            types = sorted(set(g.geometry.geom_type.dropna())) if len(g) > 0 else []
            print(s.name, types)
            if any(t in ("Polygon", "LineString", "Point") for t in types):
                any_single.append((s.name, types))
        except Exception as e:
            print("ERR", s.name, e)

    if any_single:
        print()
        print("Files containing singlepart geometries:")
        for n, t in any_single:
            print(n, t)
        return 3
    else:
        print()
        print("All SHP files contain only Multi* geometries (or are empty).")
        return 0

if __name__ == "__main__":
    sys.exit(main())



































