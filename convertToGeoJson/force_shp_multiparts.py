#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将目录下的 SHP 文件中的单部件几何强制转换为对应的多部件几何（Polygon->MultiPolygon 等），
并覆盖写回 SHP（删除旧的 .shp/.shx/.dbf/.prj/.cpg 文件以确保覆盖）。
"""
from pathlib import Path
import sys
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint
except Exception as e:
    print("ERROR: geopandas/shapely import failed:", e)
    sys.exit(2)

def _to_multipart(geom):
    if geom is None:
        return geom
    try:
        gt = geom.geom_type
    except Exception:
        return geom
    if gt == 'Polygon':
        return MultiPolygon([geom])
    if gt == 'LineString':
        return MultiLineString([geom])
    if gt == 'Point':
        return MultiPoint([geom])
    return geom

def process_dir(target_dir: Path):
    if not target_dir.exists():
        print("Target directory not found:", target_dir)
        return 1
    shps = sorted(target_dir.glob("*.shp"))
    for shp in shps:
        try:
            print("Processing", shp.name)
            g = gpd.read_file(str(shp))
            if len(g) == 0:
                print("  empty, skipping")
                continue
            g['geometry'] = g['geometry'].apply(lambda x: _to_multipart(x) if x is not None else x)
            # remove existing shapefile component files to ensure overwrite
            for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
                p = shp.with_suffix(ext)
                try:
                    if p.exists():
                        p.unlink()
                except Exception as e:
                    print("  warning: could not remove", p.name, e)
            # write back
            g.to_file(str(shp), driver='ESRI Shapefile', encoding='utf-8')
            print("  overwritten", shp.name)
        except Exception as e:
            print("  ERROR processing", shp.name, e)
    return 0

def main():
    target = Path(__file__).parent / "SourceData" / "TestSet" / "SHP" / "H50"
    return process_dir(target)

if __name__ == "__main__":
    sys.exit(main())



































