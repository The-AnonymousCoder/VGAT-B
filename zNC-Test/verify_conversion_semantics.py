#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify semantic equivalence between original GDB layers and generated GPKG layers.
Writes a short JSON report per layer and prints a summary.
"""
import json
from pathlib import Path
from collections import Counter

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint
except Exception as e:
    print("Error importing geopandas/shapely:", e)
    raise


def normalize_geom(g):
    if g is None:
        return None
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, LineString):
        return MultiLineString([g])
    if isinstance(g, Point):
        return MultiPoint([g])
    return g


def geom_wkb_key(g):
    ng = normalize_geom(g)
    if ng is None:
        return None
    try:
        return ng.wkb
    except Exception:
        return None


def analyze_pair(orig_gdf, new_gdf):
    report = {}
    report['count_orig'] = int(len(orig_gdf))
    report['count_new'] = int(len(new_gdf))
    # geometry type distribution
    report['types_orig'] = dict(Counter([ (geom.geom_type if geom is not None else None) for geom in orig_gdf.geometry ]))
    report['types_new'] = dict(Counter([ (geom.geom_type if geom is not None else None) for geom in new_gdf.geometry ]))
    # property keys
    report['props_orig'] = sorted(list(orig_gdf.columns))
    report['props_new'] = sorted(list(new_gdf.columns))
    # bbox
    try:
        report['bbox_orig'] = list(map(float, orig_gdf.total_bounds))
    except Exception:
        report['bbox_orig'] = None
    try:
        report['bbox_new'] = list(map(float, new_gdf.total_bounds))
    except Exception:
        report['bbox_new'] = None
    # area sums if polygonal
    try:
        report['area_sum_orig'] = float(sum([g.area for g in orig_gdf.geometry if g is not None]))
    except Exception:
        report['area_sum_orig'] = None
    try:
        report['area_sum_new'] = float(sum([g.area for g in new_gdf.geometry if g is not None]))
    except Exception:
        report['area_sum_new'] = None
    # multiset comparison of geometries by normalized WKB
    orig_keys = [geom_wkb_key(g) for g in orig_gdf.geometry]
    new_keys = [geom_wkb_key(g) for g in new_gdf.geometry]
    orig_counter = Counter([k for k in orig_keys if k is not None])
    new_counter = Counter([k for k in new_keys if k is not None])
    # exact match counts
    common = sum((orig_counter & new_counter).values())
    total = max(len(orig_keys), len(new_keys))
    report['geom_exact_match_count'] = int(common)
    report['geom_exact_match_ratio'] = float(common) / total if total > 0 else None
    # keys present in orig but not in new (up to 10 examples)
    missing = list((orig_counter - new_counter).elements())
    extra = list((new_counter - orig_counter).elements())
    report['missing_example_count'] = min(10, len(missing))
    report['extra_example_count'] = min(10, len(extra))
    return report


def main():
    # compute project root relative to this script
    script_root = Path(__file__).resolve().parents[1]
    base = script_root / "convertToGeoJson" / "SourceData" / "TestSet"
    gdb = base / "GDB" / "H50.gdb"
    gpkg_dir = base / "SHP" / "H50"
    layers_to_check = ["BOUA", "BRGA"]
    out_dir = Path("zNC-Test/compare_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_report = {}
    for layer in layers_to_check:
        entry = {'layer': layer}
        try:
            orig = gpd.read_file(str(gdb), layer=layer)
        except Exception as e:
            entry['error_orig'] = str(e)
            full_report[layer] = entry
            continue
        gpkg_path = gpkg_dir / f"{layer}.gpkg"
        if not gpkg_path.exists():
            entry['error_new'] = f"Missing {gpkg_path}"
            full_report[layer] = entry
            continue
        try:
            new = gpd.read_file(str(gpkg_path))
        except Exception as e:
            entry['error_new'] = str(e)
            full_report[layer] = entry
            continue

        entry['analysis'] = analyze_pair(orig, new)
        full_report[layer] = entry
        # write per-layer report
        with (out_dir / f"verify_{layer}.json").open("w", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False, indent=2)

    with (out_dir / "verify_full.json").open("w", encoding="utf-8") as fh:
        json.dump(full_report, fh, ensure_ascii=False, indent=2)

    # print summary
    for layer, info in full_report.items():
        print(f"Layer: {layer}")
        if 'analysis' in info:
            a = info['analysis']
            print(f"  counts: orig={a['count_orig']} new={a['count_new']}")
            print(f"  geom exact matches: {a['geom_exact_match_count']} / max_count={max(a['count_orig'], a['count_new'])} ratio={a['geom_exact_match_ratio']}")
            print(f"  types orig: {a['types_orig']}")
            print(f"  types new: {a['types_new']}")
            print(f"  props orig/new (examples): {a['props_orig'][:10]} / {a['props_new'][:10]}")
        else:
            print("  Error:", info.get('error_orig') or info.get('error_new'))


if __name__ == "__main__":
    main()


