#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two directories of GeoJSON files and report structural differences.

Saves a JSON report to `zNC-Test/compare_reports/compare_report.json` and
per-file detailed reports to `zNC-Test/compare_reports/{key}.json`.

Usage:
    python compare_geojson_dirs.py \
        --dir-a zNC-Test/vector-data-geojson-six \
        --dir-b zNC-Test/vector-data-geojson

The script matches files by a normalized basename (remove common prefixes like H50 and non-alphanumerics).
"""
from pathlib import Path
import json
import argparse
import re
from typing import Dict, Any, List, Tuple, Optional


def normalize_name(fname: str) -> str:
    n = fname.lower()
    # remove common prefixes like h50-, h50_, etc.
    n = re.sub(r'^h50[-_]*', '', n)
    # remove extension
    n = re.sub(r'\.geojson$', '', n)
    # keep only alnum
    n = re.sub(r'[^a-z0-9]', '', n)
    return n


def load_geojson(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        crs = None
        if isinstance(data, dict):
            crs = data.get('crs') or None
        return data, crs
    except Exception as e:
        return None, str(e)


def geom_points_from_coords(coords) -> List[Tuple[float, float]]:
    # recursively extract first-level points from nested coordinate arrays
    pts: List[Tuple[float, float]] = []

    def walk(c):
        if c is None:
            return
        if isinstance(c, (int, float)):
            return
        if isinstance(c, list):
            # coordinate pair?
            if len(c) >= 2 and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float)):
                pts.append((float(c[0]), float(c[1])))
                return
            for item in c:
                walk(item)

    walk(coords)
    return pts


def compute_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    features = data.get('features') if isinstance(data, dict) else []
    stats['feature_count'] = len(features)
    types_count: Dict[str, int] = {}
    prop_keys_set = set()
    bbox = [float('inf'), float('inf'), -float('inf'), -float('inf')]
    sample_coords: List[Tuple[float, float]] = []

    for i, feat in enumerate(features):
        geom = feat.get('geometry') if isinstance(feat, dict) else None
        gtype = geom.get('type') if geom else None
        types_count[gtype] = types_count.get(gtype, 0) + 1
        props = feat.get('properties') if isinstance(feat, dict) else {}
        if isinstance(props, dict):
            for k in props.keys():
                prop_keys_set.add(k)
        coords = geom.get('coordinates') if geom else None
        pts = geom_points_from_coords(coords)
        if pts and len(sample_coords) < 5:
            sample_coords.append(pts[0])
        for (x, y) in pts:
            if x < bbox[0]:
                bbox[0] = x
            if y < bbox[1]:
                bbox[1] = y
            if x > bbox[2]:
                bbox[2] = x
            if y > bbox[3]:
                bbox[3] = y

    if bbox[0] == float('inf'):
        bbox = None

    stats['geometry_types'] = types_count
    stats['property_keys'] = sorted(list(prop_keys_set))
    stats['bbox'] = bbox
    stats['sample_coords'] = sample_coords
    return stats


def compare_stats(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    diff = {}
    diff['feature_count_a'] = a.get('feature_count')
    diff['feature_count_b'] = b.get('feature_count')
    diff['feature_count_diff'] = (b.get('feature_count') or 0) - (a.get('feature_count') or 0)
    # geometry types
    types_a = a.get('geometry_types', {})
    types_b = b.get('geometry_types', {})
    all_types = set(list(types_a.keys()) + list(types_b.keys()))
    diff['geometry_type_counts'] = {t: {'a': types_a.get(t, 0), 'b': types_b.get(t, 0)} for t in all_types}
    # property keys
    keys_a = set(a.get('property_keys', []))
    keys_b = set(b.get('property_keys', []))
    diff['property_keys_only_in_a'] = sorted(list(keys_a - keys_b))
    diff['property_keys_only_in_b'] = sorted(list(keys_b - keys_a))
    # bbox
    diff['bbox_a'] = a.get('bbox')
    diff['bbox_b'] = b.get('bbox')
    # sample coords
    diff['sample_coords_a'] = a.get('sample_coords')
    diff['sample_coords_b'] = b.get('sample_coords')
    return diff


def main():
    parser = argparse.ArgumentParser(description="Compare two GeoJSON directories")
    parser.add_argument('--dir-a', type=str, required=False, default='zNC-Test/vector-data-geojson-six')
    parser.add_argument('--dir-b', type=str, required=False, default='zNC-Test/vector-data-geojson')
    parser.add_argument('--out', type=str, required=False, default='zNC-Test/compare_reports')
    args = parser.parse_args()

    dir_a = Path(args.dir_a).resolve()
    dir_b = Path(args.dir_b).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_a = {normalize_name(p.name): p for p in dir_a.glob('*.geojson')}
    files_b = {normalize_name(p.name): p for p in dir_b.glob('*.geojson')}

    keys = sorted(set(list(files_a.keys()) + list(files_b.keys())))
    report: Dict[str, Any] = {'dir_a': str(dir_a), 'dir_b': str(dir_b), 'pairs': {}}

    for k in keys:
        pa = files_a.get(k)
        pb = files_b.get(k)
        entry = {'name_key': k, 'path_a': str(pa) if pa else None, 'path_b': str(pb) if pb else None}
        if pa:
            da, cra = load_geojson(pa)
            if da is None:
                entry['error_a'] = cra
            else:
                entry['crs_a'] = cra
                entry['stats_a'] = compute_stats(da)
        if pb:
            db, crb = load_geojson(pb)
            if db is None:
                entry['error_b'] = crb
            else:
                entry['crs_b'] = crb
                entry['stats_b'] = compute_stats(db)
        if pa and pb and ( ('stats_a' in entry) and ('stats_b' in entry) ):
            entry['diff'] = compare_stats(entry['stats_a'], entry['stats_b'])

        report['pairs'][k] = entry
        # write per-file report
        with (out_dir / f'{k}.json').open('w', encoding='utf-8') as fh:
            json.dump(entry, fh, ensure_ascii=False, indent=2)

    # write full report
    with (out_dir / 'compare_report.json').open('w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # print a short summary
    print(f'Compared {len(keys)} keys; reports written to {out_dir}')
    for k in keys:
        e = report['pairs'][k]
        a_count = e.get('stats_a', {}).get('feature_count') if e.get('stats_a') else None
        b_count = e.get('stats_b', {}).get('feature_count') if e.get('stats_b') else None
        note = []
        if a_count is not None and b_count is not None and a_count != b_count:
            note.append(f'count A={a_count} vs B={b_count}')
        if 'diff' in e:
            gtypes = e['diff']['geometry_type_counts']
            # detect type mismatches
            mism = [t for t, v in gtypes.items() if v['a'] != v['b']]
            if mism:
                note.append(f'geom types differ: {mism}')
        if not note:
            note_msg = 'OK'
        else:
            note_msg = '; '.join(note)
        print(f' - {k}: {note_msg}')


if __name__ == '__main__':
    main()



































