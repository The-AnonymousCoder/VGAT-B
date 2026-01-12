import numpy as np
import geopandas as gpd
from shapely import GeometryCollection
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, LinearRing
import os
import random
from collections import defaultdict


def attacks1_vertex_delete(originshpfile, outshpfile, deletefactor):
    """
    实现随机删点攻击，支持MultiPolygon

    参数：
    originshpfile : str - 原始矢量地图文件路径
    outshpfile    : str - 输出文件名
    deletefactor  : float - 删除概率 (0~1)

    返回：
    str - 新文件保存路径
    """
    # 设置随机种子
    random.seed(212367)
    np.random.seed(212367)

    # 读取数据并记录原始几何类型
    gdf = gpd.read_file(originshpfile)
    original_geom_types = {idx: row.geometry.geom_type if row.geometry is not None and not row.geometry.is_empty else None
                           for idx, row in gdf.iterrows()}

    # 收集所有顶点及索引信息
    all_vertices = []
    vertex_indices = []  # (idx, polygon_part_idx, ring_type, vertex_pos)

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # 分解几何为环或线部件
        parts_info = []
        if geom.geom_type in ['MultiLineString', 'MultiPolygon']:
            geoms = list(geom.geoms)
        else:
            geoms = [geom]

        polygon_part_idx = 0  # 子多边形索引
        for sub_geom in geoms:
            if sub_geom.geom_type == 'Polygon':
                # 处理外环和内环
                exterior = sub_geom.exterior
                interiors = sub_geom.interiors
                parts_info.append((polygon_part_idx, 0, exterior.coords))
                for i, interior in enumerate(interiors, 1):
                    parts_info.append((polygon_part_idx, i, interior.coords))
                polygon_part_idx += 1
            elif sub_geom.geom_type == 'LineString':
                parts_info.append((0, 0, sub_geom.coords))
            elif sub_geom.geom_type in ['Point', 'MultiPoint']:
                continue  # 忽略点类型

        # 记录顶点信息
        for part in parts_info:
            p_idx, r_type, coords = part
            for coord in coords:
                all_vertices.append(coord)
                vertex_indices.append((idx, p_idx, r_type, len(all_vertices) - 1))

    # 生成删除标记
    keep_flags = np.random.rand(len(all_vertices)) >= deletefactor

    # 重建几何
    new_geoms = []
    for idx, row in gdf.iterrows():
        if original_geom_types.get(idx) is None:
            new_geoms.append(GeometryCollection())
            continue

        # 获取该行所有顶点信息
        row_vertices = [(vi[1], vi[2], vi[3]) for vi in vertex_indices if vi[0] == idx]
        geom_type = original_geom_types[idx]

        # 按部件分组
        parts_dict = defaultdict(lambda: defaultdict(list))
        for p_idx, r_type, v_idx in row_vertices:
            if keep_flags[v_idx]:
                parts_dict[p_idx][r_type].append(all_vertices[v_idx])

        # 根据原始类型重建
        if geom_type in ['LineString', 'MultiLineString']:
            line_parts = []
            for p_idx in parts_dict:
                for r_type in parts_dict[p_idx]:
                    coords = parts_dict[p_idx][r_type]
                    if len(coords) >= 2:
                        line_parts.append(LineString(coords))
            if len(line_parts) == 0:
                new_geom = GeometryCollection()
            elif len(line_parts) == 1:
                new_geom = line_parts[0]
            else:
                new_geom = MultiLineString(line_parts)

        elif geom_type in ['Polygon', 'MultiPolygon']:
            polygon_parts = []
            for p_idx in parts_dict:
                rings = parts_dict[p_idx]
                exterior_coords = rings.get(0, [])
                interiors = []

                # 处理外环闭合
                if len(exterior_coords) >= 2:
                    if exterior_coords[0] != exterior_coords[-1]:
                        exterior_coords.append(exterior_coords[0])
                    if len(exterior_coords) >= 4:  # 至少3个不同点
                        try:
                            exterior_ring = LinearRing(exterior_coords)
                            # 处理内环
                            interior_rings = []
                            for r_type in sorted(rings.keys()):
                                if r_type > 0:
                                    icoords = rings[r_type]
                                    if len(icoords) >= 2:
                                        if icoords[0] != icoords[-1]:
                                            icoords.append(icoords[0])
                                        if len(icoords) >= 4:
                                            interior_rings.append(LinearRing(icoords))
                            polygon = Polygon(exterior_ring, interior_rings)
                            polygon_parts.append(polygon)
                        except:
                            pass

            if geom_type == 'Polygon':
                new_geom = polygon_parts[0] if polygon_parts else GeometryCollection()
            else:
                new_geom = MultiPolygon(polygon_parts) if polygon_parts else GeometryCollection()

        else:
            new_geom = GeometryCollection()

        new_geoms.append(new_geom)

    # 更新几何列
    gdf.geometry = new_geoms

    # 保存结果：若传入的是绝对/带目录路径，则直接使用该路径；否则写入默认目录并加前缀
    if os.path.isabs(outshpfile) or os.path.dirname(outshpfile):
        output_path = outshpfile
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        output_dir = os.path.join('attacked', 'delete')
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(outshpfile)
        if not base.lower().startswith('delete_'):
            base = f'delete_{base}'
        output_path = os.path.join(output_dir, base)

    gdf.to_file(output_path)

    return output_path


def main():
    # 配置参数 MRailways MBuilding 0  MLanduse MBoundary MRoad MLake
    input_shp = "embed/Mgis_osm_railways_free_1.shp"
    base_output_name = "MLake"
    delete_factors = np.arange(0.1, 0.6, 0.1).round(2)
    output_paths = []

    for df in delete_factors:
        output_name = f"{base_output_name}_factor_{df:.1f}.shp"
        try:
            path = attacks1_vertex_delete(
                originshpfile=input_shp,
                outshpfile=output_name,
                deletefactor=df
            )
            output_paths.append(path)
            print(f"成功生成: {path}")
        except Exception as e:
            print(f"处理失败 (因子 {df:.1f}): {str(e)}")

    print("\n===== 处理完成 =====")
    print(f"原始文件: {input_shp}")
    print("生成文件列表:")
    for p in output_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()