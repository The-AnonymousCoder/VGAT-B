import numpy as np
import geopandas as gpd
from shapely import GeometryCollection
from shapely.geometry import LineString, MultiLineString
import os
import random


def attacks1_vertex_delete(originshpfile, outshpfile, deletefactor):
    """
    实现随机删点攻击

    参数：
    originshpfile : str - 原始矢量地图文件路径
    outshpfile    : str - 输出文件名
    deletefactor  : float - 删除概率 (0~1)

    返回：
    str - 新文件保存路径
    """
    # 设置随机种子 (兼容MATLAB的旧版随机状态)
    random.seed(212367)
    np.random.seed(212367)

    # 读取原始矢量数据
    gdf = gpd.read_file(originshpfile)

    # 收集所有顶点并建立全局索引
    all_vertices = []
    vertex_indices = []  # 记录每个顶点属于哪个几何体的哪个部分

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # 处理多段线 (MultiLineString)
        if geom.geom_type == 'MultiLineString':
            parts = list(geom.geoms)
        else:
            parts = [geom]

        for part_idx, part in enumerate(parts):
            if part.is_empty:
                continue
            coords = list(part.coords)
            for coord in coords:
                all_vertices.append(coord)
                vertex_indices.append((idx, part_idx, len(vertex_indices)))

    # 生成删除标记
    total_vertices = len(all_vertices)
    keep_flags = np.random.rand(total_vertices) >= deletefactor

    # 处理每个几何体
    new_geoms = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            new_geoms.append(geom)
            continue

        # 分解几何部件
        if geom.geom_type == 'MultiLineString':
            parts = list(geom.geoms)
        else:
            parts = [geom]

        new_parts = []
        for part_idx, part in enumerate(parts):
            if part.is_empty:
                new_parts.append(part)
                continue

            # 获取该部件所有顶点的全局索引
            part_vertex_indices = [vi[2] for vi in vertex_indices
                                   if vi[0] == idx and vi[1] == part_idx]

            # 过滤保留的顶点
            filtered_coords = [all_vertices[i] for i in part_vertex_indices
                               if keep_flags[i]]

            # 处理空部件的情况：保留首尾点
            if len(filtered_coords) == 0:
                original_coords = list(part.coords)
                if len(original_coords) >= 2:
                    filtered_coords = [original_coords[0], original_coords[-1]]
                else:
                    filtered_coords = original_coords  # 如果原始只有一个点则保留

            # 重建几何部件
            if len(filtered_coords) >= 2:
                new_part = LineString(filtered_coords)
            else:
                new_part = None  # 无效几何将被过滤

            if new_part and not new_part.is_empty:
                new_parts.append(new_part)

        # 重建几何体
        if len(new_parts) == 0:
            new_geom = None  # 空几何
        elif len(new_parts) == 1:
            new_geom = new_parts[0]
        else:
            new_geom = MultiLineString(new_parts)

        new_geoms.append(new_geom if new_geom else GeometryCollection())

    # 更新GeoDataFrame
    gdf['geometry'] = new_geoms

    # 创建输出目录
    output_dir = os.path.join('attacked', 'delete')
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出路径
    output_path = os.path.join(
        output_dir,
        f'delete_{outshpfile}'
    )

    # 保存结果
    gdf.to_file(output_path)

    return output_path


def main():
    # 配置参数 MRailways MBuilding 0  MLanduse MBoundary MRoad MLake
    input_shp = "embed/MBoundary.shp"  # 原始矢量地图路径
    base_output_name = "MBoundary"  # 输出文件名基础

    # 生成不同删除因子参数组 (0.1, 0.2, ..., 0.5)
    delete_factors = np.arange(0.1, 0.6, 0.1).round(2)

    # 结果路径存储列表
    output_paths = []

    # 循环处理每个删除因子
    for df in delete_factors:
        print(f"\n正在处理删除因子: {df:.1f}")

        # 生成唯一输出文件名
        output_name = f"{base_output_name}_factor_{df:.1f}.shp"

        # 执行攻击操作
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

    # 输出汇总报告
    print("\n===== 处理完成 =====")
    print(f"原始文件: {input_shp}")
    print("生成文件列表:")
    for p in output_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()