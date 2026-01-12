import shapefile
import os
import random

def attacks3_vertex_noise(originshpfile, outshpfile, randratio, noisestrength):
    # 读取原始shapefile
    sf = shapefile.Reader(originshpfile)
    shapes = sf.shapes()
    records = sf.records()
    fields = sf.fields[1:]  # 跳过DeletionFlag字段

    # 创建输出目录
    output_dir = os.path.join('attacked', 'noise')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'noise_{noisestrength}_{randratio}_{outshpfile}')

    # 创建Writer
    w = shapefile.Writer(output_path, shapeType=sf.shapeType)
    w.fields = fields

    for shape, record in zip(shapes, records):
        parts = shape.parts
        points = shape.points
        all_new_points = []

        # 处理多部分几何体
        for i in range(len(parts)):
            start = parts[i]
            end = parts[i+1] if i+1 < len(parts) else len(points)
            part_points = points[start:end]

            new_part_points = []
            for x, y in part_points:
                if random.random() < randratio:
                    noise = -noisestrength + 2 * noisestrength * random.random()
                    x += noise
                    y += noise
                new_part_points.append([x, y])

            # 检查空坐标点
            if len(new_part_points) == 0:
                print(f"警告：部分 {i} 的坐标点为空，已跳过。")
                continue

            # 确保多边形闭合
            if sf.shapeType == shapefile.POLYGON:
                if new_part_points[0] != new_part_points[-1]:
                    new_part_points.append(new_part_points[0])

            all_new_points.append(new_part_points)

        # 跳过空几何体
        if len(all_new_points) == 0:
            print(f"警告：形状 {record} 的坐标点全部为空，已跳过。")
            continue

        # 写入形状和记录
        if sf.shapeType == shapefile.POLYLINE:
            w.line(all_new_points)
        elif sf.shapeType == shapefile.POLYGON:
            w.poly(all_new_points)
        w.record(*record)

    # 保存文件
    w.close()
    return output_path