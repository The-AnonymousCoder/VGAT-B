import random
import shapefile

def attacks4_object_delete(originshpfile, outshpfile, deleteRatio):
    """
    attacks4_object_delete - 实现随机删除地理要素攻击
    该函数通过随机删除地理要素，生成新的shapefile。

    参数:
        originshpfile (str): 原始shapefile文件的路径。
        outshpfile (str): 输出shapefile文件的名称。
        deleteRatio (float): 删除地理要素的比例（介于0和1之间）。

    返回:
        str: 保存新shapefile的完整路径。
    """
    # 设置随机种子，确保结果可重复
    random.seed(31765871364233903586)

    # 读取shapefile数据
    sf = shapefile.Reader(originshpfile)

    # 获取地理要素的总数
    count = len(sf)

    # 确定保留哪些要素，通过随机数和删除比例来决定
    reservedFs = [random.random() > deleteRatio for _ in range(count)]

    # 获取保留的要素索引
    indices_to_keep = [i for i, keep in enumerate(reservedFs) if keep]

    # 提取保留的地理要素
    reservedFe = [sf.record(i) for i in indices_to_keep]

    # 设置输出文件路径，包含删除比例字符串
    output_path = f'attacked/deleteFeature/delete_{deleteRatio}_{outshpfile}'

    # 创建新的shapefile并写入保留的要素
    with shapefile.Writer(output_path) as writer:
        # 写入与原始shapefile相同的字段
        writer.fields = sf.fields[1:]  # 跳过删除标记字段
        # 写入保留的要素
        for record in reservedFe:
            writer.record(*record)
        for i in indices_to_keep:
            writer.shape(sf.shape(i))

    return output_path
