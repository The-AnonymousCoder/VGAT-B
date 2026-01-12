import os
import random
import numpy as np
import geopandas as gpd

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
    random.seed(42)
    np.random.seed(42)
    
    # 读取shapefile数据
    gdf = gpd.read_file(originshpfile)
    
    # 获取地理要素的总数
    count = len(gdf)
    
    if deleteRatio == 0:
        # 如果删除比例为0，直接复制原文件
        new_gdf = gdf.copy()
    else:
        # 确定保留哪些要素，通过随机数和删除比例来决定
        keep_indices = []
        for i in range(count):
            if random.random() > deleteRatio:
                keep_indices.append(i)
        
        # 如果没有要素被保留，至少保留一个
        if len(keep_indices) == 0 and count > 0:
            keep_indices = [0]
            
        # 提取保留的地理要素
        new_gdf = gdf.iloc[keep_indices].copy()
        new_gdf = new_gdf.reset_index(drop=True)
    
    # 创建输出目录
    output_dir = os.path.join('attacked', 'delete')
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径，包含删除比例字符串
    output_name = f'delete_{deleteRatio}_{outshpfile}'
    output_path = os.path.join(output_dir, output_name)
    
    # 保存新的shapefile
    new_gdf.to_file(output_path)
    
    print(f"对象删除攻击完成: 原始要素数={count}, 保留要素数={len(new_gdf)}, 删除比例={deleteRatio}")
    
    return output_path
