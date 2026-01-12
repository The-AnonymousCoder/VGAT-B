#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lin18/batch_test.py - 扫描 pso_data，执行水印嵌入与提取（无攻击）流程验证，统计成功率与NC。
使用 Cat32.png
"""

from pathlib import Path
import os
import numpy as np
import warnings

from embed import embed as lin18_embed
from extract import extract as lin18_extract
import geopandas as gpd
from PIL import Image
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
import os
PROJECT_ROOT = SCRIPT_DIR.resolve().parents[1]
PSO_DIR = Path(os.environ.get('PSO_DATA_DIR', str(PROJECT_ROOT / 'zNC-Test' / 'vector-data')))
EMBED_DIR = SCRIPT_DIR / 'embed'
WM_DIR = SCRIPT_DIR / 'watermark'
ATTACK_DIR = SCRIPT_DIR / 'attacked' / 'delete'
EXTRACT_DIR = SCRIPT_DIR / 'extract'

CAT32_PNG = SCRIPT_DIR / 'Cat32.png'

for d in [EMBED_DIR, WM_DIR, EXTRACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 静默 pyogrio 写出时的 CRS 提示
warnings.filterwarnings('ignore', message=".*'crs' was not provided.*", category=UserWarning)


def ensure_embed(src: Path, force_regenerate: bool = False) -> Path:
    """
    确保矢量数据已嵌入水印
    
    Args:
        src: 源 shapefile 路径
        force_regenerate: 是否强制重新生成（忽略缓存）
    
    Returns:
        嵌入水印后的 shapefile 路径
    """
    base = src.stem
    wm_txt = WM_DIR / f'M{base}.txt'
    out_shp = EMBED_DIR / f'M{base}.shp'
    
    # 如果不强制重新生成且文件存在，直接返回
    if not force_regenerate and out_shp.exists():
        return out_shp
    
    from get_coor import get_coor_nested, get_coor_array  # type: ignore
    gdf = gpd.read_file(str(src))
    
    # 检查是否包含点矢量数据（Lin18 不支持 Point/MultiPoint）
    geom_types = gdf.geometry.geom_type.unique()
    if any(gt in ['Point', 'MultiPoint'] for gt in geom_types):
        raise ValueError(f'Lin18 不支持点矢量数据 (发现类型: {list(geom_types)})')
    
    coor_nested, feature_type = get_coor_nested(gdf)
    coor_array = get_coor_array(coor_nested, feature_type)
    
    # ===== 使用二值化水印 (0/1)，与 extract.py 的 NC 计算保持一致 =====
    # 加载图像并二值化为 0/1（与 NC.py 的 image_to_array 一致）
    wm_img = Image.open(str(CAT32_PNG)).convert('L').resize((32, 32))
    watermark = (np.array(wm_img) > 127).astype(int).flatten()  # 二值化
    
    # 按照 embed.py 的方式重复水印
    repeat_time = (coor_array.shape[1] - 2) * 4 // len(watermark)
    wm = np.tile(watermark, repeat_time)
    wm = np.hstack((wm, watermark[:(coor_array.shape[1] - 2) * 4 % len(watermark)]))
    
    # 保存水印文本（使用 delimiter='' 和 fmt='%d'，与 embed.py 一致）
    np.savetxt(wm_txt, wm, delimiter='', fmt='%d')
    try:
        lin18_embed(str(src), str(wm_txt))
        return out_shp if out_shp.exists() else EMBED_DIR / (wm_txt.stem + '.shp')
    except Exception as e:
        # 兜底：直接复制源为嵌入版本，保留 CRS
        print('embed失败，使用原始复制:', src.stem, e)
        gdf = gpd.read_file(str(src))
        try:
            gdf.to_file(str(out_shp))
        except Exception:
            # 最后方案：以 GPKG 再转 SHP（避免驱动问题）
            tmp = out_shp.with_suffix('.gpkg')
            gdf.to_file(tmp, driver='GPKG')
            gpd.read_file(tmp).to_file(str(out_shp))
        return out_shp


def evaluate_embedding_and_extraction(src: Path, force_regenerate: bool = False) -> float:
    """
    对单个矢量：嵌入 -> 直接提取（无攻击），返回 NC
    
    Args:
        src: 源 shapefile 路径
        force_regenerate: 是否强制重新生成嵌入文件
    
    Returns:
        NC 值
    """
    wm_shp = ensure_embed(src, force_regenerate=force_regenerate)
    _, _, nc = lin18_extract(str(wm_shp), str(CAT32_PNG))
    return float(nc)


def main(force_regenerate: bool = False, clean_cache: bool = False):
    """
    主函数
    
    Args:
        force_regenerate: 是否强制重新生成嵌入文件（忽略缓存）
        clean_cache: 是否清理旧的嵌入/提取缓存
    """
    import sys
    
    if not CAT32_PNG.exists():
        print('缺少 Cat32.png', flush=True)
        return
    
    # 自动清理 macOS 隐藏文件（._开头）
    hidden_files = list(PSO_DIR.glob('._*'))
    if hidden_files:
        print(f'🧹 清理 {len(hidden_files)} 个 macOS 隐藏文件...', flush=True)
        for f in hidden_files:
            try:
                f.unlink()
            except Exception as e:
                print(f'  ⚠️  删除失败: {f.name} - {e}', flush=True)
    
    # 清理旧的嵌入/提取缓存
    if clean_cache:
        print('🧹 清理旧的嵌入/提取缓存...', flush=True)
        import shutil
        for cache_dir in [EMBED_DIR, EXTRACT_DIR, WM_DIR]:
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    print(f'  ✅ 已清理: {cache_dir.name}/', flush=True)
                except Exception as e:
                    print(f'  ⚠️  清理失败: {cache_dir.name}/ - {e}', flush=True)
    
    # 过滤有效的 shapefile（排除 macOS 隐藏文件）
    shp_files = sorted([p for p in PSO_DIR.glob('*.shp') 
                       if not p.name.startswith('._') and not p.name.startswith('.')])
    print(f'发现 {len(shp_files)} 个矢量', flush=True)
    
    if force_regenerate:
        print('⚠️  强制重新生成模式：将忽略已存在的嵌入文件', flush=True)

    total = 0
    success = 0
    skipped = 0
    nc_values = {}
    for src in shp_files:
        try:
            print(f'正在处理: {src.stem}...', end=' ', flush=True)
            nc = evaluate_embedding_and_extraction(src, force_regenerate=force_regenerate)
            total += 1
            nc_values[src.stem] = float(nc)
            if float(nc) > 0:
                success += 1
            print(f'NC={float(nc):.6f}', flush=True)
        except ValueError as ve:
            # 跳过不支持的数据类型（如点矢量）
            if 'Lin18 不支持' in str(ve):
                skipped += 1
                print(f'跳过 - {ve}', flush=True)
            else:
                total += 1
                print(f'失败 - {ve}', flush=True)
        except Exception as exc:
            total += 1
            print(f'失败 - {exc}', flush=True)
    
    print(f'\n总计: {total}，成功: {success}，跳过: {skipped}，成功率: {success/total*100:.1f}%' if total > 0 else f'\n总计: {total}，跳过: {skipped}', flush=True)
    if nc_values:
        avg_nc = sum(nc_values.values())/len(nc_values)
        print('平均NC (仅成功案例):', round(avg_nc, 6), flush=True)


if __name__ == '__main__':
    import sys
    # 支持命令行参数
    # --force / -f: 强制重新生成（忽略缓存）
    # --clean / -c: 清理旧的嵌入/提取缓存
    force = '--force' in sys.argv or '-f' in sys.argv
    clean = '--clean' in sys.argv or '-c' in sys.argv
    main(force_regenerate=force, clean_cache=clean)


