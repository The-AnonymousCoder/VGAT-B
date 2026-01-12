#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量更新 Fig1-Fig12.py，添加点矢量数据过滤逻辑
"""

import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# 要替换的旧代码模式
OLD_PATTERN = r'''def discover_inputs\(\) -> List\[Path\]:
    if not DIR_PSO\.exists\(\):
        return \[\]
    shp_files = \[p for p in sorted\(DIR_PSO\.glob\('\*\.shp'\)\) if not p\.name\.startswith\('\._'\)\]
    valid = \[\]
    for p in shp_files:
        if p\.with_suffix\('\.dbf'\)\.exists\(\) and p\.with_suffix\('\.shx'\)\.exists\(\):
            valid\.append\(p\)
    return valid\[:8\]'''

# 新代码
NEW_CODE = '''def discover_inputs() -> List[Path]:
    if not DIR_PSO.exists():
        return []
    shp_files = [p for p in sorted(DIR_PSO.glob('*.shp')) if not p.name.startswith('._')]
    valid = []
    for p in shp_files:
        if not (p.with_suffix('.dbf').exists() and p.with_suffix('.shx').exists()):
            continue
        # 过滤点矢量数据（Lin18 不支持 Point/MultiPoint）
        try:
            gdf = gpd.read_file(str(p))
            geom_types = gdf.geometry.geom_type.unique()
            if any(gt in ['Point', 'MultiPoint'] for gt in geom_types):
                print(f'跳过点矢量: {p.stem}')
                continue
            valid.append(p)
        except Exception:
            continue
    return valid[:6]'''


def update_fig_file(fig_path: Path):
    """更新单个 Fig 文件"""
    try:
        content = fig_path.read_text(encoding='utf-8')
        
        # 查找并替换
        pattern = re.compile(OLD_PATTERN, re.MULTILINE | re.DOTALL)
        if pattern.search(content):
            new_content = pattern.sub(NEW_CODE, content)
            fig_path.write_text(new_content, encoding='utf-8')
            print(f'✅ 已更新: {fig_path.name}')
            return True
        else:
            print(f'⚠️  未找到匹配模式: {fig_path.name}')
            return False
    except Exception as e:
        print(f'❌ 更新失败: {fig_path.name} - {e}')
        return False


def main():
    print('=== 批量更新 Lin18 Fig1-Fig12 添加点矢量过滤 ===\n')
    
    updated_count = 0
    total_count = 0
    
    # 查找所有 Fig 文件
    fig_files = sorted(SCRIPT_DIR.glob('Fig*.py'))
    
    for fig_file in fig_files:
        total_count += 1
        if update_fig_file(fig_file):
            updated_count += 1
    
    print(f'\n=== 完成：已更新 {updated_count}/{total_count} 个文件 ===')


if __name__ == '__main__':
    main()
