#!/usr/bin/env python3
"""
同步 zNC-Test/vector-data 到各个 zzContrastExperiment/*/pso_data 目录的脚本。
用法:
  python scripts/sync_pso_data.py
或设置环境变量 PSO_DATA_DIR 指向源数据（覆盖默认）
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
SOURCE = Path(os.environ.get('PSO_DATA_DIR', str(PROJECT_ROOT / 'zNC-Test' / 'vector-data')))

TARGET_PARENT = PROJECT_ROOT / 'zzContrastExperiment'

SHP_EXTS = ['.shp', '.shx', '.dbf', '.prj', '.cpg']

def sync_one(target_pso: Path):
    if not target_pso.exists():
        target_pso.mkdir(parents=True, exist_ok=True)
    # 清空目标 pso_data（仅删除支持的扩展文件）
    for f in list(target_pso.glob('*')):
        try:
            if f.suffix.lower() in SHP_EXTS:
                f.unlink()
        except Exception:
            pass
    # 复制源目录下的 shapefile 基础文件（按 basename）
    for shp in SOURCE.glob('*.shp'):
        stem = shp.stem
        for ext in SHP_EXTS:
            src = SOURCE / (stem + ext)
            if src.exists():
                dst = target_pso / (stem + ext)
                shutil.copy2(src, dst)

def main():
    if not SOURCE.exists():
        print(f"源数据目录不存在: {SOURCE}")
        return
    print(f"使用源数据: {SOURCE}")
    # 遍历子目录
    for sub in TARGET_PARENT.iterdir():
        if not sub.is_dir():
            continue
        pso = sub / 'pso_data'
        if pso.exists():
            print(f"同步到: {pso}")
            sync_one(pso)
        else:
            print(f"跳过 (无 pso_data): {sub}")
    print("同步完成。")

if __name__ == '__main__':
    main()



































