from pathlib import Path
import os
def ensure_zero_for(base_name: str):
    """
    Ensure zero_watermark/{base_name}_zero.png exists.
    If missing, try to generate all zero watermarks by running batch_test_zero_watermark().
    """
    script_dir = Path(__file__).resolve().parent
    zdir = script_dir / 'zero_watermark'
    zdir.mkdir(parents=True, exist_ok=True)
    target = zdir / f'{base_name}_zero.png'
    if target.exists():
        return
    try:
        # try to run batch generator in same package
        try:
            from .batch_test_all_maps import batch_test_zero_watermark
        except Exception:
            from batch_test_all_maps import batch_test_zero_watermark
        print(f'零水印缺失: {target}, 尝试生成 zero_watermark 目录下的缺失文件...')
        batch_test_zero_watermark()
    except Exception as e:
        print(f'尝试生成零水印失败: {e}')
    # final check: no raise here, caller will handle if still missing


