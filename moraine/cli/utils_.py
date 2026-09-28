"""internal utilities for CLI developing"""


__all__ = ['mk_clean_dir']

from pathlib import Path
import shutil

def mk_clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True,exist_ok=True)
