"""numba decorators with the compiled code cached by a hash of the sources (decision 0029)"""

__all__ = ['mjit', 'ngjit', 'ngpjit']

import hashlib
import os
import shutil
import time
from pathlib import Path

import numba
from numba import jit

# Compiled numba functions are cached on disk (decision 0029). numba invalidates a cached function only when its own
# file changes, not when a numba function it calls from another file does; so the cache directory is named after a
# hash of all moraine sources and the numba version: any change of the code uses a new, empty directory. It is set
# only while a function is decorated, so that numba code of the user keeps its own cache location.
_CACHE_KEEP_DAYS = 30

def _source_hash(root=None):
    """hash of the python sources under `root` (the moraine package by default) and the numba version"""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    h = hashlib.sha1(numba.__version__.encode())
    for p in sorted(root.rglob('*.py')):
        if '.ipynb_checkpoints' in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]

def _numba_cache_dir():
    """cache directory of this version of the sources, None if it cannot be created; directories of other versions
    not used for `_CACHE_KEEP_DAYS` days are removed"""
    base = os.environ.get('NUMBA_CACHE_DIR')
    base = Path(base)/'moraine' if base else Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'moraine'/'numba'
    path = base/_source_hash()
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.utime(path)
    except OSError:
        return None
    for old in base.iterdir():
        try:
            if old != path and old.is_dir() and time.time()-old.stat().st_mtime > _CACHE_KEEP_DAYS*86400:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return str(path)

_CACHE_DIR = _numba_cache_dir()

def mjit(**options):
    """`numba.jit` with the compiled code cached in moraine's numba cache directory; use it (or `ngjit`, `ngpjit`)
    instead of `cache=True`"""
    def decorate(func):
        if _CACHE_DIR is None:
            return jit(**options)(func)
        old = numba.config.CACHE_DIR
        numba.config.CACHE_DIR = _CACHE_DIR
        try:
            return jit(cache=True, **options)(func)
        finally:
            numba.config.CACHE_DIR = old
    return decorate

# Adapted from spatialpandas at https://github.com/holoviz/spatialpandas under BSD-2-Clause license.
ngjit = mjit(nopython=True, nogil=True)
ngpjit = mjit(nopython=True, nogil=True, parallel=True)

def mcuda_jit(**options):
    """`numba.cuda.jit` for a kernel with the compiled code cached in moraine's numba cache directory (decision 0029);
    use it instead of `cuda.jit(cache=True)`. Device functions keep `cuda.jit(device=True)`: they are compiled into
    the kernels that call them."""
    from numba import cuda
    def decorate(func):
        if _CACHE_DIR is None:
            return cuda.jit(**options)(func)
        old = numba.config.CACHE_DIR
        numba.config.CACHE_DIR = _CACHE_DIR
        try:
            return cuda.jit(cache=True, **options)(func)
        finally:
            numba.config.CACHE_DIR = old
    return decorate
