from pathlib import Path

import numba
import numpy as np

import moraine as mr
from moraine.api import utils_
from moraine.api.utils_ import _source_hash, mjit


def test_source_hash(tmp_path):
    (tmp_path / 'a.py').write_text('x = 1\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'b.py').write_text('y = 2\n')
    h = _source_hash(tmp_path)
    (tmp_path / '.ipynb_checkpoints').mkdir()
    (tmp_path / '.ipynb_checkpoints' / 'c.py').write_text('z = 3\n')
    assert _source_hash(tmp_path) == h
    (tmp_path / 'sub' / 'b.py').write_text('y = 4\n')  # a change in any file gives another cache directory
    assert _source_hash(tmp_path) != h


def test_mjit_keeps_numba_cache_dir():
    before = numba.config.CACHE_DIR

    @mjit(nopython=True)
    def f(x):
        return x + 1

    assert f(1) == 2
    assert numba.config.CACHE_DIR == before
    if utils_._CACHE_DIR is not None:
        assert str(f._cache._cache_path).startswith(utils_._CACHE_DIR)


def test_emi_kernel_cached(rng):
    # LAPACK is called by symbol name, so the EMI kernel can be saved to the cache
    from moraine.api.pl import _emi_numba
    x = rng.standard_normal((4, 30, 6)) + 1j * rng.standard_normal((4, 30, 6))
    c = np.conj(x).transpose(0, 2, 1) @ x
    d = 1 / np.sqrt(np.diagonal(c, axis1=1, axis2=2).real)
    pairs = mr.TempNet.from_bandwidth(6).image_pairs
    coh = (c[:, pairs[:, 0], pairs[:, 1]].conj() * d[:, pairs[:, 0]] * d[:, pairs[:, 1]]).astype(np.complex64)
    assert mr.emi(coh).shape == (4, 6)
    if utils_._CACHE_DIR is not None:
        assert list(Path(_emi_numba._cache._cache_path).glob('pl._emi_numba-*.nbi'))
