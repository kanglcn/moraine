from pathlib import Path

import numba
import numpy as np
import pytest

import moraine as mr
from moraine.api import utils_
from moraine.api.utils_ import _source_hash, mjit, _default_cuda_home


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


def test_default_cuda_home(tmp_path, monkeypatch):
    for v in ('CUDA_HOME', 'CUDA_PATH', 'CONDA_PREFIX'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr('sys.prefix', str(tmp_path))
    _default_cuda_home()
    assert 'CUDA_HOME' not in __import__('os').environ          # no CUDA libraries in this prefix
    (tmp_path / 'nvvm').mkdir()
    _default_cuda_home()
    assert __import__('os').environ['CUDA_HOME'] == str(tmp_path)
    monkeypatch.setenv('CUDA_HOME', '/somewhere/else')
    _default_cuda_home()
    assert __import__('os').environ['CUDA_HOME'] == '/somewhere/else'   # set by the user: kept


@pytest.mark.gpu
def test_cuda_kernels_cached():
    """GPU kernels are saved to moraine's numba cache directory (decision 0029), like the CPU functions."""
    import cupy as cp
    from moraine.api import ps
    before = numba.config.CACHE_DIR
    rslc = cp.asarray((np.random.default_rng(0).standard_normal((8, 8, 5, 2)) @ [1, 1j]).astype(np.complex64))
    mr.amp_disp(rslc)
    assert numba.config.CACHE_DIR == before
    if utils_._CACHE_DIR is not None:
        kernel = ps._amp_disp_cuda
        assert str(kernel._cache._cache_path).startswith(utils_._CACHE_DIR)
        assert list(Path(kernel._cache._cache_path).glob('ps._amp_disp_cuda-*.nbi'))
