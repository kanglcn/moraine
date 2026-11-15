"""Shared fixtures and helpers (tests import the helpers: ``from conftest import write_zarr``).

Tests marked ``gpu`` run only when cupy and a CUDA device are available; a test that takes the ``gpu`` fixture runs
on the CPU (``gpu`` False) and on the GPU (True, marked ``gpu``).
Tests that read the sample data set are skipped when it is missing. The data
root defaults to ``<repo>/data`` and can be set with ``MORAINE_TEST_DATA``;
it should contain ``rslc.zarr`` (nlines, width, nimages) and the GAMMA
output directory ``gamma/``. ``gamma/sim_orb/`` (simulated orbital phases, made by GAMMA's
``phase_sim_orb``) is optional; without it the GAMMA loading test takes ~15 min instead of seconds.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from moraine.api.utils_ import is_cuda_available

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('MORAINE_TEST_DATA', REPO / 'data'))
HAS_GPU = is_cuda_available()


def pytest_unconfigure(config):
    # numba-cuda (0.30) crashes in the interpreter's teardown after a kernel ran (segmentation fault, exit 139 instead
    # of the test result); leave the process without that teardown once the report is printed
    if HAS_GPU and 'numba.cuda' in sys.modules:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(int(getattr(config, '_moraine_exitstatus', 0)))


def pytest_sessionfinish(session, exitstatus):
    session.config._moraine_exitstatus = int(exitstatus)


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason='needs cupy and a CUDA GPU')
    for item in items:
        if 'gpu' in item.keywords and not HAS_GPU:
            item.add_marker(skip_gpu)


def data_path(*parts) -> Path:
    """Path inside the test data set, skip the test if it does not exist."""
    p = DATA.joinpath(*parts)
    if not p.exists():
        pytest.skip(f'test data not found: {p}')
    return p


def need_gamma(*programs):
    """Skip the test if one of the GAMMA programs is not on PATH."""
    for program in programs:
        if shutil.which(program) is None:
            pytest.skip(f'GAMMA command {program} not found')


def need_models(*names):
    """Skip the test if one of the trained models (n2f, n2fs3d, n2ft) is not downloaded."""
    from moraine.api.dl.model import _get_model
    for name in names:
        try:
            _get_model(name)
        except FileNotFoundError:
            pytest.skip(f'{name} model not downloaded, run moraine.download_dl_model()')


@pytest.fixture(scope='session')
def rslc():
    """The sample rslc stack, shape (2500, 1834, 17), complex64."""
    return zarr.open(data_path('rslc.zarr'), mode='r')


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(params=[False, pytest.param(True, marks=pytest.mark.gpu)])
def gpu(request):
    """Where the test runs: False on the CPU, True on the GPU."""
    return request.param


def write_zarr(path, data, chunks=None) -> str:
    """Write the array `data` to a new zarr array at `path`, in chunks of shape `chunks` (default: one chunk);
    return the path as a string."""
    z = zarr.open(str(path), mode='w', shape=data.shape, dtype=data.dtype, chunks=chunks or data.shape)
    z[...] = data
    return str(path)


def read_zarr(path) -> np.ndarray:
    """The whole zarr array at `path`."""
    return zarr.open(str(path), mode='r')[:]


def json_out(capsys) -> dict:
    """The JSON object printed (`--json`) since the last read of `capsys`."""
    return json.loads(capsys.readouterr().out)


def random_gix(rng, shape, n, sort=True) -> np.ndarray:
    """`n` different random grid indices (azimuth, range) of a raster of `shape`, (n, 2) int32, sorted in raster
    order unless `sort` is False."""
    g = rng.choice(shape[0] * shape[1], size=n, replace=False)
    if sort:
        g.sort()
    return np.stack(np.unravel_index(g, shape), axis=-1).astype(np.int32)


def phase_diff(a, b) -> np.ndarray:
    """Absolute phase differences (rad) of the complex arrays `a` and `b` where both are finite, 1D."""
    ok = np.isfinite(a) & np.isfinite(b)
    return np.abs(np.angle(a[ok] * b[ok].conj()))


def synthetic_shp(rng, shape=(5, 10), nimages=17, half_az_win=1, half_r_win=2, p_true=0.7):
    """Random rslc stack and a valid `is_shp` (center pixel always True, pixels outside the image False)."""
    rslc = (rng.random((*shape, nimages)) + 1j * rng.random((*shape, nimages))).astype(np.complex64)
    az_win, r_win = 2 * half_az_win + 1, 2 * half_r_win + 1
    is_shp = rng.random((*shape, az_win, r_win)) < p_true
    i, j, k, l = np.meshgrid(*(np.arange(n) for n in is_shp.shape), indexing='ij')
    is_shp[(k == half_az_win) & (l == half_r_win)] = True
    az, r = i + k - half_az_win, j + l - half_r_win
    is_shp[(az < 0) | (az >= shape[0]) | (r < 0) | (r >= shape[1])] = False
    return rslc, is_shp


@pytest.fixture(scope='session')
def ds_can(rslc):
    """DS candidates of a 400x400 crop of the sample data: 11x11 KS-test SHPs, >= 50 SHPs."""
    import moraine as mr
    crop = rslc[:400, :400]
    p = mr.ks_test(np.abs(crop) ** 2, az_half_win=5, r_half_win=5)
    is_shp = p >= 0.05
    is_ds_can = np.count_nonzero(is_shp, axis=(-2, -1)) >= 50
    return dict(rslc=crop, is_ds_can=is_ds_can, gix=np.stack(np.where(is_ds_can), axis=-1),
                is_shp=is_shp[is_ds_can])
