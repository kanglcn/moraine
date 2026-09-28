"""Shared fixtures.

Tests marked ``gpu`` run only when cupy and a CUDA device are available.
Tests that read the sample data set are skipped when it is missing. The data
root defaults to ``<repo>/data`` and can be set with ``MORAINE_TEST_DATA``;
it should contain ``rslc.zarr`` (nlines, width, nimages) and the GAMMA
output directory ``gamma/``.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from moraine.utils_ import is_cuda_available

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('MORAINE_TEST_DATA', REPO / 'data'))
HAS_GPU = is_cuda_available()


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


@pytest.fixture(scope='session')
def rslc():
    """The sample rslc stack, shape (2500, 1834, 17), complex64."""
    return zarr.open(data_path('rslc.zarr'), mode='r')


@pytest.fixture
def rng():
    return np.random.default_rng(0)


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
    is_shp = p < 0.05
    is_ds_can = np.count_nonzero(is_shp, axis=(-2, -1)) >= 50
    return dict(rslc=crop, is_ds_can=is_ds_can, gix=np.stack(np.where(is_ds_can), axis=-1),
                is_shp=is_shp[is_ds_can])
