import itertools
import math

import numpy as np
import pytest

import moraine as mr
from moraine.api.co import emperical_co, emperical_co_pc, ad_intf_pc, uncompress_coh, isPD, regularize_spectral
from conftest import synthetic_shp


def _ds_can(is_shp, min_shp=3):
    is_ds_can = np.count_nonzero(is_shp, axis=(-2, -1)) >= min_shp
    return is_ds_can, np.stack(np.where(is_ds_can), axis=-1), is_shp[is_ds_can]


@pytest.mark.gpu
def test_emperical_co_gpu(rng):
    import cupy as cp
    rslc, is_shp = synthetic_shp(rng, p_true=0.7)
    cov, coh = emperical_co(cp.asarray(rslc), cp.asarray(is_shp))
    cov, coh = cov.get(), coh.get()
    haz, hr = is_shp.shape[2] // 2, is_shp.shape[3] // 2
    for i, j, k, l in itertools.product(range(rslc.shape[0]), range(rslc.shape[1]), range(rslc.shape[2]), range(rslc.shape[2])):
        _cov, a2k, a2l, n = 0j, 0.0, 0.0, 0
        for m, q in itertools.product(range(is_shp.shape[2]), range(is_shp.shape[3])):
            if is_shp[i, j, m, q]:
                s = rslc[i + m - haz, j + q - hr]
                _cov += s[k] * s[l].conj(); a2k += abs(s[k]) ** 2; a2l += abs(s[l]) ** 2; n += 1
        assert abs(_cov / n - cov[i, j, k, l]) < 1e-6
        assert abs(_cov / math.sqrt(a2k * a2l) - coh[i, j, k, l]) < 1e-6


def test_emperical_co_pc_image_pairs(rng):
    rslc, is_shp = synthetic_shp(rng, p_true=0.1)
    _, idx, ds_is_shp = _ds_can(is_shp)
    full = emperical_co_pc(rslc, idx, ds_is_shp)
    assert full.shape == (idx.shape[0], 17 * 16 // 2)
    tnet = mr.TempNet.from_bandwidth(rslc.shape[2], bandwidth=3)
    sub = emperical_co_pc(rslc, idx, ds_is_shp, image_pairs=tnet.image_pairs)
    assert sub.shape == (idx.shape[0], tnet.image_pairs.shape[0])
    all_pairs = np.stack(np.triu_indices(rslc.shape[2], 1), axis=-1)
    pos = [np.flatnonzero((all_pairs == p).all(-1))[0] for p in tnet.image_pairs]
    np.testing.assert_array_almost_equal(full[:, pos], sub)


@pytest.mark.gpu
def test_emperical_co_pc_gpu(rng):
    import cupy as cp
    rslc, is_shp = synthetic_shp(rng, p_true=0.1)
    is_ds_can, idx, ds_is_shp = _ds_can(is_shp)
    cpu = emperical_co_pc(rslc, idx, ds_is_shp)
    gpu = emperical_co_pc(cp.asarray(rslc), cp.asarray(idx), cp.asarray(ds_is_shp))
    np.testing.assert_array_almost_equal(cpu, gpu.get())
    # emperical_co and emperical_co_pc agree
    _, coh = emperical_co(cp.asarray(rslc), cp.asarray(is_shp))
    pairs = cp.triu_indices(rslc.shape[2], 1)
    cp.testing.assert_array_almost_equal(coh[cp.asarray(is_ds_can)][:, pairs[0], pairs[1]], gpu)


def test_uncompress_coh(rng):
    tnet = mr.TempNet.from_bandwidth(5, bandwidth=2)
    coh = (rng.random(tnet.image_pairs.shape[0]) + 1j * rng.random(tnet.image_pairs.shape[0])).astype(np.complex64)
    full = uncompress_coh(coh, tnet.image_pairs)
    assert full.shape == (5, 5)
    for n, (i, j) in enumerate(tnet.image_pairs):
        assert full[i, j] == coh[n]
        assert full[j, i] == np.conj(coh[n])


@pytest.mark.parametrize('gpu', [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_ad_intf_pc(rng, gpu):
    rslc, is_shp = synthetic_shp(rng, p_true=0.1)
    _, idx, ds_is_shp = _ds_can(is_shp)
    if gpu:
        import cupy as cp
        rslc, idx, ds_is_shp = cp.asarray(rslc), cp.asarray(idx), cp.asarray(ds_is_shp)
    coh = emperical_co_pc(rslc, idx, ds_is_shp)
    ifg = ad_intf_pc(rslc[:, :, 0], rslc[:, :, 1], idx, ds_is_shp)
    if gpu:
        coh, ifg = coh.get(), ifg.get()
    np.testing.assert_array_equal(coh[:, 0], ifg)


@pytest.mark.gpu
def test_isPD_and_regularize_spectral(rslc):
    import cupy as cp
    s = cp.asarray(rslc[600:605, 600:610])
    p = mr.ks_test(cp.sort(cp.abs(s) ** 2, axis=-1), az_half_win=1, r_half_win=2)
    _, coh = emperical_co(s, p < 0.05)
    assert isPD(coh).shape == coh.shape[:-2]
    r1 = regularize_spectral(coh, 0.1)
    r2 = regularize_spectral(coh, cp.ones(coh.shape[:-2]) / 10)
    cp.testing.assert_array_almost_equal(r1, r2)


def test_nearestPD(rng):
    from moraine.api.co import nearestPD
    a = rng.standard_normal((4, 6, 6))
    a = (a + np.swapaxes(a, -1, -2)) / 2          # symmetric, mostly indefinite
    assert not isPD(a).all()
    pd = nearestPD(a)
    assert isPD(pd).all()


@pytest.mark.gpu
def test_isPD_cpu_gpu_agree(rng):
    import cupy as cp
    a = rng.standard_normal((50, 6, 6))
    a = (a + np.swapaxes(a, -1, -2)) / 2
    a[:25] += 6 * np.eye(6)                       # half of them positive definite
    np.testing.assert_array_equal(isPD(a), isPD(cp.asarray(a)).get())


def _correlated_speckle(rng, shape=(600, 800)):
    """Speckle filtered with a separable kernel (azimuth [1, 0.8], range [0.5, 1, 0.5]) and its true |rho|^2
    on the lags (-4..4, -6..6)."""
    from scipy.signal import convolve2d
    w = ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)).astype(np.complex64)
    k = np.outer([1.0, 0.8], [0.5, 1.0, 0.5])
    ac = np.abs(convolve2d(k, k[::-1, ::-1])) ** 2
    ac /= ac.max()
    true = np.zeros((9, 13))
    ca, cr = np.array(ac.shape) // 2
    true[4 - ca:4 + ca + 1, 6 - cr:6 + cr + 1] = ac
    return w, convolve2d(w, k, mode='valid').astype(np.complex64), true


def test_slc_correlation(rng):
    from moraine.api.co import slc_correlation
    white, slc, true = _correlated_speckle(rng)
    rho2 = slc_correlation(white)
    assert rho2.shape == (9, 13) and rho2.dtype == np.float32 and rho2[4, 6] == 1
    assert abs(rho2.sum() - 1) < 0.02
    rho2 = slc_correlation(slc)
    np.testing.assert_allclose(rho2, true, atol=0.01)
    assert abs(rho2.sum() / true.sum() - 1) < 0.03


def test_slc_correlation_robust(rng):
    """Azimuth phase ramps (TOPS), texture, 0 and NaN pixels do not change the estimate."""
    from moraine.api.co import slc_correlation
    _, slc, true = _correlated_speckle(rng)
    ref = slc_correlation(slc)
    y = np.arange(slc.shape[0])[:, None]
    ramp = (slc * np.exp(1j * 0.02 * y ** 2)).astype(np.complex64)
    np.testing.assert_allclose(slc_correlation(ramp), ref, atol=1e-4)
    texture = (slc * (1 + 3 * (np.arange(slc.shape[1]) % 200 < 100))).astype(np.complex64)
    holes = slc.copy()
    holes[rng.random(slc.shape) < 0.1] = 0
    holes[rng.random(slc.shape) < 0.02] = np.nan
    for s in (texture, holes):
        np.testing.assert_allclose(slc_correlation(s), true, atol=0.01)
