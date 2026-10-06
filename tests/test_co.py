import itertools
import math

import numpy as np
import pytest

import moraine as mr
from moraine.api.co import emperical_co, emperical_co_pc, ad_intf_pc, uncompress_coh, isPD, regularize_spectral
from moraine.api.co import _slc_correlation, _rslc_rho2, _shp_n_looks, _normalize_local_power
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
    np.testing.assert_allclose(coh[:, 0], ifg, rtol=1e-6, atol=1e-6)  # float32 sums in another order


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
    white, slc, true = _correlated_speckle(rng)
    rho2 = _slc_correlation(white)
    assert rho2.shape == (9, 13) and rho2.dtype == np.float32 and rho2[4, 6] == 1
    assert abs(rho2.sum() - 1) < 0.02
    rho2 = _slc_correlation(slc)
    np.testing.assert_allclose(rho2, true, atol=0.01)
    assert abs(rho2.sum() / true.sum() - 1) < 0.03


def test_slc_correlation_robust(rng):
    """Azimuth phase ramps (TOPS), texture, 0 and NaN pixels do not change the estimate."""
    _, slc, true = _correlated_speckle(rng)
    ref = _slc_correlation(slc)
    y = np.arange(slc.shape[0])[:, None]
    ramp = (slc * np.exp(1j * 0.02 * y ** 2)).astype(np.complex64)
    np.testing.assert_allclose(_slc_correlation(ramp), ref, atol=1e-4)
    texture = (slc * (1 + 3 * (np.arange(slc.shape[1]) % 200 < 100))).astype(np.complex64)
    holes = slc.copy()
    holes[rng.random(slc.shape) < 0.1] = 0
    holes[rng.random(slc.shape) < 0.02] = np.nan
    for s in (texture, holes):
        np.testing.assert_allclose(_slc_correlation(s), true, atol=0.01)


def _rho2_separable():
    """|rho|^2 table (lags -4..4, -6..6) of a separable correlation like Sentinel-1 IW."""
    az = {0: 1.0, 1: 0.64 ** 2, 2: 0.15 ** 2}
    rg = {0: 1.0, 1: 0.43 ** 2, 2: 0.03 ** 2}
    return np.array([[az.get(abs(a), 0) * rg.get(abs(r), 0) for r in range(-6, 7)] for a in range(-4, 5)], np.float32)


def _shp_masks(rng):
    full = np.ones((11, 11), bool)
    blob = np.zeros((11, 11), bool); blob[2:9, 2:9] = True; blob[5, 1] = True
    scattered = np.zeros(121, bool); scattered[rng.choice(121, 50, replace=False)] = True
    return np.stack([full, blob, scattered.reshape(11, 11), np.zeros((11, 11), bool)])


def test_shp_n_looks(rng):
    masks, rho2 = _shp_masks(rng), _rho2_separable()
    n_looks = _shp_n_looks(masks, rho2)
    assert n_looks.dtype == np.float32 and n_looks.shape == (4,)
    for k in range(3):                                           # n^2 / sum_{p,q} |rho(p - q)|^2 from its definition
        p = np.argwhere(masks[k]); d = p[:, None] - p[None]
        inside = (np.abs(d[..., 0]) <= 4) & (np.abs(d[..., 1]) <= 6)
        s = np.where(inside, rho2[np.clip(d[..., 0] + 4, 0, 8), np.clip(d[..., 1] + 6, 0, 12)], 0).sum()
        assert n_looks[k] == pytest.approx(len(p) ** 2 / s, rel=1e-5)
    assert n_looks[1] < n_looks[2] < 50 and n_looks[3] == 0     # compact < scattered < number of SHPs
    white = np.pad([[1.0]], ((4, 4), (6, 6))).astype(np.float32)
    np.testing.assert_array_equal(_shp_n_looks(masks, white), masks.sum(axis=(1, 2)))


def test_shp_n_looks_asymmetric_table(rng):
    """An estimated table is not exactly symmetric and has small negative values (taken as 0)."""
    masks = np.concatenate([_shp_masks(rng), rng.random((50, 11, 11)) < 0.5])
    rho2 = _rho2_separable() * (1 + 0.05 * rng.standard_normal((9, 13))).astype(np.float32) - np.float32(0.002)
    rho2[4, 6] = 1
    n_looks = _shp_n_looks(masks, rho2)
    for k in range(len(masks)):
        p = np.argwhere(masks[k]); d = p[:, None] - p[None]
        inside = (np.abs(d[..., 0]) <= 4) & (np.abs(d[..., 1]) <= 6)
        v = np.where(inside, rho2[np.clip(d[..., 0] + 4, 0, 8), np.clip(d[..., 1] + 6, 0, 12)], 0)
        expected = len(p) ** 2 / np.maximum(v, 0).sum() if len(p) else 0
        assert n_looks[k] == pytest.approx(expected, rel=1e-5)
    with pytest.raises(ValueError, match='64'):
        _shp_n_looks(np.ones((2, 3, 65), bool), rho2)


def test_normalize_local_power(rng):
    """The SLC divided by the square root of the local mean power of the valid pixels (15 x 15, symmetric
    extension at the borders), against numpy."""
    slc = ((rng.standard_normal((40, 57)) + 1j * rng.standard_normal((40, 57))) * rng.uniform(0.1, 10, (40, 57))).astype(np.complex64)
    slc[3:9, 10:20] = 0
    slc[20, 5] = np.nan
    power = np.abs(slc.astype(np.complex128)) ** 2
    valid = np.isfinite(power) & (power > 0)
    def box(x):
        c = np.pad(np.pad(x, 7, mode='symmetric').cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[15:, 15:] - c[:-15, 15:] - c[15:, :-15] + c[:-15, :-15]
    expected = np.where(valid, slc / np.sqrt(box(np.where(valid, power, 0)) / box(valid.astype(float))), 0)
    np.testing.assert_allclose(_normalize_local_power(slc, 7), expected, rtol=1e-5, atol=1e-7)


@pytest.mark.gpu
def test_shp_n_looks_gpu(rng):
    import cupy as cp
    masks = rng.random((5000, 11, 11)) < 0.6
    np.testing.assert_allclose(_shp_n_looks(cp.asarray(masks), _rho2_separable()).get(), _shp_n_looks(masks, _rho2_separable()), rtol=1e-5)


def _correlated_stack(rng, nimages=3):
    """rslc stack (600, 800, nimages) of independent speckle with the correlation of `_correlated_speckle` and
    its true |rho|^2."""
    images = [_correlated_speckle(rng) for _ in range(nimages)]
    return np.stack([s for _, s, _ in images], axis=-1), images[0][2].astype(np.float32)


def _n_looks_case(rng):
    stack, true = _correlated_stack(rng)
    idx = np.array([[100, 100], [200, 300], [300, 500]], np.int32)
    return stack, idx, _shp_masks(rng)[:3], true


def test_emperical_co_pc_n_looks(rng):
    """The effective number of looks from the SHP positions and the speckle correlation of the rslc stack."""
    stack, idx, masks, true = _n_looks_case(rng)
    coh, n_looks = emperical_co_pc(stack, idx, masks, return_n_looks=True)
    np.testing.assert_array_equal(coh, emperical_co_pc(stack, idx, masks))
    np.testing.assert_array_equal(n_looks, _shp_n_looks(masks, _rslc_rho2(stack)))
    np.testing.assert_allclose(n_looks, _shp_n_looks(masks, true), rtol=0.03)


def test_emperical_co_pc_n_looks_small(rng):
    """An rslc stack too small to estimate the speckle correlation: the number of SHPs."""
    rslc, is_shp = synthetic_shp(rng, p_true=0.5)
    _, idx, ds_is_shp = _ds_can(is_shp)
    assert _rslc_rho2(rslc) is None
    _, n_looks = emperical_co_pc(rslc, idx, ds_is_shp, return_n_looks=True)
    assert n_looks.dtype == np.float32
    np.testing.assert_array_equal(n_looks, ds_is_shp.sum(axis=(1, 2)))


@pytest.mark.gpu
def test_emperical_co_pc_n_looks_gpu(rng):
    import cupy as cp
    stack, idx, masks, _ = _n_looks_case(rng)
    rslc, is_shp = synthetic_shp(rng, p_true=0.5)
    _, small_idx, small_is_shp = _ds_can(is_shp)
    for s, i, m in ((stack, idx, masks), (rslc, small_idx, small_is_shp)):
        _, n_looks = emperical_co_pc(s, i, m, return_n_looks=True)
        _, n_looks_gpu = emperical_co_pc(cp.asarray(s), cp.asarray(i), cp.asarray(m), return_n_looks=True)
        np.testing.assert_allclose(n_looks_gpu.get(), n_looks, rtol=1e-5)


GPU = [False, pytest.param(True, marks=pytest.mark.gpu)]


@pytest.mark.parametrize('gpu', GPU)
def test_emperical_co_pc_reference(rng, gpu):
    # coherence of the SHP samples in float64; nan pixels that are not SHPs of a point must not count
    rslc, is_shp = synthetic_shp(rng, shape=(6, 9), nimages=7, half_az_win=1, half_r_win=2)
    rslc[2, 4, 3] = np.nan
    idx = np.stack(np.meshgrid(np.arange(6), np.arange(9), indexing='ij'), -1).reshape(-1, 2)
    pc_is_shp = is_shp.reshape(-1, *is_shp.shape[2:])
    pairs = mr.TempNet.from_bandwidth(7).image_pairs
    expected = np.empty((idx.shape[0], pairs.shape[0]), np.complex128)
    for i, (a, r) in enumerate(idx):
        k, l = np.nonzero(pc_is_shp[i])
        x = rslc[a + k - 1, r + l - 2].astype(np.complex128)    # (n_shp, nimages); synthetic SHPs stay inside
        for p, (m, j) in enumerate(pairs):
            expected[i, p] = (x[:, m] * x[:, j].conj()).sum() / np.sqrt((np.abs(x[:, m])**2).sum() * (np.abs(x[:, j])**2).sum())
    if gpu:
        import cupy as cp
        coh = emperical_co_pc(cp.asarray(rslc), cp.asarray(idx), cp.asarray(pc_is_shp)).get()
    else:
        coh = emperical_co_pc(rslc, idx, pc_is_shp)
    assert np.isnan(coh).any() and not np.isnan(coh).all()
    np.testing.assert_allclose(coh, expected, rtol=1e-5, atol=1e-6)
