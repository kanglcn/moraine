import itertools

import numpy as np
import pytest
from scipy import stats

from moraine.api.shp import ks_test, select_shp, _ks_p_numba


@pytest.fixture
def sample_stack(rng):
    nlines, width, nimages = 5, 5, 5
    samples = [np.sort(stats.uniform.rvs(size=nimages, random_state=rng))] + \
              [np.sort(stats.norm.rvs(size=nimages, random_state=rng)) for _ in range(nlines * width - 1)]
    return np.stack(samples).reshape(nlines, width, nimages).astype(np.float32)


def test_ks_test_matches_scipy(sample_stack):
    az_half_win = r_half_win = 5
    nlines, width, _ = sample_stack.shape
    dist, p = ks_test(sample_stack, az_half_win, r_half_win, return_dist=True)
    np.testing.assert_array_equal(p, ks_test(sample_stack, az_half_win, r_half_win, return_dist=False))
    assert dist.shape == (nlines, width, 2 * az_half_win + 1, 2 * r_half_win + 1)
    for az, r, k, l in itertools.product(range(nlines), range(width), range(2 * az_half_win + 1), range(2 * r_half_win + 1)):
        sec_az, sec_r = az + k - az_half_win, r + l - r_half_win
        if sec_az < 0 or sec_az >= nlines or sec_r < 0 or sec_r >= width:
            assert np.isnan(dist[az, r, k, l])
        else:
            scipy_dist = stats.ks_2samp(sample_stack[az, r], sample_stack[sec_az, sec_r], method='asymp').statistic
            assert abs(dist[az, r, k, l] - scipy_dist) < 1e-7


@pytest.mark.gpu
def test_ks_test_gpu(sample_stack):
    import cupy as cp
    dist, p = ks_test(sample_stack, 5, 5, return_dist=True)
    dist_cp, p_cp = ks_test(cp.asarray(sample_stack), 5, 5, return_dist=True)
    np.testing.assert_array_equal(dist, cp.asnumpy(dist_cp))
    np.testing.assert_array_equal(p, cp.asnumpy(p_cp))


def test_select_shp(rng):
    p = rng.random(100 * 100 * 5 * 5).astype(np.float32)
    p[rng.choice(p.size, size=1000, replace=False)] = np.nan
    p = p.reshape(100, 100, 5, 5)
    p = (p + np.transpose(p, axes=(0, 1, 3, 2))) / 2
    for i in range(5):
        p[:, :, i, i] = 1
    is_shp, shp_num = select_shp(p, 0.05)
    np.testing.assert_array_equal(is_shp, p >= 0.05)
    np.testing.assert_array_equal(shp_num, np.count_nonzero(p >= 0.05, axis=(-2, -1)).astype(np.int32))


def test_ks_p_identical_samples():
    # the KS statistic 0 (identical samples, e.g. the centre pixel itself) has the p value 1
    assert _ks_p_numba(0.0) == 1.0
    p = ks_test(np.sort(np.random.default_rng(0).random((3, 3, 15)), axis=-1).astype(np.float32), 1, 1)
    np.testing.assert_array_equal(p[:, :, 1, 1], 1.0)


def test_select_shp_same_distribution(rng):
    # intensities of two regions whose mean differs by a factor 10: the SHPs of a pixel are in its own region
    n_az, n_r, n = 20, 20, 30
    scale = np.where(np.arange(n_r) < n_r // 2, 1.0, 10.0)
    rmli = (rng.exponential(size=(n_az, n_r, n)) * scale[None, :, None]).astype(np.float32)
    ah = rh = 3
    is_shp, shp_num = select_shp(ks_test(rmli, ah, rh), 0.05)
    side = np.arange(n_r) < n_r // 2
    for j in (8, 9, 10, 11):  # windows across the border
        win = np.arange(j - rh, j + rh + 1)
        same = (side[win] == side[j])[None, :] & np.ones((2 * ah + 1, 1), bool)
        s = is_shp[5:15, j]
        assert s[:, ~same].mean() < 0.01      # the other region is rejected
        assert s[:, same].mean() > 0.9        # the own region is kept (5 % false rejections expected)
        assert s[:, ah, rh].all()                # the centre pixel itself


GPU = [False, pytest.param(True, marks=pytest.mark.gpu)]


@pytest.mark.parametrize('gpu', GPU)
def test_ks_test_reference(rng, gpu):
    # the KS statistic and its p value pair by pair, nan where a pixel is nan or outside the image
    n_az, n_r, n, ah, rh = 7, 9, 12, 2, 3
    rmli = rng.random((n_az, n_r, n)).astype(np.float32)
    rmli[1, 2, 5] = np.nan; rmli[4, 4, :] = np.nan
    s = np.sort(rmli, axis=-1)
    en = np.sqrt(n / 2)
    dist = np.full((n_az, n_r, 2 * ah + 1, 2 * rh + 1), np.nan, np.float32); p = dist.copy()
    for i, j, l, m in itertools.product(range(n_az), range(n_r), range(2 * ah + 1), range(2 * rh + 1)):
        si, sj = i + l - ah, j + m - rh
        if 0 <= si < n_az and 0 <= sj < n_r and not (np.isnan(s[i, j]).any() or np.isnan(s[si, sj]).any()):
            x = np.concatenate((s[i, j], s[si, sj]))
            k = np.abs(np.searchsorted(s[i, j], x, side='right') - np.searchsorted(s[si, sj], x, side='right')).max()
            dist[i, j, l, m] = k / n
            p[i, j, l, m] = _ks_p_numba((en + 0.12 + 0.11 / en) * (k / n))
    if gpu:
        import cupy as cp
        d_out, p_out = (a.get() for a in ks_test(cp.asarray(rmli), ah, rh, return_dist=True))
    else:
        d_out, p_out = ks_test(rmli, ah, rh, return_dist=True)
    np.testing.assert_array_equal(d_out, dist)
    np.testing.assert_array_equal(p_out, p)
