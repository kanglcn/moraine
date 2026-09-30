import itertools

import numpy as np
import pytest
from scipy import stats

from moraine.api.shp import ks_test, select_shp


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
    np.testing.assert_array_almost_equal(p, cp.asnumpy(p_cp))


def test_select_shp(rng):
    p = rng.random(100 * 100 * 5 * 5).astype(np.float32)
    p[rng.choice(p.size, size=1000, replace=False)] = np.nan
    p = p.reshape(100, 100, 5, 5)
    p = (p + np.transpose(p, axes=(0, 1, 3, 2))) / 2
    for i in range(5):
        p[:, :, i, i] = 1
    is_shp, shp_num = select_shp(p, 0.05)
    np.testing.assert_array_equal(is_shp, p < 0.05)
    np.testing.assert_array_equal(shp_num, np.count_nonzero(p < 0.05, axis=(-2, -1)).astype(np.int32))
