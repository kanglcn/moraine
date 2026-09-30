import math

import numpy as np
import pytest

from moraine.api.pc import (_unravel_gix, _check_idx_sorted, _pc_split_by_chunk, _gix_ras_chunk,
                        pc_hix, pc_gix, pc_sort, pc2ras, pc_union, pc_intersect, pc_diff)


def _random_gix(rng, shape, n):
    g = np.sort(rng.choice(shape[0] * shape[1], size=n, replace=False))
    return np.stack(np.unravel_index(g, shape), axis=-1).astype(np.int32)


def test_unravel_gix():
    np.testing.assert_array_equal(_unravel_gix(np.array([1, 4, 7, 9]), (4, 4)),
                                  np.array([[0, 1], [1, 0], [1, 3], [2, 1]], dtype=np.int32))


def test_check_idx_sorted(rng):
    _check_idx_sorted(np.sort(rng.choice(5000, size=1000, replace=False)))


def test_hix_gix_roundtrip(rng):
    gix = _random_gix(rng, (100, 100), 1000)
    np.testing.assert_array_equal(pc_gix(pc_hix(gix, (100, 100)), (100, 100)), gix)


def test_pc_sort():
    gix = np.stack(([0, 1, 5, 3, 4, 3], [3, 5, 2, 4, 6, 8]), axis=-1)
    np.testing.assert_equal(gix[pc_sort(gix, shape=(6, 9))],
                            np.stack(([0, 1, 3, 3, 4, 5], [3, 5, 4, 8, 6, 2]), axis=-1))
    hix = np.array([0, 1, 5, 4, 3])
    np.testing.assert_equal(hix[pc_sort(hix)], [0, 1, 3, 4, 5])


def test_pc2ras(rng):
    a = np.arange(1000 * 50, dtype=np.float32).reshape(1000, 50)
    gix = _random_gix(rng, (100, 50), 1000)
    ras = pc2ras(gix, a, shape=(100, 50))
    np.testing.assert_array_equal(ras[gix[:, 0], gix[:, 1]], a)
    hix = pc_hix(gix, shape=(100, 50))
    key = pc_sort(hix)
    np.testing.assert_array_equal(pc2ras(hix[key], a[key], shape=(100, 50)), ras)


def _xp(gpu):
    if gpu:
        import cupy as cp
        return cp
    return np


GPU = [False, pytest.param(True, marks=pytest.mark.gpu)]
RAS = np.array([[4, 3, 8, 3], [4, 7, 2, 6], [9, 0, 3, 7], [1, 4, 2, 6]])
GIX1 = np.stack(([0, 0, 1, 1, 2, 3], [2, 3, 0, 3, 1, 2]), axis=-1).astype(np.int32)
GIX2 = np.stack(([0, 0, 1, 2, 2, 3], [0, 3, 1, 1, 3, 1]), axis=-1).astype(np.int32)
D1 = np.array([3, 2, 5, 4, 32, 2]); D2 = np.array([3, 5, 6, 2, 1, 4])
HIX1 = np.array([2, 3, 4, 7, 9, 14], dtype=np.int64); HIX2 = np.array([0, 3, 5, 9, 11, 13], dtype=np.int64)


def _get(x):
    return x.get() if hasattr(x, 'get') else x


@pytest.mark.parametrize('gpu', GPU)
def test_pc_union_gix(gpu):
    xp = _xp(gpu)
    gix, inv1, inv2, iidx2 = pc_union(xp.asarray(GIX1), xp.asarray(GIX2), shape=(4, 4))
    gix, inv1, inv2, iidx2 = map(_get, (gix, inv1, inv2, iidx2))
    data = np.empty(gix.shape[0], dtype=D1.dtype)
    data[inv1] = D1; data[inv2] = D2[iidx2]
    np.testing.assert_equal(data, [3, 3, 2, 5, 6, 4, 32, 1, 4, 2])
    np.testing.assert_equal(RAS[gix[:, 0], gix[:, 1]], [4, 8, 3, 4, 7, 6, 0, 7, 4, 2])


@pytest.mark.parametrize('gpu', GPU)
def test_pc_union_hix(gpu):
    xp = _xp(gpu)
    hix, inv1, inv2, iidx2 = map(_get, pc_union(xp.asarray([2, 4, 6, 8, 10], dtype=np.int64),
                                                 xp.asarray([1, 2, 3, 6, 9], dtype=np.int64)))
    d1 = np.array([3, 2, 5, 4, 32]); d2 = np.array([3, 5, 6, 2, 1])
    data = np.empty(hix.shape[-1], dtype=d1.dtype)
    data[inv1] = d1; data[inv2] = d2[iidx2]
    np.testing.assert_equal(data, [3, 3, 6, 2, 5, 4, 1, 32])


@pytest.mark.gpu
def test_pc_union_cpu_gpu_agree(rng):
    import cupy as cp
    h1 = np.sort(rng.choice(100_000, size=1_000, replace=False))
    h2 = np.sort(rng.choice(100_000, size=1_000, replace=False))
    for a, b in zip(pc_union(h1, h2), pc_union(cp.asarray(h1), cp.asarray(h2))):
        np.testing.assert_array_equal(a, b.get())


@pytest.mark.parametrize('gpu', GPU)
def test_pc_intersect(gpu):
    xp = _xp(gpu)
    gix, i1, i2 = map(_get, pc_intersect(xp.asarray(GIX1), xp.asarray(GIX2), shape=(4, 4)))
    np.testing.assert_equal(gix, [[0, 3], [2, 1]])
    np.testing.assert_equal(RAS[gix[:, 0], gix[:, 1]], [3, 0])
    np.testing.assert_equal(D1[i1], [2, 32]); np.testing.assert_equal(D2[i2], [5, 2])
    hix, i1, i2 = map(_get, pc_intersect(xp.asarray(HIX1), xp.asarray(HIX2)))
    np.testing.assert_equal(hix, [3, 9])
    np.testing.assert_equal(D1[i1], [2, 32]); np.testing.assert_equal(D2[i2], [5, 2])


@pytest.mark.parametrize('gpu', GPU)
def test_pc_diff(gpu):
    xp = _xp(gpu)
    gix, i1 = map(_get, pc_diff(xp.asarray(GIX1), xp.asarray(GIX2), shape=(4, 4)))
    np.testing.assert_equal(gix, np.stack(([0, 1, 1, 3], [2, 0, 3, 2]), axis=-1))
    np.testing.assert_equal(i1, [0, 2, 3, 5])
    hix, i1 = map(_get, pc_diff(xp.asarray(HIX1), xp.asarray(HIX2)))
    np.testing.assert_equal(hix, [2, 4, 7, 14])
    np.testing.assert_equal(i1, [0, 2, 3, 5])


def test_pc_split_by_chunk_and_gix_ras_chunk(rng):
    shape, chunks = (100, 100), (23, 26)
    gix = _random_gix(rng, shape, 1000)
    chunk_idx, bounds, inv = _pc_split_by_chunk(gix, chunks, shape)
    az_b = np.minimum(np.arange(0, shape[0] + chunks[0], chunks[0]), shape[0])
    r_b = np.minimum(np.arange(0, shape[1] + chunks[1], chunks[1]), shape[1])
    n_r = math.ceil(shape[1] / chunks[1])
    assert len(bounds) == math.ceil(shape[0] / chunks[0]) * n_r + 1
    sorted_gix = gix[chunk_idx]
    local = _gix_ras_chunk(sorted_gix, bounds, chunks, shape)
    for i in range(len(bounds) - 1):
        a, r = divmod(i, n_r)
        assert bounds[i] <= bounds[i + 1]
        g = sorted_gix[bounds[i]:bounds[i + 1]]
        assert np.all(np.diff(np.ravel_multi_index((g[:, 0], g[:, 1]), shape)) > 0)
        assert np.all((g[:, 0] >= az_b[a]) & (g[:, 0] < az_b[a + 1]) & (g[:, 1] >= r_b[r]) & (g[:, 1] < r_b[r + 1]))
        l = local[bounds[i]:bounds[i + 1]]
        np.testing.assert_array_equal(l[:, 0] + a * chunks[0], g[:, 0])
        np.testing.assert_array_equal(l[:, 1] + r * chunks[1], g[:, 1])
    np.testing.assert_array_equal(gix[chunk_idx][inv], gix)
