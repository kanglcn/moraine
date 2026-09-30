import numpy as np
import pytest

import moraine as mr
from moraine.api.rtree import HilbertRtree, _is_outside, _is_inside, _is_inside_bf, _expand_ranges


def test_box_helpers():
    assert _is_outside(np.array([0.5, 0.5, 0.6, 0.6]), np.array([0.24330907, 0.02271734, 0.90802693, 0.33264098]))
    assert not _is_inside(np.array([0.5, 0.5, 0.6, 0.6]), np.array([0.22144525, 0.06039695, 0.97828617, 0.99201584]))
    bounds = np.array([0.5, 0.5, 0.6, 0.6])
    x = np.array([0.55, 0.58, 0.55, 0.8]); y = np.array([0.8, 0.52, 0.58, 0.55])
    np.testing.assert_array_equal(_is_inside_bf(bounds, x, y), [False, True, True, False])
    np.testing.assert_array_equal(_expand_ranges(np.array([[0, 2], [3, 6], [8, 10]])), [0, 1, 3, 4, 5, 8, 9])


def _points(rng, n, size):
    g = np.sort(rng.choice(size * size, size=n, replace=False))
    gix = np.stack(np.unravel_index(g, (size, size)), axis=-1).astype(np.int32)
    key = mr.pc_sort(mr.pc_hix(gix, shape=(size, size)))
    return gix[:, 1][key] / size, gix[:, 0][key] / size


def test_bbox_query_in_bound(rng):
    x, y = _points(rng, 1000, 100)
    rtree = HilbertRtree.build(x, y, page_size=32)
    idx = rtree.bbox_query([x.min(), y.min(), x.max(), y.max()], x, y)
    assert idx[-1] < 1000


@pytest.fixture(scope='module')
def big():
    rng = np.random.default_rng(1)
    x, y = _points(rng, 200_000, 2000)
    return x, y, HilbertRtree.build(x, y, page_size=4096)


def test_bbox_query_matches_brute_force(big):
    x, y, rtree = big
    x0, y0, xm, ym = 0.9, 0.9, 1.0, 1.0
    np.testing.assert_array_equal(rtree.bbox_query([x0, y0, xm, ym], x, y),
                                  np.where((x > x0) & (x < xm) & (y > y0) & (y < ym))[0])


def test_maybe_covered_ranges(big):
    x, y, rtree = big
    x0, y0, xm, ym = 0.9, 0.9, 1.0, 1.0
    ranges, is_covered = rtree.maybe_covered_ranges([x0, y0, xm, ym])
    for (a, b), covered in zip(ranges, is_covered):
        if covered:
            assert np.all((x[a:b] >= x0) & (x[a:b] <= xm) & (y[a:b] >= y0) & (y[a:b] <= ym))


def test_save_load(big, tmp_path):
    x, y, rtree = big
    rtree.save(str(tmp_path / 'rtree.zarr'))
    loaded = HilbertRtree.load(str(tmp_path / 'rtree.zarr'))
    np.testing.assert_array_equal(loaded.bounds_tree, rtree.bounds_tree)
    bounds = [0.2, 0.3, 0.25, 0.4]
    np.testing.assert_array_equal(loaded.bbox_query(bounds, x, y), rtree.bbox_query(bounds, x, y))
