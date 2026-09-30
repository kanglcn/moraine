import numpy as np

from moraine.api.chunk_ import (all_chunk_slices, all_chunk_slices_with_overlap,
                            chunkwise_slicing_mapping, chunkwise_knn_mapping)


def test_all_chunk_slices_cover_array():
    a = np.arange(50 * 100 * 11).reshape(50, 100, 11)
    covered = np.zeros(a.shape, dtype=int)
    for s in all_chunk_slices(a.shape, (15, 40, 6)):
        covered[s] += 1
    assert (covered == 1).all()
    assert len(all_chunk_slices((343,), (15,))) == 23


def test_all_chunk_slices_with_overlap():
    shape, chunks, depth = (50, 100, 11), (15, 40, 6), (1, 2, 0)
    plain = all_chunk_slices(shape, chunks)
    overlap = all_chunk_slices_with_overlap(shape, chunks, depth)
    assert len(plain) == len(overlap)
    for p, o in zip(plain, overlap):
        for ps, os_, d, n in zip(p, o, depth, shape):
            assert os_.start == max(ps.start - d, 0)
            assert os_.stop == min(ps.stop + d, n)


def test_chunkwise_slicing_mapping_reconstructs():
    a = np.random.default_rng(0).random((50, 100))
    out = np.full_like(a, np.nan)
    for in_s, out_s, map_s in zip(*chunkwise_slicing_mapping(a.shape, (15, 40), (1, 2))):
        out[out_s] = a[in_s][map_s]
    np.testing.assert_array_equal(out, a)


def test_chunkwise_knn_mapping():
    rng = np.random.default_rng(0)
    x, y = rng.random(5000), rng.random(5000)
    order = np.lexsort((x, y))  # any fixed order of points
    x, y = x[order], y[order]
    in_indices, out_slices, map_indices = chunkwise_knn_mapping(x, y, 1000, k=32)
    covered = np.zeros(x.shape[0], dtype=int)
    for in_idx, out_s, map_idx in zip(in_indices, out_slices, map_indices):
        covered[out_s] += 1
        # the mapped input points are exactly the output points of this chunk
        np.testing.assert_array_equal(in_idx[map_idx], np.arange(out_s.start, out_s.stop))
    assert (covered == 1).all()
