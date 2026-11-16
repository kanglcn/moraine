import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from conftest import write_zarr, read_zarr as r, random_gix


@pytest.fixture
def w(tmp_path):
    """w(name, array, chunks) writes a zarr array in tmp_path and returns its path; w(name) only returns the path."""
    def _w(name, data=None, chunks=None):
        return str(tmp_path / name) if data is None else write_zarr(tmp_path / name, data, chunks)
    return _w


def _cplx(rng, *shape):
    return (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex64)


def _hix(rng, n, sort=True):
    h = rng.choice(100 * 100, size=n, replace=False).astype(np.int64)
    if sort: h.sort()
    return h


def test_ras2pc(w, rng):
    ras1 = rng.random((100, 100)).astype(np.float32); ras2 = _cplx(rng, 100, 100, 3)
    gix = random_gix(rng, (100, 100), 1000)
    w('gix.zarr', gix, (200, 1)); w('ras1.zarr', ras1, (20, 100)); w('ras2.zarr', ras2, (20, 100, 1))
    mc.ras2pc(w('gix.zarr'), w('ras1.zarr'), w('pc1.zarr'))
    np.testing.assert_array_equal(r(w('pc1.zarr')), ras1[gix[:, 0], gix[:, 1]])
    mc.ras2pc(w('gix.zarr'), ras=[w('ras1.zarr'), w('ras2.zarr')], pc=[w('pc1.zarr'), w('pc2.zarr')])
    np.testing.assert_array_equal(r(w('pc1.zarr')), ras1[gix[:, 0], gix[:, 1]])
    np.testing.assert_array_equal(r(w('pc2.zarr')), ras2[gix[:, 0], gix[:, 1]])


def test_pc_concat(w, rng):
    pc = _cplx(rng, 1000, 3)
    w('pc1.zarr', pc[:300], (300, 1)); w('pc2.zarr', pc[300:], (700, 1))
    mc.pc_concat([w('pc1.zarr'), w('pc2.zarr')], w('pc.zarr'), chunks=500)
    np.testing.assert_array_equal(r(w('pc.zarr')), pc)


def test_ras2pc_ras_chunk(w, rng):
    shape, chunks = (100, 100), (20, 20)
    ras1 = rng.random(shape).astype(np.float32); ras2 = _cplx(rng, *shape, 3)
    gix = random_gix(rng, shape, 1000)
    w('gix.zarr', gix, (200, 1)); w('ras1.zarr', ras1, chunks); w('ras2.zarr', ras2, (*chunks, 1))
    mc.ras2pc_ras_chunk(w('gix.zarr'), w('ras1.zarr'), w('pc1'), key=w('key.zarr'))
    mc.pc_concat(w('pc1'), w('pc1.zarr'), key=w('key.zarr'), chunks=200)
    np.testing.assert_array_equal(r(w('pc1.zarr')), ras1[gix[:, 0], gix[:, 1]])
    mc.ras2pc_ras_chunk(w('gix.zarr'), ras=[w('ras1.zarr'), w('ras2.zarr')], pc=[w('pc1'), w('pc2')], key=w('key.zarr'))
    for name, ras in [('pc1', ras1), ('pc2', ras2)]:
        mc.pc_concat(w(name), w(name + '.zarr'), key=w('key.zarr'), chunks=200)
        np.testing.assert_array_equal(r(w(name + '.zarr')), ras[gix[:, 0], gix[:, 1]])
    z = zarr.open(w('pc2') + '/0.zarr', mode='r')
    assert z.chunks == (z.shape[0], 1)                           # a stack: one image per chunk
    # a window array (nlines, width, az_win, r_win): the whole window of a point in one chunk (decision 0032)
    ras3 = rng.random((*shape, 3, 3)).astype(np.float32)
    w('ras3.zarr', ras3, (*chunks, 1, 1))
    mc.ras2pc_ras_chunk(w('gix.zarr'), w('ras3.zarr'), w('pc3'), key=w('key3.zarr'))
    z = zarr.open(w('pc3') + '/0.zarr', mode='r')
    assert z.chunks == (z.shape[0], 3, 3)
    mc.pc_concat(w('pc3'), w('pc3.zarr'), key=w('key3.zarr'), chunks=200)
    np.testing.assert_array_equal(r(w('pc3.zarr')), ras3[gix[:, 0], gix[:, 1]])
    # without key the output is in chunk order
    chunk_idx = mr.api.pc.raster._pc_split_by_chunk(gix, chunks, shape)[0]
    sgix = gix[chunk_idx]
    for name, ras in [('pc1', ras1), ('pc2', ras2)]:
        mc.pc_concat(w(name), w(name + '.zarr'), chunks=200)
        np.testing.assert_array_equal(r(w(name + '.zarr')), ras[sgix[:, 0], sgix[:, 1]])


def test_pc2ras(w, rng):
    pc1 = rng.random(1000).astype(np.float32); pc2 = _cplx(rng, 1000, 3)
    gix = random_gix(rng, (100, 100), 1000)
    ras1 = np.full((100, 100), np.nan, np.float32); ras2 = np.full((100, 100, 3), np.nan, np.complex64)
    ras1[gix[:, 0], gix[:, 1]] = pc1; ras2[gix[:, 0], gix[:, 1]] = pc2
    w('gix.zarr', gix, (200, 1)); w('pc1.zarr', pc1, (200,)); w('pc2.zarr', pc2, (200, 1))
    mc.pc2ras(w('gix.zarr'), w('pc1.zarr'), w('ras1.zarr'), shape=(100, 100), chunks=(20, 100))
    np.testing.assert_array_equal(r(w('ras1.zarr')), ras1)
    mc.pc2ras(w('gix.zarr'), [w('pc1.zarr'), w('pc2.zarr')], [w('ras1.zarr'), w('ras2.zarr')], shape=(100, 100), chunks=(20, 100))
    np.testing.assert_array_equal(r(w('ras1.zarr')), ras1)
    np.testing.assert_array_equal(r(w('ras2.zarr')), ras2)


def test_pc_hix_gix(w, rng):
    gix = random_gix(rng, (100, 100), 1000)
    w('gix.zarr', gix, (100, 1))
    mc.pc_hix(w('gix.zarr'), w('hix.zarr'), shape=(100, 100))
    np.testing.assert_array_equal(r(w('hix.zarr')), mr.pc_hix(gix, (100, 100)))
    mc.pc_gix(w('hix.zarr'), w('gix_.zarr'), (100, 100))
    np.testing.assert_array_equal(r(w('gix_.zarr')), gix)


def test_pc_sort(w, rng):
    pc_in = rng.random(1000).astype(np.float32)
    gix_in = random_gix(rng, (100, 100), 1000, sort=False)
    ind = np.lexsort((gix_in[:, 1], gix_in[:, 0]))
    w('pc_in.zarr', pc_in, (100,)); w('gix_in.zarr', gix_in, (100, 1))
    mc.pc_sort(w('gix_in.zarr'), w('gix.zarr'), w('pc_in.zarr'), w('pc.zarr'), shape=(100, 100))
    np.testing.assert_array_equal(r(w('pc.zarr')), pc_in[ind])
    np.testing.assert_array_equal(r(w('gix.zarr')), gix_in[ind])
    hix_in = _hix(rng, 1000, sort=False)
    ind = np.argsort(hix_in, kind='stable')
    w('hix_in.zarr', hix_in, (100,))
    mc.pc_sort(w('hix_in.zarr'), w('hix.zarr'), w('pc_in.zarr'), w('pc.zarr'))
    np.testing.assert_array_equal(r(w('pc.zarr')), pc_in[ind])
    np.testing.assert_array_equal(r(w('hix.zarr')), hix_in[ind])


def test_pc_sort_empty_gix_needs_shape(w, tmp_path):
    """pc-sort of an empty grid index without `shape` asks for it (numpy's "zero-size array to reduction operation
    maximum which has no identity" before) and writes no output; with `shape` its outputs have no points."""
    w('e.zarr', np.zeros((0, 2), np.int32), (200, 1)); w('pc_in.zarr', np.zeros((0, 3), np.float32), (200, 1))
    with pytest.raises(ValueError, match=r'shape \(nlines, width\) is needed when a grid index is empty: '
                                         'the grid index has no points'):
        mc.pc_sort(w('e.zarr'), w('gix.zarr'), w('pc_in.zarr'), w('pc.zarr'), key=w('key.zarr'))
    assert not any((tmp_path / name).exists() for name in ('gix.zarr', 'pc.zarr', 'key.zarr'))
    mc.pc_sort(w('e.zarr'), w('gix.zarr'), w('pc_in.zarr'), w('pc.zarr'), shape=(100, 100), key=w('key.zarr'))
    assert [r(w(name)).shape for name in ('gix.zarr', 'pc.zarr', 'key.zarr')] == [(0, 2), (0, 3), (0,)]


@pytest.mark.parametrize('kind', ['gix', 'hix'])
def test_pc_union(w, rng, kind):
    pc1 = _cplx(rng, 1000, 3); pc2 = _cplx(rng, 800, 3)
    if kind == 'gix':
        i1, i2, c = random_gix(rng, (100, 100), 1000), random_gix(rng, (100, 100), 800), (200, 1)
    else:
        i1, i2, c = _hix(rng, 1000), _hix(rng, 800), (200,)
    idx, inv1, inv2, iidx2 = mr.pc_union(i1, i2)
    pc = np.empty((idx.shape[0], 3), pc1.dtype); pc[inv1] = pc1; pc[inv2] = pc2[iidx2]
    w('i1.zarr', i1, c); w('i2.zarr', i2, c); w('pc1.zarr', pc1, (200, 1)); w('pc2.zarr', pc2, (200, 1))
    mc.pc_union(w('i1.zarr'), w('i2.zarr'), w('i.zarr'), **({'shape': (100, 100)} if kind == 'gix' else {}))
    mc.pc_union(w('i1.zarr'), w('i2.zarr'), w('i.zarr'), w('pc1.zarr'), w('pc2.zarr'), w('pc.zarr'))
    np.testing.assert_array_equal(r(w('i.zarr')), idx)
    np.testing.assert_array_equal(r(w('pc.zarr')), pc)


@pytest.mark.parametrize('kind', ['gix', 'hix'])
def test_pc_intersect(w, rng, kind):
    pc1 = _cplx(rng, 1000, 3); pc2 = _cplx(rng, 800, 3)
    if kind == 'gix':
        i1, i2, c = random_gix(rng, (100, 100), 1000), random_gix(rng, (100, 100), 800), (200, 1)
    else:
        i1, i2, c = _hix(rng, 1000), _hix(rng, 800), (200,)
    idx, _, iidx2 = mr.pc_intersect(i1, i2)
    w('i1.zarr', i1, c); w('i2.zarr', i2, c); w('pc1.zarr', pc1, (200, 1)); w('pc2.zarr', pc2, (200, 1))
    mc.pc_intersect(w('i1.zarr'), w('i2.zarr'), w('i.zarr'), **({'shape': (100, 100)} if kind == 'gix' else {}))
    mc.pc_intersect(w('i1.zarr'), w('i2.zarr'), w('i.zarr'), pc2=w('pc2.zarr'), pc=w('pc.zarr'), prefer_1=False)
    np.testing.assert_array_equal(r(w('i.zarr')), idx)
    np.testing.assert_array_equal(r(w('pc.zarr')), pc2[iidx2])


@pytest.mark.parametrize('kind', ['gix', 'hix'])
def test_pc_diff(w, rng, kind):
    pc1 = _cplx(rng, 1000, 3)
    if kind == 'gix':
        i1, i2, c = random_gix(rng, (100, 100), 1000), random_gix(rng, (100, 100), 800), (200, 1)
    else:
        i1, i2, c = _hix(rng, 1000), _hix(rng, 800), (200,)
    idx, iidx1 = mr.pc_diff(i1, i2)
    w('i1.zarr', i1, c); w('i2.zarr', i2, c); w('pc1.zarr', pc1, (200, 1))
    mc.pc_diff(w('i1.zarr'), w('i2.zarr'), w('i.zarr'))
    mc.pc_diff(w('i1.zarr'), w('i2.zarr'), w('i.zarr'), pc1=w('pc1.zarr'), pc=w('pc.zarr'))
    np.testing.assert_array_equal(r(w('i.zarr')), idx)
    np.testing.assert_array_equal(r(w('pc.zarr')), pc1[iidx1])


def test_set_commands_empty_gix_need_shape(w, rng, tmp_path):
    """With an empty grid index the set commands and pc-select-data ask for `shape` (an IndexError before) and write
    no output; with `shape` they give the points of the other grid index or none."""
    gix = random_gix(rng, (100, 100), 50); pc_in = rng.random((50, 3)).astype(np.float32)
    w('e.zarr', gix[:0], (200, 1)); w('g.zarr', gix, (200, 1)); w('pc_in.zarr', pc_in, (200, 1))
    match = r'shape \(nlines, width\) is needed when a grid index is empty: the second grid index has no points'
    for fn in (mc.pc_union, mc.pc_intersect, mc.pc_diff):
        with pytest.raises(ValueError, match=match):
            fn(w('g.zarr'), w('e.zarr'), w('i.zarr'))
        assert not (tmp_path / 'i.zarr').exists()
    with pytest.raises(ValueError, match=match):
        mc.pc_select_data(w('g.zarr'), w('e.zarr'), w('pc_in.zarr'), w('pc.zarr'))
    assert not (tmp_path / 'pc.zarr').exists()
    mc.pc_union(w('e.zarr'), w('g.zarr'), w('i.zarr'), shape=(100, 100))
    np.testing.assert_array_equal(r(w('i.zarr')), gix)
    mc.pc_intersect(w('g.zarr'), w('e.zarr'), w('i.zarr'), shape=(100, 100))
    assert r(w('i.zarr')).shape == (0, 2)
    mc.pc_diff(w('g.zarr'), w('e.zarr'), w('i.zarr'), w('pc_in.zarr'), w('pc.zarr'), shape=(100, 100))
    np.testing.assert_array_equal(r(w('i.zarr')), gix); np.testing.assert_array_equal(r(w('pc.zarr')), pc_in)
    mc.pc_select_data(w('g.zarr'), w('e.zarr'), w('pc_in.zarr'), w('pc.zarr'), shape=(100, 100))
    assert r(w('pc.zarr')).shape == (0, 3)


def test_pc_logic_ras(w, rng):
    ras = rng.random((100, 100)).astype(np.float32)
    w('ras.zarr', ras, (10, 100))
    mc.pc_logic_ras(w('ras.zarr'), w('gix.zarr'), '(ras>=0.1)&(ras<=0.5)')
    np.testing.assert_array_equal(r(w('gix.zarr')), np.stack(np.where((ras >= 0.1) & (ras <= 0.5)), axis=-1))


def test_pc_logic_pc(w, rng):
    pc_in = rng.random(1000).astype(np.float32)
    keep = (pc_in >= 0.1) & (pc_in <= 0.5)
    gix_in = random_gix(rng, (100, 100), 1000); hix_in = _hix(rng, 1000)
    w('pc_in.zarr', pc_in, (100,)); w('gix_in.zarr', gix_in, (100, 1)); w('hix_in.zarr', hix_in, (100,))
    mc.pc_logic_pc(w('gix_in.zarr'), w('pc_in.zarr'), w('gix.zarr'), '(pc_in>=0.1)&(pc_in<=0.5)')
    np.testing.assert_array_equal(r(w('gix.zarr')), gix_in[keep])
    mc.pc_logic_pc(w('hix_in.zarr'), w('pc_in.zarr'), w('hix.zarr'), '(pc_in>=0.1)&(pc_in<=0.5)')
    np.testing.assert_array_equal(r(w('hix.zarr')), hix_in[keep])


def test_pc_select_data(w, rng):
    pc_in = rng.random((1000, 4)).astype(np.float32)
    iidx = np.sort(rng.choice(1000, size=500, replace=False))
    gix_in = random_gix(rng, (100, 100), 1000); hix_in = _hix(rng, 1000)
    w('pc_in.zarr', pc_in, (100, 1)); w('gix_in.zarr', gix_in, (100, 1)); w('gix.zarr', gix_in[iidx], (100, 1))
    mc.pc_select_data(w('gix_in.zarr'), w('gix.zarr'), w('pc_in.zarr'), w('pc.zarr'))
    np.testing.assert_array_equal(r(w('pc.zarr')), pc_in[iidx])
    w('pc_in1.zarr', pc_in[:, 0].copy(), (100,)); w('hix_in.zarr', hix_in, (100,)); w('hix.zarr', hix_in[iidx], (100,))
    mc.pc_select_data(w('hix_in.zarr'), w('hix.zarr'), w('pc_in1.zarr'), w('pc.zarr'))
    np.testing.assert_array_equal(r(w('pc.zarr')), pc_in[iidx, 0])


def test_pc_select_data_grid_index_by_hilbert_index_needs_shape(w, rng, tmp_path):
    """pc-select-data of a grid index `idx_in` by a hilbert index `idx` needs `shape`, as its help says: without it
    a ValueError asks for it before the indices are read (pc_hix failed with "TypeError: cannot unpack non-iterable
    NoneType object") and nothing is written; with it the points of `idx` are selected, in the order of `idx`."""
    shape = (100, 100)
    gix_in = random_gix(rng, shape, 1000); pc_in = rng.random((1000, 3)).astype(np.float32)
    hix_in = mr.pc_hix(gix_in, shape)
    iidx = rng.choice(1000, size=300, replace=False)
    iidx = iidx[np.argsort(hix_in[iidx])]                     # the points to select, in hilbert order
    w('gix_in.zarr', gix_in, (200, 1)); w('hix.zarr', hix_in[iidx], (200,)); w('pc_in.zarr', pc_in, (200, 1))
    with pytest.raises(ValueError, match=r'shape \(nlines, width\) is needed when a grid index and a hilbert index '
                                         r'are mixed: idx_in is a grid index, idx a hilbert index'):
        mc.pc_select_data(w('gix_in.zarr'), w('hix.zarr'), w('pc_in.zarr'), w('pc.zarr'))
    assert not (tmp_path / 'pc.zarr').exists()
    mc.pc_select_data(w('gix_in.zarr'), w('hix.zarr'), w('pc_in.zarr'), w('pc.zarr'), shape=shape)
    np.testing.assert_array_equal(r(w('pc.zarr')), pc_in[iidx])


def test_bool2gix(tmp_path):
    rng = np.random.default_rng(0)
    is_pc = rng.random((30, 40)) < 0.2
    write_zarr(tmp_path / 'is_pc.zarr', is_pc, (10, 40))
    mc.bool2gix(str(tmp_path / 'is_pc.zarr'), str(tmp_path / 'gix.zarr'), chunks=50)
    gix = zarr.open(str(tmp_path / 'gix.zarr'), mode='r')
    assert gix.chunks == (50, 1)  # one column per chunk, as every point cloud array
    np.testing.assert_array_equal(gix[:], np.stack(np.where(is_pc), axis=-1))
