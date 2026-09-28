import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc


@pytest.fixture
def w(tmp_path):
    """w(name, array, chunks) writes a zarr array in tmp_path and returns its path; w(name) only returns the path."""
    def _w(name, data=None, chunks=None):
        p = str(tmp_path / name)
        if data is not None:
            z = zarr.open(p, mode='w', shape=data.shape, dtype=data.dtype, chunks=chunks)
            z[:] = data
        return p
    return _w


def r(p):
    return zarr.open(p, mode='r')[:]


def _cplx(rng, *shape):
    return (rng.random(shape) + 1j * rng.random(shape)).astype(np.complex64)


def _gix(rng, n, shape=(100, 100), sort=True):
    g = rng.choice(shape[0] * shape[1], size=n, replace=False).astype(np.int32)
    if sort: g.sort()
    return np.stack(np.unravel_index(g, shape), axis=-1).astype(np.int32)


def _hix(rng, n, sort=True):
    h = rng.choice(100 * 100, size=n, replace=False).astype(np.int64)
    if sort: h.sort()
    return h


def test_ras2pc(w, rng):
    ras1 = rng.random((100, 100)).astype(np.float32); ras2 = _cplx(rng, 100, 100, 3)
    gix = _gix(rng, 1000)
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
    gix = _gix(rng, 1000, shape)
    w('gix.zarr', gix, (200, 1)); w('ras1.zarr', ras1, chunks); w('ras2.zarr', ras2, (*chunks, 1))
    mc.ras2pc_ras_chunk(w('gix.zarr'), w('ras1.zarr'), w('pc1'), key=w('key.zarr'))
    mc.pc_concat(w('pc1'), w('pc1.zarr'), key=w('key.zarr'), chunks=200)
    np.testing.assert_array_equal(r(w('pc1.zarr')), ras1[gix[:, 0], gix[:, 1]])
    mc.ras2pc_ras_chunk(w('gix.zarr'), ras=[w('ras1.zarr'), w('ras2.zarr')], pc=[w('pc1'), w('pc2')], key=w('key.zarr'))
    for name, ras in [('pc1', ras1), ('pc2', ras2)]:
        mc.pc_concat(w(name), w(name + '.zarr'), key=w('key.zarr'), chunks=200)
        np.testing.assert_array_equal(r(w(name + '.zarr')), ras[gix[:, 0], gix[:, 1]])
    # without key the output is in chunk order
    chunk_idx = mr.pc._pc_split_by_chunk(gix, chunks, shape)[0]
    sgix = gix[chunk_idx]
    for name, ras in [('pc1', ras1), ('pc2', ras2)]:
        mc.pc_concat(w(name), w(name + '.zarr'), chunks=200)
        np.testing.assert_array_equal(r(w(name + '.zarr')), ras[sgix[:, 0], sgix[:, 1]])


def test_pc2ras(w, rng):
    pc1 = rng.random(1000).astype(np.float32); pc2 = _cplx(rng, 1000, 3)
    gix = _gix(rng, 1000)
    ras1 = np.full((100, 100), np.nan, np.float32); ras2 = np.full((100, 100, 3), np.nan, np.complex64)
    ras1[gix[:, 0], gix[:, 1]] = pc1; ras2[gix[:, 0], gix[:, 1]] = pc2
    w('gix.zarr', gix, (200, 1)); w('pc1.zarr', pc1, (200,)); w('pc2.zarr', pc2, (200, 1))
    mc.pc2ras(w('gix.zarr'), w('pc1.zarr'), w('ras1.zarr'), shape=(100, 100), chunks=(20, 100))
    np.testing.assert_array_equal(r(w('ras1.zarr')), ras1)
    mc.pc2ras(w('gix.zarr'), [w('pc1.zarr'), w('pc2.zarr')], [w('ras1.zarr'), w('ras2.zarr')], shape=(100, 100), chunks=(20, 100))
    np.testing.assert_array_equal(r(w('ras1.zarr')), ras1)
    np.testing.assert_array_equal(r(w('ras2.zarr')), ras2)


def test_pc_hix_gix(w, rng):
    gix = _gix(rng, 1000)
    w('gix.zarr', gix, (100, 1))
    mc.pc_hix(w('gix.zarr'), w('hix.zarr'), shape=(100, 100))
    np.testing.assert_array_equal(r(w('hix.zarr')), mr.pc_hix(gix, (100, 100)))
    mc.pc_gix(w('hix.zarr'), w('gix_.zarr'), (100, 100))
    np.testing.assert_array_equal(r(w('gix_.zarr')), gix)


def test_pc_sort(w, rng):
    pc_in = rng.random(1000).astype(np.float32)
    gix_in = _gix(rng, 1000, sort=False)
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


@pytest.mark.parametrize('kind', ['gix', 'hix'])
def test_pc_union(w, rng, kind):
    pc1 = _cplx(rng, 1000, 3); pc2 = _cplx(rng, 800, 3)
    if kind == 'gix':
        i1, i2, c = _gix(rng, 1000), _gix(rng, 800), (200, 1)
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
        i1, i2, c = _gix(rng, 1000), _gix(rng, 800), (200, 1)
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
        i1, i2, c = _gix(rng, 1000), _gix(rng, 800), (200, 1)
    else:
        i1, i2, c = _hix(rng, 1000), _hix(rng, 800), (200,)
    idx, iidx1 = mr.pc_diff(i1, i2)
    w('i1.zarr', i1, c); w('i2.zarr', i2, c); w('pc1.zarr', pc1, (200, 1))
    mc.pc_diff(w('i1.zarr'), w('i2.zarr'), w('i.zarr'))
    mc.pc_diff(w('i1.zarr'), w('i2.zarr'), w('i.zarr'), pc1=w('pc1.zarr'), pc=w('pc.zarr'))
    np.testing.assert_array_equal(r(w('i.zarr')), idx)
    np.testing.assert_array_equal(r(w('pc.zarr')), pc1[iidx1])


def test_pc_logic_ras(w, rng):
    ras = rng.random((100, 100)).astype(np.float32)
    w('ras.zarr', ras, (10, 100))
    mc.pc_logic_ras(w('ras.zarr'), w('gix.zarr'), '(ras>=0.1)&(ras<=0.5)')
    np.testing.assert_array_equal(r(w('gix.zarr')), np.stack(np.where((ras >= 0.1) & (ras <= 0.5)), axis=-1))


def test_pc_logic_pc(w, rng):
    pc_in = rng.random(1000).astype(np.float32)
    keep = (pc_in >= 0.1) & (pc_in <= 0.5)
    gix_in = _gix(rng, 1000); hix_in = _hix(rng, 1000)
    w('pc_in.zarr', pc_in, (100,)); w('gix_in.zarr', gix_in, (100, 1)); w('hix_in.zarr', hix_in, (100,))
    mc.pc_logic_pc(w('gix_in.zarr'), w('pc_in.zarr'), w('gix.zarr'), '(pc_in>=0.1)&(pc_in<=0.5)')
    np.testing.assert_array_equal(r(w('gix.zarr')), gix_in[keep])
    mc.pc_logic_pc(w('hix_in.zarr'), w('pc_in.zarr'), w('hix.zarr'), '(pc_in>=0.1)&(pc_in<=0.5)')
    np.testing.assert_array_equal(r(w('hix.zarr')), hix_in[keep])


def test_pc_select_data(w, rng):
    pc_in = rng.random((1000, 4)).astype(np.float32)
    iidx = np.sort(rng.choice(1000, size=500, replace=False))
    gix_in = _gix(rng, 1000); hix_in = _hix(rng, 1000)
    w('pc_in.zarr', pc_in, (100, 1)); w('gix_in.zarr', gix_in, (100, 1)); w('gix.zarr', gix_in[iidx], (100, 1))
    mc.pc_select_data(w('gix_in.zarr'), w('gix.zarr'), w('pc_in.zarr'), w('pc.zarr'))
    np.testing.assert_array_equal(r(w('pc.zarr')), pc_in[iidx])
    w('pc_in1.zarr', pc_in[:, 0].copy(), (100,)); w('hix_in.zarr', hix_in, (100,)); w('hix.zarr', hix_in[iidx], (100,))
    mc.pc_select_data(w('hix_in.zarr'), w('hix.zarr'), w('pc_in1.zarr'), w('pc.zarr'))
    np.testing.assert_array_equal(r(w('pc.zarr')), pc_in[iidx, 0])


def test_data_reduce(w, rng):
    pc1 = _cplx(rng, 1000); pc2 = _cplx(rng, 800, 3)
    w('pc_in1.zarr', pc1, (200,)); w('pc_in2.zarr', pc2, (200, 1))
    mc.data_reduce(w('pc_in1.zarr'), w('pc_out1.zarr'), map_func=np.abs, reduce_func=np.sum, post_map_func=lambda x: x / 1000)
    mc.data_reduce(w('pc_in2.zarr'), w('pc_out2.zarr'), map_func=np.abs, reduce_func=np.sum, post_map_func=lambda x: x / 800)
    np.testing.assert_array_almost_equal(r(w('pc_out1.zarr'))[0], np.mean(np.abs(pc1), axis=0))
    np.testing.assert_array_almost_equal(r(w('pc_out2.zarr')), np.mean(np.abs(pc2), axis=0))
