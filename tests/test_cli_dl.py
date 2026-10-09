import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.cli.dl import _cli_pre_infer_n2f_numba

pytest.importorskip('torch')
GPU = [False, pytest.param(True, marks=pytest.mark.gpu)]


@pytest.fixture(autouse=True)
def _need_models():
    try:
        mr.api.dl._get_model('n2f'); mr.api.dl._get_model('n2ft')
    except FileNotFoundError:
        pytest.skip('models not downloaded, run moraine.download_dl_model()')


def _phase_diff(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.abs(np.angle(a[ok] * b[ok].conj()))


def test_cli_pre_infer_n2f(rslc):
    ref, sec = rslc[:500, :500, 0], rslc[:500, :500, 1]
    x, mask = _cli_pre_infer_n2f_numba(ref, sec)
    ifg = ref * sec.conj(); ifg = ifg / np.abs(ifg)
    np.testing.assert_array_almost_equal(x[:, :, ~mask], np.stack((ifg.real, ifg.imag))[None][:, :, ~mask])


@pytest.mark.gpu
def test_cli_pre_infer_n2f_gpu(rslc):
    import cupy as cp
    from moraine.cli.dl import _cli_pre_infer_n2f_cp
    ref, sec = rslc[:500, :500, 0], rslc[:500, :500, 1]
    x, mask = _cli_pre_infer_n2f_numba(ref, sec)
    x_cp, mask_cp = _cli_pre_infer_n2f_cp(cp.asarray(ref), cp.asarray(sec))
    np.testing.assert_array_equal(mask, mask_cp.get())
    np.testing.assert_array_almost_equal(x[:, :, ~mask], x_cp[:, :, ~mask_cp].get())


def test_n2f_tiles():
    from moraine.cli.dl import _n2f_tiles
    from moraine.api.chunk_ import chunkwise_slicing_mapping
    shape = (400, 500)
    # output chunks equal to the processing chunks: the tiles of the API's chunkwise processing
    tiles = _n2f_tiles(shape, (150, 200), (150, 200), (32, 16))
    in_slices, out_slices, map_slices = chunkwise_slicing_mapping(shape, (150, 200), (32, 16))
    assert len(tiles) == len(out_slices) == 9
    for out_s, tile_list in tiles.items():
        assert len(tile_list) == 1
        k = out_slices.index(out_s)
        assert tile_list[0][0] == in_slices[k] and tile_list[0][1] == map_slices[k]
        assert tile_list[0][2] == (slice(0, out_s[0].stop-out_s[0].start), slice(0, out_s[1].stop-out_s[1].start))
    # processing chunks cut at the output chunks: every pixel in exactly one tile, tiles inside their output chunk
    tiles = _n2f_tiles(shape, (150, 150), (200, 300), (10, 10))
    covered = np.zeros(shape, dtype=int)
    for (out_az, out_r), tile_list in tiles.items():
        for (in_az, in_r), (map_az, map_r), (loc_az, loc_r) in tile_list:
            az = slice(out_az.start+loc_az.start, out_az.start+loc_az.stop)
            r = slice(out_r.start+loc_r.start, out_r.start+loc_r.stop)
            covered[az, r] += 1
            assert az.stop <= out_az.stop and r.stop <= out_r.stop
            assert in_az.start == max(az.start-10, 0) and in_az.stop == min(az.stop+10, shape[0])
            assert map_az == slice(az.start-in_az.start, az.stop-in_az.start)
            assert in_r.start == max(r.start-10, 0) and map_r.stop-map_r.start == r.stop-r.start
    assert (covered == 1).all()
    assert sum(len(t) for t in tiles.values()) == 4*4  # azimuth 0,150,200,300; range 0,150,300,450


@pytest.mark.parametrize('gpu', GPU)
def test_cli_n2f_out_chunk(rslc, tmp_path, gpu):
    """one output chunk with all image pairs gives the API result of every pair; 0 in the rslc gives NaN"""
    from moraine.cli.dl import _cli_n2f_out_chunk, _n2f_tiles
    crop = rslc[:400, :400, :4].copy()
    crop[0, 0, 0] = 0  # GAMMA writes 0 where there are no data
    z = zarr.open(str(tmp_path / 'rslc.zarr'), mode='w', shape=crop.shape, dtype=crop.dtype, chunks=(200, 200, 1))
    z[:] = crop
    pairs = np.array([[0, 1], [2, 3]])
    out_zarr = zarr.open(str(tmp_path / 'intf.zarr'), mode='w', shape=(400, 400, 2), dtype=crop.dtype, chunks=(400, 400, 1))
    tiles = _n2f_tiles((400, 400), (200, 200), (400, 400), (32, 32))
    assert list(tiles) == [(slice(0, 400), slice(0, 400))]
    _cli_n2f_out_chunk(str(tmp_path / 'rslc.zarr'), str(tmp_path / 'intf.zarr'), (slice(0, 400), slice(0, 400)),
                       tiles[(slice(0, 400), slice(0, 400))], pairs, cuda=gpu)
    out = out_zarr[:]
    assert np.isnan(out[0, 0, 0]) and np.isfinite(out[0, 0, 1])
    for k, (a, b) in enumerate(pairs):
        api = mr.n2f((crop[:, :, a] * crop[:, :, b].conj()).astype(np.complex64), chunks=(200, 200), depths=(32, 32))
        np.testing.assert_array_equal(np.isnan(out[:, :, k]), np.isnan(api))
        assert np.median(_phase_diff(out[:, :, k], api)) < 1e-2


@pytest.mark.slow
@pytest.mark.parametrize('cuda', GPU)
def test_cli_n2f(rslc, tmp_path, cuda):
    crop = rslc[:400, :400, :4]
    z = zarr.open(str(tmp_path / 'rslc.zarr'), mode='w', shape=crop.shape, dtype=crop.dtype, chunks=(200, 200, 1))
    z[:] = crop
    pairs = np.array([[0, 1], [2, 3]])
    mc.n2f(str(tmp_path / 'rslc.zarr'), str(tmp_path / 'intf.zarr'), pairs, chunks=(200, 200), depths=(32, 32), cuda=cuda)
    out = zarr.open(str(tmp_path / 'intf.zarr'), mode='r')[:]
    assert out.shape == (400, 400, 2)
    api = mr.n2f((crop[:, :, 2] * crop[:, :, 3].conj()).astype(np.complex64), chunks=(200, 200), depths=(32, 32))
    # NaN pixels get random phase, so compare the typical pixel
    assert np.median(_phase_diff(out[:, :, 1], api)) < 1e-2
    # several processing chunks per output chunk: the tiles of the API's chunkwise processing
    mc.n2f(str(tmp_path / 'rslc.zarr'), str(tmp_path / 'intf2.zarr'), pairs, chunks=(150, 150), out_chunks=(300, 300),
           depths=(16, 16), cuda=cuda)
    out2 = zarr.open(str(tmp_path / 'intf2.zarr'), mode='r')
    assert out2.chunks == (300, 300, 1)
    api2 = mr.n2f((crop[:, :, 2] * crop[:, :, 3].conj()).astype(np.complex64), chunks=(150, 150), depths=(16, 16))
    np.testing.assert_array_equal(np.isnan(out2[:, :, 1]), np.isnan(api2))
    assert np.median(_phase_diff(out2[:, :, 1], api2)) < 1e-2


@pytest.fixture(scope='module')
def ps(rslc, tmp_path_factory):
    d = tmp_path_factory.mktemp('n2ft')
    crop = rslc[:1200, :1200]  # enough points for four processing chunks
    gix = np.stack(np.where(mr.amp_disp(crop) < 0.3), axis=-1)
    data = {'x.zarr': gix[:, 1].astype(np.float64), 'y.zarr': gix[:, 0].astype(np.float64),
            'rslc.zarr': crop[gix[:, 0], gix[:, 1]]}
    for name, a in data.items():
        z = zarr.open(str(d / name), mode='w', shape=a.shape, dtype=a.dtype, chunks=(a.shape[0], 1) if a.ndim == 2 else (a.shape[0],))
        z[:] = a
    return d, data


def test_n2ft_block_bounds():
    from moraine.cli.dl import _n2ft_block_bounds
    np.testing.assert_array_equal(_n2ft_block_bounds(100, 20, 40), [0, 20, 40, 60, 80, 100])
    np.testing.assert_array_equal(_n2ft_block_bounds(100, 30, 40), [0, 30, 40, 70, 80, 100])  # cut at output chunks
    np.testing.assert_array_equal(_n2ft_block_bounds(95, 20, 200), [0, 20, 40, 60, 80, 95])


@pytest.mark.slow
@pytest.mark.parametrize('cuda', GPU)
def test_cli_n2ft(ps, cuda):
    d, data = ps
    x, y, s = data['x.zarr'], data['y.zarr'], data['rslc.zarr']
    chunks = x.shape[0] // 4 + 1  # four processing chunks in two output chunks, the same chunks as the API
    pairs = np.array([[0, 1], [2, 3]])
    mc.n2ft(str(d / 'x.zarr'), str(d / 'y.zarr'), str(d / 'rslc.zarr'), str(d / f'intf_{cuda}.zarr'),
            pairs, chunks=chunks, out_chunks=2 * chunks, cuda=cuda)
    out = zarr.open(str(d / f'intf_{cuda}.zarr'), mode='r')
    assert out.chunks == (2 * chunks, 1)
    for k, (a, b) in enumerate(pairs):
        api = mr.n2ft(x, y, mr.intf(np.ascontiguousarray(s[:, a]), np.ascontiguousarray(s[:, b])), chunks=chunks, cuda=cuda)
        if cuda:
            assert np.median(_phase_diff(out[:, k], api)) < 1e-4
        else:  # torch in a single thread worker against the threads of this process: rounding only (up to 2e-5)
            np.testing.assert_allclose(out[:, k], api, atol=1e-4)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize('rmm_pool_size', [None, 0.3])
def test_cli_n2ft_compile(ps, monkeypatch, rmm_pool_size):
    """the compiled model tunes its kernels in the workers (inductor caches off: as on a new machine), also with an rmm pool"""
    monkeypatch.setenv('TORCHINDUCTOR_FORCE_DISABLE_CACHES', '1')
    d, data = ps
    x, y, s = data['x.zarr'], data['y.zarr'], data['rslc.zarr']
    chunks = x.shape[0] // 2 + 1
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    out_path = str(d / f'intf_compiled_{rmm_pool_size}.zarr')
    mc.n2ft(str(d / 'x.zarr'), str(d / 'y.zarr'), str(d / 'rslc.zarr'), out_path, pairs, chunks=chunks, cuda=True,
            compile=True, rmm_pool_size=rmm_pool_size)
    out = zarr.open(out_path, mode='r')
    for k, (a, b) in enumerate(pairs):
        api = mr.n2ft(x, y, mr.intf(np.ascontiguousarray(s[:, a]), np.ascontiguousarray(s[:, b])), chunks=chunks, cuda=True)
        assert np.median(_phase_diff(out[:, k], api)) < 1e-4
