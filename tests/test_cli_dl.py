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


@pytest.mark.parametrize('gpu', GPU)
def test_cli_n2f_keeps_input(rslc, gpu):
    # the image blocks are shared by the dask tasks of several image pairs and must not change
    from moraine.cli.dl import _cli_n2f_cpu, _cli_n2f_np_in_gpu
    ref, sec = rslc[:400, :400, 0], rslc[:400, :400, 1]
    ref[0, 0] = 0  # GAMMA writes 0 where there are no data
    ref0, sec0 = ref.copy(), sec.copy()
    out = (_cli_n2f_np_in_gpu if gpu else _cli_n2f_cpu)(ref, sec, chunks=(200, 200), depths=(32, 32))
    np.testing.assert_array_equal(ref, ref0)
    np.testing.assert_array_equal(sec, sec0)
    assert np.isnan(out[0, 0])


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
