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
        mr.dl._get_model('n2f'); mr.dl._get_model('n2ft')
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
    crop = rslc[:600, :600]
    gix = np.stack(np.where(mr.amp_disp(crop) < 0.3), axis=-1)
    data = {'x.zarr': gix[:, 1].astype(np.float64), 'y.zarr': gix[:, 0].astype(np.float64),
            'rslc.zarr': crop[gix[:, 0], gix[:, 1]]}
    for name, a in data.items():
        z = zarr.open(str(d / name), mode='w', shape=a.shape, dtype=a.dtype, chunks=(a.shape[0], 1) if a.ndim == 2 else (a.shape[0],))
        z[:] = a
    return d, data


@pytest.mark.slow
@pytest.mark.parametrize('cuda', GPU)
def test_cli_n2ft(ps, cuda):
    d, data = ps
    n = data['x.zarr'].shape[0]
    mc.n2ft(str(d / 'x.zarr'), str(d / 'y.zarr'), str(d / 'rslc.zarr'), str(d / f'intf_{cuda}.zarr'),
            np.array([[0, 1], [2, 3]]), chunks=n // 2, cuda=cuda)
    out = zarr.open(str(d / f'intf_{cuda}.zarr'), mode='r')[:]
    s = data['rslc.zarr']
    api = mr.n2ft(data['x.zarr'], data['y.zarr'], s[:, 0] * s[:, 1].conj(), chunks=n // 2)
    assert np.median(_phase_diff(out[:, 0], api)) < 1e-4
