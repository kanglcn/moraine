import numpy as np
import pytest

import moraine as mr
from moraine.dl import (_pre_infer_n2f_numba, _after_infer_n2f_numba, _pre_infer_n2fs3d_numba,
                        _load_model, _get_model, n2f, n2fs3d, n2ft)

torch = pytest.importorskip('torch')
GPU = [False, pytest.param(True, marks=pytest.mark.gpu)]


def _phase_diff(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.abs(np.angle(a[ok] * b[ok].conj()))


@pytest.fixture(scope='module')
def intf(rslc):
    return (rslc[:600, :600, 0] * rslc[:600, :600, 1].conj()).astype(np.complex64)


@pytest.fixture(scope='module')
def adi(rslc):
    return mr.amp_disp(rslc[:600, :600]).astype(np.float32)


@pytest.fixture(autouse=True)
def _need_model(request):
    name = {'test_n2fs3d': 'n2fs3d', 'test_n2ft': 'n2ft'}.get(request.node.originalname, 'n2f')
    try:
        _get_model(name)
    except FileNotFoundError:
        pytest.skip(f'{name} model not downloaded, run moraine.download_dl_model()')


def test_pre_after_infer_n2f(intf):
    x, mask = _pre_infer_n2f_numba(intf)
    ref = intf / np.abs(intf)
    np.testing.assert_array_almost_equal(x[0, :, ~mask].T, np.stack((ref.real, ref.imag))[:, ~mask], decimal=5)
    out = _after_infer_n2f_numba(x, mask)
    np.testing.assert_array_almost_equal(np.angle(out[~mask]), np.angle(intf[~mask]))
    assert np.isnan(out[mask]).all()


@pytest.mark.gpu
def test_pre_after_infer_gpu(intf, adi):
    import cupy as cp
    from moraine.dl import _pre_infer_n2f_cp, _after_infer_n2f_cp, _pre_infer_n2fs3d_cp
    x, mask = _pre_infer_n2f_numba(intf)
    x_cp, mask_cp = _pre_infer_n2f_cp(cp.asarray(intf))
    np.testing.assert_array_equal(mask, mask_cp.get())
    np.testing.assert_array_almost_equal(x[:, :, ~mask], x_cp[:, :, ~mask_cp].get())
    np.testing.assert_array_almost_equal(np.angle(_after_infer_n2f_cp(cp.asarray(x), cp.asarray(mask)).get()[~mask]),
                                         np.angle(intf[~mask]))
    x, mask = _pre_infer_n2fs3d_numba(adi, intf)
    x_cp, mask_cp = _pre_infer_n2fs3d_cp(cp.asarray(adi), cp.asarray(intf))
    np.testing.assert_array_equal(mask, mask_cp.get())
    np.testing.assert_array_almost_equal(x[:, :, ~mask], x_cp[:, :, ~mask_cp].get())


def test_model_is_cached():
    assert _get_model('n2f') is _get_model('n2f')


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='download_dl_model'):
        _load_model('n2f', str(tmp_path / 'nothing.pth'))


@pytest.mark.parametrize('gpu', GPU)
def test_n2f(intf, gpu):
    out = n2f(intf.copy(), chunks=(300, 300), depths=(32, 32))
    np.testing.assert_array_equal(np.isnan(out), np.isnan(intf) | (np.abs(intf) < 1e-30))
    np.testing.assert_allclose(np.abs(out[np.isfinite(out)]), 1, rtol=1e-4)
    if gpu:
        import cupy as cp
        out_cp = n2f(cp.asarray(intf), chunks=(300, 300), depths=(32, 32))
        assert isinstance(out_cp, cp.ndarray)
        np.testing.assert_array_equal(np.isnan(out_cp.get()), np.isnan(out))
        # NaN pixels are filled with random phase, compare medians; TF32 adds ~1e-3 rad
        assert np.median(_phase_diff(out, out_cp.get())) < 1e-2
        np.testing.assert_array_equal(np.isnan(mr.dl._n2f_np_in_gpu(intf.copy())), np.isnan(out))


@pytest.mark.parametrize('gpu', GPU)
def test_n2fs3d(adi, intf, gpu):
    out = n2fs3d(adi, intf.copy())
    assert np.isnan(out).sum() >= np.isnan(intf).sum()
    if gpu:
        import cupy as cp
        out_cp = n2fs3d(cp.asarray(adi), cp.asarray(intf))
        np.testing.assert_array_equal(np.isnan(out_cp.get()), np.isnan(out))
        assert np.median(_phase_diff(out, out_cp.get())) < 1e-2


@pytest.fixture(scope='module')
def points(rslc, adi):
    """PS-like point cloud: pixels of the crop with low amplitude dispersion."""
    gix = np.stack(np.where(adi < 0.3), axis=-1)
    s = rslc[:600, :600]
    return gix[:, 1].astype(np.float64), gix[:, 0].astype(np.float64), s[gix[:, 0], gix[:, 1]]


@pytest.mark.parametrize('gpu', GPU)
def test_n2ft(points, gpu):
    x, y, s = points
    ifg = s[:, 0] * s[:, 1].conj()
    out = n2ft(x, y, ifg, cuda=gpu)
    assert out.shape == ifg.shape
    np.testing.assert_allclose(np.abs(out), 1, rtol=1e-4)
    np.testing.assert_array_equal(n2ft(x, y, ifg, cuda=gpu), out)             # reproducible
    stack = s[:, [0]] * s[:, 1:4].conj()
    out_stack = n2ft(x, y, stack, chunks=x.shape[0] // 3, cuda=gpu)
    assert out_stack.shape == stack.shape
    if gpu:
        assert np.median(_phase_diff(out, n2ft(x, y, ifg))) < 1e-3
