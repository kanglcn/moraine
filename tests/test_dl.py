import importlib.resources
import numpy as np
import pytest

import moraine as mr
from moraine.api.dl import (_pre_infer_n2f_numba, _after_infer_n2f_numba, _pre_infer_n2fs3d_numba,
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
    from moraine.api.dl import _pre_infer_n2f_cp, _after_infer_n2f_cp, _pre_infer_n2fs3d_cp
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
    x = intf.copy(); x[0, 0] = 0  # GAMMA writes 0 where there are no data
    x0 = x.copy()
    out = n2f(x, chunks=(300, 300), depths=(32, 32))
    np.testing.assert_array_equal(x, x0)  # the input is not changed
    assert np.isnan(out[0, 0])
    np.testing.assert_array_equal(np.isnan(out), np.isnan(x) | (np.abs(x) < 1e-30))
    np.testing.assert_allclose(np.abs(out[np.isfinite(out)]), 1, rtol=1e-4)
    if gpu:
        import cupy as cp
        x_cp = cp.asarray(x)
        out_cp = n2f(x_cp, chunks=(300, 300), depths=(32, 32))
        assert isinstance(out_cp, cp.ndarray)
        np.testing.assert_array_equal(x_cp.get(), x0)
        np.testing.assert_array_equal(np.isnan(out_cp.get()), np.isnan(out))
        # NaN pixels are filled with random phase, compare medians; TF32 adds ~1e-3 rad
        assert np.median(_phase_diff(out, out_cp.get())) < 1e-2
        np.testing.assert_array_equal(np.isnan(mr.api.dl._n2f_np_in_gpu(x)), np.isnan(out))
        np.testing.assert_array_equal(x, x0)


@pytest.mark.parametrize('gpu', GPU)
def test_n2fs3d(adi, intf, gpu):
    x = intf.copy(); x[0, 0] = 0  # GAMMA writes 0 where there are no data
    x0 = x.copy()
    out = n2fs3d(adi, x)
    np.testing.assert_array_equal(x, x0)  # the input is not changed
    assert np.isnan(out[0, 0])
    assert np.isnan(out).sum() >= np.isnan(intf).sum()
    if gpu:
        import cupy as cp
        x_cp = cp.asarray(x)
        out_cp = n2fs3d(cp.asarray(adi), x_cp)
        np.testing.assert_array_equal(x_cp.get(), x0)
        np.testing.assert_array_equal(np.isnan(out_cp.get()), np.isnan(out))
        assert np.median(_phase_diff(out, out_cp.get())) < 1e-2
        np.testing.assert_array_equal(np.isnan(mr.api.dl._n2fs3d_np_in_gpu(adi, x)), np.isnan(out))
        np.testing.assert_array_equal(x, x0)


@pytest.fixture(scope='module')
def points(rslc, adi):
    """PS-like point cloud: pixels of the crop with low amplitude dispersion."""
    gix = np.stack(np.where(adi < 0.3), axis=-1)
    s = rslc[:600, :600]
    return gix[:, 1].astype(np.float64), gix[:, 0].astype(np.float64), s[gix[:, 0], gix[:, 1]]


def test_prefetched_keeps_order():
    from moraine.api.dl import _prefetched
    import time
    def prepare(i):
        time.sleep(0.01 * max(0, 5 - i))               # the first items take longest
        return i * i
    assert [(i, v) for i, v in _prefetched(range(6), prepare, n_prefetch=3)] == [(i, i * i) for i in range(6)]
    assert list(_prefetched([], prepare)) == []
    assert list(_prefetched([7], prepare, n_prefetch=0)) == [(7, 49)]


def test_n2ft_compile_default():
    from moraine.api.dl import _n2ft_compile_default
    assert not _n2ft_compile_default(590_667, 91)        # Campi Flegrei: compiling costs more than it saves
    assert _n2ft_compile_default(2_000_000, 91)
    assert not _n2ft_compile_default(0, 91)


@pytest.mark.parametrize('gpu', GPU)
def test_n2ft_phasors(rng, gpu):
    """unit phasors of the interferograms on the device, (m, n, 2) float32, as real / |z| and imag / |z|"""
    import torch
    from moraine.api.dl import _n2ft_phasors
    intf = ((rng.standard_normal((50, 3, 2)) @ [1, 1j]) * rng.uniform(0.1, 3, (50, 3))).astype(np.complex64)
    x = _n2ft_phasors(intf, torch.device('cuda' if gpu else 'cpu')).cpu().numpy()
    assert x.shape == (3, 50, 2) and x.dtype == np.float32
    expected = np.stack([(intf.real / np.abs(intf)).T, (intf.imag / np.abs(intf)).T], -1)
    np.testing.assert_allclose(x, expected, atol=1e-6)


@pytest.mark.parametrize('gpu', GPU)
def test_n2ft_batched_interferograms(points, gpu):
    """several interferograms of the same points per model call give the results of one call per interferogram"""
    from moraine.api.dl import _n2ft_structure, _infer_n2ft_structure, _get_model
    x, y, s = points
    stack = s[:, [0]] * s[:, 1:6].conj()
    model = _get_model('n2ft', None, 'cuda' if gpu else 'cpu')
    structure = _n2ft_structure(x, y, next(model.parameters()).device)
    one = _infer_n2ft_structure(structure, stack, model, max_point_intfs=1)          # one interferogram per call
    batched = _infer_n2ft_structure(structure, stack, model)                        # all five at once
    assert np.median(_phase_diff(batched, one)) < 1e-5
    np.testing.assert_allclose(batched, one, atol=1e-3)


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


def test_n2ft_batches():
    from moraine.api.dl import _n2ft_batches
    assert _n2ft_batches(0, 8) == []
    assert _n2ft_batches(1, 8) == [(0, 1)]
    assert _n2ft_batches(3, 2) == [(0, 3)]
    assert _n2ft_batches(2, 1) == [(0, 1), (1, 2)]
    for m in range(1, 60):
        for batch in range(1, 12):
            batches = _n2ft_batches(m, batch)
            sizes = [stop-start for start, stop in batches]
            assert batches[0][0] == 0 and batches[-1][1] == m
            assert all(a[1] == b[0] for a, b in zip(batches[:-1], batches[1:]))
            assert max(sizes)-min(sizes) <= 1
            assert max(sizes) <= max(batch, 3)
            if batch >= 2 and m >= 2:
                assert min(sizes) >= 2


@pytest.mark.parametrize('gpu', GPU)
def test_n2ft_fold_batch_norms(points, gpu):
    """the loaded n2ft model has its batch norms folded into multiply-adds and gives the output of the original model"""
    import torch
    from moraine.api.dl import _n2ft_structure, _infer_n2ft_structure, _get_model, _model_files
    from moraine.api.n2ft_torch_ import N2FT, PointTransformerBlock, ChannelAffine
    x, y, s = points
    stack = s[:, [0]] * s[:, 1:4].conj()
    folded = _get_model('n2ft', None, 'cuda' if gpu else 'cpu')
    device = next(folded.parameters()).device
    original = N2FT(PointTransformerBlock, [1, 1, 1, 1, 1])
    original.load_state_dict(torch.load(importlib.resources.files('moraine')/'dl_model'/_model_files['n2ft'],
                                        map_location='cpu', weights_only=True))
    original.eval().to(device)
    kinds = lambda model: {type(m) for m in model.modules()}
    assert torch.nn.BatchNorm1d in kinds(original) and ChannelAffine not in kinds(original)
    assert torch.nn.BatchNorm1d not in kinds(folded) and ChannelAffine in kinds(folded)
    structure = _n2ft_structure(x, y, device)
    np.testing.assert_allclose(_infer_n2ft_structure(structure, stack, folded),
                               _infer_n2ft_structure(structure, stack, original), atol=1e-4)


def test_n2ft_max_point_intfs():
    from moraine.api.dl import _n2ft_max_point_intfs, _N2FT_MAX_POINT_INTFS, _N2FT_MEMORY_FRACTION, _N2FT_POINT_INTF_BYTES
    assert _n2ft_max_point_intfs(torch.device('cpu')) == _N2FT_MAX_POINT_INTFS == 200_000
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        assert _n2ft_max_point_intfs(torch.device('cuda', 0)) == max(50_000, int(total*_N2FT_MEMORY_FRACTION)//_N2FT_POINT_INTF_BYTES)
