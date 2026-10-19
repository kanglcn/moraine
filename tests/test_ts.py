import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.command import main


def _ts(n_points=50, nimages=7, seed=0):
    """image phases: a common offset per image, stable points 0..29, moving points 30.., noise"""
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-3, 3, nimages).astype(np.float32)
    motion = np.zeros((n_points, nimages), dtype=np.float32)
    motion[30:] = np.arange(nimages, dtype=np.float32) * rng.uniform(0.2, 0.5, (n_points - 30, 1)).astype(np.float32)
    noise = 0.01 * rng.standard_normal((n_points, nimages)).astype(np.float32)
    stable = np.zeros(n_points, dtype=bool); stable[:30] = True
    return motion + offset + noise, offset, stable


def test_pc_reference_api_median_mean_point():
    ts, offset, stable = _ts()
    out, value = mr.pc_reference(ts, stable)
    assert out.shape == ts.shape and out.dtype == ts.dtype and value.shape == (ts.shape[1],)
    np.testing.assert_allclose(np.median(out[stable], axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(value, offset, atol=0.02)
    np.testing.assert_allclose(out, ts - np.median(ts[stable], axis=0), atol=1e-6)
    out_mean, value_mean = mr.pc_reference(ts, stable, method='mean')
    np.testing.assert_allclose(out_mean[stable].mean(axis=0), 0, atol=1e-5)
    out_point, value_point = mr.pc_reference(ts, 5)
    np.testing.assert_array_equal(out_point[5], 0)
    np.testing.assert_array_equal(value_point, ts[5])


def test_pc_reference_api_median_robust_to_unwrapping_error():
    ts, offset, stable = _ts()
    ts[3, 4] += 2 * np.pi                       # one reference point off by a whole cycle in one image
    value = mr.pc_reference(ts, stable)[1]
    value_mean = mr.pc_reference(ts, stable, method='mean')[1]
    assert abs(value[4] - offset[4]) < 0.02
    assert abs(value_mean[4] - offset[4]) > 0.2


def test_pc_reference_api_nan_and_1d():
    ts, offset, stable = _ts()
    ts[0] = np.nan                              # a reference point without values is left out
    out, value = mr.pc_reference(ts, stable)
    np.testing.assert_allclose(value, offset, atol=0.02)
    assert np.isnan(out[0]).all()
    out1, value1 = mr.pc_reference(ts[:, 2], stable)
    assert out1.shape == (ts.shape[0],) and value1.shape == ()
    np.testing.assert_allclose(out1, out[:, 2])
    ts[:30, 3] = np.nan
    with pytest.raises(ValueError, match=r'all nan\) in column\(s\) \[3\]'):
        mr.pc_reference(ts, stable)


@pytest.mark.parametrize('ref, method, match', [
    (np.zeros(50, dtype=bool), 'median', 'no point'),
    (np.ones(49, dtype=bool), 'median', 'shape'),
    (np.ones(50, dtype=np.int8), 'median', 'bool'),
    (True, 'median', 'single bool'),
    (50, 'median', 'not one of the 50 points'),
    (-1, 'median', 'not one of the 50 points'),
    (np.ones(50, dtype=bool), 'mode', "'median' or 'mean'"),
])
def test_pc_reference_api_errors(ref, method, match):
    ts = _ts()[0]
    with pytest.raises(ValueError, match=match):
        mr.pc_reference(ts, ref, method=method)


def test_pc_reference_api_integer_data():
    with pytest.raises(ValueError, match='floating point'):
        mr.pc_reference(np.zeros((5, 2), dtype=np.int32), 0)


@pytest.mark.gpu
def test_pc_reference_api_gpu():
    import cupy as cp
    ts, offset, stable = _ts()
    out, value = mr.pc_reference(cp.asarray(ts), cp.asarray(stable))
    assert isinstance(out, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(out), mr.pc_reference(ts, stable)[0], atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(mr.pc_reference(cp.asarray(ts), 5)[0]), mr.pc_reference(ts, 5)[0], atol=1e-6)


def _zarr(path, a, chunks):
    z = zarr.open(str(path), mode='w', shape=a.shape, dtype=a.dtype, chunks=chunks)
    z[:] = a
    return str(path)


def test_pc_reference(tmp_path):
    ts, offset, stable = _ts(n_points=50, nimages=7)
    pc = _zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    mask = _zarr(tmp_path / 'mask.zarr', stable, (20,))
    mc.pc_reference(pc, str(tmp_path / 'out.zarr'), ref_mask=mask, ref_value=str(tmp_path / 'ref.zarr'), out_chunks=10)
    out = zarr.open(str(tmp_path / 'out.zarr'), mode='r')
    assert out.shape == ts.shape and out.dtype == ts.dtype and out.chunks == (10, 1)
    expected, value = mr.pc_reference(ts, stable)
    np.testing.assert_allclose(out[:], expected, atol=1e-6)
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'ref.zarr'), mode='r')[:], value, atol=1e-6)
    mc.pc_reference(pc, str(tmp_path / 'mean.zarr'), ref_mask=mask, method='mean', processes=True, n_workers=2)
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'mean.zarr'), mode='r')[:],
                               mr.pc_reference(ts, stable, method='mean')[0], atol=1e-6)


def test_pc_reference_point_1d_and_command_line(tmp_path):
    ts = _ts()[0]
    pc1 = _zarr(tmp_path / 'col.zarr', ts[:, 3].copy(), (20,))
    mc.pc_reference(pc1, str(tmp_path / 'out1.zarr'), ref_point=7, ref_value=str(tmp_path / 'ref1.zarr'))
    out1 = zarr.open(str(tmp_path / 'out1.zarr'), mode='r')
    assert out1.shape == (50,) and out1.chunks == (20,)
    np.testing.assert_allclose(out1[:], ts[:, 3] - ts[7, 3], atol=1e-6)
    assert zarr.open(str(tmp_path / 'ref1.zarr'), mode='r').shape == (1,)
    pc = _zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    assert main(['pc-reference', '--pc', pc, '--out', str(tmp_path / 'out.zarr'), '--ref-point', '7']) == 0
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'out.zarr'), mode='r')[:], ts - ts[7], atol=1e-6)


def test_pc_reference_argument_errors(tmp_path):
    ts, offset, stable = _ts()
    pc = _zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    mask = _zarr(tmp_path / 'mask.zarr', stable, (20,))
    with pytest.raises(ValueError, match='exactly one'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'))
    with pytest.raises(ValueError, match='exactly one'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=mask, ref_point=1)
    with pytest.raises(ValueError, match='not one of the 50 points'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_point=50)
    short = _zarr(tmp_path / 'short.zarr', stable[:40], (20,))
    with pytest.raises(ValueError, match=r'bool of shape \(50,\)'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=short)
    empty = _zarr(tmp_path / 'empty.zarr', np.zeros(50, dtype=bool), (20,))
    with pytest.raises(ValueError, match='no reference point'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=empty)
    complex_pc = _zarr(tmp_path / 'c.zarr', ts.astype(np.complex64), (20, 1))
    with pytest.raises(ValueError, match='floating point'):
        mc.pc_reference(complex_pc, str(tmp_path / 'o.zarr'), ref_point=1)


S1_WAVELENGTH = 0.055465759531382094


def test_pc_disp_api():
    mm_per_rad = 1000 * S1_WAVELENGTH / (4 * np.pi)
    phase = np.array([[0, np.pi, 2 * np.pi], [-1, 1, np.nan]], dtype=np.float32)
    disp = mr.pc_disp(phase, S1_WAVELENGTH)
    assert disp.shape == phase.shape and disp.dtype == np.float32
    np.testing.assert_allclose(disp[0, 2], 1000 * S1_WAVELENGTH / 2, rtol=1e-6)     # one cycle is half a wavelength
    np.testing.assert_allclose(disp[1, :2], [-mm_per_rad, mm_per_rad], rtol=1e-6)  # positive phase: towards the satellite
    assert np.isnan(disp[1, 2])
    disp64 = mr.pc_disp(phase.astype(np.float64), S1_WAVELENGTH)
    assert disp64.dtype == np.float32
    np.testing.assert_allclose(mr.pc_disp(phase[0], S1_WAVELENGTH), disp[0])


@pytest.mark.parametrize('phase, wavelength, match', [
    (np.ones(3, dtype=np.complex64), S1_WAVELENGTH, 'floating point'),
    (np.ones(3, dtype=np.int32), S1_WAVELENGTH, 'floating point'),
    (np.ones(3, dtype=np.float32), 0.0, 'between 0 and 1'),
    (np.ones(3, dtype=np.float32), -0.05, 'between 0 and 1'),
    (np.ones(3, dtype=np.float32), 5.55, 'between 0 and 1'),
    (np.ones(3, dtype=np.float32), np.nan, 'between 0 and 1'),
])
def test_pc_disp_api_errors(phase, wavelength, match):
    with pytest.raises(ValueError, match=match):
        mr.pc_disp(phase, wavelength)


@pytest.mark.gpu
def test_pc_disp_api_gpu():
    import cupy as cp
    phase = _ts()[0]
    disp = mr.pc_disp(cp.asarray(phase), S1_WAVELENGTH)
    assert isinstance(disp, cp.ndarray) and disp.dtype == np.float32
    np.testing.assert_allclose(cp.asnumpy(disp), mr.pc_disp(phase, S1_WAVELENGTH), rtol=1e-6)


def test_pc_disp(tmp_path):
    ts = _ts()[0].astype(np.float64)
    ts[2, 3] = np.nan
    pc = _zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    mc.pc_disp(pc, str(tmp_path / 'disp.zarr'), wavelength=S1_WAVELENGTH, out_chunks=15)
    disp = zarr.open(str(tmp_path / 'disp.zarr'), mode='r')
    assert disp.shape == ts.shape and disp.dtype == np.float32 and disp.chunks == (15, 1)
    np.testing.assert_allclose(disp[:], mr.pc_disp(ts, S1_WAVELENGTH), rtol=1e-6)
    assert np.isnan(disp[2, 3])
    mc.pc_disp(pc, str(tmp_path / 'disp_p.zarr'), wavelength=S1_WAVELENGTH, processes=True, n_workers=2)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'disp_p.zarr'), mode='r')[:], disp[:])


def test_pc_disp_1d_and_command_line(tmp_path):
    ts = _ts()[0]
    pc1 = _zarr(tmp_path / 'col.zarr', ts[:, 3].copy(), (20,))
    mc.pc_disp(pc1, str(tmp_path / 'd1.zarr'), wavelength=S1_WAVELENGTH)
    d1 = zarr.open(str(tmp_path / 'd1.zarr'), mode='r')
    assert d1.shape == (50,) and d1.chunks == (20,)
    np.testing.assert_allclose(d1[:], mr.pc_disp(ts[:, 3], S1_WAVELENGTH), rtol=1e-6)
    pc = _zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    assert main(['pc-disp', '--pc', pc, '--out', str(tmp_path / 'd.zarr'), '--wavelength', str(S1_WAVELENGTH)]) == 0
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'd.zarr'), mode='r')[:], mr.pc_disp(ts, S1_WAVELENGTH), rtol=1e-6)


def test_pc_disp_argument_errors(tmp_path):
    ts = _ts()[0]
    pc = _zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    with pytest.raises(ValueError, match='between 0 and 1'):
        mc.pc_disp(pc, str(tmp_path / 'o.zarr'), wavelength=5.55)
    complex_pc = _zarr(tmp_path / 'c.zarr', ts.astype(np.complex64), (20, 1))
    with pytest.raises(ValueError, match='floating point'):
        mc.pc_disp(complex_pc, str(tmp_path / 'o.zarr'), wavelength=S1_WAVELENGTH)
    cube = _zarr(tmp_path / 'cube.zarr', np.zeros((4, 3, 2), dtype=np.float32), (4, 3, 1))
    with pytest.raises(ValueError, match='shape'):
        mc.pc_disp(cube, str(tmp_path / 'o.zarr'), wavelength=S1_WAVELENGTH)
