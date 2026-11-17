import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.command import main
from moraine.api.utils_ import ArgumentError
from conftest import write_zarr


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
    with pytest.raises(ValueError, match=r'all nan\) in column\(s\) \[3\]') as e:
        mr.pc_reference(ts, stable)
    assert not isinstance(e.value, ArgumentError)          # a problem of the data, not of the arguments (0052)


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
    """wrong arguments are ArgumentErrors; a mask without points is a problem of the data, a ValueError (decision
    0052)"""
    ts = _ts()[0]
    with pytest.raises(ValueError, match=match) as e:
        mr.pc_reference(ts, ref, method=method)
    assert isinstance(e.value, ArgumentError) == (match != 'no point')


def test_pc_reference_api_integer_data():
    with pytest.raises(ArgumentError, match='floating point'):
        mr.pc_reference(np.zeros((5, 2), dtype=np.int32), 0)


@pytest.mark.gpu
def test_pc_reference_api_gpu():
    import cupy as cp
    ts, offset, stable = _ts()
    out, value = mr.pc_reference(cp.asarray(ts), cp.asarray(stable))
    assert isinstance(out, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(out), mr.pc_reference(ts, stable)[0], atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(mr.pc_reference(cp.asarray(ts), 5)[0]), mr.pc_reference(ts, 5)[0], atol=1e-6)


def test_pc_reference(tmp_path):
    ts, offset, stable = _ts(n_points=50, nimages=7)
    pc = write_zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    mask = write_zarr(tmp_path / 'mask.zarr', stable, (20,))
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
    pc1 = write_zarr(tmp_path / 'col.zarr', ts[:, 3].copy(), (20,))
    mc.pc_reference(pc1, str(tmp_path / 'out1.zarr'), ref_point=7, ref_value=str(tmp_path / 'ref1.zarr'))
    out1 = zarr.open(str(tmp_path / 'out1.zarr'), mode='r')
    assert out1.shape == (50,) and out1.chunks == (20,)
    np.testing.assert_allclose(out1[:], ts[:, 3] - ts[7, 3], atol=1e-6)
    assert zarr.open(str(tmp_path / 'ref1.zarr'), mode='r').shape == (1,)
    pc = write_zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    assert main(['pc-reference', '--pc', pc, '--out', str(tmp_path / 'out.zarr'), '--ref-point', '7']) == 0
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'out.zarr'), mode='r')[:], ts - ts[7], atol=1e-6)


def test_pc_reference_argument_errors(tmp_path):
    ts, offset, stable = _ts()
    pc = write_zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    mask = write_zarr(tmp_path / 'mask.zarr', stable, (20,))
    with pytest.raises(ArgumentError, match='exactly one'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'))
    with pytest.raises(ArgumentError, match='exactly one'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=mask, ref_point=1)
    with pytest.raises(ArgumentError, match='not one of the 50 points'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_point=50)
    short = write_zarr(tmp_path / 'short.zarr', stable[:40], (20,))
    with pytest.raises(ArgumentError, match=r'bool of shape \(50,\)'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=short)
    empty = write_zarr(tmp_path / 'empty.zarr', np.zeros(50, dtype=bool), (20,))
    with pytest.raises(ValueError, match='no reference point') as e:
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=empty)
    assert not isinstance(e.value, ArgumentError)          # the values of the mask: a problem of the data
    complex_pc = write_zarr(tmp_path / 'c.zarr', ts.astype(np.complex64), (20, 1))
    with pytest.raises(ArgumentError, match='floating point'):
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
    with pytest.raises(ArgumentError, match=match):
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
    pc = write_zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    mc.pc_disp(pc, str(tmp_path / 'disp.zarr'), wavelength=S1_WAVELENGTH, out_chunks=15)
    disp = zarr.open(str(tmp_path / 'disp.zarr'), mode='r')
    assert disp.shape == ts.shape and disp.dtype == np.float32 and disp.chunks == (15, 1)
    np.testing.assert_allclose(disp[:], mr.pc_disp(ts, S1_WAVELENGTH), rtol=1e-6)
    assert np.isnan(disp[2, 3])
    mc.pc_disp(pc, str(tmp_path / 'disp_p.zarr'), wavelength=S1_WAVELENGTH, processes=True, n_workers=2)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'disp_p.zarr'), mode='r')[:], disp[:])


def test_pc_disp_1d_and_command_line(tmp_path):
    ts = _ts()[0]
    pc1 = write_zarr(tmp_path / 'col.zarr', ts[:, 3].copy(), (20,))
    mc.pc_disp(pc1, str(tmp_path / 'd1.zarr'), wavelength=S1_WAVELENGTH)
    d1 = zarr.open(str(tmp_path / 'd1.zarr'), mode='r')
    assert d1.shape == (50,) and d1.chunks == (20,)
    np.testing.assert_allclose(d1[:], mr.pc_disp(ts[:, 3], S1_WAVELENGTH), rtol=1e-6)
    pc = write_zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    assert main(['pc-disp', '--pc', pc, '--out', str(tmp_path / 'd.zarr'), '--wavelength', str(S1_WAVELENGTH)]) == 0
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'd.zarr'), mode='r')[:], mr.pc_disp(ts, S1_WAVELENGTH), rtol=1e-6)


def test_pc_disp_argument_errors(tmp_path):
    ts = _ts()[0]
    pc = write_zarr(tmp_path / 'ts.zarr', ts, (20, 1))
    with pytest.raises(ArgumentError, match='between 0 and 1'):
        mc.pc_disp(pc, str(tmp_path / 'o.zarr'), wavelength=5.55)
    complex_pc = write_zarr(tmp_path / 'c.zarr', ts.astype(np.complex64), (20, 1))
    with pytest.raises(ArgumentError, match='floating point'):
        mc.pc_disp(complex_pc, str(tmp_path / 'o.zarr'), wavelength=S1_WAVELENGTH)
    cube = write_zarr(tmp_path / 'cube.zarr', np.zeros((4, 3, 2), dtype=np.float32), (4, 3, 1))
    with pytest.raises(ArgumentError, match='shape'):
        mc.pc_disp(cube, str(tmp_path / 'o.zarr'), wavelength=S1_WAVELENGTH)


# ---------------------------------------------------------------- DEM error

def _dates(nimages, step=12, gaps=False, seed=0):
    """YYYYMMDD dates `step` days apart, with some steps of 2 to 5 times `step` with `gaps`"""
    steps = np.full(nimages - 1, step)
    if gaps:
        rng = np.random.default_rng(seed)
        steps[rng.choice(nimages - 1, nimages // 4, replace=False)] *= rng.integers(2, 6, nimages // 4)
    days = np.concatenate([[0], np.cumsum(steps)])
    return [str(d).replace('-', '') for d in np.datetime64('2019-01-02') + days], days


def _geometry(n_points, nimages, seed=0):
    """perpendicular baselines (m, relative to image 3), slant range (m), look vector elevation (rad), mm per m"""
    rng = np.random.default_rng(seed)
    bperp = (rng.normal(0, 60, (1, nimages)) + rng.normal(0, 3, (n_points, nimages))).astype(np.float32)
    bperp -= bperp[:, 3:4]
    slant_range = rng.uniform(800e3, 900e3, n_points).astype(np.float32)
    theta = rng.uniform(0.85, 1.1, n_points).astype(np.float32)
    c = 1000 * bperp.astype(np.float64) / (slant_range.astype(np.float64) * np.cos(theta.astype(np.float64)))[:, None]
    return bperp, slant_range, theta, c


@pytest.mark.parametrize('gaps', [False, True])
@pytest.mark.parametrize('ref', [0, 5])
def test_pc_sc_dem_error_api_exact(gaps, ref):
    # a constant velocity and the DEM error, no noise: dh exact, the velocity left
    n_points, nimages = 300, 25
    dates, days = _dates(nimages, gaps=gaps)
    bperp, slant_range, theta, c = _geometry(n_points, nimages)
    rng = np.random.default_rng(1)
    dh_true = rng.normal(0, 20, n_points)
    v = rng.normal(0, 0.1, (n_points, 1))                     # mm / day
    motion = v * (days - days[ref])
    disp = (motion + (c - c[:, ref:ref + 1]) * dh_true[:, None]).astype(np.float32)
    dh, out = mr.pc_sc_dem_error(disp, bperp, slant_range, theta, dates, ref=ref)
    assert dh.shape == (n_points,) and dh.dtype == np.float32 and out.shape == disp.shape and out.dtype == np.float32
    np.testing.assert_allclose(dh, dh_true, atol=2e-3)       # float32 input: 1e-6 of about 100 mm
    np.testing.assert_allclose(out, motion, atol=1e-4)
    np.testing.assert_array_equal(out[:, ref], disp[:, ref])  # 0: the reference image keeps its value


@pytest.mark.parametrize('gaps', [False, True])
def test_pc_sc_dem_error_api_is_the_least_squares_of_the_differences(gaps):
    n_points, nimages = 50, 20
    dates, days = _dates(nimages, gaps=gaps)
    bperp, slant_range, theta, c = _geometry(n_points, nimages)
    rng = np.random.default_rng(2)
    disp = np.cumsum(rng.normal(0, 2, (n_points, nimages)), axis=1).astype(np.float32)
    disp -= disp[:, :1]
    dh = mr.pc_sc_dem_error(disp, bperp, slant_range, theta, dates)[0]
    dt = np.diff(days).astype(np.float64)
    for i in range(n_points):      # [1, dt, dc] is rank deficient for equal steps: dh is still determined
        G = np.stack([np.ones(nimages - 1), dt, np.diff(c[i])], axis=1)
        x = np.linalg.lstsq(G, np.diff(disp[i].astype(np.float64)), rcond=None)[0]
        assert abs(dh[i] - x[2]) <= 1e-5 * max(1, abs(x[2])), (i, dh[i], x[2])


def test_pc_sc_dem_error_api_errors_follow_the_noise_model():
    # white noise of the displacement and a random walk (deformation): dh is unbiased and its error has the
    # covariance of its least squares estimate of the differences
    n_points, nimages = 4000, 40
    dates, days = _dates(nimages, gaps=True, seed=3)
    bperp, slant_range, theta, c = _geometry(n_points, nimages, seed=3)
    rng = np.random.default_rng(4)
    dt = np.diff(days).astype(np.float64)
    sw, srw = 2.0, 0.5                                          # mm, mm per sqrt(day)
    walk = np.concatenate([np.zeros((n_points, 1)), np.cumsum(rng.normal(0, 1, (n_points, nimages - 1)) * srw * np.sqrt(dt), axis=1)], axis=1)
    disp = walk + rng.normal(0, sw, (n_points, nimages)) + (c - c[:, :1]) * 10.0
    disp = (disp - disp[:, :1]).astype(np.float32)
    dh = mr.pc_sc_dem_error(disp, bperp, slant_range, theta, dates)[0].astype(np.float64)
    D = np.diff(np.eye(nimages), axis=0)
    cov = sw**2 * D @ D.T + srw**2 * np.diag(dt)               # covariance of the differences
    A = np.stack([np.ones(nimages - 1), dt], axis=1)
    dc = np.diff(c, axis=1)
    dc_rest = dc - (A @ np.linalg.lstsq(A, dc.T, rcond=None)[0]).T
    std = np.sqrt(np.einsum('ij,jk,ik->i', dc_rest, cov, dc_rest)) / np.einsum('ij,ij->i', dc_rest, dc_rest)
    z = (dh - 10.0) / std
    assert abs(z.mean()) < 0.06 and 0.95 < z.std() < 1.05, (z.mean(), z.std())


def test_pc_sc_dem_error_api_nan():
    n_points, nimages = 40, 12
    dates, days = _dates(nimages)
    bperp, slant_range, theta, c = _geometry(n_points, nimages)
    disp = ((c - c[:, :1]) * 5.0).astype(np.float32)
    disp[3, 7] = np.nan
    bperp = bperp.copy(); bperp[8, 2] = np.nan
    dh, out = mr.pc_sc_dem_error(disp, bperp, slant_range, theta, dates)
    assert np.isnan(dh[[3, 8]]).all() and np.isnan(out[[3, 8]]).all()
    keep = np.setdiff1d(np.arange(n_points), [3, 8])
    np.testing.assert_allclose(dh[keep], 5.0, atol=1e-3)
    dh_keep, out_keep = mr.pc_sc_dem_error(disp[keep], bperp[keep], slant_range[keep], theta[keep], dates)
    np.testing.assert_array_equal(dh[keep], dh_keep)
    np.testing.assert_array_equal(out[keep], out_keep)


@pytest.mark.parametrize('change, match', [
    (dict(disp=np.zeros(10, np.float32)), r'disp must have shape \(n_points, nimages\)'),
    (dict(disp=np.zeros((10, 6), np.int32)), 'disp must be floating point'),
    (dict(bperp=np.zeros((10, 5), np.float32)), r'bperp must have shape \(10, 6\)'),
    (dict(slant_range=np.zeros(9, np.float32)), r'slant_range must have shape \(10,\)'),
    (dict(theta=np.zeros(10, np.int16)), 'theta must be floating point'),
    (dict(dates=['20190102', '20190114']), '2 dates for 6 images'),
    (dict(dates=['20190102', '20190126', '20190114', '20190207', '20190219', '20190303']), '20190126 is followed by 20190114'),
    (dict(dates=['20190102', '20190102', '20190114', '20190207', '20190219', '20190303']), '20190102 is followed by 20190102'),
    (dict(dates=['2019-01-02', '20190114', '20190126', '20190207', '20190219', '20190303']), "YYYYMMDD, got '2019-01-02'"),
    (dict(dates=['20190132', '20190214', '20190226', '20190307', '20190319', '20190331']), "YYYYMMDD, got '20190132'"),
    (dict(ref=6), 'one of the 6 images'),
    (dict(ref=-1), 'one of the 6 images'),
    (dict(ref=True), 'one of the 6 images'),
])
def test_pc_sc_dem_error_api_errors(change, match):
    bperp, slant_range, theta, c = _geometry(10, 6)
    args = dict(disp=np.zeros((10, 6), np.float32), bperp=bperp, slant_range=slant_range, theta=theta,
                dates=_dates(6)[0], ref=0)
    args.update(change)
    with pytest.raises(ArgumentError, match=match):
        mr.pc_sc_dem_error(**args)


def test_pc_sc_dem_error_api_needs_4_images():
    bperp, slant_range, theta, c = _geometry(10, 4)
    with pytest.raises(ArgumentError, match='at least 4 images, got 3'):
        mr.pc_sc_dem_error(np.zeros((10, 3), np.float32), bperp[:, :3], slant_range, theta, _dates(3)[0])
    dates = ' 20190102,20190114 20190126,20190207 '                # text with commas or spaces
    assert mr.pc_sc_dem_error(((c - c[:, :1]) * 3).astype(np.float32), bperp, slant_range, theta, dates)[0].shape == (10,)


@pytest.mark.gpu
def test_pc_sc_dem_error_api_gpu():
    import cupy as cp
    n_points, nimages = 200, 15
    dates, days = _dates(nimages, gaps=True)
    bperp, slant_range, theta, c = _geometry(n_points, nimages)
    disp = (np.random.default_rng(5).normal(0, 2, (n_points, nimages)) + (c - c[:, :1]) * 7).astype(np.float32)
    dh, out = mr.pc_sc_dem_error(*(cp.asarray(a) for a in (disp, bperp, slant_range, theta)), dates)
    assert isinstance(dh, cp.ndarray) and isinstance(out, cp.ndarray)
    dh_cpu, out_cpu = mr.pc_sc_dem_error(disp, bperp, slant_range, theta, dates)
    np.testing.assert_allclose(cp.asnumpy(dh), dh_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(out), out_cpu, rtol=1e-5, atol=1e-4)


def _dem_error_inputs(tmp_path, n_points=50, nimages=9):
    dates, days = _dates(nimages, gaps=True)
    bperp, slant_range, theta, c = _geometry(n_points, nimages)
    rng = np.random.default_rng(6)
    disp = (0.03 * days + rng.normal(0, 1, (n_points, nimages)) + (c - c[:, :1]) * rng.normal(0, 15, (n_points, 1)))
    disp = (disp - disp[:, :1]).astype(np.float32)
    disp[7, 4] = np.nan
    paths = dict(disp=write_zarr(tmp_path / 'disp.zarr', disp, (20, 1)), bperp=write_zarr(tmp_path / 'bperp.zarr', bperp, (20, 1)),
                 slant_range=write_zarr(tmp_path / 'range.zarr', slant_range, (20,)),
                 theta=write_zarr(tmp_path / 'theta.zarr', theta, (20,)))
    return paths, dates, mr.pc_sc_dem_error(disp, bperp, slant_range, theta, dates)


def test_pc_sc_dem_error(tmp_path):
    paths, dates, (dh, out) = _dem_error_inputs(tmp_path)
    mc.pc_sc_dem_error(**paths, dh=str(tmp_path / 'dh.zarr'), out=str(tmp_path / 'out.zarr'), dates=dates, out_chunks=10)
    dh_z, out_z = (zarr.open(str(tmp_path / f), mode='r') for f in ('dh.zarr', 'out.zarr'))
    assert dh_z.shape == dh.shape and dh_z.dtype == np.float32 and dh_z.chunks == (10,)
    assert out_z.shape == out.shape and out_z.dtype == np.float32 and out_z.chunks == (10, 1)
    np.testing.assert_array_equal(dh_z[:], dh)
    np.testing.assert_array_equal(out_z[:], out)
    assert np.isnan(dh_z[7]) and np.isnan(out_z[7]).all()
    mc.pc_sc_dem_error(**paths, dh=str(tmp_path / 'dh_p.zarr'), out=str(tmp_path / 'out_p.zarr'), dates=dates,
                    processes=True, n_workers=2)
    assert zarr.open(str(tmp_path / 'out_p.zarr'), mode='r').chunks == (20, 1)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'dh_p.zarr'), mode='r')[:], dh)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'out_p.zarr'), mode='r')[:], out)


def test_pc_sc_dem_error_command_line_and_pipeline(tmp_path):
    paths, dates, (dh, out) = _dem_error_inputs(tmp_path)
    args = [a for k, v in paths.items() for a in (f'--{k}', v)]
    assert main(['pc-sc-dem-error', *args, '--dh', str(tmp_path / 'dh.zarr'), '--out', str(tmp_path / 'out.zarr'),
                 '--dates', *dates]) == 0
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'dh.zarr'), mode='r')[:], dh)
    # the dates of a pipeline from the metadata: a list of a TOML file
    (tmp_path / 'meta.toml').write_text(f'dates = {dates}\n'.replace("'", '"'))
    lines = ['[vars]', 'dates = { file = "meta.toml", key = "dates" }', '', '[[step]]', 'name = "dem_error"',
             'run = "pc-sc-dem-error"', *(f'{k} = "{v}"' for k, v in paths.items()), 'dh = "dh2.zarr"',
             'out = "out2.zarr"', 'dates = "${dates}"']
    (tmp_path / 'p.toml').write_text('\n'.join(lines) + '\n')
    assert main(['run', str(tmp_path / 'p.toml'), '-q']) == 0
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'out2.zarr'), mode='r')[:], out)


def test_pc_sc_dem_error_argument_errors(tmp_path):
    paths, dates, _ = _dem_error_inputs(tmp_path)
    outputs = dict(dh=str(tmp_path / 'dh.zarr'), out=str(tmp_path / 'o.zarr'))
    with pytest.raises(ArgumentError, match='8 dates for 9 images'):
        mc.pc_sc_dem_error(**paths, **outputs, dates=dates[:-1])
    with pytest.raises(ArgumentError, match='one of the 9 images'):
        mc.pc_sc_dem_error(**paths, **outputs, dates=dates, ref=9)
    short = write_zarr(tmp_path / 'short.zarr', np.zeros(49, np.float32), (20,))
    with pytest.raises(ArgumentError, match=r'theta must have shape \(50,\)'):
        mc.pc_sc_dem_error(**{**paths, 'theta': short}, **outputs, dates=dates)
    three = write_zarr(tmp_path / 'three.zarr', np.zeros((50, 3), np.float32), (20, 1))
    with pytest.raises(ArgumentError, match=r'bperp must have shape \(50, 3\)'):
        mc.pc_sc_dem_error(**{**paths, 'disp': three}, **outputs, dates=dates[:3])
    with pytest.raises(ArgumentError, match='at least 4 images'):
        mc.pc_sc_dem_error(**{**paths, 'disp': three, 'bperp': three}, **outputs, dates=dates[:3])


# ---------------------------------------------------------------- velocity

def _design(days, periods=()):
    t = np.asarray(days, dtype=np.float64) / 365.25
    columns = [np.ones_like(t), t]
    for period in periods:
        columns += [np.cos(2 * np.pi * t / period), np.sin(2 * np.pi * t / period)]
    return np.stack(columns, axis=1)


@pytest.mark.parametrize('gaps', [False, True])
@pytest.mark.parametrize('periods', [None, [1], [1, 0.5]])
def test_pc_velocity_api_exact(gaps, periods):
    # the model itself, no noise: the coefficients exact, the residuals 0
    n_points, nimages = 200, 40
    dates, days = _dates(nimages, gaps=gaps)
    g = _design(days, periods or [])
    coef_true = np.random.default_rng(2).normal(0, 10, (n_points, g.shape[1]))
    disp = (coef_true @ g.T).astype(np.float32)
    velocity, velocity_std, rms, coef, residual = mr.pc_velocity(disp, dates, periods=periods)
    for a, shape in ((velocity, (n_points,)), (velocity_std, (n_points,)), (rms, (n_points,)),
                     (coef, (n_points, g.shape[1])), (residual, (n_points, nimages))):
        assert a.shape == shape and a.dtype == np.float32
    np.testing.assert_allclose(coef, coef_true, atol=2e-4)          # float32 input: 1e-6 of about 100 mm
    np.testing.assert_array_equal(velocity, coef[:, 1])
    np.testing.assert_allclose(residual, 0, atol=2e-5)
    np.testing.assert_allclose(rms, 0, atol=2e-5)
    np.testing.assert_allclose(velocity_std, 0, atol=2e-5)


def test_pc_velocity_api_is_the_least_squares():
    dates, days = _dates(30, gaps=True)
    rng = np.random.default_rng(3)
    disp = rng.normal(0, 5, (20, 30)).astype(np.float32)
    g = _design(days, [1])
    velocity, velocity_std, rms, coef, residual = mr.pc_velocity(disp, ','.join(dates), periods='1')
    expected, rss = np.linalg.lstsq(g, disp.T.astype(np.float64), rcond=None)[:2]
    np.testing.assert_allclose(coef, expected.T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(residual, disp - expected.T @ g.T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rms, np.sqrt(rss / 30), rtol=1e-5)
    np.testing.assert_allclose(velocity_std, np.sqrt(rss / (30 - 4) * np.linalg.inv(g.T @ g)[1, 1]), rtol=1e-5)


def test_pc_velocity_api_std_follows_the_noise():
    # white noise of 2 mm: the standard deviation from the residuals is the spread of the velocity errors
    n_points, nimages = 20000, 60
    dates, days = _dates(nimages, gaps=True)
    rng = np.random.default_rng(4)
    v = rng.normal(0, 20, (n_points, 1))
    disp = (v * days / 365.25 + 3 * np.cos(2 * np.pi * days / 365.25) + rng.normal(0, 2, (n_points, nimages)))
    velocity, velocity_std = mr.pc_velocity(disp.astype(np.float32), dates, periods=[1])[:2]
    g = _design(days, [1])
    theory = 2 * np.sqrt(np.linalg.inv(g.T @ g)[1, 1])
    errors = velocity - v[:, 0]
    assert abs(errors.mean()) < 3 * theory / np.sqrt(n_points)
    assert abs(errors.std() / theory - 1) < 0.03
    assert abs(np.sqrt(np.mean(velocity_std.astype(np.float64) ** 2)) / theory - 1) < 0.01


def test_pc_velocity_api_exclude():
    # one image with a large error of every point: excluded, the fit is that of the other images, and its residual
    # shows the error
    n_points, nimages = 100, 25
    dates, days = _dates(nimages)
    rng = np.random.default_rng(5)
    disp = 4 * days / 365.25 + rng.normal(0, 0.5, (n_points, nimages))
    disp[:, -1] += 30
    disp = disp.astype(np.float32)
    velocity = mr.pc_velocity(disp, dates)[0]
    assert np.median(velocity) > 8                               # the last image pulls the line
    velocity, velocity_std, rms, coef, residual = mr.pc_velocity(disp, dates, exclude=[dates[-1]])
    kept = mr.pc_velocity(disp[:, :-1], dates[:-1])
    for a, b in zip((velocity, velocity_std, rms, coef), kept[:4]):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(residual[:, :-1], kept[4], rtol=1e-5, atol=1e-5)
    assert abs(np.median(velocity) - 4) < 0.2
    assert 28 < np.median(residual[:, -1]) < 32
    assert np.sqrt(np.mean(residual[:, -1] ** 2)) > 20 * np.sqrt(np.mean(residual[:, :-1] ** 2))
    # the first date excluded: the time is still counted from it
    first = mr.pc_velocity(disp, dates, exclude=f'{dates[0]} {dates[-1]}')[3]
    expected = np.linalg.lstsq(_design(days[1:-1]), disp[:, 1:-1].T.astype(np.float64), rcond=None)[0].T
    np.testing.assert_allclose(first, expected, rtol=1e-5, atol=1e-4)


def test_pc_velocity_api_nan():
    n_points, nimages = 8, 12
    dates, days = _dates(nimages, gaps=True)
    rng = np.random.default_rng(6)
    disp = rng.normal(0, 3, (n_points, nimages)).astype(np.float32)
    disp[1] = np.nan                                       # no value
    disp[2, 3:] = np.nan                                   # 3 values for 2 coefficients: fewer than p + 2
    disp[3, 4:] = np.nan                                   # 4 values: enough
    disp[4, [0, 5, 6]] = np.nan                            # some values missing
    velocity, velocity_std, rms, coef, residual = mr.pc_velocity(disp, dates)
    for a in (velocity, velocity_std, rms, coef[:, 0], coef[:, 1]):
        assert np.isnan(a[[1, 2]]).all() and np.isfinite(a[[0, 3, 4, 5, 6, 7]]).all()
    assert np.isnan(residual[[1, 2]]).all()
    np.testing.assert_array_equal(np.isnan(residual[[0, 3, 4, 5, 6, 7]]), np.isnan(disp[[0, 3, 4, 5, 6, 7]]))
    g = _design(days)
    for i in (3, 4):
        ok = np.isfinite(disp[i])
        expected = np.linalg.lstsq(g[ok], disp[i, ok].astype(np.float64), rcond=None)[0]
        np.testing.assert_allclose(coef[i], expected, rtol=1e-5, atol=1e-5)
    # the points with all values are those of the fit without the others
    np.testing.assert_array_equal(velocity[[0, 5, 6, 7]], mr.pc_velocity(disp[[0, 5, 6, 7]], dates)[0])
    # values left only at dates one year apart: the annual term is not defined there, nan
    one_year = [str(d).replace('-', '') for d in np.datetime64('2019-01-02') + np.arange(5) * 365]
    yearly = rng.normal(0, 3, (2, 5)).astype(np.float32)
    dense = np.concatenate([yearly, rng.normal(0, 3, (2, 4)).astype(np.float32)], axis=1)
    extra = ['20190301', '20190601', '20190901', '20191201']
    order = np.argsort(one_year + extra)
    dense = dense[:, order]
    dense[0, np.isin(np.array(one_year + extra)[order], extra)] = np.nan
    v = mr.pc_velocity(dense, list(np.array(one_year + extra)[order]), periods=[1])[0]
    assert np.isnan(v[0]) and np.isfinite(v[1])


@pytest.mark.parametrize('change, match', [
    (dict(disp=np.zeros(12, np.float32)), r'disp must have shape \(n_points, nimages\)'),
    (dict(disp=np.zeros((3, 12), np.int32)), 'disp must be floating point'),
    (dict(dates='20190102'), '1 dates for 12 images'),
    (dict(exclude=['20190103']), 'exclude: 20190103 not among the dates'),
    (dict(periods=[1, -0.5]), 'periods must be positive numbers of years'),
    (dict(periods='annual'), 'periods must be numbers of years'),
    (dict(periods=[1, 1]), 'cannot be told apart'),
    (dict(periods=[12 / 365.25]), 'cannot be told apart'),        # the period of the sampling: a constant
    (dict(periods=[1, 0.5], exclude=_dates(12)[0][:5]), r'7 dates to fit \(12 images, 5 excluded\) for 6 coefficients'),
])
def test_pc_velocity_api_errors(change, match):
    args = dict(disp=np.zeros((3, 12), np.float32), dates=_dates(12)[0])
    args.update(change)
    with pytest.raises(ArgumentError, match=match):
        mr.pc_velocity(**args)


@pytest.mark.gpu
def test_pc_velocity_api_gpu():
    import cupy as cp
    n_points, nimages = 300, 30
    dates, days = _dates(nimages, gaps=True)
    rng = np.random.default_rng(7)
    disp = (rng.normal(0, 10, (n_points, 1)) * days / 365.25 + rng.normal(0, 2, (n_points, nimages))).astype(np.float32)
    disp[3, 4:9] = np.nan
    out = mr.pc_velocity(cp.asarray(disp), dates, exclude=[dates[7]], periods=[1])
    assert all(isinstance(a, cp.ndarray) for a in out)
    for a, b in zip(out, mr.pc_velocity(disp, dates, exclude=[dates[7]], periods=[1])):
        np.testing.assert_allclose(cp.asnumpy(a), b, rtol=1e-5, atol=1e-4)


def _velocity_inputs(tmp_path, n_points=50, nimages=20):
    dates, days = _dates(nimages, gaps=True)
    rng = np.random.default_rng(8)
    disp = rng.normal(0, 10, (n_points, 1)) * days / 365.25 + rng.normal(0, 1, (n_points, nimages))
    disp[:, 6] += rng.normal(0, 8, n_points)               # a noisy image
    disp[:25, 11] += 20                                    # an image with an error of the first half of the points
    disp = disp.astype(np.float32)
    disp[7, 4] = np.nan
    disp[9] = np.nan
    mask = np.zeros(n_points, dtype=bool); mask[25:] = True
    return (write_zarr(tmp_path / 'disp.zarr', disp, (20, 1)), write_zarr(tmp_path / 'mask.zarr', mask, (20,)),
            disp, mask, dates)


def test_pc_velocity(tmp_path):
    disp_path, mask_path, disp, mask, dates = _velocity_inputs(tmp_path)
    names = ('velocity', 'velocity_std', 'rms', 'image_rms', 'coef')
    outputs = {name: str(tmp_path / f'{name}.zarr') for name in names}
    mc.pc_velocity(disp_path, **outputs, dates=dates, exclude=[dates[6]], periods=[1, 0.5], out_chunks=10)
    expected = mr.pc_velocity(disp, dates, exclude=[dates[6]], periods=[1, 0.5])
    z = {name: zarr.open(path, mode='r') for name, path in outputs.items()}
    for name, a in zip(('velocity', 'velocity_std', 'rms', 'coef'), expected[:4]):
        assert z[name].shape == a.shape and z[name].dtype == np.float32
        assert z[name].chunks == ((10, 1) if name == 'coef' else (10,))
        np.testing.assert_array_equal(z[name][:], a)
    assert np.isnan(z['velocity'][9]) and np.isfinite(z['velocity'][7])
    residual = expected[4]
    image_rms = np.sqrt(np.nanmean(residual ** 2, axis=0))
    assert z['image_rms'].shape == (20,) and z['image_rms'].dtype == np.float32
    np.testing.assert_allclose(z['image_rms'][:], image_rms, rtol=1e-5)
    assert np.argsort(image_rms)[-2:].tolist() == [11, 6]       # the excluded image is still there
    # the mask: the image with the error of the points outside it is like the others
    mc.pc_velocity(disp_path, str(tmp_path / 'v2.zarr'), str(tmp_path / 's2.zarr'), dates,
                   image_rms=str(tmp_path / 'i2.zarr'), mask=mask_path, exclude=[dates[6]], periods=[1, 0.5], processes=True, n_workers=2)
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'i2.zarr'), mode='r')[:],
                               np.sqrt(np.nanmean(residual[mask] ** 2, axis=0)), rtol=1e-5)
    assert zarr.open(str(tmp_path / 'v2.zarr'), mode='r').chunks == (20,)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'v2.zarr'), mode='r')[:], expected[0])


def test_pc_velocity_command_line_and_pipeline(tmp_path):
    disp_path, mask_path, disp, mask, dates = _velocity_inputs(tmp_path)
    expected = mr.pc_velocity(disp, dates, exclude=[dates[6], dates[11]], periods=[1])
    assert main(['pc-velocity', '--disp', disp_path, '--velocity', str(tmp_path / 'v.zarr'), '--velocity_std',
                 str(tmp_path / 's.zarr'), '--dates', *dates, '--exclude', dates[6], dates[11], '--periods', '1']) == 0
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'v.zarr'), mode='r')[:], expected[0])
    # one excluded date, the periods of a list and the dates of the metadata
    (tmp_path / 'meta.toml').write_text(f'dates = {dates}\n'.replace("'", '"'))
    lines = ['[vars]', 'dates = { file = "meta.toml", key = "dates" }', '', '[[step]]', 'name = "velocity"',
             'run = "pc-velocity"', f'disp = "{disp_path}"', 'velocity = "v2.zarr"', 'velocity_std = "s2.zarr"',
             'dates = "${dates}"', f'exclude = ["{dates[6]}"]', 'periods = [1, 0.5]']
    (tmp_path / 'p.toml').write_text('\n'.join(lines) + '\n')
    assert main(['run', str(tmp_path / 'p.toml'), '-q']) == 0
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'v2.zarr'), mode='r')[:],
                                  mr.pc_velocity(disp, dates, exclude=[dates[6]], periods=[1, 0.5])[0])


def test_pc_velocity_argument_errors(tmp_path):
    disp_path, mask_path, disp, mask, dates = _velocity_inputs(tmp_path)
    out = dict(velocity=str(tmp_path / 'v.zarr'), velocity_std=str(tmp_path / 's.zarr'))
    short = write_zarr(tmp_path / 'short.zarr', np.zeros(49, bool), (20,))
    for kwargs, match in [
        (dict(dates=dates[:-1]), '19 dates for 20 images'),
        (dict(dates=dates, exclude=['20190101']), 'exclude: 20190101 not among the dates'),
        (dict(dates=dates, periods=[0]), 'periods must be positive'),
        (dict(dates=dates, exclude=dates[3:]), r'3 dates to fit \(20 images, 17 excluded\) for 2 coefficients'),
        (dict(dates=dates, mask=short), r'mask must be bool of shape \(50,\)'),
        (dict(dates=dates, mask=disp_path), r'mask must be bool of shape \(50,\)'),
    ]:
        with pytest.raises(ArgumentError, match=match):
            mc.pc_velocity(disp_path, **out, **kwargs)
        assert not (tmp_path / 'v.zarr').exists() and not (tmp_path / 's.zarr').exists()
    one = write_zarr(tmp_path / 'one.zarr', disp[:, 0], (20,))
    with pytest.raises(ArgumentError, match=r'disp must have shape \(n_points, nimages\)'):
        mc.pc_velocity(one, **out, dates=dates)
