import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.command import main
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
    with pytest.raises(ValueError, match='exactly one'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'))
    with pytest.raises(ValueError, match='exactly one'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=mask, ref_point=1)
    with pytest.raises(ValueError, match='not one of the 50 points'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_point=50)
    short = write_zarr(tmp_path / 'short.zarr', stable[:40], (20,))
    with pytest.raises(ValueError, match=r'bool of shape \(50,\)'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=short)
    empty = write_zarr(tmp_path / 'empty.zarr', np.zeros(50, dtype=bool), (20,))
    with pytest.raises(ValueError, match='no reference point'):
        mc.pc_reference(pc, str(tmp_path / 'o.zarr'), ref_mask=empty)
    complex_pc = write_zarr(tmp_path / 'c.zarr', ts.astype(np.complex64), (20, 1))
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
    with pytest.raises(ValueError, match='between 0 and 1'):
        mc.pc_disp(pc, str(tmp_path / 'o.zarr'), wavelength=5.55)
    complex_pc = write_zarr(tmp_path / 'c.zarr', ts.astype(np.complex64), (20, 1))
    with pytest.raises(ValueError, match='floating point'):
        mc.pc_disp(complex_pc, str(tmp_path / 'o.zarr'), wavelength=S1_WAVELENGTH)
    cube = write_zarr(tmp_path / 'cube.zarr', np.zeros((4, 3, 2), dtype=np.float32), (4, 3, 1))
    with pytest.raises(ValueError, match='shape'):
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
    with pytest.raises(ValueError, match=match):
        mr.pc_sc_dem_error(**args)


def test_pc_sc_dem_error_api_needs_4_images():
    bperp, slant_range, theta, c = _geometry(10, 4)
    with pytest.raises(ValueError, match='at least 4 images, got 3'):
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
    with pytest.raises(ValueError, match='8 dates for 9 images'):
        mc.pc_sc_dem_error(**paths, **outputs, dates=dates[:-1])
    with pytest.raises(ValueError, match='one of the 9 images'):
        mc.pc_sc_dem_error(**paths, **outputs, dates=dates, ref=9)
    short = write_zarr(tmp_path / 'short.zarr', np.zeros(49, np.float32), (20,))
    with pytest.raises(ValueError, match=r'theta must have shape \(50,\)'):
        mc.pc_sc_dem_error(**{**paths, 'theta': short}, **outputs, dates=dates)
    three = write_zarr(tmp_path / 'three.zarr', np.zeros((50, 3), np.float32), (20, 1))
    with pytest.raises(ValueError, match=r'bperp must have shape \(50, 3\)'):
        mc.pc_sc_dem_error(**{**paths, 'disp': three}, **outputs, dates=dates[:3])
    with pytest.raises(ValueError, match='at least 4 images'):
        mc.pc_sc_dem_error(**{**paths, 'disp': three, 'bperp': three}, **outputs, dates=dates[:3])
