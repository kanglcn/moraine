import numpy as np
import pytest

import moraine as mr
from moraine.api.pqm import temp_coh


@pytest.mark.gpu
def test_temp_coh_ds_gpu(ds_can):
    import cupy as cp
    coh = mr.emperical_co_pc(ds_can['rslc'], ds_can['gix'], ds_can['is_shp'])
    ph = mr.emi(coh)
    np.testing.assert_array_almost_equal(temp_coh(coh, ph), temp_coh(cp.asarray(coh), cp.asarray(ph)).get())


@pytest.mark.gpu
def test_temp_coh_ps_gpu(rslc):
    import cupy as cp
    s = rslc[:300, :300]
    pairs = mr.TempNet.from_bandwidth(s.shape[-1], 1).image_pairs
    intf = s[..., pairs[:, 0]] * s[..., pairs[:, 1]].conj()
    t = temp_coh(intf, s, pairs)
    assert t.shape == s.shape[:2]
    np.testing.assert_array_almost_equal(t, temp_coh(cp.asarray(intf), cp.asarray(s), pairs).get())


def test_temp_coh_keeps_inputs(rng):
    # contiguous raster and point cloud stacks of non unit amplitude; the inputs must not change
    pairs = mr.TempNet.from_bandwidth(6, 2).image_pairs
    s = (rng.standard_normal((4, 5, 6)) + 1j * rng.standard_normal((4, 5, 6))).astype(np.complex64)
    noise = np.exp(1j * rng.normal(0, 0.5, (4, 5, pairs.shape[0])))
    intf = (s[..., pairs[:, 0]] * s[..., pairs[:, 1]].conj() * noise).astype(np.complex64)
    u = lambda z: z / np.abs(z)
    expected = np.abs((u(intf) * u(s[..., pairs[:, 0]]).conj() * u(s[..., pairs[:, 1]])).mean(axis=-1))
    for shape in (s.shape[:2], (s.shape[0] * s.shape[1],)):
        intf_in = intf.reshape(*shape, -1).copy(); s_in = s.reshape(*shape, -1).copy()
        t = temp_coh(intf_in, s_in, pairs)
        np.testing.assert_allclose(t, expected.reshape(shape), rtol=1e-5)
        np.testing.assert_array_equal(intf_in, intf.reshape(*shape, -1))
        np.testing.assert_array_equal(s_in, s.reshape(*shape, -1))


# ---------------------------------------------------------------- DEM error relative to the neighbourhood

S1_WAVELENGTH = 0.0555


def _su_points(n=3000, nimages=30, dh_max=10.0, sigma=0.0, seed=1):
    """points dh_true above a neighbourhood whose phase (deformation, atmosphere) is that of the filtered
    interferograms of the sequential image pairs, plus white noise of `sigma` radians"""
    from moraine.api.pqm import _phase_per_m
    rng = np.random.default_rng(seed)
    pairs = mr.TempNet.from_bandwidth(nimages, bandwidth=1).image_pairs
    bperp = rng.normal(0, 80, (n, nimages)); bperp -= bperp[:, 5:6]
    slant_range, theta = rng.uniform(800e3, 900e3, n), rng.uniform(0.85, 1.1, n)
    f = _phase_per_m(slant_range, theta, S1_WAVELENGTH)
    common = rng.uniform(-np.pi, np.pi, (n, nimages))
    dh_true = rng.uniform(-dh_max, dh_max, n)
    intf = np.exp(1j * (common[:, pairs[:, 0]] - common[:, pairs[:, 1]])).astype(np.complex64)
    rslc = (3 * np.exp(1j * (common + f[:, None] * bperp * dh_true[:, None]
                             + sigma * rng.standard_normal((n, nimages))))).astype(np.complex64)
    a = f[:, None] * (bperp[:, pairs[:, 0]] - bperp[:, pairs[:, 1]])   # phase per meter of every pair
    return dict(intf=intf, rslc=rslc, bperp=bperp.astype(np.float32), slant_range=slant_range.astype(np.float32),
                theta=theta.astype(np.float32), wavelength=S1_WAVELENGTH, image_pairs=pairs), dh_true, a


def test_pc_su_dem_error_exact():
    from moraine.api.pqm import _remove_su_dem_error
    args, dh_true, _ = _su_points(n=500)
    dh, t_coh = mr.pc_su_dem_error(**args)
    assert dh.dtype == np.float32 and t_coh.dtype == np.float32 and dh.shape == t_coh.shape == (500,)
    np.testing.assert_allclose(dh, dh_true, atol=1e-4)
    np.testing.assert_allclose(t_coh, 1, atol=1e-5)
    assert mr.temp_coh(args['intf'], args['rslc'], args['image_pairs']).min() < 0.9     # without the fit: noise
    # the phase without the DEM error agrees with the filtered interferograms: plain temporal coherence 1
    fixed = _remove_su_dem_error(args['rslc'], args['bperp'], args['slant_range'], args['theta'], S1_WAVELENGTH, dh)
    assert fixed.dtype == np.complex64
    np.testing.assert_allclose(np.abs(fixed), 3, rtol=1e-6)
    np.testing.assert_allclose(mr.temp_coh(args['intf'], fixed, args['image_pairs']), t_coh, atol=1e-5)


def test_pc_su_dem_error_follows_the_noise_model():
    # white noise of the phase of the points: dh unbiased, with the error of least squares of the phases of the pairs
    sigma = 0.2
    args, dh_true, a = _su_points(sigma=sigma, seed=2)
    dh = mr.pc_su_dem_error(**args, max_dh=20)[0]
    v = a - a.mean(1, keepdims=True)
    nimages = args['rslc'].shape[1]
    D = np.diff(np.eye(nimages), axis=0)                            # the pairs of neighbouring images
    std = np.sqrt(np.einsum('ij,jk,ik->i', v, sigma**2 * D @ D.T, v)) / np.einsum('ij,ij->i', v, v)
    z = (dh - dh_true) / std
    assert abs(z.mean()) < 0.08 and 0.93 < z.std() < 1.07, (z.mean(), z.std())


def test_pc_su_dem_error_random_phases():
    # fitting the DEM error raises the temporal coherence of noise only a little
    args, _, _ = _su_points(seed=3)
    args['rslc'] = np.exp(2j * np.pi * np.random.default_rng(4).random(args['rslc'].shape)).astype(np.complex64)
    t_coh = mr.pc_su_dem_error(**args)[1]
    plain = mr.temp_coh(args['intf'], args['rslc'], args['image_pairs'])
    assert np.all(t_coh >= plain - 1e-6)
    assert np.percentile(t_coh, 99.9) < 0.65 and np.mean(t_coh - plain) < 0.1


def test_pc_su_dem_error_search_range_and_brute_force():
    args, _, a = _su_points(n=200, dh_max=0, sigma=0.3, seed=5)
    dh_true = 40.0                                                  # a roof 40 m above its neighbourhood
    from moraine.api.pqm import _phase_per_m
    f = _phase_per_m(args['slant_range'], args['theta'], S1_WAVELENGTH)
    args['rslc'] = (args['rslc'] * np.exp(1j * f[:, None] * args['bperp'] * dh_true)).astype(np.complex64)
    near, t_near = mr.pc_su_dem_error(**args, max_dh=20)
    far, t_far = mr.pc_su_dem_error(**args, max_dh=60)
    assert np.median(np.abs(far - dh_true)) < 2 and np.median(t_far) > np.median(t_near) + 0.2
    # the coherence is the largest on a fine grid of the search range
    diff = (args['intf'] / np.abs(args['intf']) * np.conj(args['rslc'][:, args['image_pairs'][:, 0]])
            * args['rslc'][:, args['image_pairs'][:, 1]] / np.abs(args['rslc'][:, args['image_pairs'][:, 0]]
                                                                * args['rslc'][:, args['image_pairs'][:, 1]]))
    grid = np.arange(-60, 60.001, 0.01)
    best = np.max(np.abs(np.einsum('ik,ikg->ig', diff, np.exp(1j * a[:, :, None] * grid))), axis=1) / a.shape[1]
    assert np.all(np.abs(near) <= 20) and np.all(np.abs(far) <= 60)
    np.testing.assert_allclose(t_far, best, atol=5e-4)


def test_pc_su_dem_error_nan():
    args, dh_true, _ = _su_points(n=50)
    args['rslc'][3, 7] = np.nan
    args['bperp'][8, 2] = np.nan
    args['slant_range'][12] = np.nan
    args['bperp'][20] = 4.0                                         # equal baselines: no DEM error to fit
    dh, t_coh = mr.pc_su_dem_error(**args)
    assert np.isnan(dh[[3, 8, 12, 20]]).all() and np.isnan(t_coh[[3, 8, 12]]).all()
    plain = mr.temp_coh(args['intf'], args['rslc'], args['image_pairs'])
    np.testing.assert_allclose(t_coh[20], plain[20], rtol=1e-5)
    keep = np.setdiff1d(np.arange(50), [3, 8, 12, 20])
    np.testing.assert_allclose(dh[keep], dh_true[keep], atol=1e-4)


@pytest.mark.parametrize('change, match', [
    (dict(rslc=np.ones((10, 6), np.float32)), 'rslc must be complex'),
    (dict(intf=np.ones(10, np.complex64)), 'shape'),
    (dict(bperp=np.zeros((10, 5), np.float32)), r'bperp must have shape \(10, 6\)'),
    (dict(theta=np.zeros(9, np.float32)), r'theta must have shape \(10,\)'),
    (dict(slant_range=np.zeros(10, np.int32)), 'slant_range must be floating point'),
    (dict(wavelength=5.55), 'between 0 and 1'),
    (dict(max_dh=0), 'max_dh must be positive'),
    (dict(image_pairs=np.array([[0, 1], [1, 2]])), 'intf has 5 interferograms of 6 images'),
    (dict(image_pairs=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 6]])), 'intf has 5 interferograms'),
])
def test_pc_su_dem_error_errors(change, match):
    args = _su_points(n=10, nimages=6)[0]
    args.update(change)
    with pytest.raises(ValueError, match=match):
        mr.pc_su_dem_error(**args)


def _zarr(path, a, chunks):
    import zarr
    z = zarr.open(str(path), mode='w', shape=a.shape, dtype=a.dtype, chunks=chunks)
    z[:] = a
    return str(path)


def test_pc_su_dem_error_command(tmp_path):
    import zarr
    import moraine.cli as mc
    from moraine.command import main
    from moraine.api.pqm import _remove_su_dem_error
    args, dh_true, _ = _su_points(n=70, sigma=0.2)
    args['rslc'][5, 3] = np.nan
    pairs = args.pop('image_pairs')
    np.savetxt(tmp_path / 'pairs.txt', pairs, fmt='%d')
    paths = {k: _zarr(tmp_path / f'{k}.zarr', v, (30, 1) if v.ndim == 2 else (30,))
             for k, v in args.items() if k != 'wavelength'}
    dh, t_coh = mr.pc_su_dem_error(**args, image_pairs=pairs, max_dh=30)
    fixed = _remove_su_dem_error(args['rslc'], args['bperp'], args['slant_range'], args['theta'], S1_WAVELENGTH, dh)
    out = {k: str(tmp_path / f'{k}_o.zarr') for k in ('dh', 't_coh', 'out')}
    mc.pc_su_dem_error(**paths, **out, wavelength=S1_WAVELENGTH, image_pairs=pairs, max_dh=30, chunks=20)
    res = {k: zarr.open(v, mode='r') for k, v in out.items()}
    assert res['dh'].chunks == (20,) and res['out'].chunks == (20, 1) and res['out'].dtype == np.complex64
    np.testing.assert_array_equal(res['dh'][:], dh)
    np.testing.assert_array_equal(res['t_coh'][:], t_coh)
    np.testing.assert_array_equal(res['out'][:], fixed)
    assert np.isnan(res['dh'][5]) and np.isnan(res['out'][5, 3])
    # without `out`, in processes, from the command line
    mc.pc_su_dem_error(**paths, dh=str(tmp_path / 'dh_p.zarr'), t_coh=str(tmp_path / 't_p.zarr'),
                       wavelength=S1_WAVELENGTH, image_pairs=pairs, max_dh=30, processes=True, n_workers=2)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'dh_p.zarr'), mode='r')[:], dh)
    cli = [a for k, v in paths.items() for a in (f'--{k}', v)]
    assert main(['pc-su-dem-error', *cli, '--dh', str(tmp_path / 'dh_c.zarr'), '--t_coh', str(tmp_path / 't_c.zarr'),
                 '--wavelength', str(S1_WAVELENGTH), '--image_pairs', str(tmp_path / 'pairs.txt'),
                 '--max_dh', '30', '-q']) == 0
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 't_c.zarr'), mode='r')[:], t_coh)
    with pytest.raises(ValueError, match='between 0 and 1'):
        mc.pc_su_dem_error(**paths, **out, wavelength=5.55, image_pairs=pairs)
