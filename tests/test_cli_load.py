"""Loading GAMMA results. Needs the GAMMA software on PATH and the sample GAMMA output."""
import os
import shutil

import numpy as np
import pytest
import toml
import zarr

import moraine.cli as mc
from conftest import data_path

NLINES, WIDTH, NIMAGES, REF = 2500, 1834, 17, '20220620'


@pytest.fixture(scope='module')
def gamma():
    for cmd in ('base_calc', 'geocode', 'phase_sim_orb', 'create_offset'):
        if shutil.which(cmd) is None:
            pytest.skip(f'GAMMA command {cmd} not found')
    return data_path('gamma')


def test_load_gamma_range(gamma, tmp_path):
    mc.load_gamma_range(str(gamma / 'rslc' / f'{REF}.rslc.par'), str(tmp_path / 'range.zarr'))
    r = zarr.open(str(tmp_path / 'range.zarr'), mode='r')[:]
    assert r.shape == (NLINES, WIDTH)
    assert np.all(np.diff(r, axis=1) > 0)          # slant range increases with the range index


def test_load_gamma_metadata(gamma, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # base_calc writes base.out / base_calc.log to the working directory
    mc.load_gamma_metadata(str(gamma / 'rslc'), str(gamma / 'DEM' / 'dem_seg_par'), REF, str(tmp_path / 'meta.toml'))
    meta = toml.load(tmp_path / 'meta.toml')
    assert len(meta['dates']) == NIMAGES
    assert REF in meta['dates']


def test_load_gamma_lat_lon_hgt(gamma, tmp_path):
    g = gamma / 'geocoding'
    out = [str(tmp_path / f) for f in ('lat.zarr', 'lon.zarr', 'hgt.zarr')]
    mc.load_gamma_lat_lon_hgt(str(g / '20210802.diff_par'), str(gamma / 'rslc' / f'{REF}.rslc.par'),
                              str(gamma / 'DEM' / 'dem_seg_par'), str(g / '20210802.hgt'), str(tmp_path / 'scratch'), *out)
    lat, lon = zarr.open(out[0], mode='r')[:], zarr.open(out[1], mode='r')[:]
    assert lat.shape == lon.shape == (NLINES, WIDTH)
    assert np.nanmin(lat) >= -90 and np.nanmax(lat) <= 90 and np.nanmin(lon) >= -180 and np.nanmax(lon) <= 360


def test_load_gamma_look_vector(gamma, tmp_path):
    g = gamma / 'geocoding'
    out = [str(tmp_path / f) for f in ('theta.zarr', 'phi.zarr')]
    mc.load_gamma_look_vector(str(g / '20210802.lv_theta'), str(g / '20210802.lv_phi'), str(g / '20210802.lt_fine'),
                              str(gamma / 'rslc' / f'{REF}.rslc.par'), str(gamma / 'DEM' / 'dem_seg_par'),
                              str(tmp_path / 'scratch'), *out)
    assert zarr.open(out[0], mode='r').shape == zarr.open(out[1], mode='r').shape == (NLINES, WIDTH)


@pytest.mark.slow
def test_load_gamma_flatten_rslc(gamma, tmp_path):
    # phase_sim_orb takes ~50 s per image; reuse the simulated orbital phases of the sample data
    # (data/gamma/sim_orb/<ref>_<date>.sim_orb), which load_gamma_flatten_rslc skips if present
    (tmp_path / 'scratch').mkdir()
    for f in sorted((gamma / 'sim_orb').glob('*.sim_orb')):
        (tmp_path / 'scratch' / f.name).symlink_to(f)
    mc.load_gamma_flatten_rslc(str(gamma / 'rslc'), REF, str(gamma / 'geocoding' / '20210802.hgt'),
                               str(tmp_path / 'scratch'), str(tmp_path / 'rslc.zarr'), chunks=(1000, 1000))
    z = zarr.open(str(tmp_path / 'rslc.zarr'), mode='r')
    assert z.shape == (NLINES, WIDTH, NIMAGES)
    assert np.iscomplexobj(z[:10, :10, 0])


class _Stop(Exception):
    pass


def test_load_gamma_flatten_rslc_gamma_threads(tmp_path, monkeypatch):
    rslc_dir = data_path('gamma', 'rslc')
    commands = []

    def system(command):
        commands.append(command)
        if 'phase_sim_orb' in command:
            raise _Stop

    monkeypatch.setattr(os, 'system', system)
    for kwargs, threads in (({'gamma_threads': 3}, 3), ({}, min(64, os.cpu_count() or 1))):
        commands.clear()
        with pytest.raises(_Stop):
            mc.load_gamma_flatten_rslc(str(rslc_dir), REF, str(tmp_path / 'hgt'), str(tmp_path / 'scratch'),
                                       str(tmp_path / 'rslc.zarr'), **kwargs)
        assert commands[-1].startswith(f'OMP_NUM_THREADS={threads} phase_sim_orb ')
        assert not any('OMP_NUM_THREADS' in c for c in commands if 'create_offset' in c)


# perpendicular baseline: synthetic GAMMA par files with straight orbits along z (no GAMMA needed)
_DATES = ['20200101', '20200113', '20200125']
_NL, _W = 20, 30


def _write_par(path, nlines, width, t0, pos, vel, dt=10.0):
    lines = [f'range_samples:   {width}', f'azimuth_lines:   {nlines}', f'number_of_state_vectors:   {len(pos)}',
             f'time_of_first_state_vector:   {float(t0)!r}   s', f'state_vector_interval:   {float(dt)!r}   s']
    for k, (p, v) in enumerate(zip(pos, vel), 1):
        lines.append(f'state_vector_position_{k}:  {float(p[0])!r}  {float(p[1])!r}  {float(p[2])!r}   m   m   m')
        lines.append(f'state_vector_velocity_{k}:  {float(v[0])!r}  {float(v[1])!r}  {float(v[2])!r}   m/s m/s m/s')
    path.write_text('\n'.join(lines) + '\n')


def _synthetic_gamma(tmp_path, n_sv=(15, 15, 15)):
    """rslc directory of 3 images with straight orbits along z, and lat / lon / hgt rasters chunked (8, 16)"""
    rslc_dir = tmp_path / 'rslc'
    rslc_dir.mkdir()
    a, h, v = 6378137.0, 700e3, 7000.0
    orbits = []
    offsets = [(0.0, 0.0, 18000.0), (40.0, 150.0, 18010.5), (-25.0, -80.0, 17990.0)]   # x, y (m), first time (s)
    for date, (dx, dy, t0), n in zip(_DATES, offsets, n_sv):
        t = np.arange(n)*10.0
        pos = np.stack([np.full(n, a + h + dx), np.full(n, dy), v*(t - 70.0)], -1)
        vel = np.tile([0.0, 0.0, v], (n, 1))
        (rslc_dir / f'{date}.rslc').write_bytes(b'')
        _write_par(rslc_dir / f'{date}.rslc.par', _NL, _W, t0, pos, vel)
        orbits.append((t0 + t, pos, vel))
    lat, lon = np.meshgrid(np.linspace(-0.5, 0.5, _NL), np.linspace(3.0, 5.0, _W), indexing='ij')
    lat[3, 4] = np.nan
    hgt = np.random.default_rng(0).uniform(0, 1500, (_NL, _W)).astype(np.float32)
    rasters = []
    for name, x in (('lat', lat), ('lon', lon), ('hgt', hgt)):
        z = zarr.open(str(tmp_path / f'{name}.zarr'), mode='w', shape=x.shape, dtype=x.dtype, chunks=(8, 16))
        z[:] = x
        rasters.append(str(tmp_path / f'{name}.zarr'))
    return rslc_dir, rasters, orbits


def test_load_gamma_bperp_synthetic(tmp_path):
    from moraine.api.utils_.orbit import bperp
    rslc_dir, rasters, orbits = _synthetic_gamma(tmp_path)
    mc.load_gamma_bperp(str(rslc_dir), '20200113', *rasters, str(tmp_path / 'bperp.zarr'))
    out = zarr.open(str(tmp_path / 'bperp.zarr'), mode='r')
    assert out.shape == (_NL, _W, 3) and out.chunks == (8, 16, 1) and out.dtype == np.float32
    lat, lon, hgt = (zarr.open(r, mode='r')[:] for r in rasters)
    t, pos, vel = (np.stack(a) for a in zip(*orbits))
    np.testing.assert_array_equal(out[:], bperp(lon, lat, hgt, t, pos, vel, ref=1))
    assert (out[:, :, 1][~np.isnan(lat)] == 0).all() and np.isnan(out[3, 4]).all()
    assert np.isfinite(out[:, :, 0][~np.isnan(lat)]).all()


def test_load_gamma_bperp_errors(tmp_path):
    rslc_dir, rasters, _ = _synthetic_gamma(tmp_path)
    with pytest.raises(ValueError, match='not a date'):
        mc.load_gamma_bperp(str(rslc_dir), '20200102', *rasters, str(tmp_path / 'b.zarr'))
    zarr.open(str(tmp_path / 'bad.zarr'), mode='w', shape=(3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='shape'):
        mc.load_gamma_bperp(str(rslc_dir), '20200113', rasters[0], rasters[1], str(tmp_path / 'bad.zarr'),
                            str(tmp_path / 'b.zarr'))
    other = tmp_path / 'other'
    other.mkdir()
    rslc_dir, rasters, _ = _synthetic_gamma(other, n_sv=(15, 14, 15))
    with pytest.raises(ValueError, match='different numbers of orbit state vectors'):
        mc.load_gamma_bperp(str(rslc_dir), '20200113', *rasters, str(other / 'b.zarr'))


def test_load_gamma_bperp(gamma, tmp_path, monkeypatch):
    """At the scene centre the perpendicular baselines are close to those of GAMMA's base_calc: same images, order,
    reference and sign. base_calc gives a scene centre model (3 cm from the exact baseline on the ellipsoid on this
    sample, whose state vectors are 60 s apart), and the centre pixel is 380 m above the ellipsoid, where the baseline
    differs by B_par h / (R sin(incidence)), another 6 cm."""
    monkeypatch.chdir(tmp_path)   # base_calc writes base.out / base_calc.log to the working directory
    g = gamma / 'geocoding'
    rasters = [str(tmp_path / f) for f in ('lat.zarr', 'lon.zarr', 'hgt.zarr')]
    mc.load_gamma_lat_lon_hgt(str(g / '20210802.diff_par'), str(gamma / 'rslc' / f'{REF}.rslc.par'),
                              str(gamma / 'DEM' / 'dem_seg_par'), str(g / '20210802.hgt'), str(tmp_path / 'scratch'),
                              *rasters)
    mc.load_gamma_bperp(str(gamma / 'rslc'), REF, *rasters, str(tmp_path / 'bperp.zarr'))
    mc.load_gamma_metadata(str(gamma / 'rslc'), str(gamma / 'DEM' / 'dem_seg_par'), REF, str(tmp_path / 'meta.toml'))
    out = zarr.open(str(tmp_path / 'bperp.zarr'), mode='r')
    assert out.shape == (NLINES, WIDTH, NIMAGES)
    centre = out[NLINES//2, WIDTH//2]
    base = np.array(toml.load(tmp_path / 'meta.toml')['perpendicular_baseline'])
    np.testing.assert_allclose(centre, base, rtol=0, atol=0.2)


@pytest.mark.slow
def test_load_gamma_bperp_phase_sim_orb(gamma, tmp_path):
    """The perpendicular baseline is the height sensitivity of GAMMA's simulated phase (phase_sim_orb at the heights
    -+ 500 m): dphi/dh = -4 pi / lambda bperp / (rho_j up.n), rho_j the range of the satellite of the image, up.n the
    change of height per length on the range circle of the reference. phase_sim_orb places a pixel at the range and
    time of the reference par at zero Doppler, as pt2geo does with a diff_par without offsets; there they agree to
    3e-6, the error of the finite difference over 1000 m. At the positions of the 01 step, shifted by the offsets of
    the geocoding (-3.15 range pixels, -2.04 lines), they differ by 9e-5; a wrong sign, image or reference gives
    errors of order 1."""
    from scipy.constants import speed_of_light
    from moraine.api.utils_ import orbit
    from moraine.api.utils_.gamma_io import read_gamma_image, write_gamma_image
    from moraine.cli.load import _fetch_slc_par_date, _read_gamma_orbit
    g, rslc = gamma / 'geocoding', gamma / 'rslc'
    ref_par = rslc / f'{REF}.rslc.par'
    diff_par = tmp_path / 'diff_par'                  # the geocoding diff_par without its offsets
    with open(g / '20210802.diff_par') as f, open(diff_par, 'w') as out:
        for line in f:
            key, _, values = line.partition(':')
            if key in ('range_offset_polynomial', 'azimuth_offset_polynomial'):
                line = f'{key}:' + '  0.0'*len(values.split()) + '\n'
            out.write(line)
    rasters = [str(tmp_path / f) for f in ('lat.zarr', 'lon.zarr', 'hgt.zarr')]
    mc.load_gamma_lat_lon_hgt(str(diff_par), str(ref_par), str(gamma / 'DEM' / 'dem_seg_par'),
                              str(g / '20210802.hgt'), str(tmp_path / 'scratch'), *rasters)
    mc.load_gamma_bperp(str(rslc), REF, *rasters, str(tmp_path / 'bperp.zarr'))
    dates, _, pars = _fetch_slc_par_date(rslc)
    kr = dates.index(REF)
    b = zarr.open(str(tmp_path / 'bperp.zarr'), mode='r')
    j = int(np.nanargmax(np.abs(b[NLINES//2, WIDTH//2])))           # the image with the longest baseline
    par = rslc / f'{dates[j]}.rslc.par'
    hgt = read_gamma_image(str(g / '20210802.hgt'), WIDTH, dtype='float')
    assert os.system(f'create_offset {ref_par} {par} {tmp_path}/off 1 1 1 0 > {tmp_path}/gamma.log 2>&1') == 0
    sims = []
    for dh in (-500, 500):
        write_gamma_image((hgt + dh).astype(np.float32), str(tmp_path / 'hgt_dh'))
        assert os.system(f'phase_sim_orb {ref_par} {par} {tmp_path}/off {tmp_path}/hgt_dh {tmp_path}/sim {ref_par} '
                         f'- - 1 1 >> {tmp_path}/gamma.log 2>&1') == 0
        sims.append(read_gamma_image(str(tmp_path / 'sim'), WIDTH, dtype='float').astype(np.float64))
    lam = speed_of_light/float(next(l.split()[1] for l in open(ref_par) if l.startswith('radar_frequency')))
    lat, lon, h = (zarr.open(x, mode='r')[:] for x in rasters)
    bj = b[:, :, j]
    iy, ix = np.nonzero(np.isfinite(bj) & np.isfinite(sims[0]) & np.isfinite(sims[1]))
    pick = np.random.default_rng(0).choice(len(iy), 300, replace=False)
    t, p, v = (np.stack(a) for a in zip(*(_read_gamma_orbit(x) for x in pars)))
    n_out = int(np.ceil((t[:, -1] - t[:, 0]).max()/orbit._STEP)) + 1
    t, p, v = (np.stack(a) for a in zip(*(orbit._resample(*o, n_out) for o in zip(t, p, v))))
    rel = []
    for y, x in zip(iy[pick], ix[pick]):
        P = np.array(orbit._geodetic_to_ecef(lon[y, x], lat[y, x], np.float64(h[y, x])))
        up, P = P[3:], P[:3]
        zr = orbit._zero_doppler(t[kr], p[kr], v[kr], *P)
        u = P - np.array(zr[1:4])
        n = np.cross(u, np.array(zr[4:7]))
        n *= np.sign(n @ up)/np.linalg.norm(n)
        rho_j = np.linalg.norm(P - np.array(orbit._zero_doppler(t[j], p[j], v[j], *P)[1:4]))
        k = -4*np.pi/lam*bj[y, x]/(rho_j*(up @ n))
        rel.append(((sims[1][y, x] - sims[0][y, x])/1000 - k)/abs(k))
    assert np.abs(rel).max() < 5e-6, (np.median(rel), np.abs(rel).max())
