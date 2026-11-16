import datetime
import re
import sys
import types

import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.api import tropo as tr
from moraine.api.utils_ import ArgumentError
from moraine.cli import tropo as ctr
from moraine.command import commands, main
from conftest import write_zarr, json_out

LEVELS = np.array(ctr._LEVELS, dtype=np.float64) * 100        # Pa
G0 = 9.80665


def _geopotential(h, lat):
    """geopotential (m^2/s^2) of the height h above the geoid (m) at latitude lat (degrees)"""
    phi = np.deg2rad(lat)
    g = 9.80616 * (1 - 0.002637 * np.cos(2 * phi) + 0.0000059 * np.cos(2 * phi) ** 2)
    a, b = 6378137.0, 6356752.3142
    re = np.sqrt(1 / (np.cos(phi) ** 2 / a ** 2 + np.sin(phi) ** 2 / b ** 2))
    return G0 * h * (g / G0) * re / (re + h)


def _exp_atmosphere(grid_lat, grid_lon, P0=101325., Hp=8000., T0=270., e0=1500., He=2000.):
    """z, t, q (n_levels, n_lat, n_lon) float32 of an isothermal atmosphere whose pressure and water vapour pressure
    fall exponentially with the height above the geoid; P0 may vary over the grid, an array (n_lat, n_lon)"""
    P0 = np.broadcast_to(np.asarray(P0, dtype=np.float64), (grid_lat.size, grid_lon.size))
    h = -Hp * np.log(LEVELS[:, None, None] / P0[None])                    # heights of the levels
    z = _geopotential(h, grid_lat[None, :, None])
    t = np.full(h.shape, T0)
    e = e0 * np.exp(-h / He)
    q = tr._EPS * e / (LEVELS[:, None, None] - (1 - tr._EPS) * e)
    return z.astype(np.float32), t.astype(np.float32), q.astype(np.float32)


def _exp_delay(h, lat, P0=101325., Hp=8000., T0=270., e0=1500., He=2000.):
    """the zenith delay (m) of `_exp_atmosphere` at the height h, up to the highest level"""
    h_top = -Hp * np.log(LEVELS.min() / P0)
    gm = 9.784 * (1 - 0.00266 * np.cos(2 * np.deg2rad(lat)) - 0.00028e-3 * h)
    zhd = 1e-6 * tr._K1 * tr._RD * P0 * np.exp(-h / Hp) / gm
    zwd = 1e-6 * (tr._K2P / T0 + tr._K3 / T0 ** 2) * e0 * He * (np.exp(-h / He) - np.exp(-h_top / He))
    return zhd + zwd


GRID_LAT = np.array([41.25, 41.0, 40.75, 40.5])           # decreasing, as in the files of ERA5
GRID_LON = np.array([13.75, 14.0, 14.25, 14.5])


def _points(n=2000, seed=0, hmin=-50.0, hmax=3000.0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(13.76, 14.49, n), rng.uniform(40.51, 41.24, n), rng.uniform(hmin, hmax, n))


# ---------------------------------------------------------------- era5_zenith_delay

def test_zenith_delay_exponential_atmosphere():
    lon, lat, h = _points()                      # also below the lowest level (1000 hPa is at 105 m)
    z, t, q = _exp_atmosphere(GRID_LAT, GRID_LON)
    ztd = mr.era5_zenith_delay(lon, lat, h, LEVELS, GRID_LON, GRID_LAT, z, t, q)
    assert ztd.dtype == np.float64 and ztd.shape == lon.shape
    np.testing.assert_allclose(ztd, _exp_delay(h, lat), rtol=0, atol=1e-6)


def test_zenith_delay_bilinear_between_nodes():
    # surface pressure linear in longitude and latitude: the delay at a point is that of its own surface pressure
    P0 = 100000 + 800 * (GRID_LAT[:, None] - 40.5) + 1200 * (GRID_LON[None, :] - 13.75)
    z, t, q = _exp_atmosphere(GRID_LAT, GRID_LON, P0=P0)
    lon, lat, h = _points(seed=1)
    ztd = mr.era5_zenith_delay(lon, lat, h, LEVELS, GRID_LON, GRID_LAT, z, t, q)
    p0 = 100000 + 800 * (lat - 40.5) + 1200 * (lon - 13.75)
    np.testing.assert_allclose(ztd, _exp_delay(h, lat, P0=p0), rtol=0, atol=2e-6)


def _standard_atmosphere(lat, dT=0.0, rh0=0.7):
    """z, t, q (n_levels, 2, 2) of the standard atmosphere shifted by dT with a relative humidity falling from rh0 at
    the ground to 0.1 at 10 km, dry above 12 km; and the true profile: geometric heights, pressure, temperature,
    water vapour pressure on a fine grid"""
    H = np.arange(-1000, 50000, 0.5)              # geopotential height
    layers = [(0, 288.15, -6.5e-3), (11000, 216.65, 0.0), (20000, 216.65, 1.0e-3), (32000, 228.65, 2.8e-3),
              (47000, 270.65, 0.0)]
    T, P, p_base = np.empty_like(H), np.empty_like(H), 101325.0
    for i, (hb, tb, lapse) in enumerate(layers):
        top = layers[i + 1][0] if i + 1 < len(layers) else np.inf
        sel = (H < top) if i == 0 else (H >= hb) & (H < top)
        tb = tb + dT
        T[sel] = tb + lapse * (H[sel] - hb)
        k = G0 / (tr._RD * lapse) if lapse else 0
        P[sel] = p_base * (np.exp(-G0 * (H[sel] - hb) / (tr._RD * tb)) if lapse == 0 else (T[sel] / tb) ** -k)
        if np.isfinite(top):
            p_base *= np.exp(-G0 * (top - hb) / (tr._RD * tb)) if lapse == 0 else ((tb + lapse * (top - hb)) / tb) ** -k
    esat = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    e = np.where(H < 12000, np.clip(rh0 - (rh0 - 0.1) * H / 10000, 0.1, None) * esat, 0.0)
    h = tr._geometric_height(G0 * H, lat)
    H_lev = np.interp(np.log(LEVELS), np.log(P[::-1]), H[::-1])
    t_lev, e_lev = np.interp(H_lev, H, T), np.interp(H_lev, H, e)
    q_lev = tr._EPS * e_lev / (LEVELS - (1 - tr._EPS) * e_lev)
    fields = [np.broadcast_to(a[:, None, None], (LEVELS.size, 2, 2)).astype(np.float32)
              for a in (G0 * H_lev, t_lev, q_lev)]
    return fields, (h, P, T, e, tr._geometric_height(G0 * H_lev.max(), lat))


@pytest.mark.parametrize('dT, rh0', [(0, 0.7), (8, 0.9), (-10, 0.4)])
def test_zenith_delay_standard_atmosphere_within_half_a_millimeter(dT, rh0):
    lat0 = 40.8
    (z, t, q), (hf, P, T, e, h_top) = _standard_atmosphere(lat0, dT, rh0)
    h = np.arange(0, 3001, 50.0)
    ztd = mr.era5_zenith_delay(np.full(h.size, 14.1), np.full(h.size, lat0), h, LEVELS, np.array([14.0, 14.25]),
                               np.array([40.75, 41.0]), z, t, q)
    # the truth from the fine profile: hydrostatic from the pressure at h, wet integral up to the highest level
    nw = tr._K2P * e / T + tr._K3 * e / T ** 2
    cum = np.concatenate([[0], np.cumsum(0.5 * (nw[1:] + nw[:-1]) * np.diff(hf))])
    zwd = 1e-6 * (np.interp(h_top, hf, cum) - np.interp(h, hf, cum))
    gm = 9.784 * (1 - 0.00266 * np.cos(2 * np.deg2rad(lat0)) - 0.00028e-3 * h)
    zhd = 1e-6 * tr._K1 * tr._RD * np.exp(np.interp(h, hf, np.log(P))) / gm
    np.testing.assert_allclose(ztd, zhd + zwd, rtol=0, atol=5e-4)


def test_zenith_delay_grid_orders_and_longitude_conventions():
    z, t, q = _exp_atmosphere(GRID_LAT, GRID_LON, P0=100000 + 500 * np.arange(16).reshape(4, 4))
    lon, lat, h = _points(n=500, seed=2)
    ref = mr.era5_zenith_delay(lon, lat, h, LEVELS, GRID_LON, GRID_LAT, z, t, q)
    flipped = mr.era5_zenith_delay(lon, lat, h, LEVELS[::-1], GRID_LON[::-1], GRID_LAT[::-1],
                                   z[::-1, ::-1, ::-1], t[::-1, ::-1, ::-1], q[::-1, ::-1, ::-1])
    np.testing.assert_allclose(flipped, ref, rtol=0, atol=1e-12)
    shifted = mr.era5_zenith_delay(lon - 360, lat, h, LEVELS, GRID_LON, GRID_LAT, z, t, q)
    np.testing.assert_allclose(shifted, ref, rtol=0, atol=1e-12)


def test_zenith_delay_nan_and_outside_the_grid():
    z, t, q = _exp_atmosphere(GRID_LAT, GRID_LON)
    lon, lat, h = (np.array([14.0, 14.1, 14.2]), np.array([40.8, np.nan, 40.9]), np.array([10.0, 20.0, np.nan]))
    ztd = mr.era5_zenith_delay(lon, lat, h, LEVELS, GRID_LON, GRID_LAT, z, t, q)
    assert np.isfinite(ztd[0]) and np.isnan(ztd[1:]).all()
    with pytest.raises(ValueError, match='outside the grid'):
        mr.era5_zenith_delay(np.array([14.6]), np.array([40.8]), np.array([0.0]), LEVELS, GRID_LON, GRID_LAT, z, t, q)
    with pytest.raises(ValueError, match='equally spaced'):
        mr.era5_zenith_delay(lon, lat, h, LEVELS, np.array([13.75, 14.0, 14.5, 14.75]), GRID_LAT, z, t, q)


# ---------------------------------------------------------------- pc_tropo_era5

def _hours_fields(hours, p0s):
    fields = [_exp_atmosphere(GRID_LAT, GRID_LON, P0=p0) for p0 in p0s]
    return [np.stack([f[i] for f in fields]) for i in range(3)]


def test_pc_tropo_time_interpolation_sign_and_reference():
    lon, lat, h = _points(n=300, seed=3, hmin=0, hmax=500)
    theta = np.full(lon.shape, 0.9)
    times = ['2019-01-02T05:15:00Z', '2019-01-14T17:00:00+00:00', datetime.datetime(2019, 1, 26, 5, 30)]
    hours = [datetime.datetime(2019, 1, 2, 5), datetime.datetime(2019, 1, 2, 6), datetime.datetime(2019, 1, 14, 17),
             datetime.datetime(2019, 1, 26, 5), datetime.datetime(2019, 1, 26, 6)]
    p0s = [101000., 101400., 100200., 101800., 101600.]
    z, t, q = _hours_fields(hours, p0s)
    out = mr.pc_tropo_era5(lon, lat, h, theta, times, 0.0, hours, LEVELS, GRID_LON, GRID_LAT, z, t, q, ref=1)
    assert out.dtype == np.float32 and out.shape == (300, 3)
    np.testing.assert_array_equal(out[:, 1], 0)
    d = [0.75 * _exp_delay(h, lat, p0s[0]) + 0.25 * _exp_delay(h, lat, p0s[1]), _exp_delay(h, lat, p0s[2]),
         0.5 * _exp_delay(h, lat, p0s[3]) + 0.5 * _exp_delay(h, lat, p0s[4])]
    expect = np.stack([-1000 * (dk - d[1]) / np.sin(theta) for dk in d], axis=1)
    np.testing.assert_allclose(out, expect, rtol=0, atol=2e-3)
    # a larger pressure (delay) than in the reference image is an apparent motion away from the satellite
    assert (out[:, 0] < 0).all() and (out[:, 2] < 0).all()
    assert mr.pc_tropo_era5(lon, lat, h, theta, times[1:2], 0.0, hours[2:3], LEVELS, GRID_LON, GRID_LAT,
                            z[2:3], t[2:3], q[2:3]).shape == (300, 1)


def test_pc_tropo_geoid_nan_and_errors():
    lon, lat, h = _points(n=100, seed=4, hmin=0, hmax=500)
    theta = np.full(lon.shape, 0.9)
    times, hours = ['2019-01-02T05:00:00Z', '2019-01-14T05:00:00Z'], ['2019-01-02T05:00Z', '2019-01-14T05:00Z']
    z, t, q = _hours_fields(hours, [101000., 100500.])
    args = (LEVELS, GRID_LON, GRID_LAT, z, t, q)
    out = mr.pc_tropo_era5(lon, lat, h, theta, times, 0.0, hours, *args)
    np.testing.assert_allclose(mr.pc_tropo_era5(lon, lat, h + 46.0, theta, times, 46.0, hours, *args), out,
                               rtol=0, atol=1e-4)
    theta_nan = theta.copy(); theta_nan[3] = np.nan
    out_nan = mr.pc_tropo_era5(lon, lat, h, theta_nan, times, 0.0, hours, *args)
    assert np.isnan(out_nan[3]).all() and np.isfinite(np.delete(out_nan, 3, axis=0)).all()
    with pytest.raises(ValueError, match='no ERA5 fields of 1 hours'):
        mr.pc_tropo_era5(lon, lat, h, theta, ['2019-01-02T05:00:00Z', '2019-01-14T05:20:00Z'], 0.0, hours, *args)
    with pytest.raises(ValueError, match='elevation angle'):
        mr.pc_tropo_era5(lon, lat, h, np.full(lon.shape, 50.0), times, 0.0, hours, *args)
    with pytest.raises(ValueError, match='geoid'):
        mr.pc_tropo_era5(lon, lat, h, theta, times, 4600.0, hours, *args)
    with pytest.raises(ValueError, match='ISO 8601'):
        mr.pc_tropo_era5(lon, lat, h, theta, ['20190102', '20190114'], 0.0, hours, *args)


def test_argument_errors_of_the_array_functions():
    """The checks of the arguments of era5_zenith_delay and pc_tropo_era5, and of the shapes and dtypes of their arrays,
    raise ArgumentError (decision 0052); they raised plain ValueErrors. Problems in the values of the data stay plain
    ValueErrors: points outside the grid, a grid that is not equally spaced, pressures that are not positive or not
    different, fields that are not finite, temperatures that are not positive, theta outside (0, pi/2], hours of the
    images without fields."""
    z, t, q = _exp_atmosphere(GRID_LAT, GRID_LON)
    lon, lat, h = _points(n=10, seed=7, hmin=0, hmax=500)
    zenith = dict(lon=lon, lat=lat, hgt=h, pressure=LEVELS, grid_lon=GRID_LON, grid_lat=GRID_LAT, geopotential=z,
                  temperature=t, specific_humidity=q)
    for kw, error in [
            (dict(lat=lat[:5]), 'lon, lat and hgt must have one shape (n_points,), got lat of shape (5,)'),
            (dict(hgt=h.reshape(2, 5)), 'lon, lat and hgt must have one shape (n_points,), got hgt of shape (2, 5)'),
            (dict(hgt=np.arange(10, dtype=np.int32)), 'hgt must be floating point, got int32'),
            (dict(pressure=LEVELS[:1]), 'pressure must have shape (n_levels,) with n_levels >= 2, got shape (1,)'),
            (dict(grid_lon=GRID_LON[:1]), 'grid_lon must have shape (n,) with n >= 2, got shape (1,)'),
            (dict(grid_lat=GRID_LAT[:, None]), 'grid_lat must have shape (n,) with n >= 2, got shape (4, 1)'),
            (dict(temperature=t[:, :3]), 'temperature must have shape (n_levels, n_lat, n_lon) = (37, 4, 4), got '
                                         '(37, 3, 4)')]:
        with pytest.raises(ArgumentError, match=f'^{re.escape(error)}$'):
            mr.era5_zenith_delay(**{**zenith, **kw})
    for kw, error in [
            (dict(lon=lon + 1.0), '10 points are outside the grid'),
            (dict(grid_lon=np.array([13.75, 14.0, 14.5, 14.75])), 'grid_lon must be equally spaced'),
            (dict(pressure=-LEVELS), 'pressure must be (n_levels,) different positive pressures in pascals'),
            (dict(pressure=np.full(LEVELS.size, 50000.0)), 'pressure must be (n_levels,) different positive'),
            (dict(temperature=np.full_like(t, np.nan)), 'temperature has values that are not finite'),
            (dict(temperature=-t), 'temperature must be in kelvin, positive')]:
        with pytest.raises(ValueError, match=re.escape(error)) as e:
            mr.era5_zenith_delay(**{**zenith, **kw})
        assert type(e.value) is ValueError, error

    times, hours = ['2019-01-02T05:00:00Z', '2019-01-14T05:00:00Z'], ['2019-01-02T05:00Z', '2019-01-14T05:00Z']
    z, t, q = _hours_fields(hours, [101000., 100500.])
    tropo = dict(lon=lon, lat=lat, hgt=h, theta=np.full(10, 0.9), times=times, geoid=0.0, era5_times=hours,
                 pressure=LEVELS, grid_lon=GRID_LON, grid_lat=GRID_LAT, geopotential=z, temperature=t,
                 specific_humidity=q)
    for kw, error in [
            (dict(times=[]), 'times must have the time of at least one image'),
            (dict(times=['20190102', '20190114']), 'times must be date-times in UTC (ISO 8601, e.g. '
                                                   "2019-01-02T05:06:07.123456Z), got '20190102'"),
            (dict(ref=2), 'ref must be the index of one of the 2 images, got 2'),
            (dict(ref=True), 'ref must be the index of one of the 2 images, got True'),
            (dict(geoid=4600.0), 'geoid must be the height of the geoid above the WGS84 ellipsoid in meters, between '
                                 '-150 and 150, got 4600.0'),
            (dict(theta=np.full(5, 0.9)), 'theta must have the shape (n_points,) of lon, got shape (5,)'),
            (dict(theta=np.ones(10, np.int32)), 'theta must be floating point, got int32'),
            (dict(era5_times=[hours[0], hours[0]]), 'era5_times must be different hours'),
            (dict(era5_times=['2019-01-02T05:30Z', hours[1]]), 'era5_times must be whole hours'),
            (dict(geopotential=z[:1]), 'geopotential must have shape (n_hours, n_levels, n_lat, n_lon) = '
                                       '(2, 37, 4, 4), got (1, 37, 4, 4)'),
            (dict(lat=lat[:5]), 'lon, lat and hgt must have one shape (n_points,), got lat of shape (5,)')]:
        with pytest.raises(ArgumentError, match=f'^{re.escape(error)}$'):
            mr.pc_tropo_era5(**{**tropo, **kw})
    for kw, error in [
            (dict(theta=np.full(10, 50.0)), 'theta must be the elevation angle of the look vector in radians'),
            (dict(times=[times[0], '2019-01-14T05:20:00Z']), 'no ERA5 fields of 1 hours that the images need')]:
        with pytest.raises(ValueError, match=re.escape(error)) as e:
            mr.pc_tropo_era5(**{**tropo, **kw})
        assert type(e.value) is ValueError, error


# ---------------------------------------------------------------- CLI

def _write_era5_file(path, hour, lat, lon, z, t, q):
    """an ERA5 file of one hour as the Climate Data Store writes it (netCDF4: valid_time, pressure_level from 1000 hPa,
    latitude decreasing, longitude, z / t / q with dimensions (time, level, latitude, longitude))"""
    import h5py
    order = np.argsort(-LEVELS)
    with h5py.File(path, 'w') as f:
        seconds = (hour - datetime.datetime(1970, 1, 1)).total_seconds()
        scales = [f.create_dataset('valid_time', data=np.array([seconds], dtype=np.int64)),
                  f.create_dataset('pressure_level', data=LEVELS[order] / 100),
                  f.create_dataset('latitude', data=lat), f.create_dataset('longitude', data=lon)]
        scales[0].attrs['units'] = 'seconds since 1970-01-01'
        scales[1].attrs['units'] = 'hPa'
        for s in scales:
            s.make_scale(s.name.strip('/'))
        for name, a in (('z', z), ('t', t), ('q', q)):
            d = f.create_dataset(name, data=a[order][None].astype(np.float32))
            for i, s in enumerate(scales):
                d.dims[i].attach_scale(s)


def test_cli_pc_tropo_era5(tmp_path):
    pytest.importorskip('h5py')
    lon, lat, h = _points(n=1000, seed=5, hmin=0, hmax=500)
    theta = np.full(lon.shape, 0.9)
    paths = {name: write_zarr(tmp_path / f'{name}.zarr', a.astype(np.float32), chunks=(256,))
             for name, a in (('lon', lon), ('lat', lat), ('hgt', h + 46.0), ('theta', theta))}
    times = ['2019-01-02T05:15:00Z', '2019-01-14T05:00:00Z']
    hours = mr.api.tropo._era5_hours(times)
    assert hours == [datetime.datetime(2019, 1, 2, 5), datetime.datetime(2019, 1, 2, 6),
                     datetime.datetime(2019, 1, 14, 5)]
    p0s = [101000., 101400., 100200.]
    era5 = tmp_path / 'era5'
    era5.mkdir()
    for hour, p0 in zip(hours, p0s):
        _write_era5_file(era5 / f'era5_{hour:%Y%m%d_%H}.nc', hour, GRID_LAT, GRID_LON,
                         *_exp_atmosphere(GRID_LAT, GRID_LON, P0=p0))
    out = str(tmp_path / 'tropo.zarr')
    mc.pc_tropo_era5(str(era5), paths['lon'], paths['lat'], paths['hgt'], paths['theta'], out, times, 46.0,
                     out_chunks=512)
    z_out = zarr.open(out, mode='r')
    assert z_out.dtype == np.float32 and z_out.shape == (1000, 2) and z_out.chunks == (512, 1)
    z, t, q = _hours_fields(hours, p0s)
    expect = mr.pc_tropo_era5(lon.astype(np.float32), lat.astype(np.float32), (h + 46.0).astype(np.float32),
                              theta.astype(np.float32), times, 46.0, hours, LEVELS, GRID_LON, GRID_LAT, z, t, q)
    np.testing.assert_allclose(z_out[:], expect, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(z_out[:, 0], 0)
    (era5 / f'era5_{hours[1]:%Y%m%d_%H}.nc').unlink()
    with pytest.raises(ValueError, match='download-era5'):
        mc.pc_tropo_era5(str(era5), paths['lon'], paths['lat'], paths['hgt'], paths['theta'], out, times, 46.0)


def test_cli_argument_errors_are_usage_errors(tmp_path, capsys, monkeypatch):
    """download-era5 and pc-tropo-era5 check their arguments, and the shapes and dtypes of their inputs, before they
    download, read or write anything: a wrong one is an ArgumentError, on the command line a usage error (exit code 2,
    the JSON error; decision 0052). They were processing errors (exit code 1, after `ValueError: `), a `geoid` out
    of range was found only by the tasks, after the output was created, and `times` without a time failed with an
    IndexError. ERA5 files that are missing stay a processing error. Nothing is downloaded: the downloads of both
    sources fail in this test."""
    pytest.importorskip('h5py')

    def no_download(*args, **kwargs):
        raise AssertionError('no download in this test')
    monkeypatch.setattr(ctr, '_ncar_file', no_download)
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=no_download))
    monkeypatch.setenv('CDSAPI_KEY', 'test')
    era5 = tmp_path / 'era5'
    times, bbox = ['2019-01-02T05:06:07Z', '2019-01-14T05:00:00Z'], ['14.0', '40.7', '14.2', '40.9']
    for args, error in [
            (['--times', *times, '--bbox', *bbox[:3]], "bbox must be [west, south, east, north] in degrees, got "
                                                       "['14.0', '40.7', '14.2']"),
            (['--times', *times, '--bbox', '14.2', '40.7', '14.0', '40.9'],
             'bbox must be [west, south, east, north] in degrees with west < east and south < north, got '
             '[14.2, 40.7, 14.0, 40.9]'),
            (['--times', *times, '--bbox', *bbox, '--source', 'aws'], "source must be 'ncar' or 'cds', not 'aws'"),
            (['--times', '20190102', '--bbox', *bbox, '--source', 'cds'],
             "times must be date-times in UTC (ISO 8601, e.g. 2019-01-02T05:06:07.123456Z), got '20190102'")]:
        assert main(['download-era5', '--out_dir', str(era5), *args, '--json', '-q']) == 2, args
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not era5.exists()
    with pytest.raises(ArgumentError, match='west < east'):
        mc.download_era5(str(era5), times, [14.2, 40.7, 14.0, 40.9])

    lon, lat, h = _points(n=10, seed=8, hmin=0, hmax=500)
    paths = {name: write_zarr(tmp_path / f'{name}.zarr', a)
             for name, a in (('lon', lon.astype(np.float32)), ('lat', lat.astype(np.float32)),
                             ('hgt', h.astype(np.float32)), ('theta', np.full(10, 0.9, np.float32)),
                             ('lat_int', lat.astype(np.int32)), ('hgt5', h[:5].astype(np.float32)))}
    out = tmp_path / 'tropo.zarr'
    inputs = dict(lon=paths['lon'], lat=paths['lat'], hgt=paths['hgt'], theta=paths['theta'])

    def tropo(**kw):
        kw = {'era5_dir': str(era5), **inputs, 'out': str(out), 'times': times, 'geoid': '46', **kw}
        argv = [a for k, v in kw.items() for a in (f'--{k}', *(v if isinstance(v, list) else [v]))]
        return main(['pc-tropo-era5', *argv, '--json', '-q'])
    for kw, error in [
            (dict(lat=paths['lat_int']), 'lat must be floating point, got int32'),
            (dict(hgt=paths['hgt5']), 'hgt must have shape (10,), got shape (5,)'),
            (dict(times=['20190102', '20190114']), "times must be date-times in UTC (ISO 8601, e.g. "
                                                   "2019-01-02T05:06:07.123456Z), got '20190102'"),
            (dict(times=''), 'times must have the time of at least one image'),
            (dict(ref='2'), 'ref must be the index of one of the 2 images, got 2'),
            (dict(geoid='4600'), 'geoid must be the height of the geoid above the WGS84 ellipsoid in meters, between '
                                 '-150 and 150, got 4600.0')]:
        assert tropo(**kw) == 2, kw
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not out.exists() and not era5.exists()
    with pytest.raises(ArgumentError, match='geoid must be'):
        mc.pc_tropo_era5(str(era5), *inputs.values(), str(out), times, 4600.0)
    assert tropo() == 1                                       # no ERA5 files: a processing error
    assert json_out(capsys)['error'].startswith(f'ValueError: {era5}: no ERA5 file of 3 of the 3 hours')
    assert not out.exists()


def test_download_era5_without_times_is_a_usage_error(tmp_path, capsys, monkeypatch):
    """download-era5 without a time (`--times ''`, times = [] in Python) stops with an ArgumentError, a usage error
    (exit code 2) on the command line, before it makes the output directory, as pc-tropo-era5 does; it made an empty
    directory and succeeded. Nothing is downloaded: the downloads fail in this test."""
    pytest.importorskip('h5py')

    def no_download(*args, **kwargs):
        raise AssertionError('no download in this test')
    monkeypatch.setattr(ctr, '_ncar_file', no_download)
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=no_download))
    era5, bbox = tmp_path / 'era5', ['14.0', '40.7', '14.2', '40.9']
    for times in ([''], [' ']):
        assert main(['download-era5', '--out_dir', str(era5), '--times', *times, '--bbox', *bbox, '--json', '-q']) == 2
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': 'times must have the time of at least one image'}
        assert not era5.exists()
    for times in ([], ''):
        with pytest.raises(ArgumentError, match='^times must have the time of at least one image$'):
            mc.download_era5(str(era5), times, bbox, source='cds')
        assert not era5.exists()


def test_argument_errors_before_the_h5py_check(tmp_path, capsys, monkeypatch):
    """Without h5py, download-era5 and pc-tropo-era5 report a wrong argument as such, a usage error (exit code 2),
    and only right arguments as the missing h5py (an ImportError, exit code 1); the wrong argument was reported as the
    missing h5py. Nothing is downloaded: the downloads fail in this test, and h5py cannot be imported."""
    def no_download(*args, **kwargs):
        raise AssertionError('no download in this test')
    monkeypatch.setattr(ctr, '_ncar_file', no_download)
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=no_download))
    monkeypatch.setitem(sys.modules, 'h5py', None)                    # import h5py raises an ImportError
    era5, times, bbox = tmp_path / 'era5', ['2019-01-02T05:06:07Z'], ['14.0', '40.7', '14.2', '40.9']
    for args, code, error in [
            (['--bbox', '14.2', '40.7', '14.0', '40.9'], 2, 'bbox must be [west, south, east, north] in degrees with '
                                                            'west < east and south < north, got [14.2, 40.7, 14.0, 40.9]'),
            (['--bbox', *bbox, '--times', ''], 2, 'times must have the time of at least one image'),
            (['--bbox', *bbox], 1, "ImportError: download-era5 needs cdsapi and h5py: pip install 'moraine[era5]'")]:
        argv = ['download-era5', '--out_dir', str(era5), '--times', *times, *args, '--json', '-q']
        assert main(argv) == code, args
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not era5.exists()
    lon, lat, h = _points(n=10, seed=8, hmin=0, hmax=500)
    inputs = [write_zarr(tmp_path / f'{name}.zarr', a.astype(np.float32))
              for name, a in (('lon', lon), ('lat', lat), ('hgt', h), ('theta', np.full(10, 0.9)))]
    argv = ['pc-tropo-era5', '--era5_dir', str(era5), *[a for name, path in zip(('lon', 'lat', 'hgt', 'theta'), inputs)
                                                         for a in (f'--{name}', path)],
            '--out', str(tmp_path / 'tropo.zarr'), '--times', *times, '--json', '-q']
    for geoid, code, error in [
            ('4600', 2, 'geoid must be the height of the geoid above the WGS84 ellipsoid in meters, between -150 and '
                        '150, got 4600.0'),
            ('46', 1, "ImportError: pc-tropo-era5 needs h5py: pip install 'moraine[era5]'")]:
        assert main([*argv, '--geoid', geoid]) == code, geoid
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not (tmp_path / 'tropo.zarr').exists()


class _FakeClient:
    """cdsapi.Client of the tests: writes the file of the requested hour and area"""
    requests = []
    wrong_hour = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def retrieve(self, name, request, target):
        type(self).requests.append((name, request))
        north, west, south, east = request['area']
        lat = np.arange(north, south - 1e-9, -0.25)
        lon = np.arange(west, east + 1e-9, 0.25)
        hour = datetime.datetime.strptime(f'{request["year"][0]}{request["month"][0]}{request["day"][0]}'
                                          f'{request["time"][0]}', '%Y%m%d%H:%M')
        if type(self).wrong_hour:
            hour += datetime.timedelta(hours=1)
        _write_era5_file(target, hour, lat, lon, *_exp_atmosphere(lat, lon))


def test_cli_download_era5(tmp_path, monkeypatch):
    pytest.importorskip('h5py')
    monkeypatch.setitem(sys.modules, 'cdsapi', types.SimpleNamespace(Client=_FakeClient))
    monkeypatch.setenv('CDSAPI_KEY', 'test')
    _FakeClient.requests = []
    out = tmp_path / 'era5'
    times = ['2019-01-02T05:06:07.123456Z', '2019-01-14T05:00:00Z']
    bbox = [14.0055556, 40.753376, 14.2455380, 40.9077778]
    mc.download_era5(str(out), times, bbox, source='cds', n_workers=2)
    assert sorted(p.name for p in out.iterdir()) == ['era5_20190102_05.nc', 'era5_20190102_06.nc',
                                                    'era5_20190114_05.nc']
    assert len(_FakeClient.requests) == 3
    name, request = _FakeClient.requests[0]
    assert name == 'reanalysis-era5-pressure-levels'
    assert request['area'] == [41.25, 13.75, 40.5, 14.5]
    assert sorted(request['variable']) == ['geopotential', 'specific_humidity', 'temperature']
    assert len(request['pressure_level']) == 37 and request['data_format'] == 'netcdf'
    assert sorted(r['time'][0] for _, r in _FakeClient.requests) == ['05:00', '05:00', '06:00']
    era5 = ctr._read_era5(out / 'era5_20190102_06.nc')
    assert era5['time'] == datetime.datetime(2019, 1, 2, 6) and era5['z'].shape == (37, 4, 4)
    np.testing.assert_allclose(np.sort(era5['pressure']), LEVELS)

    _FakeClient.requests = []
    mc.download_era5(str(out), ','.join(times), '14.0055556, 40.753376, 14.245538, 40.9077778', source='cds')
    assert _FakeClient.requests == []                                    # the files cover the area: kept
    mc.download_era5(str(out), times, [14.0, 40.7, 14.6, 40.9], source='cds')
    assert len(_FakeClient.requests) == 3                                 # a larger area: downloaded again

    _FakeClient.wrong_hour = True
    try:
        with pytest.raises(RuntimeError, match='does not hold'):
            mc.download_era5(str(tmp_path / 'other'), times[1:], bbox, source='cds')
    finally:
        _FakeClient.wrong_hour = False
    assert list((tmp_path / 'other').iterdir()) == []                   # no file, no partial file
    monkeypatch.delenv('CDSAPI_KEY')
    monkeypatch.setenv('CDSAPI_RC', str(tmp_path / 'missing_cdsapirc'))
    _FakeClient.requests = []
    mc.download_era5(str(out), times, [14.0, 40.7, 14.6, 40.9], source='cds')          # nothing to download: no account needed
    assert _FakeClient.requests == []
    with pytest.raises(RuntimeError, match='API key'):
        mc.download_era5(str(tmp_path / 'third'), times, bbox, source='cds')


def test_commands_of_the_troposphere():
    cmds = commands()
    download, tropo = cmds['download-era5'], cmds['pc-tropo-era5']
    assert download.module == tropo.module == 'tropo'
    kinds = {p.name: (p.kind, p.required) for p in download.params}
    assert kinds['times'] == ('list', True) and kinds['bbox'] == ('list', True) and kinds['n_workers'][1] is False
    assert kinds['source'] == ('str', False) and next(p for p in download.params if p.name == 'source').default == 'ncar'
    kinds = {p.name: (p.kind, p.required) for p in tropo.params}
    assert kinds['times'] == ('list', True) and kinds['geoid'] == ('value', True) and kinds['ref'] == ('value', False)


# ---------------------------------------------------------------- the NSF NCAR archive

def _ncar_fields(lat, lon, day):
    """{name: (24, n_levels, n_lat, n_lon)} of a day: the exponential atmosphere with a pressure growing by the hour"""
    out = {'z': [], 't': [], 'q': []}
    for hour in range(24):
        z, t, q = _exp_atmosphere(lat, lon, P0=100000 + 50 * hour + 10 * day.day)
        for name, a in zip('ztq', (z, t, q)):
            out[name].append(a)
    return {k: np.stack(v) for k, v in out.items()}


def _write_ncar_day(path, day, lat, lon, name, field):
    """a daily file of one variable as in the NSF NCAR archive (levels from 1 hPa, upper case variable names)"""
    import h5py
    order = np.argsort(LEVELS)                                       # 1 .. 1000 hPa
    with h5py.File(path, 'w') as f:
        f.create_dataset('latitude', data=lat)
        f.create_dataset('longitude', data=lon)
        f.create_dataset('level', data=LEVELS[order] / 100).attrs['units'] = 'hPa'
        start = (datetime.datetime(day.year, day.month, day.day) - datetime.datetime(1900, 1, 1)) // datetime.timedelta(hours=1)
        f.create_dataset('time', data=np.arange(start, start + 24, dtype=np.int32))
        f.create_dataset('utc_date', data=np.array([int(f'{day:%Y%m%d}{h:02d}') for h in range(24)], dtype=np.int32))
        d = f.create_dataset(name.upper(), data=field[:, order].astype(np.float32), chunks=(1, 37, lat.size, lon.size),
                             compression='gzip')
        d.attrs['_FillValue'] = np.float32(9.999e20)


@pytest.fixture
def ncar_archive(tmp_path, monkeypatch):
    """a local NSF NCAR archive of some days; returns (directory, {url: path}); downloads run in threads"""
    files = {}

    def add(day, lat, lon):
        fields = _ncar_fields(lat, lon, day)
        for name, code, _ in ctr._NCAR_FIELDS:
            path = tmp_path / f'ncar_{day:%Y%m%d}_{name}.nc'
            _write_ncar_day(path, day, lat, lon, name, fields[name])
            files[ctr._NCAR_URL.format(day=day, code=code)] = path
        return fields

    def opener(url):
        if url not in files:
            raise FileNotFoundError(url)
        return open(files[url], 'rb')
    monkeypatch.setattr(ctr, '_ncar_file', opener)
    monkeypatch.setattr(ctr, 'Executor', lambda n_workers=None, threads_per_worker=None, processes=None:
                        mc.executor.Executor(n_workers=n_workers, threads_per_worker=threads_per_worker))
    return add


def test_cli_download_era5_ncar(tmp_path, ncar_archive):
    pytest.importorskip('h5py')
    lat, lon = np.arange(41.5, 40.25 - 1e-9, -0.25), np.arange(13.5, 14.75 + 1e-9, 0.25)
    fields = {d: ncar_archive(d, lat, lon) for d in (datetime.date(2019, 1, 2), datetime.date(2019, 1, 14))}
    times = ['2019-01-02T05:12:13Z', '2019-01-14T06:00:00Z']
    out = tmp_path / 'era5'
    mc.download_era5(str(out), times, [14.0055556, 40.753376, 14.245538, 40.9077778])
    assert sorted(p.name for p in out.iterdir()) == ['era5_20190102_05.nc', 'era5_20190102_06.nc', 'era5_20190114_06.nc']
    rows, cols = slice(1, 5), slice(1, 5)                              # 41.25 .. 40.5 N, 13.75 .. 14.5 E
    for day, hour in ((datetime.date(2019, 1, 2), 5), (datetime.date(2019, 1, 2), 6), (datetime.date(2019, 1, 14), 6)):
        era5 = ctr._read_era5(out / f'era5_{day:%Y%m%d}_{hour:02d}.nc')
        assert era5['time'] == datetime.datetime(day.year, day.month, day.day, hour)
        np.testing.assert_array_equal(era5['lat'], lat[rows])
        np.testing.assert_array_equal(era5['lon'], lon[cols])
        np.testing.assert_allclose(era5['pressure'], LEVELS)
        for name in 'ztq':
            np.testing.assert_array_equal(era5[name], fields[day][name][hour][:, rows, cols])
    # the files of the archive give the delays of the points as the API does with its fields
    lon_p, lat_p, h_p = _points(n=200, seed=6, hmin=0, hmax=500)
    lon_p, lat_p = 13.8 + (lon_p - 13.76) * 0.65 / 0.73, 40.55 + (lat_p - 40.51) * 0.65 / 0.73
    paths = {name: write_zarr(tmp_path / f'{name}.zarr', a.astype(np.float32))
             for name, a in (('lon', lon_p), ('lat', lat_p), ('hgt', h_p), ('theta', np.full(200, 0.9)))}
    mc.pc_tropo_era5(str(out), paths['lon'], paths['lat'], paths['hgt'], paths['theta'], str(tmp_path / 't.zarr'), times, 0.0)
    assert np.isfinite(zarr.open(str(tmp_path / 't.zarr'), mode='r')[:]).all()


def test_cli_download_era5_ncar_across_longitude_0_and_missing_day(tmp_path, ncar_archive):
    pytest.importorskip('h5py')
    lat, lon = np.array([41.5, 41.25, 41.0, 40.75, 40.5, 40.25]), np.arange(0, 360, 0.25)
    fields = ncar_archive(datetime.date(2019, 3, 1), lat, lon)
    out = tmp_path / 'era5'
    mc.download_era5(str(out), ['2019-03-01T05:00:00Z'], [-0.4, 40.6, 0.4, 41.1])
    era5 = ctr._read_era5(out / 'era5_20190301_05.nc')
    np.testing.assert_allclose(era5['lon'], np.arange(-0.75, 0.75 + 1e-9, 0.25))
    cols = [1437, 1438, 1439, 0, 1, 2, 3]
    np.testing.assert_array_equal(era5['lat'], lat)                    # 41.5 .. 40.25 N
    np.testing.assert_array_equal(era5['t'], fields['t'][5][:, :, cols])
    with pytest.raises(FileNotFoundError, match='source cds'):
        mc.download_era5(str(out), ['2019-03-02T05:00:00Z'], [-0.4, 40.6, 0.4, 41.1])
    with pytest.raises(ValueError, match="'ncar' or 'cds'"):
        mc.download_era5(str(out), ['2019-03-01T05:00:00Z'], [-0.4, 40.6, 0.4, 41.1], source='aws')


def test_http_file_reads_hdf5_by_ranges(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    path = tmp_path / 'remote.h5'
    data = np.random.default_rng(0).standard_normal((3, 400, 500)).astype(np.float32)
    with h5py.File(path, 'w') as f:
        f.create_dataset('a', data=data, chunks=(1, 400, 500), compression='gzip')
        f.create_dataset('small', data=np.arange(10))
    blob = path.read_bytes()
    requests = []

    class Response:
        def __init__(self, body, start, stop):
            self.body, self.headers = body, {'Content-Range': f'bytes {start}-{stop}/{len(blob)}'}
        def read(self):
            return self.body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def urlopen(request, timeout=None):
        start, stop = (int(v) for v in request.headers['Range'].split('=')[1].split('-'))
        stop = min(stop, len(blob) - 1)
        requests.append(stop - start + 1)
        return Response(blob[start:stop + 1], start, stop)
    monkeypatch.setattr(ctr.urllib.request, 'urlopen', urlopen)
    with ctr._HttpFile('https://example.org/remote.h5', block=1 << 12) as fobj, h5py.File(fobj, 'r') as f:
        np.testing.assert_array_equal(f['small'][:], np.arange(10))
        np.testing.assert_array_equal(f['a'][1, 100:110, 7:9], data[1, 100:110, 7:9])
    assert max(requests) > 4 << 12 and sum(r > 4 << 12 for r in requests) == 1    # the chunk in one request
