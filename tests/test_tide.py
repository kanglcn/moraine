import datetime
import re

import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.api import tide as td
from moraine.api.utils_ import ArgumentError
from moraine.command import main
from conftest import write_zarr, json_out

# Test cases of the IERS routine DEHANTTIDEINEL (IERS Conventions 2010, version of 2017): station, Sun and Moon in
# meters (ECEF), the day (0 h UTC) and the displacement in meters (ECEF). Its fourth test case is left out: its Sun is
# 1.4e10 m from the Earth and its expected output is that of the third case.
IERS_CASES = [
    ((4075578.385, 931852.890, 4801570.154), (137859926952.015, 54228127881.4350, 23509422341.6960),
     (-179996231.920342, -312468450.131567, -169288918.592160), (2009, 4, 13),
     (0.7700420357108125891e-01, 0.6304056321824967613e-01, 0.5516568152597246810e-01)),
    ((1112189.660, -4842955.026, 3985352.284), (-54537460436.2357, 130244288385.279, 56463429031.5996),
     (300396716.912, 243238281.451, 120548075.939), (2012, 7, 13),
     (-0.2036831479592075833e-01, 0.5658254776225972449e-01, -0.7597679676871742227e-01)),
    ((1112200.5696, -4842957.8511, 3985345.9122), (100210282451.6279, 103055630398.3160, 56855096480.4475),
     (369817604.4348, 1897917.5258, 120804980.8284), (2015, 7, 15),
     (0.00509570869172363845, 0.0828663025983528700, -0.0636634925404189617)),
]

# PySolid 0.3.4 (calc_solid_earth_tides_point): latitude, longitude (degrees), time (UTC), east, north, up (mm)
PYSOLID = [
    (40.83, 14.14, '2015-03-10T06:00:00Z', -15.3327, -12.9327, -113.6106),
    (40.83, 14.14, '2019-01-02T05:06:07Z', 9.7355, -15.4182, -107.6475),
    (40.83, 14.14, '2021-12-29T17:30:00Z', 49.7297, -30.7899, 91.4326),
    (0.0, 100.0, '2015-03-10T06:00:00Z', 33.0012, 9.5117, 168.8871),
    (0.0, 100.0, '2019-01-02T05:06:07Z', -36.7894, -32.9950, 139.9830),
    (0.0, 100.0, '2021-12-29T17:30:00Z', -43.4804, 28.6245, 40.9024),
    (-45.0, 170.0, '2015-03-10T06:00:00Z', -27.6360, 28.3037, -73.7779),
    (-45.0, 170.0, '2019-01-02T05:06:07Z', -15.6756, 1.2835, -111.9069),
    (-45.0, 170.0, '2021-12-29T17:30:00Z', 51.6188, 15.9900, -30.7469),
    (75.0, 20.0, '2015-03-10T06:00:00Z', 5.4831, 4.0928, -142.8933),
    (75.0, 20.0, '2019-01-02T05:06:07Z', -18.2277, 7.5499, -133.9036),
    (75.0, 20.0, '2021-12-29T17:30:00Z', 34.2563, -42.5071, -66.9610),
]

TIMES = ['2019-01-02T05:06:07Z', '2019-01-14T05:06:07Z', '2019-01-26T05:06:07Z', '2019-02-07T05:06:07Z']


def _points(n=500, seed=0):
    """points around the Campi Flegrei with the look vector of its descending track"""
    rng = np.random.default_rng(seed)
    lon = 14.0 + 0.4 * rng.random(n)
    lat = 40.7 + 0.3 * rng.random(n)
    theta = np.radians(55.4 + rng.random(n))
    phi = np.radians(-10.6 + 0.1 * rng.random(n))
    return lon, lat, theta, phi


@pytest.mark.parametrize('xsta, sun, moon, day, want', IERS_CASES)
def test_displacement_of_the_iers_test_cases(xsta, sun, moon, day, want):
    step2 = td._step2(datetime.datetime(*day))
    got = td._displacement(*xsta, np.array([*sun, td._MASS_SUN]), np.array([*moon, td._MASS_MOON]), step2)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_solid_tide_against_pysolid():
    """PySolid follows the IERS Conventions 2003 with the Sun and the Moon of the same series; it differs in the
    terms of step 2 (the table of 2003, the time counted from half a day before J2000): at most 0.3 mm over 10 places
    and 6 days every 5 minutes"""
    for lat, lon, t, e, n, u in PYSOLID:
        got = mr.solid_tide(np.array([lon]), np.array([lat]), t)
        assert got.shape == (1, 3) and got.dtype == np.float64
        np.testing.assert_allclose(got[0] * 1000, [e, n, u], rtol=0, atol=0.5, err_msg=f'{lat} {lon} {t}')


def test_solid_tide_times_nan_and_longitudes():
    lon = np.array([14.14, -345.86, np.nan, 14.14])
    lat = np.array([40.83, 40.83, 40.83, np.nan])
    for t in ('2019-01-02T05:06:07Z', '2019-01-02T06:06:07+01:00', np.datetime64('2019-01-02T05:06:07'),
              datetime.datetime(2019, 1, 2, 5, 6, 7)):
        got = mr.solid_tide(lon, lat, t)
        np.testing.assert_allclose(got[1], got[0], rtol=0, atol=1e-9)
        assert np.isnan(got[2:]).all()
        np.testing.assert_allclose(got[0] * 1000, PYSOLID[1][3:], rtol=0, atol=0.5)


def test_leap_seconds():
    assert td._tai_utc(datetime.datetime(1972, 1, 1)) == 10
    assert td._tai_utc(datetime.datetime(2016, 12, 31, 23, 59, 59)) == 36
    assert td._tai_utc(datetime.datetime(2017, 1, 1)) == 37
    assert td._tai_utc(datetime.datetime(2026, 11, 17)) == 37


def test_pc_solid_tide_line_of_sight_and_reference():
    lon, lat, theta, phi = _points(n=50)
    enu = np.stack([mr.solid_tide(lon, lat, t) for t in TIMES], axis=1)       # (n, nimages, 3)
    for ref in (0, 2):
        got = mr.pc_solid_tide(lon, lat, theta, phi, TIMES, ref=ref)
        assert got.shape == (50, 4) and got.dtype == np.float32
        np.testing.assert_array_equal(got[:, ref], 0)
        look = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], axis=1)
        los = np.einsum('nkc,nc->nk', enu, look) * 1000
        np.testing.assert_allclose(got, los - los[:, ref:ref + 1], rtol=0, atol=1e-4)
    # up, east and north alone
    n = lon.size
    for th, ph, c in ((np.pi / 2, 0.0, 2), (0.0, 0.0, 0), (0.0, np.pi / 2, 1)):
        got = mr.pc_solid_tide(lon, lat, np.full(n, th), np.full(n, ph), TIMES)
        np.testing.assert_allclose(got, (enu[:, :, c] - enu[:, :1, c]) * 1000, rtol=0, atol=1e-4)
    # a few millimeters over the area of about 30 km: the part a spatial reference leaves
    got = mr.pc_solid_tide(lon, lat, theta, phi, TIMES)
    spread = np.ptp(got, axis=0)[1:]
    assert (spread > 0.1).all() and (spread < 5).all(), spread


def test_pc_solid_tide_nan_and_float32_inputs():
    lon, lat, theta, phi = (a.astype(np.float32) for a in _points(n=20))
    lon[3], lat[4], theta[5], phi[6] = np.nan, np.nan, np.nan, np.nan
    got = mr.pc_solid_tide(lon, lat, theta, phi, TIMES, ref=1)
    assert np.isnan(got[3:7]).all() and np.isfinite(got[:3]).all() and np.isfinite(got[7:]).all()


def test_argument_errors_of_the_array_functions():
    lon, lat, theta, phi = _points(n=5)
    for kwargs, error in [
            (dict(lon=lon.astype(int)), 'lon must be floating point, got int64'),
            (dict(lat=lat[:4]), 'lat must have the shape (5,) of lon, got shape (4,)'),
            (dict(theta=theta[:, None]), 'theta must have the shape (5,) of lon, got shape (5, 1)'),
            (dict(phi=phi.astype(np.int32)), 'phi must be floating point, got int32'),
            (dict(times=[]), 'times must have the time of at least one image'),
            (dict(ref=4), 'ref must be the index of one of the 4 images, got 4'),
            (dict(ref=True), 'ref must be the index of one of the 4 images, got True'),
            (dict(times=['1970-06-01T00:00:00Z']), 'times must be from 1972-01-01 on (UTC), got 1970-06-01 00:00:00'),
            (dict(times=['20190102']), "times must be date-times in UTC (ISO 8601, e.g. 2019-01-02T05:06:07.123456Z), "
                                       "got '20190102'")]:
        args = dict(lon=lon, lat=lat, theta=theta, phi=phi, times=TIMES) | kwargs
        with pytest.raises(ArgumentError, match=f'^{re.escape(error)}$'):
            mr.pc_solid_tide(**args)
    for kwargs, error in [
            (dict(lat=np.full(5, 91.0)), 'lat must be latitudes in degrees between -90 and 90'),
            (dict(theta=np.degrees(theta)), 'theta must be the elevation angle of the look vector in radians'),
            (dict(phi=np.full(5, -10.6)), 'phi must be the orientation angle of the look vector in radians')]:
        args = dict(lon=lon, lat=lat, theta=theta, phi=phi, times=TIMES) | kwargs
        with pytest.raises(ValueError, match=re.escape(error)) as e:
            mr.pc_solid_tide(**args)
        assert not isinstance(e.value, ArgumentError)
    with pytest.raises(ArgumentError, match='^time must be one time, got 2$'):
        mr.solid_tide(lon, lat, TIMES[:2])


def test_cli_pc_solid_tide(tmp_path):
    lon, lat, theta, phi = _points(n=1000, seed=3)
    paths = {name: write_zarr(tmp_path / f'{name}.zarr', a.astype(np.float32), chunks=(256,))
             for name, a in (('lon', lon), ('lat', lat), ('theta', theta), ('phi', phi))}
    out = str(tmp_path / 'tide.zarr')
    mc.pc_solid_tide(paths['lon'], paths['lat'], paths['theta'], paths['phi'], out, TIMES, ref=1, out_chunks=512)
    z = zarr.open(out, mode='r')
    assert z.dtype == np.float32 and z.shape == (1000, 4) and z.chunks == (512, 1)
    expect = mr.pc_solid_tide(lon.astype(np.float32), lat.astype(np.float32), theta.astype(np.float32),
                              phi.astype(np.float32), TIMES, ref=1)
    np.testing.assert_array_equal(z[:], expect)
    np.testing.assert_array_equal(z[:, 1], 0)


def test_cli_argument_errors_are_usage_errors(tmp_path, capsys):
    """pc-solid-tide checks its arguments and the shapes and dtypes of its inputs before it writes anything: a wrong
    one is a usage error (exit code 2, decision 0052)"""
    lon, lat, theta, phi = _points(n=10)
    paths = {name: write_zarr(tmp_path / f'{name}.zarr', a) for name, a in
             (('lon', lon), ('lat', lat), ('theta', theta), ('phi', phi), ('short', lon[:9]),
              ('int', np.arange(10)))}
    out = tmp_path / 'tide.zarr'
    base = ['pc-solid-tide', '--lon', paths['lon'], '--lat', paths['lat'], '--theta', paths['theta'], '--phi',
            paths['phi'], '--out', str(out)]
    for args, error in [
            (['--times', *TIMES, '--ref', '4'], 'ref must be the index of one of the 4 images, got 4'),
            (['--times', '1971-12-31T23:00:00Z'], 'times must be from 1972-01-01 on (UTC), got 1971-12-31 23:00:00'),
            (['--times', '20190102'], "times must be date-times in UTC (ISO 8601, e.g. 2019-01-02T05:06:07.123456Z), "
                                      "got '20190102'")]:
        assert main([*base, *args, '--json', '-q']) == 2, args
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not out.exists()
    for name, error in (('short', 'lat must have shape (10,), got shape (9,)'),
                        ('int', 'lat must be floating point, got int64')):
        args = [a if a != paths['lat'] else paths[name] for a in base]
        assert main([*args, '--times', *TIMES, '--json', '-q']) == 2
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not out.exists()
