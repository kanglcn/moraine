import datetime
import gzip
import io
import re
import urllib.error
import urllib.request

import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.cli import iono as ci
from moraine.api.utils_ import ArgumentError
from moraine.command import main
from conftest import write_zarr, json_out

WAVELENGTH = 0.0554658                                      # m, Sentinel-1
MM_PER_TECU = 1000 * 40.31e16 * WAVELENGTH ** 2 / 299792458.0 ** 2      # 13.80 mm per TEC unit (slant)
LAT = 87.5 - 2.5 * np.arange(71)                            # the grid of the maps of JPL
LON = -180.0 + 5.0 * np.arange(73)
TIMES = ['2019-01-02T05:12:13Z', '2019-01-14T05:12:13Z', '2019-01-26T05:12:13Z']


def _points(n=500, seed=0):
    """points around the Campi Flegrei with the look vector of its descending track"""
    rng = np.random.default_rng(seed)
    lon = 14.0 + 0.4 * rng.random(n)
    lat = 40.7 + 0.3 * rng.random(n)
    theta = np.radians(55.4 + rng.random(n))
    phi = np.radians(-10.6 + 0.1 * rng.random(n))
    return lon, lat, theta, phi


def _rec(content, label):
    return f'{content:<60}{label:<20}'


def _epoch(t):
    return ''.join(f'{v:6d}' for v in (t.year, t.month, t.day, t.hour, t.minute, t.second))


def _write_ionex(path, times, tec, lat=LAT, lon=LON, height=450.0, radius=6371.0, exponent=-1, rms=True,
                 dimension=2):
    """an IONEX file of the maps `tec` (nmaps, nlat, nlon) in TEC units at the naive UTC datetimes `times` (nan: no
    value), gzip if the name ends with .gz"""
    nlon = lon.size
    lines = [_rec('     1.0            IONOSPHERE MAPS     GPS', 'IONEX VERSION / TYPE'),
             _rec('test', 'PGM / RUN BY / DATE'),
             _rec(_epoch(times[0]), 'EPOCH OF FIRST MAP'), _rec(_epoch(times[-1]), 'EPOCH OF LAST MAP'),
             _rec(f'{len(times):6d}', '# OF MAPS IN FILE'), _rec('  NONE', 'MAPPING FUNCTION'),
             _rec('a comment with LAT/LON1/LON2/DLON/H in it', 'COMMENT'),
             _rec(f'{radius:8.1f}', 'BASE RADIUS'), _rec(f'{dimension:6d}', 'MAP DIMENSION'),
             _rec(f'  {height:6.1f}{height:6.1f}{0:6.1f}', 'HGT1 / HGT2 / DHGT'),
             _rec(f'  {lat[0]:6.1f}{lat[-1]:6.1f}{lat[1] - lat[0]:6.1f}', 'LAT1 / LAT2 / DLAT'),
             _rec(f'  {lon[0]:6.1f}{lon[-1]:6.1f}{lon[1] - lon[0]:6.1f}', 'LON1 / LON2 / DLON'),
             _rec(f'{exponent:6d}', 'EXPONENT'), _rec('', 'END OF HEADER')]
    for kind in ('TEC', 'RMS') if rms else ('TEC',):
        for k, (t, m) in enumerate(zip(times, tec)):
            values = np.where(np.isnan(m), 9999, np.round(m / 10.0 ** exponent)).astype(int)
            if kind == 'RMS':
                values = np.full_like(values, 7)
            lines += [_rec(f'{k + 1:6d}', f'START OF {kind} MAP'), _rec(_epoch(t), 'EPOCH OF CURRENT MAP')]
            for j, la in enumerate(lat):
                lines.append(_rec(f'  {la:6.1f}{lon[0]:6.1f}{lon[-1]:6.1f}{lon[1] - lon[0]:6.1f}{height:6.1f}',
                                  'LAT/LON1/LON2/DLON/H'))
                lines += [''.join(f'{v:5d}' for v in values[j, r:r + 16]) for r in range(0, nlon, 16)]
            lines.append(_rec(f'{k + 1:6d}', f'END OF {kind} MAP'))
    lines.append(_rec('', 'END OF FILE'))
    text = '\n'.join(lines) + '\n'
    data = gzip.compress(text.encode()) if str(path).endswith('.gz') else text.encode()
    with open(path, 'wb') as f:
        f.write(data)
    return str(path)


def _field(t, lat, lon):
    """a smooth global field of the vertical TEC in TEC units at the naive UTC datetime t, fixed to the Sun (it turns
    360 degrees to the west per day)"""
    hours = (t - datetime.datetime(2019, 1, 1)).total_seconds() / 3600
    lam = np.radians(lon + 15 * hours)
    return 12 + 6 * np.cos(np.radians(lat)) ** 2 * (1 + np.cos(lam)) + 0.02 * hours


def _day_maps(day, step=2):
    """the maps of one day (00:00 to 24:00 UTC every `step` hours) of `_field`, rounded to 0.1 TEC units"""
    times = [datetime.datetime.combine(day, datetime.time()) + datetime.timedelta(hours=h) for h in range(0, 25, step)]
    tec = np.stack([np.round(_field(t, LAT[:, None], LON[None, :]), 1) for t in times])
    return times, tec


def test_read_ionex_plain_and_gzip(tmp_path):
    times, tec = _day_maps(datetime.date(2019, 1, 2))
    tec[3, 10, 20] = np.nan
    for name in ('maps.19I', 'maps.19I.gz'):
        got = mr.read_ionex(_write_ionex(tmp_path / name, times, tec))
        assert got['times'] == times and got['height'] == 450e3 and got['radius'] == 6371e3
        np.testing.assert_array_equal(got['lat'], LAT)
        np.testing.assert_array_equal(got['lon'], LON)
        assert got['tec'].dtype == np.float32 and got['tec'].shape == (13, 71, 73)
        np.testing.assert_allclose(got['tec'], tec, rtol=0, atol=1e-5)      # also the nan, the RMS maps skipped
    # an exponent of the maps (here -2) and a grid that is not that of JPL
    lat, lon = np.array([50.0, 45.0, 40.0]), np.arange(0.0, 31.0, 2.0)
    tec = np.full((2, 3, 16), 3.25)
    got = mr.read_ionex(_write_ionex(tmp_path / 'small.19I', times[:2], tec, lat=lat, lon=lon, exponent=-2, rms=False))
    np.testing.assert_allclose(got['tec'], 3.25, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(got['lon'], lon)


def test_read_ionex_errors(tmp_path):
    times, tec = _day_maps(datetime.date(2019, 1, 2))
    (tmp_path / 'text').write_text('not an IONEX file\n')
    with pytest.raises(ValueError, match='not an IONEX file'):
        mr.read_ionex(tmp_path / 'text')
    with pytest.raises(ValueError, match='maps of 3 dimensions'):
        mr.read_ionex(_write_ionex(tmp_path / 'three.19I', times, tec, dimension=3))
    path = _write_ionex(tmp_path / 'bad.19I', times, tec)
    text = open(path).read().replace('  85.0-180.0 180.0   5.0 450.0', '  85.0-175.0 180.0   5.0 450.0', 1)
    open(path, 'w').write(text)
    with pytest.raises(ValueError, match='are not of the grid of the header'):
        mr.read_ionex(path)


def test_ionex_vtec_follows_the_sun():
    """maps of a field fixed to the Sun, linear between the nodes: the interpolation between the maps turned with the
    Sun gives it exactly at any time between them (Schaer et al., 1998); between maps not turned it would not"""
    def g(lat, lon):
        return 10 + 0.1 * lat + 0.05 * lon
    t_maps = [datetime.datetime(2019, 1, 2, h) for h in (0, 2, 4)]
    tec = np.stack([g(LAT[:, None], LON[None, :] + 15 * t.hour) for t in t_maps])
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-60, 0, 100), rng.uniform(-80, 80, 100)
    for minutes in (0, 37, 120, 199, 240):
        t = t_maps[0] + datetime.timedelta(minutes=minutes)
        got = mr.ionex_vtec(lon, lat, t, tec, t_maps, LAT, LON)
        np.testing.assert_allclose(got, g(lat, lon + 15 * minutes / 60), rtol=0, atol=1e-4)
    # at the times of the maps: bilinear in the map; on the nodes, their values; longitudes wrap around
    t = t_maps[1]
    got = mr.ionex_vtec(np.array([15.0, -180.0, 180.0, 540.0 - 360]), np.array([40.0, 87.5, -87.5, 0.0]), t, tec,
                        t_maps, LAT, LON)
    np.testing.assert_allclose(got, [tec[1, 19, 39], tec[1, 0, 0], tec[1, 70, 0], tec[1, 35, 0]], rtol=0, atol=1e-5)
    # outside the latitudes of the maps, or a nan position: nan
    assert np.isnan(mr.ionex_vtec(np.array([0.0, np.nan]), np.array([88.0, 0.0]), t, tec, t_maps, LAT, LON)).all()
    with pytest.raises(ValueError, match='do not hold the time 2019-01-02 04:00:01'):
        mr.ionex_vtec(lon, lat, '2019-01-02T04:00:01Z', tec, t_maps, LAT, LON)


def _pierce_bisection(lon, lat, theta, phi, rs):
    """pierce points by bisection along the line from the point (an independent way)"""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    rl, pl = np.radians(lon), np.radians(lat)
    n = a / np.sqrt(1 - e2 * np.sin(pl) ** 2)
    p = np.stack([n * np.cos(pl) * np.cos(rl), n * np.cos(pl) * np.sin(rl), n * (1 - e2) * np.sin(pl)], 1)
    rot = np.stack([np.stack([-np.sin(rl), np.cos(rl), np.zeros_like(rl)], 1),
                    np.stack([-np.sin(pl) * np.cos(rl), -np.sin(pl) * np.sin(rl), np.cos(pl)], 1),
                    np.stack([np.cos(pl) * np.cos(rl), np.cos(pl) * np.sin(rl), np.sin(pl)], 1)], 1)
    enu = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], 1)
    look = np.einsum('nij,ni->nj', rot, enu)
    lo, hi = np.zeros(lon.size), np.full(lon.size, 4e6)
    for _ in range(80):
        mid = (lo + hi) / 2
        out = np.linalg.norm(p + mid[:, None] * look, axis=1) > rs
        hi, lo = np.where(out, mid, hi), np.where(out, lo, mid)
    q = p + lo[:, None] * look
    return (np.degrees(np.arctan2(q[:, 1], q[:, 0])), np.degrees(np.arcsin(q[:, 2] / rs)),
            rs / np.einsum('ni,ni->n', q, look))


def test_pierce_point():
    rng = np.random.default_rng(2)
    n = 2000
    lon, lat = rng.uniform(-180, 180, n), rng.uniform(-89, 89, n)
    theta, phi = np.radians(rng.uniform(0, 90, n)), np.radians(rng.uniform(-360, 360, n))
    for height, radius in ((450e3, 6371e3), (350e3, 6371e3)):
        got = mr.pierce_point(lon, lat, theta, phi, height, radius)
        want = _pierce_bisection(lon, lat, theta, phi, radius + height)
        np.testing.assert_allclose(got[0], want[0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(got[1], want[1], rtol=0, atol=1e-9)
        np.testing.assert_allclose(got[2], want[2], rtol=1e-12)
    # the Campi Flegrei, descending: about 280 km east of the point, mapping factor 1/cos(asin(R/(R+H) sin(34.6)))
    ipp_lon, ipp_lat, mapping = mr.pierce_point(np.array([14.14]), np.array([40.83]), np.radians([55.4]),
                                                np.radians([-10.6]))
    assert 17.3 < ipp_lon[0] < 17.5 and 40.0 < ipp_lat[0] < 40.2
    np.testing.assert_allclose(mapping, 1 / np.cos(np.arcsin(6371 / 6821 * np.sin(np.radians(34.6)))), rtol=2e-3)
    # straight up: the pierce point above the point, mapping factor 1 (to the deflection of the vertical)
    _, _, mapping = mr.pierce_point(lon, lat, np.full(n, np.pi / 2), phi)
    assert (mapping > 1 - 1e-12).all() and (mapping < 1 + 1e-5).all()
    got = mr.pierce_point(np.array([np.nan, 1.0]), np.array([0.0, 1.0]), np.array([0.5, np.nan]), np.zeros(2))
    assert all(np.isnan(a).all() for a in got)


def test_pc_iono_tec_scale_sign_and_reference():
    """maps constant in space and linear in time: the line of sight displacement is the change of the vertical TEC
    times the mapping factor and 13.80 mm per TEC unit (Sentinel-1), positive when the TEC grows"""
    lon, lat, theta, phi = _points(n=50)
    t_maps = [datetime.datetime(2019, 1, 2, 4), datetime.datetime(2019, 1, 2, 6)]
    tec = np.stack([np.full((71, 73), 5.0), np.full((71, 73), 9.0)])
    times = ['2019-01-02T04:00:00Z', '2019-01-02T05:00:00Z', '2019-01-02T05:30:00Z']    # vertical TEC 5, 7, 8
    _, _, mapping = mr.pierce_point(lon, lat, theta, phi)
    for ref, ratio in ((0, 1.0), (2, 0.8)):
        got = mr.pc_iono_tec(lon, lat, theta, phi, times, WAVELENGTH, tec, t_maps, LAT, LON, ratio=ratio, ref=ref)
        assert got.shape == (50, 3) and got.dtype == np.float32
        np.testing.assert_array_equal(got[:, ref], 0)
        vtec = np.array([5.0, 7.0, 8.0])
        want = ratio * MM_PER_TECU * mapping[:, None] * (vtec - vtec[ref])[None, :]
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-5)
    assert (mr.pc_iono_tec(lon, lat, theta, phi, times, WAVELENGTH, tec, t_maps, LAT, LON)[:, 1:] > 0).all()
    np.testing.assert_allclose(MM_PER_TECU, 13.80, atol=0.005)


def test_pc_iono_tec_spread_and_nan():
    lon, lat, theta, phi = _points(n=200)
    times, tec = _day_maps(datetime.date(2019, 1, 2))
    imgs = ['2019-01-02T05:12:13Z', '2019-01-02T11:12:13Z', '2019-01-02T17:12:13Z']
    got = mr.pc_iono_tec(lon, lat, theta, phi, imgs, WAVELENGTH, tec, times, LAT, LON)
    # the same from the vertical TEC at the pierce points
    ipp_lon, ipp_lat, mapping = mr.pierce_point(lon, lat, theta, phi)
    stec = np.stack([mapping * mr.ionex_vtec(ipp_lon, ipp_lat, t, tec, times, LAT, LON) for t in imgs], axis=1)
    np.testing.assert_allclose(got, MM_PER_TECU * (stec - stec[:, :1]), rtol=1e-6, atol=1e-4)
    spread = np.ptp(got, axis=0)[1:]
    assert (spread > 0.01).all() and (spread < 20).all(), spread
    # nan inputs and float32 inputs; a map without value next to the pierce points
    lon32, lat32, theta32, phi32 = (a.astype(np.float32) for a in (lon, lat, theta, phi))
    lon32[3], lat32[4], theta32[5], phi32[6] = np.nan, np.nan, np.nan, np.nan
    got = mr.pc_iono_tec(lon32, lat32, theta32, phi32, imgs, WAVELENGTH, tec, times, LAT, LON)
    assert np.isnan(got[3:7]).all() and np.isfinite(got[:3]).all() and np.isfinite(got[7:]).all()
    tec[:, 19, :] = np.nan                              # the row below the pierce points (40.1 N)
    assert np.isnan(mr.pc_iono_tec(lon, lat, theta, phi, imgs, WAVELENGTH, tec, times, LAT, LON)).all()


def test_argument_errors_of_the_array_functions():
    lon, lat, theta, phi = _points(n=5)
    times, tec = _day_maps(datetime.date(2019, 1, 2))
    base = dict(lon=lon, lat=lat, theta=theta, phi=phi, times=TIMES[:1], wavelength=WAVELENGTH, tec=tec,
                tec_times=times, tec_lat=LAT, tec_lon=LON)
    for kwargs, error in [
            (dict(lon=lon.astype(int)), 'lon must be floating point, got int64'),
            (dict(phi=phi[:4]), 'phi must have the shape (5,) of lon, got shape (4,)'),
            (dict(times=[]), 'times must have the time of at least one image'),
            (dict(ref=1), 'ref must be the index of one of the 1 images, got 1'),
            (dict(wavelength=5.55), 'wavelength must be in meters, between 0 and 1, got 5.55'),
            (dict(ratio=0), 'ratio must be the part of the electron content below the satellite, 0 < ratio <= 1, '
                            'got 0.0'),
            (dict(ratio=1.5), 'ratio must be the part of the electron content below the satellite, 0 < ratio <= 1, '
                              'got 1.5'),
            (dict(height=450), 'height and radius must be in meters, height between 5e4 and 3e7 and radius between '
                               '6e6 and 7e6 (e.g. 450e3 and 6371e3), got 450.0, 6371000.0'),
            (dict(height=6e4, radius=6.3e6), 'the layer (radius + height 6360000 m) must be above the points (the '
                                             'equatorial radius of WGS84, 6378137 m)'),
            (dict(tec=tec[0]), 'tec must be floating point of shape (nmaps, nlat, nlon), got float64 of shape '
                               '(71, 73)'),
            (dict(tec_times=times[::-1]), 'tec_times must be the increasing times of the 13 maps of tec (at least '
                                          '2), got 13 times'),
            (dict(tec_lon=LON[::-1], tec=tec[:, :, ::-1]), 'tec_lon must increase'),
            (dict(tec_lat=LAT[:-1]), 'tec_lat must be the 71 nodes of an evenly spaced grid of the maps, got shape '
                                     '(70,)')]:
        with pytest.raises(ArgumentError, match=f'^{re.escape(error)}$'):
            mr.pc_iono_tec(**(base | kwargs))
    for kwargs, error in [
            (dict(lat=np.full(5, 91.0)), 'lat must be latitudes in degrees between -90 and 90'),
            (dict(theta=np.degrees(theta)), 'theta must be the elevation angle of the look vector in radians'),
            (dict(times=['2019-01-03T00:00:01Z']), 'the maps from 2019-01-02 00:00 to 2019-01-03 00:00 do not hold '
                                                   'the time 2019-01-03 00:00:01 (image 0)')]:
        with pytest.raises(ValueError, match=re.escape(error)) as e:
            mr.pc_iono_tec(**(base | kwargs))
        assert not isinstance(e.value, ArgumentError)
    with pytest.raises(ArgumentError, match='^time must be one time, got 2$'):
        mr.ionex_vtec(lon, lat, TIMES[:2], tec, times, LAT, LON)


@pytest.fixture
def jpl_server(tmp_path, monkeypatch):
    """the server of JPL: the maps of _field for the days in `files` (url -> bytes); the urls asked for in
    `requests`"""
    files, requests = {}, []

    def add(day, data=None):
        times, tec = _day_maps(day)
        path = _write_ionex(tmp_path / ci._ionex_name(day), times, tec)
        files[ci._JPL_URL.format(day=day, name=ci._ionex_name(day))] = data or open(path, 'rb').read()

    def urlopen(url, timeout=None):
        requests.append(url)
        if url not in files:
            raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)
        return io.BytesIO(files[url])

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    return add, requests


def test_cli_download_ionex(tmp_path, jpl_server):
    add, requests = jpl_server
    for day in (datetime.date(2019, 1, 2), datetime.date(2019, 1, 14)):
        add(day)
    out = tmp_path / 'ionex'
    times = ['2019-01-02T05:12:13Z', '2019-01-14T05:12:13Z', '2019-01-02T17:00:00Z']
    mc.download_ionex(str(out), times)
    assert sorted(p.name for p in out.iterdir()) == ['JPLG0020.19I.gz', 'JPLG0140.19I.gz']
    assert len(requests) == 2
    assert mr.read_ionex(out / 'JPLG0020.19I.gz')['times'][-1] == datetime.datetime(2019, 1, 3)
    mc.download_ionex(str(out), times)                  # all there: nothing downloaded
    assert len(requests) == 2
    path = out / 'JPLG0140.19I.gz'                      # a cut file is downloaded again
    path.write_bytes(path.read_bytes()[:5000])
    mc.download_ionex(str(out), times)
    assert len(requests) == 3 and mr.read_ionex(path)['times'][0] == datetime.datetime(2019, 1, 14)
    # a day not on the server; a file that is not IONEX: no file left
    with pytest.raises(FileNotFoundError, match='JPLG0260.19I.gz is not on the server of JPL'):
        mc.download_ionex(str(out), ['2019-01-26T05:12:13Z'])
    add(datetime.date(2019, 2, 7), data=b'<html>maintenance</html>')
    with pytest.raises(ValueError, match='is not an IONEX file with the maps of 2019-02-07'):
        mc.download_ionex(str(out), ['2019-02-07T05:12:13Z'])
    assert sorted(p.name for p in out.iterdir()) == ['JPLG0020.19I.gz', 'JPLG0140.19I.gz']


def test_cli_pc_iono_tec(tmp_path):
    """across midnight: the images take the maps of their days, which agree at 24:00 / 00:00 here"""
    ionex = tmp_path / 'ionex'
    ionex.mkdir()
    all_times, all_tec = [], []
    for day in (datetime.date(2019, 1, 2), datetime.date(2019, 1, 3)):
        times, tec = _day_maps(day)
        _write_ionex(ionex / ci._ionex_name(day), times, tec)
        all_times += times[1:] if all_times else times
        all_tec += list(tec[1:] if all_tec else tec)
    lon, lat, theta, phi = _points(n=1000, seed=3)
    paths = {name: write_zarr(tmp_path / f'{name}.zarr', a.astype(np.float32), chunks=(256,))
             for name, a in (('lon', lon), ('lat', lat), ('theta', theta), ('phi', phi))}
    out = str(tmp_path / 'iono.zarr')
    imgs = ['2019-01-02T05:12:13Z', '2019-01-02T23:30:00Z', '2019-01-03T00:00:00Z', '2019-01-03T00:45:00Z']
    mc.pc_iono_tec(str(ionex), paths['lon'], paths['lat'], paths['theta'], paths['phi'], out, imgs, WAVELENGTH,
                   ratio=0.9, ref=1, out_chunks=512)
    z = zarr.open(out, mode='r')
    assert z.dtype == np.float32 and z.shape == (1000, 4) and z.chunks == (512, 1)
    expect = mr.pc_iono_tec(*(a.astype(np.float32) for a in (lon, lat, theta, phi)), imgs, WAVELENGTH,
                            np.stack(all_tec), all_times, LAT, LON, ratio=0.9, ref=1)
    np.testing.assert_allclose(z[:], expect, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(z[:, 1], 0)
    # a day without file: a processing error
    with pytest.raises(ValueError, match=re.escape(f'{ionex}: no IONEX file of 1 of the 2 days of the images, e.g. '
                                                   'JPLG0040.19I.gz; download them with `download-ionex`')):
        mc.pc_iono_tec(str(ionex), paths['lon'], paths['lat'], paths['theta'], paths['phi'], out,
                       ['2019-01-02T05:12:13Z', '2019-01-04T05:12:13Z'], WAVELENGTH)
    # a file without the maps of the time of an image
    times, tec = _day_maps(datetime.date(2019, 1, 5))
    _write_ionex(ionex / ci._ionex_name(datetime.date(2019, 1, 5)), times[:7], tec[:7])        # 00:00 to 12:00
    with pytest.raises(ValueError, match='do not hold the image of 2019-01-05 13:00:00'):
        mc.pc_iono_tec(str(ionex), paths['lon'], paths['lat'], paths['theta'], paths['phi'], out,
                       ['2019-01-05T13:00:00Z'], WAVELENGTH)


def test_cli_argument_errors_are_usage_errors(tmp_path, capsys, jpl_server):
    """both commands check their arguments and the shapes and dtypes of the inputs before they write or download
    anything: a wrong one is a usage error (exit code 2, decision 0052)"""
    _, requests = jpl_server
    lon, lat, theta, phi = _points(n=10)
    paths = {name: write_zarr(tmp_path / f'{name}.zarr', a) for name, a in
             (('lon', lon), ('lat', lat), ('theta', theta), ('phi', phi), ('short', lon[:9]),
              ('int', np.arange(10)))}
    out = tmp_path / 'iono.zarr'
    base = ['pc-iono-tec', '--ionex-dir', str(tmp_path / 'ionex'), '--lon', paths['lon'], '--lat', paths['lat'],
            '--theta', paths['theta'], '--phi', paths['phi'], '--out', str(out)]
    for args, error in [
            (['--times', *TIMES, '--wavelength', '0.0554658', '--ref', '3'],
             'ref must be the index of one of the 3 images, got 3'),
            (['--times', *TIMES, '--wavelength', '5.55'], 'wavelength must be in meters, between 0 and 1, got 5.55'),
            (['--times', *TIMES, '--wavelength', '0.0554658', '--ratio', '0'],
             'ratio must be the part of the electron content below the satellite, 0 < ratio <= 1, got 0.0'),
            (['--times', '20190102', '--wavelength', '0.0554658'],
             "times must be date-times in UTC (ISO 8601, e.g. 2019-01-02T05:06:07.123456Z), got '20190102'")]:
        assert main([*base, *args, '--json', '-q']) == 2, args
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not out.exists()
    for name, error in (('short', 'lat must have shape (10,), got shape (9,)'),
                        ('int', 'lat must be floating point, got int64')):
        args = [a if a != paths['lat'] else paths[name] for a in base]
        assert main([*args, '--times', *TIMES, '--wavelength', '0.0554658', '--json', '-q']) == 2
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not out.exists()
    assert main([*base, '--times', *TIMES, '--wavelength', '0.0554658', '--json', '-q']) == 1
    assert json_out(capsys)['error'].startswith('ValueError: ')                 # no IONEX files: a processing error
    for args, error in [
            (['--times', '2008-11-18T23:00:00Z'], 'times must be from 2008-11-19 on (the maps of JPL on its server), '
                                                  'got 2008-11-18 23:00:00'),
            (['--times', *TIMES, '--n-workers', '0'], 'n_workers must be a positive integer, got 0')]:
        assert main(['download-ionex', '--out-dir', str(tmp_path / 'dl'), *args, '--json', '-q']) == 2, args
        assert json_out(capsys) == {'version': 1, 'ok': False, 'error': error}
        assert not (tmp_path / 'dl').exists()
    assert requests == []
