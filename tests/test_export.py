import sqlite3
import struct

import numpy as np
import pytest

import moraine.cli as mc
from moraine.cli.export import _decimal
from moraine.command import main
from moraine.api.utils_ import ArgumentError
from conftest import write_zarr

DATES = ['20190102', '20190114', '20190126', '20190207', '20190219']


def _inputs(tmp_path, n=500, chunks=128):
    rng = np.random.default_rng(0)
    d = dict(lon=14 + rng.random(n), lat=40.5 + rng.random(n))
    d['vel'] = rng.standard_normal(n).astype(np.float32) * 30
    d['vel'][[3, 77]] = np.nan
    d['vel'][5] = 1.6
    d['hgt'] = rng.standard_normal(n) * 100                 # float64
    d['flag'] = rng.random(n) > 0.5
    d['cnt'] = rng.integers(-1000, 1000, n).astype(np.int16)
    d['ts'] = rng.standard_normal((n, len(DATES))).astype(np.float32)
    d['ts'][7, 2] = np.nan
    d['mask'] = rng.random(n) > 0.3
    d['mask'][[3, 5, 7]] = True
    paths = {k: write_zarr(tmp_path / f'{k}.zarr', v, (chunks, 1) if v.ndim == 2 else (chunks,)) for k, v in d.items()}
    return d, paths


def _export(paths, out, **kwargs):
    args = dict(ts=paths['ts'], dates=DATES, mask=paths['mask'], VEL=paths['vel'], HEIGHT=paths['hgt'],
                FLAG=paths['flag'], CNT=paths['cnt'])
    args.update(kwargs)
    mc.pc_export(str(out), paths['lon'], paths['lat'], **{k: v for k, v in args.items() if v is not None})


def _point(blob):
    """lon, lat of a GeoPackageBinary point without envelope"""
    magic, version, flags, srs_id = struct.unpack('<2sBBi', blob[:8])
    assert (magic, version, flags, srs_id) == (b'GP', 0, 1, 4326)
    order, typ, x, y = struct.unpack('<BIdd', blob[8:])
    assert (order, typ) == (1, 1)
    return x, y


def test_pc_export_geopackage(tmp_path):
    d, paths = _inputs(tmp_path)
    out = tmp_path / 'res' / 'points.gpkg'
    _export(paths, out)
    assert not (tmp_path / 'res' / 'points.gpkg.part').exists()
    con = sqlite3.connect(out)
    assert con.execute('PRAGMA application_id').fetchone()[0] == 0x47504B47
    assert con.execute('PRAGMA user_version').fetchone()[0] == 10200
    assert con.execute('SELECT srs_id FROM gpkg_spatial_ref_sys ORDER BY srs_id').fetchall() == [(-1,), (0,), (4326,)]
    assert con.execute('SELECT * FROM gpkg_geometry_columns').fetchall() == [('points', 'geom', 'POINT', 4326, 0, 0)]
    idx = np.flatnonzero(d['mask'])
    contents = con.execute('SELECT table_name, data_type, min_x, min_y, max_x, max_y, srs_id FROM gpkg_contents')
    assert contents.fetchall() == [('points', 'features', d['lon'][idx].min(), d['lat'][idx].min(),
                                    d['lon'][idx].max(), d['lat'][idx].max(), 4326)]
    assert con.execute('SELECT feature_count FROM gpkg_ogr_contents').fetchall() == [(idx.size,)]
    columns = [(r[1], r[2]) for r in con.execute('PRAGMA table_info(points)')]
    assert columns == [('fid', 'INTEGER'), ('geom', 'POINT'), ('point', 'INTEGER'), ('VEL', 'FLOAT'),
                       ('HEIGHT', 'REAL'), ('FLAG', 'BOOLEAN'), ('CNT', 'SMALLINT')] + [(f'D{t}', 'FLOAT') for t in DATES]
    rows = con.execute('SELECT * FROM points ORDER BY fid').fetchall()
    assert [r[0] for r in rows] == list(range(1, idx.size + 1))
    assert [r[2] for r in rows] == idx.tolist()
    np.testing.assert_array_equal([_point(r[1]) for r in rows], np.stack([d['lon'][idx], d['lat'][idx]], axis=1))
    vel = np.array([np.nan if r[3] is None else r[3] for r in rows])
    np.testing.assert_array_equal(vel.astype(np.float32), d['vel'][idx])       # nan as NULL
    assert rows[list(idx).index(5)][3] == 1.6                                  # not 1.600000023841858
    assert [r[4] for r in rows] == d['hgt'][idx].tolist()                     # float64 as it is
    assert [r[5] for r in rows] == d['flag'][idx].astype(int).tolist()
    assert [r[6] for r in rows] == d['cnt'][idx].tolist()
    ts = np.array([[np.nan if v is None else v for v in r[7:]] for r in rows])
    np.testing.assert_array_equal(ts.astype(np.float32), d['ts'][idx])
    assert rows[list(idx).index(7)][7 + 2] is None
    # the R-tree holds every point (in float32 rounded outwards)
    rtree = np.array(con.execute('SELECT id, minx, maxx, miny, maxy FROM rtree_points_geom ORDER BY id').fetchall())
    np.testing.assert_array_equal(rtree[:, 0], np.arange(1, idx.size + 1))
    assert (rtree[:, 1] <= d['lon'][idx]).all() and (d['lon'][idx] <= rtree[:, 2]).all()
    assert (rtree[:, 3] <= d['lat'][idx]).all() and (d['lat'][idx] <= rtree[:, 4]).all()
    assert np.abs(rtree[:, 1:] - np.stack([d['lon'][idx]] * 2 + [d['lat'][idx]] * 2, axis=1)).max() < 1e-5
    triggers = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert {f'rtree_points_geom_{s}' for s in ('insert', 'update1', 'update2', 'update3', 'update4', 'delete')} <= triggers


def test_pc_export_gdal(tmp_path):
    ogr = pytest.importorskip('osgeo.ogr')
    ogr.UseExceptions()
    d, paths = _inputs(tmp_path)
    out = tmp_path / 'points.gpkg'
    _export(paths, out, mask=None)
    try:
        from osgeo_utils.samples import validate_gpkg
    except ImportError:
        validate_gpkg = None
    if validate_gpkg is not None:
        assert validate_gpkg.check(str(out), abort_at_first_error=False, extra_checks=True, warning_msg=False,
                                   warning_as_error=True) == []
    ds = ogr.Open(str(out))
    layer = ds.GetLayer(0)
    assert layer.GetName() == 'points' and layer.GetGeomType() == ogr.wkbPoint
    assert layer.GetFeatureCount() == 500
    defn = layer.GetLayerDefn()
    assert [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())] == \
        ['point', 'VEL', 'HEIGHT', 'FLAG', 'CNT'] + [f'D{t}' for t in DATES]
    for f in layer:
        i = f.GetField('point')
        assert (f.GetGeometryRef().GetX(), f.GetGeometryRef().GetY()) == (d['lon'][i], d['lat'][i])
        vel = f.GetField('VEL')
        assert (vel is None) if np.isnan(d['vel'][i]) else np.float32(vel) == d['vel'][i]
        assert f.GetField('D20190126') is None if i == 7 else np.float32(f.GetField('D20190126')) == d['ts'][i, 2]
    layer.SetSpatialFilterRect(14.2, 40.7, 14.4, 40.9)
    inside = (d['lon'] >= 14.2) & (d['lon'] <= 14.4) & (d['lat'] >= 40.7) & (d['lat'] <= 40.9)
    assert sorted(f.GetField('point') for f in layer) == np.flatnonzero(inside).tolist()


def test_pc_export_replaces_and_keeps(tmp_path):
    d, paths = _inputs(tmp_path)
    out = tmp_path / 'points.gpkg'
    _export(paths, out, ts=None, dates=None)
    _export(paths, out, mask=None, chunks=100)           # replaces it
    con = sqlite3.connect(out)
    assert con.execute('SELECT count(*) FROM points').fetchone()[0] == 500
    assert len(con.execute('PRAGMA table_info(points)').fetchall()) == 7 + len(DATES)
    con.close()
    # an error while writing leaves the file as it was and no partial file
    lon = d['lon'].copy(); lon[300] = np.nan
    bad = write_zarr(tmp_path / 'bad_lon.zarr', lon, (128,))
    with pytest.raises(ValueError, match='lon / lat of point 300 is not finite'):
        mc.pc_export(str(out), bad, paths['lat'], VEL=paths['vel'])
    assert not (tmp_path / 'points.gpkg.part').exists()
    assert sqlite3.connect(out).execute('SELECT count(*) FROM points').fetchone()[0] == 500
    meters = write_zarr(tmp_path / 'e.zarr', d['lon'] * 1e5, (128,))
    with pytest.raises(ValueError, match='lon / lat must be in degrees'):
        mc.pc_export(str(out), meters, paths['lat'])
    # no point selected: an empty layer
    none = write_zarr(tmp_path / 'none.zarr', np.zeros(500, bool), (128,))
    mc.pc_export(str(tmp_path / 'empty.gpkg'), paths['lon'], paths['lat'], mask=none, VEL=paths['vel'])
    con = sqlite3.connect(tmp_path / 'empty.gpkg')
    assert con.execute('SELECT count(*) FROM empty').fetchone()[0] == 0
    assert con.execute('SELECT min_x, max_y FROM gpkg_contents').fetchone() == (None, None)


def test_pc_export_decimal():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(100_000) * 10.0 ** rng.integers(-46, 38, 100_000)).astype(np.float32)
    bits = rng.integers(-2**31, 2**31, 100_000).astype(np.int32).view(np.float32)   # every magnitude, subnormals
    a = np.concatenate([a, bits[np.isfinite(bits)],
                        np.array([0, -0.0, 1.6, 0.1, -7.45, 3.4028235e38, 1e-45, np.inf, np.nan], np.float32)])
    out = _decimal(a)
    ok = np.isfinite(a)
    np.testing.assert_array_equal(out[ok].astype(np.float32), a[ok])            # the same float32 values
    assert np.array_equal(out[~ok], a[~ok].astype(np.float64), equal_nan=True)
    np.testing.assert_array_equal(out[ok], [float(str(v)) for v in a[ok]])        # the shortest decimal
    assert _decimal(np.float32([[1.6, 2.5], [np.nan, 0.3]])).tolist()[0] == [1.6, 2.5]


def test_pc_export_command_line_and_pipeline(tmp_path):
    d, paths = _inputs(tmp_path)
    assert main(['pc-export', '--out', str(tmp_path / 'a.gpkg'), '--lon', paths['lon'], '--lat', paths['lat'],
                 '--ts', paths['ts'], '--dates', *DATES, '--kw', f'VEL={paths["vel"]}', '--kw', f'FLAG={paths["flag"]}',
                 '-q']) == 0
    con = sqlite3.connect(tmp_path / 'a.gpkg')
    assert [r[1] for r in con.execute('PRAGMA table_info(a)')] == \
        ['fid', 'geom', 'point', 'VEL', 'FLAG'] + [f'D{t}' for t in DATES]
    assert con.execute('SELECT count(*) FROM a').fetchone()[0] == 500
    (tmp_path / 'meta.toml').write_text(f'dates = {DATES}\n'.replace("'", '"'))
    lines = ['[vars]', 'dates = { file = "meta.toml", key = "dates" }', '', '[[step]]', 'name = "export"',
             'run = "pc-export"', 'out = "gis/points.gpkg"', f'lon = "{paths["lon"]}"', f'lat = "{paths["lat"]}"',
             f'ts = "{paths["ts"]}"', 'dates = "${dates}"', '[step.kw]', f'VEL = "{paths["vel"]}"',
             f'HEIGHT = "{paths["hgt"]}"']
    (tmp_path / 'p.toml').write_text('\n'.join(lines) + '\n')
    assert main(['run', str(tmp_path / 'p.toml'), '-q']) == 0
    out = tmp_path / 'gis' / 'points.gpkg'
    con = sqlite3.connect(out)
    assert [r[1] for r in con.execute('PRAGMA table_info(points)')][2:5] == ['point', 'VEL', 'HEIGHT']
    assert con.execute('SELECT count(*) FROM points').fetchone()[0] == 500
    con.close()
    mtime = out.stat().st_mtime_ns
    assert main(['run', str(tmp_path / 'p.toml'), '-q']) == 0          # nothing changed: not written again
    assert out.stat().st_mtime_ns == mtime


def test_pc_export_argument_errors(tmp_path):
    d, paths = _inputs(tmp_path)
    write_zarr(tmp_path / 'c.zarr', np.ones(500, np.complex64), (128,))
    write_zarr(tmp_path / 'short.zarr', np.ones(499, np.float32), (128,))
    out = str(tmp_path / 'points.gpkg')
    for kwargs, match in [
        (dict(out=str(tmp_path / 'points.shp')), 'out must be a GeoPackage file ending with .gpkg'),
        (dict(out=str(tmp_path / 'gpkg_x.gpkg')), 'must not start with gpkg_'),
        (dict(out=str(tmp_path / 'a"b.gpkg')), 'nor have quotes'),
        (dict(lon=paths['ts']), r'lon must be floating point of shape \(n_points,\)'),
        (dict(lat=str(tmp_path / 'short.zarr')), r'lat must have the shape of lon \(500,\)'),
        (dict(**{'1VEL': paths['vel']}), "field name '1VEL': letters, digits and _"),
        (dict(**{'V-EL': paths['vel']}), "field name 'V-EL': letters, digits and _"),
        (dict(Point=paths['vel']), "field name 'Point': fid, geom and point are written by pc-export"),
        (dict(VEL=paths['vel'], vel=paths['hgt']), "field name 'vel' is used twice"),
        (dict(VEL=str(tmp_path / 'short.zarr')), r'VEL: shape \(500,\) needed'),
        (dict(VEL=paths['ts']), r'VEL: shape \(500,\) needed'),
        (dict(VEL=str(tmp_path / 'c.zarr')), 'VEL: bool, integer or floating point needed'),
        (dict(ts=paths['vel'], dates=DATES), r'ts must be floating point of shape \(500, nimages\)'),
        (dict(ts=paths['ts']), 'dates: the date of every image of ts is needed'),
        (dict(dates=DATES), 'dates is given without ts'),
        (dict(ts=paths['ts'], dates=DATES[:-1]), '4 dates for 5 images'),
        (dict(ts=paths['ts'], dates=DATES, d20190114=paths['vel']), "field name 'd20190114' is also the field of"),
        (dict(mask=paths['vel']), r'mask must be bool of shape \(500,\)'),
        (dict(chunks=0), 'chunks must be a positive number of points'),
    ]:
        args = dict(out=out, lon=paths['lon'], lat=paths['lat'])
        args.update(kwargs)
        with pytest.raises(ArgumentError, match=match):
            mc.pc_export(**args)
        assert not (tmp_path / 'points.gpkg').exists() and not (tmp_path / 'points.gpkg.part').exists()
