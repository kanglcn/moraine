"""Polygon files, point in polygon tests and the polygon-mask command."""
import json

import numpy as np
import pytest
import zarr

import moraine as mr
import moraine.cli as mc
from moraine.command import main


def _zarr(path, a, chunks=None):
    z = zarr.open(str(path), mode='w', shape=a.shape, dtype=a.dtype, chunks=chunks or a.shape)
    z[...] = a
    return z


SQUARE = [[2, 2], [6, 2], [6, 5], [2, 5]]                      # range 2 - 6, azimuth 2 - 5
TRIANGLE = [[10, 0], [14, 0], [10, 4]]


def test_write_read_polygons(tmp_path):
    mr.write_polygons(tmp_path / 'p.geojson', [SQUARE, TRIANGLE], 'radar_grid')
    data = json.loads((tmp_path / 'p.geojson').read_text())
    assert data['type'] == 'FeatureCollection' and data['moraine_coordinates'] == 'radar_grid'
    ring = data['features'][0]['geometry']['coordinates'][0]
    assert ring[0] == ring[-1] and len(ring) == 5              # closed ring in the file
    polys, coords = mr.read_polygons(tmp_path / 'p.geojson')
    assert coords == 'radar_grid'
    np.testing.assert_array_equal(polys[0], SQUARE)
    # plain GeoJSON of another program: longitude / latitude, MultiPolygon
    (tmp_path / 'q.geojson').write_text(json.dumps({'type': 'Feature', 'geometry': {
        'type': 'MultiPolygon', 'coordinates': [[[[0, 0], [1, 0], [1, 1], [0, 0]]], [[[5, 5], [6, 5], [6, 6]]]]}}))
    polys, coords = mr.read_polygons(tmp_path / 'q.geojson')
    assert coords == 'lonlat' and [len(q) for q in polys] == [3, 3]
    with pytest.raises(ValueError, match='3 vertices'):
        mr.write_polygons(tmp_path / 'r.geojson', [[[0, 0], [1, 1]]])
    with pytest.raises(ValueError, match='coordinates'):
        mr.write_polygons(tmp_path / 'r.geojson', [SQUARE], 'utm')


def test_polygons_contain():
    polys = [np.array(SQUARE, float), np.array(TRIANGLE, float)]
    x = np.array([[3, 7, 11], [np.nan, 4, 13.5]])
    y = np.array([[3, 3, 1], [3, 4.5, 3.5]])
    np.testing.assert_array_equal(mr.polygons_contain(polys, x, y), [[True, False, True], [False, True, False]])


def test_polygon_mask_raster_keep_inside_outside(tmp_path):
    mr.write_polygons(tmp_path / 'p.geojson', [SQUARE, TRIANGLE], 'radar_grid')
    mc.polygon_mask(str(tmp_path / 'p.geojson'), str(tmp_path / 'in.zarr'), shape=(8, 16), chunks=(3, 5))
    mask = zarr.open(str(tmp_path / 'in.zarr'), mode='r')
    assert mask.shape == (8, 16) and mask.chunks == (3, 5) and mask.dtype == bool
    yi, xi = np.mgrid[:8, :16]
    expected = mr.polygons_contain([np.array(SQUARE, float), np.array(TRIANGLE, float)], xi, yi)
    np.testing.assert_array_equal(mask[:], expected)
    assert mask[3, 4] and not mask[3, 8] and mask[1, 11]       # inside the square, between, in the triangle
    mc.polygon_mask(str(tmp_path / 'p.geojson'), str(tmp_path / 'out.zarr'), shape=(8, 16), keep='outside')
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'out.zarr'), mode='r')[:], ~expected)


def test_polygon_mask_points(tmp_path):
    mr.write_polygons(tmp_path / 'grid.geojson', [SQUARE], 'radar_grid')
    gix = np.array([[3, 3], [3, 8], [4, 5], [0, 0], [6, 4]], dtype=np.int32)    # (azimuth, range)
    _zarr(tmp_path / 'gix.zarr', gix, (2, 2))
    mc.polygon_mask(str(tmp_path / 'grid.geojson'), str(tmp_path / 'm.zarr'), gix=str(tmp_path / 'gix.zarr'))
    m = zarr.open(str(tmp_path / 'm.zarr'), mode='r')
    assert m.shape == (5,) and m.chunks == (2,)
    np.testing.assert_array_equal(m[:], [True, False, True, False, False])
    # longitude / latitude polygons with the longitude / latitude of the points (a raster works the same)
    mr.write_polygons(tmp_path / 'geo.geojson', [[[-148, 61], [-147, 61], [-147, 62], [-148, 62]]])
    lon, lat = np.array([-147.5, -146.5, -147.9]), np.array([61.5, 61.5, 61.1])
    _zarr(tmp_path / 'lon.zarr', lon); _zarr(tmp_path / 'lat.zarr', lat)
    mc.polygon_mask(str(tmp_path / 'geo.geojson'), str(tmp_path / 'g.zarr'), x=str(tmp_path / 'lon.zarr'),
                    y=str(tmp_path / 'lat.zarr'), keep='outside')
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'g.zarr'), mode='r')[:], [False, True, False])
    with pytest.raises(ValueError, match='longitude'):
        mc.polygon_mask(str(tmp_path / 'geo.geojson'), str(tmp_path / 'x.zarr'), gix=str(tmp_path / 'gix.zarr'))
    with pytest.raises(ValueError, match='exactly one'):
        mc.polygon_mask(str(tmp_path / 'grid.geojson'), str(tmp_path / 'x.zarr'), gix=str(tmp_path / 'gix.zarr'),
                        shape=(8, 16))


def test_polygon_mask_command(tmp_path, capsys):
    mr.write_polygons(tmp_path / 'p.geojson', [SQUARE], 'radar_grid')
    assert main(['polygon-mask', '--polygons', str(tmp_path / 'p.geojson'), '--mask', str(tmp_path / 'm.zarr'),
                 '--shape', '8', '16', '--keep', 'outside', '--json', '-q']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['ok'] and str(tmp_path / 'm.zarr') in out['outputs']
    assert not zarr.open(str(tmp_path / 'm.zarr'), mode='r')[3, 4]
