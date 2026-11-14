import numpy as np
import pytest
import zarr

import moraine.cli as mc

pyproj = pytest.importorskip('pyproj')


def _write(path, data, chunks):
    z = zarr.open(str(path), mode='w', shape=data.shape, dtype=data.dtype, chunks=chunks)
    z[:] = data
    return str(path)


def _pyproj(x, y, epsg_in, epsg_out):
    return pyproj.Transformer.from_crs(epsg_in, epsg_out, always_xy=True).transform(x, y)


# 32633: UTM zone 33N; the EPSG codes were ignored before (the outputs were always web mercator)
@pytest.mark.parametrize('epsg_in, epsg_out', [(4326, 3857), (4326, 32633), (32633, 4326)])
def test_transform(tmp_path, epsg_in, epsg_out):
    lon, lat = np.meshgrid(np.linspace(14.0, 14.3, 50), np.linspace(40.9, 40.7, 40))
    x, y = _pyproj(lon, lat, 4326, epsg_in)
    xx_in = _write(tmp_path / 'x.zarr', x, (15, 20))
    yy_in = _write(tmp_path / 'y.zarr', y, (10, 50))      # other chunks than xx_in
    mc.transform(xx_in, yy_in, str(tmp_path / 'xo.zarr'), str(tmp_path / 'yo.zarr'), epsg_in=epsg_in, epsg_out=epsg_out)
    for path, expected in zip(['xo.zarr', 'yo.zarr'], _pyproj(x, y, epsg_in, epsg_out)):
        z = zarr.open(str(tmp_path / path), mode='r')
        assert z.shape == x.shape and z.chunks == (15, 20) and z.dtype == x.dtype
        np.testing.assert_array_equal(z[:], expected)


# the last chunk has one point: pyproj takes it for a point and numpy warns about its conversion to a float
@pytest.mark.filterwarnings('ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning')
def test_transform_point_cloud_nan(tmp_path):
    lon = np.array([14.1, np.nan, 14.2, 14.3, 14.15], dtype=np.float32)
    lat = np.array([40.8, 40.8, np.nan, 40.85, 40.75], dtype=np.float32)
    mc.transform(_write(tmp_path / 'lon.zarr', lon, (2,)), _write(tmp_path / 'lat.zarr', lat, (2,)),
                 str(tmp_path / 'e.zarr'), str(tmp_path / 'n.zarr'))
    e, n = (zarr.open(str(tmp_path / p), mode='r') for p in ['e.zarr', 'n.zarr'])
    assert e.dtype == n.dtype == np.float32 and e.chunks == n.chunks == (2,)
    e, n = e[:], n[:]
    ee, nn = _pyproj(lon.astype(np.float64), lat.astype(np.float64), 4326, 3857)
    np.testing.assert_array_equal(e, ee.astype(np.float32))
    np.testing.assert_array_equal(n, nn.astype(np.float32))
    nan = np.isnan(lon) | np.isnan(lat)
    assert np.isnan(e[nan]).all() and np.isnan(n[nan]).all() and np.isfinite(e[~nan]).all() and np.isfinite(n[~nan]).all()


def test_transform_errors(tmp_path):
    xx = _write(tmp_path / 'lon.zarr', np.full((4, 5), 14.0), (2, 5))
    yy = _write(tmp_path / 'lat.zarr', np.full((4, 5), 40.8), (2, 5))
    e, n = str(tmp_path / 'e.zarr'), str(tmp_path / 'n.zarr')
    with pytest.raises(ValueError, match='shape'):
        mc.transform(xx, _write(tmp_path / 'lat_t.zarr', np.zeros((5, 4)), (5, 4)), e, n)
    for outputs in [(xx, n), (e, yy), (e, e)]:
        with pytest.raises(ValueError, match='other than the inputs'):
            mc.transform(xx, yy, *outputs)
    with pytest.raises(pyproj.exceptions.CRSError):
        mc.transform(xx, yy, e, n, epsg_out=1)
    np.testing.assert_array_equal(zarr.open(xx, mode='r')[:], 14.0)    # the inputs are untouched
    np.testing.assert_array_equal(zarr.open(yy, mode='r')[:], 40.8)
