import numpy as np
import pytest

from moraine.api.utils_.gamma_io import read_gamma_image, read_gamma_pdata
from conftest import data_path


def test_read_gamma_image():
    pg = pytest.importorskip('py_gamma')
    f = str(data_path('gamma', 'rslc', '20210802.rslc'))
    np.testing.assert_array_equal(read_gamma_image(f, 1834, dtype='fcomplex'),
                                  pg.read_image(f, width=1834, dtype='fcomplex'))


@pytest.mark.parametrize('dtype, big', [('float', '>f4'), ('double', '>f8'), ('fcomplex', '>c8'), ('int', '>i4')])
def test_read_gamma_pdata(tmp_path, dtype, big):
    """Point data are read as a raster of width 1: GAMMA's no data (absolute values below 1e-30) are nan, except
    for int"""
    if dtype == 'int':
        x = np.array([0, 1, -3, 7], dtype=big)
    elif dtype == 'fcomplex':
        x = np.array([0, 1e-35, 1e-31+1e-31j, -2e-31j, 1e-29, 1.5-2j], dtype=big)
    else:
        x = np.array([0, -0.0, 1e-35, -2e-31, 1e-29, 1.5], dtype=big)
    x.tofile(tmp_path / 'p')
    p = read_gamma_pdata(str(tmp_path / 'p'), dtype)
    assert p.shape == x.shape and p.dtype == np.dtype(big).newbyteorder('=')
    np.testing.assert_array_equal(p, read_gamma_image(str(tmp_path / 'p'), 1, dtype)[:, 0])
    nodata = abs(x) < 1e-30 if dtype != 'int' else np.zeros(len(x), bool)
    np.testing.assert_array_equal(p[~nodata], x[~nodata])
    assert np.isnan(p[nodata].real).all() and np.isnan(p[nodata].imag if dtype == 'fcomplex' else p[nodata]).all()
