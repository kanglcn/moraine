import numpy as np
import pytest

from moraine.gamma_ import read_gamma_image
from conftest import data_path


def test_read_gamma_image():
    pg = pytest.importorskip('py_gamma')
    f = str(data_path('gamma', 'rslc', '20210802.rslc'))
    np.testing.assert_array_equal(read_gamma_image(f, 1834, dtype='fcomplex'),
                                  pg.read_image(f, width=1834, dtype='fcomplex'))
