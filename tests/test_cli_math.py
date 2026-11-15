import numpy as np
import zarr

import moraine.cli as mc
from conftest import write_zarr


def test_math(tmp_path, rng):
    a = rng.random((100, 100)).astype(np.float32); b = rng.random((100, 100)).astype(np.float32)
    for name, x in [('a', a), ('b', b)]:
        write_zarr(tmp_path / f'{name}.zarr', x, (10, 10))
    mc.math(str(tmp_path / 'c.zarr'), 'sin(a)*exp(b)/2', a=str(tmp_path / 'a.zarr'), b=str(tmp_path / 'b.zarr'))
    np.testing.assert_array_almost_equal(zarr.open(str(tmp_path / 'c.zarr'), mode='r')[:], np.sin(a) * np.exp(b) / 2)


def test_math_dtype(tmp_path, rng):
    # numpy's rules: a number does not change the dtype of the data (numexpr wrote float64)
    a = rng.random((20, 30)).astype(np.float32)
    write_zarr(tmp_path / 'a.zarr', a, (10, 10))
    mc.math(str(tmp_path / 'b.zarr'), 'a*2.5', a=str(tmp_path / 'a.zarr'))
    mc.math(str(tmp_path / 'm.zarr'), '(a>=0.2)&(a<0.6)', a=str(tmp_path / 'a.zarr'))
    b, m = (zarr.open(str(tmp_path / p), mode='r') for p in ('b.zarr', 'm.zarr'))
    assert (b.dtype, b.chunks, m.dtype) == (np.float32, (10, 10), bool)
    np.testing.assert_array_equal(b[:], a * 2.5)
    np.testing.assert_array_equal(m[:], (a >= 0.2) & (a < 0.6))


def test_get_logger(tmp_path):
    logger = mc.get_logger(str(tmp_path / 'process.log'))
    logger.info('hello')
    assert (tmp_path / 'process.log').exists()
