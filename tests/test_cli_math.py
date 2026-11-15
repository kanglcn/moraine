import numpy as np
import zarr

import moraine.cli as mc


def test_math(tmp_path, rng):
    a = rng.random((100, 100)).astype(np.float32); b = rng.random((100, 100)).astype(np.float32)
    for name, x in [('a', a), ('b', b)]:
        z = zarr.open(str(tmp_path / f'{name}.zarr'), mode='w', shape=x.shape, dtype=x.dtype, chunks=(10, 10))
        z[:] = x
    mc.math(str(tmp_path / 'c.zarr'), 'sin(a)*exp(b)/2', a=str(tmp_path / 'a.zarr'), b=str(tmp_path / 'b.zarr'))
    np.testing.assert_array_almost_equal(zarr.open(str(tmp_path / 'c.zarr'), mode='r')[:], np.sin(a) * np.exp(b) / 2)


def test_math_dtype(tmp_path, rng):
    # numpy's rules: a number does not change the dtype of the data (numexpr wrote float64)
    a = rng.random((20, 30)).astype(np.float32)
    z = zarr.open(str(tmp_path / 'a.zarr'), mode='w', shape=a.shape, dtype=a.dtype, chunks=(10, 10))
    z[:] = a
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
