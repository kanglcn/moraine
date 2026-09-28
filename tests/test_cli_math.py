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


def test_get_logger(tmp_path):
    logger = mc.get_logger(str(tmp_path / 'process.log'))
    logger.info('hello')
    assert (tmp_path / 'process.log').exists()
