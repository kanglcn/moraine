from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr

from moraine.cli.dask_ import (parallel_read_zarr, parallel_write_zarr, dask_from_zarr, dask_from_zarr_overlap,
                               dask_to_zarr, ZarrDir, _parallel_read_pc_dir, _dask_from_pc_zarr_dir)


def _stack(tmp_path, rng, shape=(130, 170, 4), chunks=(50, 170, 1)):
    data = rng.random(shape).astype(np.float32)
    data = (data + 1j * data).astype(np.complex64)
    z = zarr.open(str(tmp_path / 'data.zarr'), mode='w', shape=data.shape, dtype=data.dtype, chunks=chunks)
    z[:] = data
    return data, z


def test_parallel_read_zarr(tmp_path, rng):
    data, z = _stack(tmp_path, rng)
    np.testing.assert_array_equal(parallel_read_zarr(z, (slice(0, 130), slice(0, 170), slice(2, 3))), data[:, :, 2:3])
    np.testing.assert_array_equal(parallel_read_zarr(z, (slice(34, 111), slice(29, 150), slice(2, 3))), data[34:111, 29:150, 2:3])


def test_parallel_write_zarr(tmp_path, rng):
    data, z = _stack(tmp_path, rng)
    out = zarr.open(str(tmp_path / 'out.zarr'), mode='w', shape=z.shape, dtype=z.dtype, chunks=z.chunks)
    parallel_write_zarr(data[:, :, 0:1], out, (slice(0, 130), slice(0, 170), slice(0, 1)))
    np.testing.assert_array_equal(out[:, :, 0:1], data[:, :, 0:1])


def test_dask_from_zarr(tmp_path, rng):
    data, _ = _stack(tmp_path, rng)
    path = str(tmp_path / 'data.zarr')
    np.testing.assert_array_equal(dask_from_zarr(path, parallel_dims=(0, 1)).compute(), data)
    np.testing.assert_array_equal(dask_from_zarr(path, chunks=(40, 40, 1)).compute(), data)
    darr = dask_from_zarr(path, chunks=(40, 40, 2))
    expected = da.overlap.overlap(darr, depth=(5, 5, 0), boundary={0: 'none', 1: 'none', 2: 'none'})
    np.testing.assert_array_equal(dask_from_zarr_overlap(path, chunks=(40, 40, 2), depth=(5, 5, 0)).compute(),
                                  expected.compute())


def test_dask_to_zarr(tmp_path, rng):
    data, _ = _stack(tmp_path, rng)
    darr = dask_from_zarr(str(tmp_path / 'data.zarr'), parallel_dims=(1, 2)).persist()
    da.compute(dask_to_zarr(darr, str(tmp_path / 'copy.zarr'), chunks=(darr.chunksize[0], darr.shape[1], 1)))
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'copy.zarr'), mode='r')[:], data)


def test_pc_zarr_dir(tmp_path, rng):
    d1 = rng.random((32, 10)).astype(np.float32); d2 = rng.random((64, 10)).astype(np.float32)
    for name, d in [('1.zarr', d1), ('2.zarr', d2)]:
        z = zarr.open(str(tmp_path / 'pc' / name), mode='w', shape=d.shape, dtype=d.dtype, chunks=(d.shape[0], 1))
        z[:] = d
    both = np.concatenate((d1, d2), axis=0)
    np.testing.assert_array_equal(_parallel_read_pc_dir(ZarrDir.from_dir(str(tmp_path / 'pc')), 2), both[:, 2])
    np.testing.assert_array_equal(_dask_from_pc_zarr_dir(str(tmp_path / 'pc')).compute(), both)


def _corrupt_chunks(path):
    """Overwrite every chunk file of a zarr array with bytes that cannot be decoded."""
    for f in Path(path).rglob('*'):
        if f.is_file() and f.name not in ('zarr.json', '.zarray', '.zattrs', '.zgroup', '.zmetadata'):
            f.write_bytes(b'not a compressed chunk')


def test_parallel_read_zarr_raises(tmp_path, rng):
    _stack(tmp_path, rng)
    _corrupt_chunks(tmp_path / 'data.zarr')
    z = zarr.open(str(tmp_path / 'data.zarr'), mode='r')
    with pytest.raises(Exception) as direct:
        z[:, :, 2:3]
    with pytest.raises(direct.type):
        parallel_read_zarr(z, (slice(0, 130), slice(0, 170), slice(2, 3)))


def test_parallel_write_zarr_raises(tmp_path, rng):
    data, _ = _stack(tmp_path, rng)
    z = zarr.open(str(tmp_path / 'data.zarr'), mode='r')
    with pytest.raises(Exception) as direct:
        z[:, :, 0:1] = data[:, :, 0:1]
    with pytest.raises(direct.type):
        parallel_write_zarr(data[:, :, 0:1], z, (slice(0, 130), slice(0, 170), slice(0, 1)))


def test_parallel_read_pc_dir_raises(tmp_path, rng):
    for name in ('1.zarr', '2.zarr'):
        d = rng.random((32, 10)).astype(np.float32)
        z = zarr.open(str(tmp_path / 'pc' / name), mode='w', shape=d.shape, dtype=d.dtype, chunks=(d.shape[0], 1))
        z[:] = d
    _corrupt_chunks(tmp_path / 'pc' / '2.zarr')
    with pytest.raises(Exception) as direct:
        zarr.open(str(tmp_path / 'pc' / '2.zarr'), mode='r')[:, 2]
    with pytest.raises(direct.type):
        _parallel_read_pc_dir(ZarrDir.from_dir(str(tmp_path / 'pc')), 2)
