"""The executor of the CLI commands: tasks in threads, processes or on GPUs, chunk tasks on zarr."""
import numpy as np
import pytest
import zarr

import moraine as mr
from moraine.cli.executor import Chunk, Device, Executor, chunk_task, check_aligned


def _square(a):
    return a * a


def _two(a, b, scale=1.0):
    return a + b, (a - b) * scale


def _fail(a):
    raise ValueError('bad task')


def _stack(path, rng, shape=(64, 48), chunks=(16, 48)):
    data = rng.random(shape).astype(np.float32)
    z = zarr.open(str(path), mode='w', shape=shape, dtype=data.dtype, chunks=chunks)
    z[:] = data
    return data


@pytest.mark.parametrize('processes', [False, True])
def test_map_order_and_errors(processes):
    with Executor(processes=processes, n_workers=2, threads_per_worker=2) as ex:
        assert 'workers' in ex.describe()
        assert ex.map(_square, [(np.float32(i),) for i in range(7)]) == [i * i for i in range(7)]
        assert ex.map(_square, []) == []
        with pytest.raises(ValueError, match='bad task'):
            ex.map(_fail, [(1,), (2,)])


def test_put_and_run_on_workers(tmp_path, rng):
    big = rng.random(1000).astype(np.float32)
    with Executor(threads_per_worker=2) as ex:
        shared = ex.put(big)
        out = ex.map(_square, [(shared,)] * 3)
        for o in out:
            np.testing.assert_array_equal(o, big * big)
        ex.run_on_workers(_square, np.float32(2))     # runs without error in every worker


def test_chunk_task(tmp_path, rng):
    a = _stack(tmp_path / 'a.zarr', rng)
    b = _stack(tmp_path / 'b.zarr', rng)
    zarr.open(str(tmp_path / 'sum.zarr'), mode='w', shape=a.shape, dtype=a.dtype, chunks=(16, 48))
    zarr.open(str(tmp_path / 'diff.zarr'), mode='w', shape=a.shape, dtype=a.dtype, chunks=(16, 48))
    tasks = []
    for start in range(0, 64, 32):           # processing chunks of 2 output chunks
        sl = (slice(start, start + 32), slice(0, 48))
        tasks.append(([Chunk(str(tmp_path / 'a.zarr'), sl), Chunk(str(tmp_path / 'b.zarr'), sl)],
                      [Chunk(str(tmp_path / 'sum.zarr'), sl), Chunk(str(tmp_path / 'diff.zarr'), sl)]))
    with Executor(threads_per_worker=2) as ex:
        ex.map_chunks(_two, tasks, scale=2.0)
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'sum.zarr'), mode='r')[:], a + b)
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'diff.zarr'), mode='r')[:], (a - b) * 2)
    # a discarded output, a Device argument and a whole array chunk
    zarr.open(str(tmp_path / 'c.zarr'), mode='w', shape=a.shape, dtype=a.dtype, chunks=(16, 48))
    chunk_task(_two, [Chunk(str(tmp_path / 'a.zarr')), Device(b)], [None, Chunk(str(tmp_path / 'c.zarr'))], kwargs={'scale': 0.5})
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'c.zarr'), mode='r')[:], (a - b) * 0.5)


def test_write_must_cover_whole_chunks(tmp_path, rng):
    _stack(tmp_path / 'a.zarr', rng)
    z = zarr.open(str(tmp_path / 'a.zarr'), mode='r')
    check_aligned(z, (slice(16, 48), slice(0, 48)))
    check_aligned(z, (slice(48, 64), slice(0, 48)))      # the last chunk may be short
    with pytest.raises(ValueError, match='whole chunks'):
        check_aligned(z, (slice(8, 24), slice(0, 48)), 'a.zarr')
    with pytest.raises(ValueError, match='whole chunks'):
        Chunk(str(tmp_path / 'a.zarr'), (slice(0, 24), slice(0, 48))).write(np.zeros((24, 48), np.float32))


@pytest.mark.gpu
def test_chunk_task_gpu(tmp_path, rng):
    rslc = (rng.random((40, 30, 3)) + 1j * rng.random((40, 30, 3))).astype(np.complex64)
    z = zarr.open(str(tmp_path / 'rslc.zarr'), mode='w', shape=rslc.shape, dtype=rslc.dtype, chunks=(20, 30, 1))
    z[:] = rslc
    zarr.open(str(tmp_path / 'adi.zarr'), mode='w', shape=rslc.shape[:2], dtype=np.float32, chunks=(20, 30))
    tasks = [([Chunk(str(tmp_path / 'rslc.zarr'), (slice(s, s + 20), slice(0, 30), slice(0, 3)))],
              [Chunk(str(tmp_path / 'adi.zarr'), (slice(s, s + 20), slice(0, 30)))]) for s in (0, 20)]
    with Executor(cuda=True) as ex:
        assert 'GPU' in ex.describe()
        ex.map_chunks(mr.amp_disp, tasks)
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'adi.zarr'), mode='r')[:], mr.amp_disp(rslc), rtol=1e-5)
