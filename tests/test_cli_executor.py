import os
import subprocess
import sys
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


@pytest.mark.parametrize('processes', [False, True])
def test_put_and_run_on_workers(tmp_path, rng, processes):
    big = rng.random(1000).astype(np.float32)
    with Executor(threads_per_worker=2, processes=processes) as ex:
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


def test_chunk_task_processes_read_ahead(tmp_path, rng):
    """process workers read the chunks of the next task while a task runs; the results are those of the threads"""
    a = _stack(tmp_path / 'a.zarr', rng)
    row = _stack(tmp_path / 'row.zarr', rng, shape=(1, 48), chunks=(1, 48))
    zarr.open(str(tmp_path / 'sum.zarr'), mode='w', shape=a.shape, dtype=a.dtype, chunks=(16, 48))
    zarr.open(str(tmp_path / 'diff.zarr'), mode='w', shape=a.shape, dtype=a.dtype, chunks=(16, 48))
    tasks = []
    for start in range(0, 64, 16):
        sl = (slice(start, start + 16), slice(0, 48))
        tasks.append(([Chunk(str(tmp_path / 'a.zarr'), sl)], [Chunk(str(tmp_path / 'sum.zarr'), sl), Chunk(str(tmp_path / 'diff.zarr'), sl)]))
    with Executor(processes=True, n_workers=2, threads_per_worker=1) as ex:
        ex.map_chunks(_two, tasks, scale=2.0, b=Chunk(str(tmp_path / 'row.zarr')))   # a chunk as a keyword argument too
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'sum.zarr'), mode='r')[:], a + row)
    np.testing.assert_allclose(zarr.open(str(tmp_path / 'diff.zarr'), mode='r')[:], (a - row) * 2)
    # a chunk that cannot be read is the error of its task
    bad = [([Chunk(str(tmp_path / 'missing.zarr'), (slice(0, 16), slice(0, 48)))], [Chunk(str(tmp_path / 'sum.zarr'), (slice(0, 16), slice(0, 48)))])]
    with Executor(processes=True, n_workers=1) as ex, pytest.raises(Exception):
        ex.map_chunks(_square, bad)


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


def test_unknown_worker_arguments_are_errors():
    with pytest.raises(TypeError, match='memory_limit'):
        Executor(memory_limit='1GB')


def _exit_task(code):
    os._exit(code)


def test_dead_worker_is_an_error():
    """a worker killed or crashed without reporting raises instead of waiting without end"""
    with Executor(processes=True, n_workers=1) as ex:
        with pytest.raises(RuntimeError, match='died with exit code 3'):
            ex.map(_exit_task, [(3,)])


def test_script_without_main_guard_fails_fast():
    """spawn imports the main module again: a stdin script cannot be imported, the worker dies, the error is raised"""
    code = 'from moraine.cli.executor import Executor\nwith Executor(processes=True) as ex:\n    pass\n'
    r = subprocess.run([sys.executable, '-'], input=code, capture_output=True, text=True, timeout=120,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert r.returncode != 0
    assert 'died with exit code' in r.stderr and '__main__' in r.stderr
