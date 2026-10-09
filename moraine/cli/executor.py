"""The workers of the CLI commands.

A command splits its work into tasks: a function and its arguments. A task reads its input chunks from zarr, computes
them (numpy on the CPU, cupy on a GPU), writes its output chunks to zarr and returns at most a small result. `Executor`
runs the tasks of a command in threads of this process (CPU) or in one process per GPU (``cuda``) and logs the
progress; `chunk_task` is the task of the most common kind (read chunks, call an API function, write chunks) and
`Chunk` names a part of a zarr array to read or write.

Backend: dask (``LocalCluster``, ``dask_cuda.LocalCUDACluster``), as before the executor existed. The commands do not
see it: a task is a plain function with picklable arguments, so the backend can change (decision 0034).
"""

__all__ = ['Executor', 'Chunk', 'Device', 'chunk_task', 'check_aligned']

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import zarr

from ..api.utils_ import is_cuda_available
from .dask_ import parallel_read_zarr, parallel_write_zarr

if is_cuda_available():
    import cupy as cp


# ---------------------------------------------------------------- tasks

@dataclass(frozen=True)
class Chunk:
    """A part of a zarr array: the `path` of the array and the `slices` of the part, one slice per dimension
    (``None``: the whole array)."""
    path: str
    slices: tuple = None

    def _slices(self, z):
        if self.slices is None:
            return tuple(slice(0, n) for n in z.shape)
        return tuple(slice(s.start or 0, z.shape[i] if s.stop is None else s.stop) for i, s in enumerate(self.slices))

    def read(self):
        """The part as a numpy array (the chunks of the array read in threads)."""
        z = zarr.open(str(self.path), mode='r')
        return parallel_read_zarr(z, self._slices(z), fill_slice=False)

    def write(self, data):
        """Write `data` to the part; the part must cover whole chunks of the array (`check_aligned`)."""
        z = zarr.open(str(self.path), mode='r+')
        slices = self._slices(z)
        check_aligned(z, slices, self.path)
        parallel_write_zarr(np.asarray(data), z, slices, fill_slice=False)


@dataclass(frozen=True)
class Device:
    """An argument moved to the device of the task: a cupy array on a GPU task, unchanged on the CPU."""
    value: object


def check_aligned(z, slices, path=''):
    """Raise when `slices` of the zarr array `z` do not cover whole chunks: tasks writing parts of one chunk at the
    same time would overwrite each other."""
    for i, (s, n, c) in enumerate(zip(slices, z.shape, z.chunks)):
        start, stop = s.start or 0, n if s.stop is None else s.stop
        if (start, stop) == (0, n):
            continue
        if start % c != 0 or (stop % c != 0 and stop != n):
            raise ValueError(f'{path}: the part {start}:{stop} of dimension {i} does not cover whole chunks of {c}; '
                             f'the processing chunks must be multiples of the output chunks')


def chunk_task(fn, inputs, outputs, cuda=False, kwargs=None):
    """The task of most commands: read the `Chunk` inputs, call ``fn(*inputs, **kwargs)`` and write the results.

    Parameters
    ----------
    fn
        function of the arrays, e.g. an API function; it returns one array or a tuple of arrays
    inputs : list
        arguments of `fn`: a `Chunk` is read (a cupy array with `cuda`), a `Device` value is moved to the device,
        anything else is passed as it is
    outputs : list
        where the results go, one `Chunk` per returned array; ``None`` discards a result. A single array result is
        written to the first output
    cuda : bool, default: False
        compute on the GPU: the inputs are cupy arrays, the results are moved back before they are written
    kwargs : dict, optional
        keyword arguments of `fn`; `Chunk` and `Device` values are resolved like the inputs

    Returns
    -------
    None
    """
    xp = cp if cuda else np

    def resolve(a):
        if isinstance(a, Chunk):
            return xp.asarray(a.read()) if cuda else a.read()
        if isinstance(a, Device):
            return xp.asarray(a.value) if cuda else a.value
        return a
    args = [resolve(a) for a in inputs]
    kwargs = {k: resolve(v) for k, v in (kwargs or {}).items()}
    results = fn(*args, **kwargs)
    if not isinstance(results, tuple):
        results = (results,)
    if len(results) < len([o for o in outputs if o is not None]):
        raise ValueError(f'{fn.__name__} returned {len(results)} arrays for {len(outputs)} outputs')
    for result, out in zip(results, outputs):
        if out is None:
            continue
        if cuda and not isinstance(result, np.ndarray):
            result = cp.asnumpy(result)
        out.write(result)
    return None


# ---------------------------------------------------------------- the executor

class Executor:
    """Runs the tasks of a command.

    Parameters
    ----------
    cuda : bool, default: False
        run the tasks on the GPUs of ``CUDA_VISIBLE_DEVICES``, one worker process per GPU; otherwise in this process
    n_workers : int, optional
        number of workers; one per GPU with `cuda`, 1 otherwise
    threads_per_worker : int, optional
        tasks a worker runs at the same time (CPU only), 1 by default
    processes : bool, optional
        CPU workers as processes instead of threads (for tasks that hold the GIL), False by default
    rmm_pool_size : float, optional
        with `cuda`: fraction of the memory of each GPU taken by an rmm memory pool, from which cupy allocates; 0.9
        by default, ``None`` for no pool (cupy's own allocator)
    **cluster_kw
        other arguments of the dask ``LocalCluster`` / ``LocalCUDACluster``
    """

    def __init__(self, cuda:bool=False, n_workers:int=None, threads_per_worker:int=None, processes:bool=None,
                 rmm_pool_size:float=0.9, **cluster_kw):
        self.cuda = bool(cuda)
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self.rmm_pool_size = rmm_pool_size
        self.cluster_kw = cluster_kw
        self._cluster = self._client = None
        self.logger = logging.getLogger(__name__)

    # ---- life cycle
    def __enter__(self):
        self.logger.info('starting the workers')
        if self.cuda:
            from dask_cuda import LocalCUDACluster
            self._cluster = LocalCUDACluster(n_workers=self.n_workers, rmm_pool_size=self.rmm_pool_size, **self.cluster_kw)
        else:
            from dask.distributed import LocalCluster
            self._cluster = LocalCluster(processes=bool(self.processes), n_workers=self.n_workers or 1,
                                         threads_per_worker=self.threads_per_worker or 1, **self.cluster_kw)
        from dask.distributed import Client
        self._client = Client(self._cluster)
        if self.cuda and self.rmm_pool_size:
            from rmm.allocators.cupy import rmm_cupy_allocator
            self.run_on_workers(_set_rmm_allocator, rmm_cupy_allocator)
        self.logger.info(self.describe())
        return self

    def __exit__(self, *exc):
        if self._client is not None:
            self._client.close()
        if self._cluster is not None:
            self._cluster.close()
        self._client = self._cluster = None
        self.logger.info('workers stopped')
        return False

    def describe(self) -> str:
        """One line on the workers, for the log."""
        info = self._cluster.scheduler_info['workers']
        n = len(info)
        threads = sum(w['nthreads'] for w in info.values())
        if self.cuda:
            pool = f', rmm pool {self.rmm_pool_size:.0%} of the GPU memory' if self.rmm_pool_size else ', no memory pool'
            return f'workers: {n} GPU process{"es" if n != 1 else ""}{pool} (dask)'
        kind = 'process' if self.processes else 'thread'
        return f'workers: {n} {kind}{"es" if self.processes and n != 1 else ("s" if n != 1 else "")} x {threads // max(n, 1)} ' \
               f'task{"s" if threads // max(n, 1) != 1 else ""} at a time (dask)'

    # ---- work
    def put(self, obj):
        """Share one object with all tasks without copying it for every task; the tasks get the object itself."""
        return self._client.scatter(obj, broadcast=True)

    def run_on_workers(self, fn, *args):
        """Call ``fn(*args)`` once in every worker, e.g. to set up a GPU memory allocator."""
        self._client.run(fn, *args)

    def map(self, fn, tasks, desc:str='tasks') -> list:
        """Run ``fn(*args)`` for the argument tuples `tasks`, as many at a time as the workers allow.

        Returns the results in the order of `tasks`; the first failing task raises its exception. The progress is
        logged at every tenth of the tasks.
        """
        from dask.distributed import as_completed
        tasks = list(tasks)
        n = len(tasks)
        if n == 0:
            return []
        t0 = time.time()
        futures = [self._client.submit(fn, *args, pure=False) for args in tasks]
        step = max(1, math.ceil(n / 10))
        done = 0
        for future in as_completed(futures):
            if future.status == 'error':
                error = future.exception()
                for f in futures:       # the other tasks are not needed any more
                    f.cancel()
                raise error
            done += 1
            if done == 1 or done % step == 0 or done == n:
                self.logger.info(f'{desc}: {done}/{n} done, {time.time() - t0:.1f} s')
        return self._client.gather(futures)

    def map_chunks(self, fn, tasks, cuda:bool=None, desc:str='chunks', **kwargs) -> None:
        """`chunk_task` for every ``(inputs, outputs)`` of `tasks` (see `chunk_task`), with `kwargs` for `fn`."""
        cuda = self.cuda if cuda is None else cuda
        return self.map(chunk_task, [(fn, inputs, outputs, cuda, kwargs) for inputs, outputs in tasks], desc=desc)


def _set_rmm_allocator(allocator):
    import cupy
    cupy.cuda.set_allocator(allocator)
