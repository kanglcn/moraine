"""The workers of the CLI commands.

A command splits its work into tasks: a function and its arguments. A task reads its input chunks from zarr, computes
them (numpy on the CPU, cupy on a GPU), writes its output chunks to zarr and returns at most a small result. `Executor`
runs the tasks of a command in threads of this process (CPU) or in one process per GPU (``cuda``) and logs the
progress; `chunk_task` is the task of the most common kind (read chunks, call an API function, write chunks) and
`Chunk` names a part of a zarr array to read or write.

The workers are moraine's own: threads of the command's process for CPU tasks, or processes started with ``spawn``
for CPU tasks that hold the GIL and for the GPUs, one process per GPU of ``CUDA_VISIBLE_DEVICES``. The commands do
not see them: a task is a plain function with picklable arguments (decisions 0033 and 0034).
"""

__all__ = ['Executor', 'Chunk', 'Device', 'chunk_task', 'check_aligned']

import concurrent.futures
import logging
import math
import multiprocessing
import os
import pickle
import queue
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from ..api.utils_ import is_cuda_available
from .zarr_ import parallel_read_zarr, parallel_write_zarr

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


@dataclass(frozen=True)
class _Loaded:
    """A `Chunk` input read ahead by the worker (while the task before it ran); resolved like the chunk itself."""
    array: object


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
        if isinstance(a, _Loaded):
            return xp.asarray(a.array) if cuda else a.array
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


# ---------------------------------------------------------------- own backend

@dataclass(frozen=True)
class _Ref:
    """A shared object stored in a file by `Executor.put`; every worker process loads it once."""
    path: str


class TaskError(RuntimeError):
    """A task failed in a worker process with an error that could not be sent back as it was."""


def _deref(a, cache):
    if isinstance(a, _Ref):
        if a.path not in cache:
            with open(a.path, 'rb') as f:
                cache[a.path] = pickle.load(f)
        return cache[a.path]
    if isinstance(a, list):
        return [_deref(x, cache) for x in a]
    if isinstance(a, tuple):
        return tuple(_deref(x, cache) for x in a)
    if isinstance(a, dict):
        return {k: _deref(v, cache) for k, v in a.items()}
    return a


def _read_ahead(args):
    """The arguments of a `chunk_task` with its `Chunk` inputs (and `Chunk` keyword arguments) read into `_Loaded`."""
    fn, inputs, outputs, cuda, kwargs = args
    load = lambda a: _Loaded(a.read()) if isinstance(a, Chunk) else a
    return (fn, [load(a) for a in inputs], outputs, cuda, {k: load(v) for k, v in (kwargs or {}).items()} if kwargs else kwargs)


def _gpu_setup():
    """The GPU of a worker process: cupy on the one GPU of its ``CUDA_VISIBLE_DEVICES``, allocating from cupy's own
    memory pool (no pool is reserved: the memory grows with the tasks and freed blocks are reused)."""
    import cupy
    cupy.cuda.Device(0).use()


def _send_error(results, wid, idx, e):
    try:
        pickle.dumps(e)
        err = e
    except Exception:
        err = TaskError(f'{type(e).__name__}: {e}')
    results.put(('error', wid, idx, err, traceback.format_exc()))


def _worker_main(wid, tasks, results, n_threads, env):
    """A worker process: runs the tasks of its queue, `n_threads` at a time, and reports to the results queue."""
    os.environ.update(env)
    try:
        if env.get('CUDA_VISIBLE_DEVICES'):
            _gpu_setup()
    except BaseException as e:
        _send_error(results, wid, None, e)
        return
    results.put(('ready', wid, None, None, None))
    cache = {}

    def run(idx, fn, args):
        try:
            results.put(('ok', wid, idx, fn(*_deref(args, cache)), None))
        except BaseException as e:
            _send_error(results, wid, idx, e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
        while True:
            msg = tasks.get()
            if msg[0] == 'stop':
                break
            if msg[0] == 'run':       # a setup function, once per worker
                try:
                    msg[1](*_deref(msg[2], cache))
                    results.put(('ran', wid, None, None, None))
                except BaseException as e:
                    _send_error(results, wid, None, e)
            else:
                # the chunk inputs of a task are read here, while the threads run the tasks before it (the main process
                # sends one task more than the threads run at a time): the reads overlap the computing, and the GPU is
                # touched by the task threads only
                idx, fn, args = msg[1:]
                if fn is chunk_task:
                    try:
                        args = _read_ahead(args)
                    except BaseException as e:
                        _send_error(results, wid, idx, e)
                        continue
                pool.submit(run, idx, fn, args)


class _Processes:
    """Worker processes of moraine's own backend, started with `spawn`; with `gpus`, one per GPU."""

    def __init__(self, n_workers, threads_per_worker, gpus):
        ctx = multiprocessing.get_context('spawn')
        self.results = ctx.Queue()
        self.threads = max(1, int(threads_per_worker or 1))
        self.tmp = Path(tempfile.mkdtemp(prefix='moraine-executor-'))
        self.workers = []
        for i in range(n_workers):
            q = ctx.Queue()
            env = {'CUDA_VISIBLE_DEVICES': gpus[i]} if gpus else {}
            p = ctx.Process(target=_worker_main, args=(i, q, self.results, self.threads, env), daemon=True)
            p.start()
            self.workers.append((p, q))
        try:
            for _ in self.workers:
                kind, wid, _, err, tb = self._recv(starting=True)
                if kind == 'error':
                    raise RuntimeError(f'worker {wid} failed to start:\n{tb}') from err
        except BaseException:
            self.close()
            raise
        self.n_workers = n_workers
        self.gpus = gpus

    def _recv(self, starting=False):
        """The next message of the workers; a worker that dies without a message (killed, crashed) is an error
        instead of a wait without end."""
        while True:
            try:
                return self.results.get(timeout=1)
            except queue.Empty:
                pass
            for wid, (p, q) in enumerate(self.workers):
                if not p.is_alive():
                    hint = (' (a worker process is started with spawn and imports the main module again: a script must '
                            'guard its code with if __name__ == "__main__")' if starting else '')
                    raise RuntimeError(f'worker {wid} died with exit code {p.exitcode}{hint}')

    def close(self):
        for p, q in self.workers:
            if p.is_alive():
                q.put(('stop',))
        for p, q in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for f in self.tmp.glob('*'):
            f.unlink()
        self.tmp.rmdir()

    def put(self, obj):
        path = self.tmp / f'{uuid.uuid4().hex}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return _Ref(str(path))

    def run_on_workers(self, fn, *args):
        for p, q in self.workers:
            q.put(('run', fn, args))
        for _ in self.workers:
            kind, wid, _, err, tb = self._recv()
            if kind == 'error':
                raise RuntimeError(f'setup failed in worker {wid}:\n{tb}') from err

    def map(self, fn, tasks, on_done):
        n = len(tasks)
        out = [None] * n
        pending = list(range(n))
        in_flight = {wid: 0 for wid in range(len(self.workers))}
        running = 0

        def feed(wid):
            nonlocal running
            while pending and in_flight[wid] < self.threads + 1:     # one more: its chunks are read while the others run
                idx = pending.pop(0)
                self.workers[wid][1].put(('task', idx, fn, tasks[idx]))
                in_flight[wid] += 1
                running += 1
        for wid in in_flight:
            feed(wid)
        done = 0
        while done < n:
            kind, wid, idx, payload, tb = self._recv()
            if kind == 'error':
                pending.clear()
                raise payload
            out[idx] = payload
            in_flight[wid] -= 1
            done += 1
            on_done(done)
            feed(wid)
        return out


class _Threads:
    """Threads of this process: the backend of CPU commands whose kernels release the GIL."""

    def __init__(self, n_threads):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(n_threads)))
        self.n_threads = max(1, int(n_threads))

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def put(self, obj):
        return obj

    def run_on_workers(self, fn, *args):
        fn(*args)

    def map(self, fn, tasks, on_done):
        futures = [self.pool.submit(fn, *args) for args in tasks]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            if future.exception() is not None:
                for f in futures:
                    f.cancel()
                raise future.exception()
            done += 1
            on_done(done)
        return [f.result() for f in futures]


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
        tasks a worker runs at the same time, 1 by default. A worker process reads the chunks of one more task while
        it runs them, so it holds the inputs of ``threads_per_worker + 1`` tasks. Keep 1 for the GPUs: numba loads a
        kernel wrongly when two threads launch it for the first time at once
    processes : bool, optional
        CPU workers as processes instead of threads (for tasks that hold the GIL), False by default
    """

    def __init__(self, cuda:bool=False, n_workers:int=None, threads_per_worker:int=None, processes:bool=None):
        self.cuda = bool(cuda)
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.processes = processes
        self._workers = None
        self.logger = logging.getLogger(__name__)

    # ---- life cycle
    def __enter__(self):
        self.logger.info('starting the workers')
        if self.cuda:
            gpus = [g.strip() for g in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if g.strip()]
            if not gpus:
                raise RuntimeError('cuda: CUDA_VISIBLE_DEVICES names no GPU')
            n = self.n_workers or len(gpus)
            if n > len(gpus):
                raise ValueError(f'n_workers {n} GPU workers for {len(gpus)} GPUs in CUDA_VISIBLE_DEVICES')
            self._workers = _Processes(n, self.threads_per_worker, gpus[:n])
        elif self.processes:
            self._workers = _Processes(self.n_workers or 1, self.threads_per_worker, None)
        else:
            self._workers = _Threads((self.n_workers or 1) * (self.threads_per_worker or 1))
        self.logger.info(self.describe())
        return self

    def __exit__(self, *exc):
        if self._workers is not None:
            self._workers.close()
        self._workers = None
        self.logger.info('workers stopped')
        return False

    def describe(self) -> str:
        """One line on the workers, for the log."""
        w = self._workers
        if isinstance(w, _Threads):
            return f'workers: {w.n_threads} thread{"s" if w.n_threads != 1 else ""} of this process'
        if w.gpus:
            return f'workers: {w.n_workers} GPU process{"es" if w.n_workers != 1 else ""} ({", ".join(w.gpus)}) x ' \
                   f'{w.threads} task{"s" if w.threads != 1 else ""} at a time'
        return f'workers: {w.n_workers} process{"es" if w.n_workers != 1 else ""} x {w.threads} task{"s" if w.threads != 1 else ""} at a time'

    # ---- work
    def put(self, obj):
        """Share one object with all tasks without copying it for every task; the tasks get the object itself."""
        return self._workers.put(obj)

    def run_on_workers(self, fn, *args):
        """Call ``fn(*args)`` once in every worker, e.g. to set up a GPU memory allocator."""
        self._workers.run_on_workers(fn, *args)

    def map(self, fn, tasks, desc:str='tasks') -> list:
        """Run ``fn(*args)`` for the argument tuples `tasks`, as many at a time as the workers allow.

        Returns the results in the order of `tasks`; the first failing task raises its exception. The progress is
        logged at every tenth of the tasks.
        """
        tasks = [tuple(args) for args in tasks]
        n = len(tasks)
        if n == 0:
            return []
        t0 = time.time()
        step = max(1, math.ceil(n / 10))

        def on_done(done):
            if done == 1 or done % step == 0 or done == n:
                self.logger.info(f'{desc}: {done}/{n} done, {time.time() - t0:.1f} s')
        return self._workers.map(fn, tasks, on_done)

    def map_chunks(self, fn, tasks, cuda:bool=None, desc:str='chunks', **kwargs) -> None:
        """`chunk_task` for every ``(inputs, outputs)`` of `tasks` (see `chunk_task`), with `kwargs` for `fn`."""
        cuda = self.cuda if cuda is None else cuda
        return self.map(chunk_task, [(fn, inputs, outputs, cuda, kwargs) for inputs, outputs in tasks], desc=desc)
