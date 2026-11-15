"""The pyramids render in processes that are not forks of the calling process (decision 0046)."""
import tempfile
import threading
import warnings

import numpy as np
import zarr

import moraine.cli as mc
from moraine.command.summary import pyramid_levels


def _zarr(path, a, chunks=None):
    z = zarr.open(str(path), mode='w', shape=a.shape, dtype=a.dtype, chunks=chunks or a.shape)
    z[...] = a


def test_pyramids_do_not_fork_the_calling_process(tmp_path, monkeypatch):
    """A fork copies only the thread that calls it: the locks of the other threads (zarr's I/O thread, numba's
    threads, ...) stay locked in the child, which may hang, and Python warns about it. The pyramids of a process
    with threads render without such a fork; the indices pc_pyramid shares with its processes are temporary
    files, removed at the end."""
    rng = np.random.default_rng(0)
    ras = rng.random((64, 48, 3), dtype=np.float32)
    _zarr(tmp_path / 'ras.zarr', ras, chunks=(16, 16, 1))
    n = 300
    g = np.stack(np.unravel_index(np.sort(rng.choice(40 * 30, n, replace=False)), (40, 30)), -1)
    _zarr(tmp_path / 'x.zarr', g[:, 1].astype(float)); _zarr(tmp_path / 'y.zarr', g[:, 0].astype(float))
    _zarr(tmp_path / 'pc.zarr', rng.random((n, 3), dtype=np.float32), chunks=(n, 1))
    tmp = tmp_path / 'tmp'; tmp.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp))
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)     # the calling process has threads
    thread.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            mc.ras_pyramid(str(tmp_path / 'ras.zarr'), str(tmp_path / 'ras_pyr'), n_workers=2)
            for method in ('first', 'mean'):
                mc.pc_pyramid(str(tmp_path / 'pc.zarr'), str(tmp_path / f'pc_pyr_{method}'), x=str(tmp_path / 'x.zarr'),
                              y=str(tmp_path / 'y.zarr'), ras_resolution=1, method=method, n_workers=2)
    finally:
        stop.set(); thread.join()
    assert not [str(w.message) for w in caught if 'fork()' in str(w.message)]
    assert not list(tmp.glob('moraine-pyramid-*'))
    np.testing.assert_array_equal(zarr.open(str(tmp_path / 'ras_pyr' / '0.zarr'), mode='r')[:], ras)
    for method in ('first', 'mean'):              # every channel, with the indices of the points and cells
        pyr = tmp_path / f'pc_pyr_{method}'
        pc = zarr.open(str(pyr / 'pc.zarr'), mode='r')[:]
        for level in pyramid_levels(pyr):
            values = zarr.open(str(pyr / f'{level}.zarr'), mode='r')[:]
            idx = zarr.open(str(pyr / f'idx_{level}.zarr'), mode='r')[:]
            assert np.isnan(values[idx == -1]).all()
            if method == 'first' or level == 0:
                np.testing.assert_array_equal(values[idx != -1], pc[idx[idx != -1]])
            else:
                assert np.isfinite(values[idx != -1]).all()
