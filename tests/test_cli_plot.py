"""The pyramids render in processes that are not forks of the calling process (decision 0046), with the memory of a
band of lines of a raster at a time."""
import tempfile
import threading
import tracemalloc
import warnings
import weakref

import numpy as np
import pytest
import zarr

import moraine.cli as mc
from moraine.command.summary import pyramid_levels
from conftest import write_zarr, random_gix


def test_pyramids_do_not_fork_the_calling_process(tmp_path, monkeypatch):
    """A fork copies only the thread that calls it: the locks of the other threads (zarr's I/O thread, numba's
    threads, ...) stay locked in the child, which may hang, and Python warns about it. The pyramids of a process
    with threads render without such a fork; the indices pc_pyramid shares with its processes are temporary
    files, removed at the end."""
    rng = np.random.default_rng(0)
    ras = rng.random((64, 48, 3), dtype=np.float32)
    write_zarr(tmp_path / 'ras.zarr', ras, chunks=(16, 16, 1))
    n = 300
    g = random_gix(rng, (40, 30), n)
    write_zarr(tmp_path / 'x.zarr', g[:, 1].astype(float)); write_zarr(tmp_path / 'y.zarr', g[:, 0].astype(float))
    write_zarr(tmp_path / 'pc.zarr', rng.random((n, 3), dtype=np.float32), chunks=(n, 1))
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


@pytest.mark.parametrize('method', ['decimate', 'mean'])
def test_raster_pyramid_memory_does_not_grow_with_the_bands(tmp_path, method):
    """A process renders a channel of a raster band by band: the memory it needs is that of a band (and of the sample
    of the statistics), not that of the channel. Measured in this process on channels of 16 and 64 bands."""
    from moraine.cli import plot
    rows, width = 32, 512
    band = rows * width * 4                                    # bytes of a band of float32

    def peak(n_bands):
        ras, pyr = tmp_path / f'ras_{n_bands}.zarr', tmp_path / f'pyr_{n_bands}'
        write_zarr(ras, np.random.default_rng(0).random((rows * n_bands, width), dtype=np.float32), (rows, width))
        mc.ras_pyramid(str(ras), str(pyr), rows=rows, method=method, n_workers=1)
        plot._ras_pyramid_init(str(ras), str(pyr), len(pyramid_levels(pyr)) - 1, 64, rows, method)
        tracemalloc.start()
        try:
            plot._ras_pyramid_channel(())
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert peak(64) - peak(16) < 8 * band          # 48 more bands; c99c11b kept them all: 98 - 145 bands more


@pytest.mark.parametrize('method', ['decimate', 'mean'])
def test_raster_pyramid_holds_one_band_at_a_time(tmp_path, monkeypatch, method):
    """A process renders a channel of a raster with one band in memory at a time: a band is released before the next
    one is read. The previous band stayed alive while the next one was read (both methods) and, with decimate, while
    its statistics were computed, as the decimated level of the band was a view of it: a band more at the peak.
    Checked by the bands still alive when the next one is read and, with decimate, by the peak memory traced while a
    channel of complex64 renders: a band and the working memory of its statistics, plus half a band for the chunks
    read and written and the levels (without the sample of the statistics)."""
    from moraine.cli import plot
    rows, width, n_bands = 256, 1024, 8
    rng = np.random.default_rng(0)
    data = (rng.random((rows * n_bands, width), dtype=np.float32)
            + 1j * rng.random((rows * n_bands, width), dtype=np.float32)).astype(np.complex64)
    ras, pyr = tmp_path / 'ras.zarr', tmp_path / 'pyr'
    write_zarr(ras, data, (rows, width))
    mc.ras_pyramid(str(ras), str(pyr), rows=rows, method=method, n_workers=1)
    step = rows * width                                        # one value of every band in the sample
    plot._ras_pyramid_init(str(ras), str(pyr), len(pyramid_levels(pyr)) - 1, step, rows, method)
    bands, alive = [], []
    read = plot.parallel_read_zarr

    def read_band(*args, **kwargs):
        alive.append(sum(b() is not None for b in bands))     # the bands still in memory when the next one is read
        band = read(*args, **kwargs)
        bands.append(weakref.ref(band))
        return band
    monkeypatch.setattr(plot, 'parallel_read_zarr', read_band)
    tracemalloc.start()
    try:
        plot._ras_pyramid_channel(())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert alive == [0] * n_bands
    if method == 'decimate':
        band = data[:rows]
        tracemalloc.start()
        try:
            plot._part_stats(band, step)
            stats = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 1.5 * band.nbytes + stats, (peak / band.size, stats / band.size)   # bytes a pixel of a band
