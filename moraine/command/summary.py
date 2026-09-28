"""Summaries and quicklook images of zarr arrays, for checking results without writing code."""

__all__ = ['summarize', 'quicklook']

import math
from pathlib import Path

import numpy as np
import zarr


def _round(x):
    x = float(x)
    if not math.isfinite(x) or x == 0:
        return x
    return round(x, 5 - int(math.floor(math.log10(abs(x)))))


def _sample(z, max_elements):
    """Read `z` with strides on its long axes so that at most about `max_elements` values are read."""
    step = [1] * z.ndim
    if z.size > max_elements:
        long_axes = [i for i, n in enumerate(z.shape) if n > 16] or list(range(z.ndim))
        s = math.ceil((z.size / max_elements) ** (1 / len(long_axes)))
        for i in long_axes:
            step[i] = s
    return z[tuple(slice(None, None, s) for s in step)], step


def _stats(a):
    out = {}
    if a.dtype == bool:
        out['true_fraction'] = _round(a.mean()) if a.size else None
        return out
    if np.iscomplexobj(a):
        out['nan_fraction'] = _round(np.isnan(a).mean()) if a.size else None
        a = np.abs(a)
        prefix = 'amplitude_'
    else:
        prefix = ''
        if np.issubdtype(a.dtype, np.floating):
            out['nan_fraction'] = _round(np.isnan(a).mean()) if a.size else None
    v = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a.ravel()
    if v.size == 0:
        return out
    p = np.percentile(v, [1, 50, 99])
    out.update({prefix + 'min': _round(v.min()), prefix + 'max': _round(v.max()),
                prefix + 'mean': _round(v.mean()), prefix + 'std': _round(v.std()),
                prefix + 'p01': _round(p[0]), prefix + 'p50': _round(p[1]), prefix + 'p99': _round(p[2])})
    return out


def summarize(
    path:str,
    max_elements:int=2_000_000,
)->dict:
    """Summarize a zarr array (or a directory / file) as a small JSON-able dict.

    Parameters
    ----------
    path : str
        path to a zarr array, a zarr group, a directory of zarr arrays or any file
    max_elements : int, default: 2_000_000
        large arrays are read with strides so that at most about this many values are used

    Returns
    -------
    dict
        path, kind, and for arrays: shape, dtype, chunks and value statistics (``sampled_step`` is
        given when the statistics are computed on a strided sample)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'{path} does not exist')
    try:
        z = zarr.open(str(p), mode='r')
    except Exception:
        z = None
    if isinstance(z, zarr.Array):
        a, step = _sample(z, max_elements)
        out = {'path': str(path), 'kind': 'array', 'shape': list(z.shape), 'dtype': str(z.dtype),
               'chunks': list(z.chunks)}
        if any(s > 1 for s in step):
            out['sampled_step'] = step
        out.update(_stats(np.asarray(a)))
        return out
    if z is not None:  # group
        return {'path': str(path), 'kind': 'group', 'members': sorted(k for k, _ in z.members())}
    if p.is_dir():
        zarrs = sorted(str(q.relative_to(p)) for q in p.rglob('*.zarr'))
        return {'path': str(path), 'kind': 'directory', 'n_zarr': len(zarrs), 'zarr': zarrs[:20]}
    return {'path': str(path), 'kind': 'file', 'bytes': p.stat().st_size}


def _read(path):
    return np.asarray(zarr.open(str(path), mode='r')[:])


def quicklook(
    path:str,
    out:str,
    index:int=0,
    gix:str=None,
    x:str=None,
    y:str=None,
    max_pixels:int=2000,
)->str:
    """Save a quicklook PNG of a zarr array.

    Rasters (nlines, width[, k]) are shown as images (complex data as phase), point clouds (n[, k]) as
    scatter plots when their coordinates are given, anything else as a histogram.

    Parameters
    ----------
    path : str
        path to the zarr array
    out : str
        output PNG file
    index : int, default: 0
        index along the last axis for 3D rasters and 2D point clouds
    gix : str, optional
        zarr with the grid index (n, 2) of a point cloud, used as its coordinates
    x : str, optional
        zarr with the x coordinates of a point cloud
    y : str, optional
        zarr with the y coordinates of a point cloud
    max_pixels : int, default: 2000
        rasters are shown with strides so that each side has at most this many pixels

    Returns
    -------
    str
        the output path
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    z = zarr.open(str(path), mode='r')
    title = f'{Path(path).name}  {z.shape} {z.dtype}'
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    def _values(a):
        if np.iscomplexobj(a):
            return np.angle(a), dict(cmap='twilight', vmin=-np.pi, vmax=np.pi), 'phase (rad)'
        if a.dtype == bool:
            return a.astype(np.float32), dict(cmap='gray', vmin=0, vmax=1), ''
        a = a.astype(np.float64)
        f = a[np.isfinite(a)]
        lim = np.percentile(f, [1, 99]) if f.size else (0, 1)
        return a, dict(cmap='viridis', vmin=lim[0], vmax=lim[1]), 'value (1-99 percentile)'

    is_pc = z.ndim in (1, 2) and (gix or (x and y))
    if not is_pc and z.ndim in (2, 3) and min(z.shape[:2]) > 1:
        s = max(1, math.ceil(max(z.shape[:2]) / max_pixels))
        a = z[::s, ::s, index] if z.ndim == 3 else z[::s, ::s]
        v, kw, label = _values(np.asarray(a))
        im = ax.imshow(v, interpolation='nearest', **kw)
        fig.colorbar(im, ax=ax, label=label)
        ax.set(xlabel=f'range index / {s}', ylabel=f'azimuth index / {s}')
        if z.ndim == 3:
            title += f'  [..., {index}]'
    elif is_pc:
        a = np.asarray(z[:, index] if z.ndim == 2 else z[:])
        if gix:
            g = _read(gix); xx, yy = g[:, 1], g[:, 0]
        else:
            xx, yy = _read(x), _read(y)
        v, kw, label = _values(a)
        sc = ax.scatter(xx, yy, c=v, s=max(0.1, min(4.0, 2e4 / max(len(a), 1))), linewidths=0, **kw)
        fig.colorbar(sc, ax=ax, label=label)
        if gix:
            ax.invert_yaxis(); ax.set(xlabel='range index', ylabel='azimuth index')
        ax.set_aspect('equal', adjustable='datalim')
        if z.ndim == 2:
            title += f'  [:, {index}]'
    else:
        a, _ = _sample(z, 2_000_000)
        a = np.asarray(a)
        a = np.abs(a) if np.iscomplexobj(a) else a.astype(np.float64)
        a = a[np.isfinite(a)]
        try:
            ax.hist(a.ravel(), bins=100)
        except ValueError:  # (nearly) constant values
            ax.hist(a.ravel(), bins=1)
        ax.set(xlabel='amplitude' if np.iscomplexobj(z[:1]) else 'value', ylabel='count')
        title += '  histogram'
    ax.set_title(title, fontsize=10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return str(out)
