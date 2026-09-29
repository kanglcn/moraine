"""Metadata of results, statistics, quicklook images and interactive views of pyramids (drawn with the
holoviews plots of moraine)."""

__all__ = ['summarize', 'quicklook', 'view', 'view_pyramid', 'pyramid_levels']

import math
from pathlib import Path

import numpy as np
import zarr


def pyramid_levels(path)->list:
    """Zoom levels [0, 1, ...] of a pyramid made by `ras_pyramid` / `pc_pyramid`, [] if `path` is not one.

    Pyramids carry ``moraine_pyramid = {version, kind}`` in the attributes of ``0.zarr``
    (docs/contracts/pyramid.md). Older pyramids without it are recognized by their levels halving in size,
    which also tells them apart from directories of per-chunk arrays (``0.zarr``, ``1.zarr``, ... made by
    `ras2pc_ras_chunk`).
    """
    p = Path(path)
    if not p.is_dir() or not (p / '0.zarr').exists():
        return []
    try:
        z0 = zarr.open(str(p / '0.zarr'), mode='r')
    except Exception:
        return []
    levels = sorted(int(q.stem) for q in p.glob('*.zarr') if q.stem.isdigit())
    meta = z0.attrs.get('moraine_pyramid') if hasattr(z0, 'attrs') else None
    if meta:
        from ..cli.plot import PYRAMID_VERSION
        if meta.get('version', 0) > PYRAMID_VERSION:
            raise ValueError(f'{path}: pyramid layout version {meta["version"]} is newer than this moraine '
                             f'supports ({PYRAMID_VERSION}); update moraine')
        return levels
    if not (p / '1.zarr').exists():
        return []
    try:
        z1 = zarr.open(str(p / '1.zarr'), mode='r')
    except Exception:
        return []
    if z0.ndim < 2 or z1.ndim != z0.ndim or \
       tuple(z1.shape[:2]) != tuple(-(-n // 2) for n in z0.shape[:2]):
        return []
    return levels


def _round(x):
    x = float(x)
    if not math.isfinite(x) or x == 0:
        return x
    return round(x, 5 - int(math.floor(math.log10(abs(x)))))


def _stats(a):
    """Statistics and warnings of a sample of values (1D array)."""
    out, warnings = {}, []
    if a.size == 0:
        return {'warnings': ['no values']}
    if a.dtype == bool:
        out['true_fraction'] = _round(a.mean())
        if a.all() or not a.any():
            warnings.append(f'all values are {bool(a[0])}')
        return {**out, 'warnings': warnings} if warnings else out
    prefix = ''
    if np.iscomplexobj(a):
        nan = np.isnan(a.real) | np.isnan(a.imag)
        a, prefix = np.abs(a), 'amplitude_'
    else:
        a = a.astype(np.float64)
        nan = np.isnan(a)
    out['nan_fraction'] = _round(nan.mean())
    inf = np.isinf(a)
    v = a[~nan & ~inf]
    if inf.any():
        warnings.append(f'{int(inf.sum())} infinite values in the sample')
    if v.size == 0:
        warnings.append('all values are nan')
    else:
        p = np.percentile(v, [1, 50, 99])
        out.update({prefix + 'min': _round(v.min()), prefix + 'max': _round(v.max()), prefix + 'mean': _round(v.mean()),
                    prefix + 'std': _round(v.std()), prefix + 'p01': _round(p[0]), prefix + 'p50': _round(p[1]),
                    prefix + 'p99': _round(p[2])})
        if v.min() == v.max():
            warnings.append(f'all values are {_round(v.min())}')
    if warnings:
        out['warnings'] = warnings
    return out


def _pyramid_stats(p, levels, max_bytes):
    """Statistics from the finest pyramid level read within `max_bytes`: a regular decimation of the scene."""
    level = levels[-1]
    for lv in levels:
        if zarr.open(str(p / f'{lv}.zarr'), mode='r').nbytes <= max_bytes:
            level = lv
            break
    a = np.asarray(zarr.open(str(p / f'{level}.zarr'), mode='r')[...])
    idx_path = p / f'idx_{level}.zarr'
    if idx_path.exists():   # point cloud pyramid: skip the cells without points (idx == -1)
        a = a[np.asarray(zarr.open(str(idx_path), mode='r')[...]) != -1]
    return {'stats_level': level, **_stats(a.ravel())}


def summarize(
    path:str,
    max_bytes:int=64 * 2**20,
)->dict:
    """Metadata of a result, plus value statistics for pyramids.

    Only pyramids are read: the statistics come from their finest level of at most `max_bytes`, a regular
    decimation of the whole scene, so the cost does not grow with the data. Point cloud pyramids skip the
    cells without points. ``warnings`` lists obvious anomalies (all nan, infinite values, constant values).

    Parameters
    ----------
    path : str
        path to a zarr array, a pyramid made by `ras_pyramid` / `pc_pyramid`, a zarr group, a directory of
        zarr arrays or any file
    max_bytes : int, default: 64 MiB
        size limit of the pyramid level used for the statistics

    Returns
    -------
    dict
        path, kind and, for arrays and pyramids, shape and dtype; chunks for arrays; for pyramids the number
        of levels, ``stats_level``, statistics (nan_fraction, min, max, mean, std, p01, p50, p99; amplitude
        for complex data, true_fraction for bool) and ``warnings`` if any
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'{path} does not exist')
    levels = pyramid_levels(p)
    if levels:
        base = zarr.open(str(p / '0.zarr'), mode='r')
        kind = 'point cloud pyramid' if (p / 'bounds.toml').exists() else 'raster pyramid'
        out = {'path': str(path), 'kind': kind, 'shape': list(base.shape), 'dtype': str(base.dtype),
               'levels': len(levels)}
        out.update(_pyramid_stats(p, levels, max_bytes))
        return out
    try:
        z = zarr.open(str(p), mode='r')
    except Exception:
        z = None
    if isinstance(z, zarr.Array):
        return {'path': str(path), 'kind': 'array', 'shape': list(z.shape), 'dtype': str(z.dtype),
                'chunks': list(z.chunks)}
    if z is not None:
        return {'path': str(path), 'kind': 'group', 'members': sorted(k for k, _ in z.members())}
    if p.is_dir():
        zarrs = sorted(str(q.relative_to(p)) for q in p.rglob('*.zarr'))
        return {'path': str(path), 'kind': 'directory', 'n_zarr': len(zarrs), 'zarr': zarrs[:20]}
    return {'path': str(path), 'kind': 'file', 'bytes': p.stat().st_size}


def _phase_2d(data_zarr, xslice, yslice):
    return np.angle(data_zarr[yslice, xslice])


def quicklook(
    pyramid:str,
    out:str,
    index:tuple=(),
    post_proc:str=None,
    width:int=1000,
)->str:
    """Save a PNG of the whole scene of a pyramid, rendered with `moraine.cli.ras_plot` / `pc_plot`.

    The plots pick the pyramid level that matches the image size, so only a small level is read however
    large the data are. Build the pyramid first with `ras_pyramid` (rasters) or `pc_pyramid` (point clouds).

    Parameters
    ----------
    pyramid : str
        pyramid directory made by `ras_pyramid` or `pc_pyramid`
    out : str
        output PNG file
    index : tuple, default: ()
        value of the key dimensions of a stack: (i,) for 3D data or `intf_0` / `intf_seq` / `phase`,
        (i, j) for `intf_all`; 0 is used for missing values
    post_proc : str, optional
        'phase', 'intf_0', 'intf_seq' or 'intf_all', see `ras_plot`; the phase is shown by default for complex data
    width : int, default: 1000
        image width in pixels

    Returns
    -------
    str
        the output path
    """
    import holoviews as hv
    from ..cli.plot import ras_plot, pc_plot

    p = Path(pyramid)
    levels = pyramid_levels(p)
    if not levels:
        raise ValueError(f'{pyramid} is not a pyramid; build one first with `moraine ras-pyramid --ras {pyramid} '
                         f'--out_dir <dir>` (rasters) or `moraine pc-pyramid` (point clouds)')
    hv.extension('bokeh', 'matplotlib', logo=False)   # the plots use bokeh defaults; the PNG is drawn by matplotlib
    base = zarr.open(str(p / '0.zarr'), mode='r')
    complex_data = np.iscomplexobj(np.empty(0, base.dtype))
    if post_proc is None and complex_data:
        post_proc = 'phase' if base.ndim == 3 else _phase_2d
    is_pc = (p / 'bounds.toml').exists()
    # pc_plot gives an image layer (coarse zoom) and a points layer (finest zoom); one of them is empty
    plots = list(pc_plot(str(p), post_proc_ras=post_proc, post_proc_pc=post_proc)) if is_pc \
        else [ras_plot(str(p), post_proc=post_proc)]
    ny, nx = base.shape[:2]
    for plot in plots:
        size = next(s for s in plot.streams if type(s).__name__ == 'PlotSize')
        size.event(width=width, height=max(1, round(width * ny / nx)))   # decides the pyramid level to read
    kdims = plots[0].kdims
    index = (tuple(index) + (0,) * len(kdims))[:len(kdims)]
    frames = [plot[index] if kdims else plot[()] for plot in plots]

    phase_like = complex_data or post_proc in ('phase', 'intf_0', 'intf_seq', 'intf_all')
    style = dict(cmap=_cyclic_cmap(), clim=(-np.pi, np.pi)) if phase_like else dict(cmap='viridis')
    title = f'{p.name}  {tuple(base.shape)} {base.dtype}'
    if kdims:
        title += '  ' + ', '.join(f'{d.name}={v}' for d, v in zip(kdims, index))
    frames = [f.opts(colorbar=len(f) > 0, backend='matplotlib', **style) for f in frames]
    if is_pc and len(frames[1]):
        frames[1] = frames[1].opts(color='z', s=4, backend='matplotlib')
    frame = hv.Overlay(frames).opts(aspect='equal', fig_inches=8, title=title, invert_yaxis=not is_pc,
                                    backend='matplotlib')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    hv.save(frame, str(out), backend='matplotlib')
    return str(out)


# ---------------------------------------------------------------- interactive views

_PHASE_POST_PROC = ('phase', 'intf_0', 'intf_seq', 'intf_all')
_FRAME = (900, 700)            # largest plot area (width, height) in screen pixels
_MAX_RATIO = 4                 # scenes more elongated than 1:4 are drawn with this ratio


def _cyclic_cmap():
    import colorcet
    return colorcet.colorwheel


def _phase_pc_1d(data_zarr, idx_array):
    return np.angle(data_zarr[idx_array])


def _kdim_ranges(shape, post_proc):
    """Range of the image indices i (and j) of a stack for the sliders."""
    extra = shape[2:]
    if not extra:
        return {}
    n = extra[0]
    if post_proc == 'intf_seq':
        return {'i': (0, n - 2)}
    if post_proc == 'intf_all':
        return {'i': (0, n - 1), 'j': (0, n - 1)}
    if post_proc in ('phase', 'intf_0'):
        return {'i': (0, n - 1)}
    return {name: (0, m - 1) for name, m in zip(('i', 'j'), extra)}


def _frame_size(width, height):
    """Plot area with the aspect of the scene, fitted into _FRAME; very elongated scenes are squeezed."""
    ratio = min(max(width / height, 1 / _MAX_RATIO), _MAX_RATIO)
    if ratio >= _FRAME[0] / _FRAME[1]:
        return _FRAME[0], max(1, round(_FRAME[0] / ratio))
    return max(1, round(_FRAME[1] * ratio)), _FRAME[1]


def view_pyramid(
    pyramid:str,
    post_proc:str=None,
):
    """Interactive plot of a pyramid for a Jupyter notebook: zoom and pan read only the data on screen.

    Parameters
    ----------
    pyramid : str
        pyramid directory made by `ras_pyramid` or `pc_pyramid`
    post_proc : str, optional
        how to show a stack of images: 'phase', 'intf_0' (interferograms with the first image),
        'intf_seq' (sequential interferograms) or 'intf_all' (any pair); the phase for complex data by
        default

    Returns
    -------
    holoviews object
        the plot, with a colour bar (cyclic for phases, viridis otherwise), axes of the right orientation
        and sliders for the images of a stack
    """
    import holoviews as hv
    from ..cli.plot import ras_plot, pc_plot

    p = Path(pyramid)
    levels = pyramid_levels(p)
    if not levels:
        raise ValueError(f'{pyramid} is not a pyramid; build one first with `moraine ras-pyramid` (rasters) '
                         f'or `moraine pc-pyramid` (point clouds)')
    base = zarr.open(str(p / '0.zarr'), mode='r')
    is_pc = (p / 'bounds.toml').exists()
    complex_data = np.iscomplexobj(np.empty(0, base.dtype))
    ras_proc = pc_proc = post_proc
    if post_proc is None and complex_data:
        if base.ndim == 3:
            ras_proc = pc_proc = post_proc = 'phase'
        else:
            ras_proc, pc_proc = _phase_2d, _phase_pc_1d
    phase_like = complex_data or post_proc in _PHASE_POST_PROC

    # colours: cyclic over (-pi, pi] for phases, else viridis over the 1 % - 99 % range of the values
    if phase_like:
        cmap, clim, label = _cyclic_cmap(), (-np.pi, np.pi), 'phase (rad)'
    else:
        stats = summarize(str(p))
        cmap, label = 'viridis', p.name
        if 'true_fraction' in stats:
            clim = (0, 1)
        elif stats.get('p01') is not None and stats['p01'] < stats['p99']:
            clim = (stats['p01'], stats['p99'])
        else:
            clim = (np.nan, np.nan)

    # axes: rasters and point clouds on the radar grid have range to the right and azimuth down;
    # point clouds on map coordinates keep north up
    if is_pc:
        import toml
        x0, y0, xm, ym = toml.load(p / 'bounds.toml')['bounds']
        grid = all(float(v).is_integer() and v >= 0 for v in (x0, y0, xm, ym))
        width, height = xm - x0, ym - y0
    else:
        grid = True
        height, width = base.shape[:2]
    xlabel, ylabel = ('range', 'azimuth') if grid else ('x', 'y')
    frame_width, frame_height = _frame_size(max(width, 1), max(height, 1))

    if is_pc:
        layers = list(pc_plot(str(p), post_proc_ras=ras_proc, post_proc_pc=pc_proc))
    else:
        layers = [ras_plot(str(p), post_proc=ras_proc)]
    ranges = _kdim_ranges(base.shape, post_proc if isinstance(post_proc, str) else None)
    if ranges:
        layers = [layer.redim.range(**{k: v for k, v in ranges.items() if k in [d.name for d in layer.kdims]})
                  for layer in layers]
    style = dict(cmap=cmap, clim=clim, tools=['hover'])
    layers[0] = layers[0].opts(colorbar=True, colorbar_opts={'title': label}, **style)
    if is_pc:        # the points of the finest zoom share the colours of the image layer
        layers[1] = layers[1].opts(color='z', size=5, **style)
    title = f'{p.name}  {tuple(base.shape)} {base.dtype}' + (f'  {post_proc}' if isinstance(post_proc, str) else '')
    plot = layers[0] * layers[1] if is_pc else layers[0]     # `*` keeps the slider dimensions of both layers
    return plot.opts(frame_width=frame_width, frame_height=frame_height, xlabel=xlabel, ylabel=ylabel,
                     invert_yaxis=grid, title=title)


def view(
    pyramids:list,
    out:str,
    post_proc:str=None,
    overwrite:bool=False,
)->str:
    """Write a Jupyter notebook with an interactive plot of every pyramid.

    Open it in Jupyter or VS Code, choose the python environment of moraine as kernel and run all cells.
    No server or port forwarding is needed: the kernel reads the data on the machine that holds them.

    Parameters
    ----------
    pyramids : list
        pyramid directories made by `ras_pyramid` / `pc_pyramid`
    out : str
        notebook to write (.ipynb)
    post_proc : str, optional
        how to show stacks, see `view_pyramid`
    overwrite : bool, default: False
        replace an existing notebook

    Returns
    -------
    str
        the notebook path
    """
    import json
    for pyr in pyramids:
        if not pyramid_levels(pyr):
            raise ValueError(f'{pyr} is not a pyramid; build one first with `moraine ras-pyramid` (rasters) '
                             f'or `moraine pc-pyramid` (point clouds)')
    out = Path(out)
    if out.suffix != '.ipynb':
        raise ValueError(f'{out}: the notebook name must end with .ipynb')
    if out.exists() and not overwrite:
        raise FileExistsError(f'{out} exists; choose another name or overwrite it')

    def cell(kind, text):
        c = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(True)}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        return c

    arg = f', post_proc={post_proc!r}' if post_proc else ''
    cells = [cell('markdown', 'Interactive views of moraine pyramids: zoom and pan to load details, use the '
                              'sliders to change the image of a stack.'),
             cell('code', 'import holoviews as hv\nfrom moraine.command.summary import view_pyramid\n'
                          "hv.extension('bokeh')\nhv.output(widget_location='bottom')   # sliders below the plots")]
    cells += [cell('code', f'view_pyramid({str(Path(pyr).resolve())!r}{arg})') for pyr in pyramids]
    nb = {'cells': cells, 'metadata': {'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + '\n')
    return str(out)
