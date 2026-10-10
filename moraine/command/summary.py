"""Metadata of results, statistics, quicklook images and notebooks with interactive maps of pyramids (drawn
with `moraine.cli.view`)."""

__all__ = ['summarize', 'quicklook', 'view', 'pyramid_levels']

from pathlib import Path

import zarr

# pyramid reading lives in the CLI layer, next to the pyramids and the views
from ..cli.plot import pyramid_levels, _pyramid_stats, _channel_warnings


def summarize(
    path:str,
    max_bytes:int=64 * 2**20,
)->dict:
    """Metadata of a result, plus value statistics for pyramids.

    Only pyramids are read: the statistics are those of all the data, computed when the pyramid was built
    (``stats_level`` 0); for a pyramid made without them they come from its finest level of at most
    `max_bytes`, a regular decimation of the whole scene, so the cost does not grow with the data (point cloud
    pyramids skip the cells without points). ``warnings`` lists obvious anomalies (all nan, infinite values,
    constant values), also of single images or channels of a stack.

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
        more = _channel_warnings(p)
        if more:
            out['warnings'] = out.get('warnings', []) + more
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


def quicklook(
    pyramid:str,
    out:str,
    index:tuple=(),
    show:str=None,
    width:int=1000,
    extent:str=None,
)->str:
    """Save a PNG of a pyramid, drawn like `moraine.cli.view` (``view(pyramid).png(out)``).

    Only the pyramid level matching the image size is read, however large the data are. Build the pyramid
    first with `ras_pyramid` (rasters) or `pc_pyramid` (point clouds).

    Parameters
    ----------
    pyramid : str
        pyramid directory made by `ras_pyramid` or `pc_pyramid`
    out : str
        output PNG file
    index : tuple, default: ()
        slider values in the order of the sliders: (image,) for 3D data or 'phase' / 'intf_0' / 'intf_seq',
        (ref, sec) for 'intf_all'; the first images by default
    show : str, optional
        what to show of a stack, see `moraine.cli.view`; the phase for complex data by default
    width : int, default: 1000
        image width in pixels
    extent : str, optional
        the part to draw, four numbers separated by commas: "west,south,east,north" in degrees on a web
        mercator map, "range_min,azimuth_min,range_max,azimuth_max" in pixels on the radar grid; the whole
        scene by default. A smaller part shows finer pyramid levels, down to the data

    Returns
    -------
    str
        the output path
    """
    from ..cli.tiles import view as _view
    if not pyramid_levels(Path(pyramid)):
        raise ValueError(f'{pyramid} is not a pyramid; build one first with `moraine ras-pyramid --ras {pyramid} '
                         f'--out_dir <dir>` (rasters) or `moraine pc-pyramid` (point clouds)')
    if extent is not None and not isinstance(extent, (tuple, list)):
        extent = [float(v) for v in str(extent).replace('(', ' ').replace(')', ' ').replace(',', ' ').split()]
    return _view(str(pyramid), show=show).png(out, width=width, index=tuple(index), extent=extent)


def view(
    pyramids:list,
    out:str,
    show:str=None,
    dates:str=None,
    overwrite:bool=False,
)->str:
    """Write a Jupyter notebook with an interactive map of every pyramid (`moraine.cli.view`).

    Open it in Jupyter or VS Code, choose the python environment of moraine as kernel and run all cells.
    No server or port forwarding is needed: the kernel reads the data on the machine that holds them.

    Parameters
    ----------
    pyramids : list
        pyramid directories made by `ras_pyramid` / `pc_pyramid`
    out : str
        notebook to write (.ipynb)
    show : str, optional
        what to show of stacks, see `moraine.cli.view`
    dates : str, optional
        toml file with the image ``dates`` (e.g. of `load_gamma_metadata`), shown with the sliders
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

    arg = (f', show={show!r}' if show else '') + (f', dates={str(Path(dates).resolve())!r}' if dates else '')
    cells = [cell('markdown', 'Interactive maps of moraine pyramids: zoom and pan to load details, use the sliders '
                              'to change the image of a stack, click a pixel / point for its time series (double '
                              'click: reference). Combine views with `*` (overlay) and `+` (side by side).'),
             cell('code', 'import moraine.cli as mc')]
    cells += [cell('code', f'mc.view({str(Path(pyr).resolve())!r}{arg})') for pyr in pyramids]
    nb = {'cells': cells, 'metadata': {'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + '\n')
    return str(out)
