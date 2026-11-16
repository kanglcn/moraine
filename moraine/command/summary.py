"""Metadata of results, statistics, quicklook images and notebooks with interactive maps of pyramids (drawn
with `moraine.cli.view`)."""

__all__ = ['summarize', 'quicklook', 'view', 'pyramid_levels']

import os
from pathlib import Path

import zarr

from . import UsageError
# pyramid reading lives in the CLI layer, next to the pyramids and the views
from ..cli.plot import pyramid_levels, _channel_warnings


def summarize(
    path:str,
)->dict:
    """Metadata of a result, plus value statistics for pyramids.

    Only pyramids are read: the statistics are those of all the data, computed when the pyramid was built.
    ``warnings`` lists obvious anomalies (all nan, infinite values, constant values), also of single images or
    channels of a stack. A pyramid made before moraine 0.10.0, without these statistics, or one whose command did
    not finish raises a ValueError: rebuild it.

    Parameters
    ----------
    path : str
        path to a zarr array, a pyramid made by `ras_pyramid` / `pc_pyramid`, a zarr group, a directory of
        zarr arrays or any file

    Returns
    -------
    dict
        path, kind and, for arrays and pyramids, shape and dtype (of the points for a point cloud pyramid);
        chunks for arrays; for pyramids the number of levels, ``method`` (how the levels are made), ``stats_level``
        (always 0: the statistics are those of all the data), statistics (nan_fraction, min, max, mean, std, p01,
        p50, p99; amplitude for complex data, true_fraction for bool) and ``warnings`` if any
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'{path} does not exist')
    levels = pyramid_levels(p)
    if levels:
        base = zarr.open(str(p / '0.zarr'), mode='r')
        meta = base.attrs['moraine_pyramid']          # with method and stats: checked by pyramid_levels
        kind = 'point cloud pyramid' if (p / 'bounds.toml').exists() else 'raster pyramid'
        dtype = zarr.open(str(p / 'pc.zarr'), mode='r').dtype if kind == 'point cloud pyramid' else base.dtype
        # stats_level: the level of the statistics in version 1 of the JSON output, 0 for those of all the data
        out = {'path': str(path), 'kind': kind, 'shape': list(base.shape), 'dtype': str(dtype),
               'levels': len(levels), 'method': meta['method'], 'stats_level': 0, **meta['stats']}
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


def _pyramid_input(path, pyramid:bool=True)->list:
    """Levels of a pyramid given to quicklook, view or info: a UsageError for a pyramid to rebuild or of a newer
    moraine (the ValueError of `pyramid_levels`) and, with `pyramid`, for a path that is not a pyramid."""
    try:
        levels = pyramid_levels(Path(path))
    except ValueError as e:
        raise UsageError(str(e)) from None
    if pyramid and not levels:
        raise UsageError(f'{path} is not a pyramid; build one first with `moraine ras-pyramid --ras {path} --out_dir '
                         f'<dir>` (rasters) or `moraine pc-pyramid` (point clouds)')
    return levels


def _numbers(text)->list:
    """The values of an option of numbers separated by commas or spaces, e.g. '-11,6' or '(-11, 6)': floats, and the
    words that are not numbers as they are written."""
    values = []
    for word in str(text).replace('(', ' ').replace(')', ' ').replace(',', ' ').split():
        try:
            values.append(float(word))
        except ValueError:
            values.append(word)
    return values


def _clim(clim):
    """(low, high) floats of a colour range: a text 'LOW,HIGH' or two numbers, finite and LOW < HIGH; None for None.
    UsageError otherwise, with the message of `clim` of the views."""
    from ..cli.tiles import _clim as _view_clim
    try:
        return _view_clim(_numbers(clim) if isinstance(clim, str) else clim, clim)
    except ValueError as e:
        raise UsageError(str(e)) from None


def _png_name(out)->str:
    """The PNG file of quicklook: `out` ending with .png, or with .png appended without a suffix; UsageError for
    another suffix."""
    suffix = os.path.splitext(str(out))[1]
    if not suffix:          # matplotlib would write OUT.png (or the format of its configuration) and not say so
        return f'{out}.png'
    if suffix.lower() != '.png':
        raise UsageError(f'{out}: the image name must end with .png, or have no suffix (then .png is appended)')
    return str(out)


def _pairs_input(pairs):
    """(n_pairs, 2) int64 image pairs of `pairs`, read and checked as the `image_pairs` arguments of the processing
    commands (`moraine.command.image_pairs`): a text or .npy file, or a list; UsageError otherwise, also for a file that
    is not two columns of integers."""
    from . import image_pairs
    return image_pairs(pairs)


def _notebook_name(out, overwrite:bool)->str:
    """The notebook file of view: `out` ending with .ipynb, or with .ipynb appended without a suffix; UsageError for
    another suffix and for an existing file without `overwrite`."""
    out = str(out)
    suffix = os.path.splitext(out)[1]
    if not suffix:          # Jupyter and VS Code open a file as a notebook by its suffix
        out = f'{out}.ipynb'
    elif suffix != '.ipynb':
        raise UsageError(f'{out}: the notebook name must end with .ipynb, or have no suffix (then .ipynb is '
                         f'appended)')
    if Path(out).exists() and not overwrite:
        raise UsageError(f'{out} exists; choose another name or overwrite it')
    return out


def quicklook(
    pyramid:str,
    out:str,
    index:tuple=(),
    show:str=None,
    width:int=1000,
    extent:str=None,
    clim:str=None,
    image_pairs:str=None,
)->str:
    """Save a PNG of a pyramid, drawn like `moraine.cli.view` (``view(pyramid).png(out)``).

    Only the pyramid level matching the image size is read, however large the data are. Build the pyramid
    first with `ras_pyramid` (rasters) or `pc_pyramid` (point clouds). Wrong inputs raise a UsageError (the
    command exits with 2) before anything is drawn: a path that is not a pyramid or one to rebuild, a stack without
    images, a `show` that does not fit the data (the default fits data with at most two stack dimensions), a wrong
    `extent`, `clim` or `image_pairs`, an `index` beyond the sliders, an `out` that is not a PNG.

    Parameters
    ----------
    pyramid : str
        pyramid directory made by `ras_pyramid` or `pc_pyramid`
    out : str
        output PNG file, a name ending with .png; .png is appended to a name without a suffix, another suffix is
        a UsageError
    index : tuple, default: ()
        slider values in the order of the sliders: (image,) for 3D data or 'phase' / 'intf_0' / 'intf_seq',
        (i, j) for 4D data (..., n, n), (ref, sec) for 'intf_all' / 'coh' / 'coh_abs', at most one value per slider
        and each within the range of its slider; the first images by default
    show : str, optional
        what to show of a stack, see `moraine.cli.view`; the phase for complex data, the values otherwise by
        default
    width : int, default: 1000
        image width in pixels
    extent : str, optional
        the part to draw, four numbers separated by commas: "west,south,east,north" in degrees on a web
        mercator map, "range_min,azimuth_min,range_max,azimuth_max" in pixels on the radar grid; the whole
        scene by default. A smaller part shows finer pyramid levels, down to the data
    clim : str or tuple, optional
        the colour range, two numbers "LOW,HIGH" (or (low, high)) with LOW < HIGH: the values of the first and the
        last colour, as `clim` of `moraine.cli.view` (e.g. "-11,6" for differences of a few millimetres); by default
        (-pi, pi) for phases, the 1 % - 99 % range of the data otherwise and +-(p99 - p01) for the differences of two
        images of a real stack
    image_pairs : str, optional
        image pairs of the compressed coherence shown with 'coh' / 'coh_abs' (a UsageError with another show), one
        column per pair: a text file of two columns, reference and secondary image index, as made by `moraine
        image-pairs`, read and checked like the `image_pairs` of the processing commands; all pairs of the images by
        default

    Returns
    -------
    str
        the PNG file written: `out`, with .png appended to a name without a suffix
    """
    from ..cli.tiles import view as _view, view_index, _data_extent
    out = _png_name(out)                            # UsageError
    _pyramid_input(pyramid)                         # UsageError
    pairs = None if image_pairs is None else _pairs_input(image_pairs)     # UsageError
    try:        # the inputs, before anything is drawn: their errors are a wrong command line, with their messages
        clim = _clim(clim)                          # UsageError
        if extent is not None and not isinstance(extent, (tuple, list)):
            extent = _numbers(extent)               # the words that are not numbers make the error of _data_extent
        # ValueError for a show that does not fit the data, e.g. image pairs that are not its columns
        layer = _view(str(pyramid), show=show, clim=clim, image_pairs=pairs)
        if extent is not None:
            _data_extent(extent, layer.crs)
    except ValueError as e:
        raise UsageError(str(e)) from None
    try:
        view_index([[layer]], tuple(index))
    except ValueError as e:         # more values than sliders, values outside their ranges: a wrong command line
        raise UsageError(f'--index {" ".join(str(i) for i in index)}: {e}') from None
    return layer.png(out, width=width, index=tuple(index), extent=extent)


def view(
    pyramids:list,
    out:str,
    show:str=None,
    dates:str=None,
    terrain:bool=False,
    overwrite:bool=False,
    image_pairs:str=None,
)->str:
    """Write a Jupyter notebook with an interactive map of every pyramid (`moraine.cli.view`).

    Open it in Jupyter or VS Code, choose the python environment of moraine as kernel and run all cells.
    No server or port forwarding is needed: the kernel reads the data on the machine that holds them. Wrong inputs
    raise a UsageError (the command exits with 2), as for `quicklook`: an `out` that is not a notebook name or that
    exists without `overwrite` (before anything is read), a path that is not a pyramid or a pyramid to rebuild, an
    `image_pairs` without the show 'coh' / 'coh_abs', that cannot be read or that breaks the data conventions.

    Parameters
    ----------
    pyramids : list
        pyramid directories made by `ras_pyramid` / `pc_pyramid`
    out : str
        notebook to write, a name ending with .ipynb; .ipynb is appended to a name without a suffix, another suffix is
        a UsageError
    show : str, optional
        what to show of stacks, see `moraine.cli.view`; the phase for complex data, the values otherwise by default
    dates : str, optional
        toml file with the image ``dates`` (e.g. of `load_gamma_metadata`), shown with the sliders
    terrain : bool, default: False
        3D maps over the terrain (public elevation tiles, see `moraine.cli.view`); web mercator pyramids only
    overwrite : bool, default: False
        replace an existing notebook; an existing notebook is a UsageError without it
    image_pairs : str, optional
        image pairs of the compressed coherence shown with 'coh' / 'coh_abs' (a UsageError with another show), a text
        file as for `quicklook`, given to the maps by its absolute path; all pairs of the images by default

    Returns
    -------
    str
        the notebook written: `out`, with .ipynb appended to a name without a suffix
    """
    import json
    from ..cli.tiles import _image_pairs_show
    out = Path(_notebook_name(out, overwrite))      # UsageError, before anything is read
    for pyr in pyramids:
        _pyramid_input(pyr)                         # UsageError
    try:        # the maps would raise it when the notebook runs
        _image_pairs_show(show, image_pairs)
    except ValueError as e:
        raise UsageError(str(e)) from None
    pairs = None if image_pairs is None else _pairs_input(image_pairs)     # UsageError

    def cell(kind, text):
        c = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(True)}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        return c

    arg = (f', show={show!r}' if show else '') + (f', dates={str(Path(dates).resolve())!r}' if dates else '') + \
        (', terrain=True' if terrain else '')
    if pairs is not None:       # the maps read a text file; the pairs of a list or of a .npy file are written out
        text = isinstance(image_pairs, (str, Path)) and Path(image_pairs).is_file() and \
            not str(image_pairs).endswith('.npy')
        arg += f', image_pairs={str(Path(image_pairs).resolve()) if text else pairs.tolist()!r}'
    cells = [cell('markdown', 'Interactive maps of moraine pyramids: zoom and pan to load details, use the sliders '
                              'to change the image of a stack, click a pixel / point for its time series (double '
                              'click: reference). Combine views with `*` (overlay) and `+` (side by side).'),
             cell('code', 'import moraine.cli as mc')]
    cells += [cell('code', f'mc.view({str(Path(pyr).resolve())!r}{arg})') for pyr in pyramids]
    nb = {'cells': cells, 'metadata': {'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + '\n')
    return str(out)
