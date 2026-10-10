"""Views of results: rasters and point clouds, from pyramids or arrays in memory, shown as interactive tile maps
in notebooks and as PNG images (decision 0018). ``mc.view(data)`` makes a layer; ``a * b`` overlays layers on
one map, ``a + b`` shows maps side by side."""

__all__ = ['view']

import inspect
import io
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import zarr

from .plot import pyramid_levels, _pyramid_stats, _resolve_post_proc, _kdim_ranges, _stats, _cyclic_cmap

TILE = 256               # tile edge in screen pixels
PROBE = 4                # the value under the cursor is of the nearest point within this many screen pixels
WORLD = 2 * np.pi * 6378137.0   # extent of the web mercator (EPSG:3857) plane in metres
STATS_BYTES = 64 * 2**20        # size of the sample for the default colour range
MEMORY_BYTES = 512 * 2**20      # zarr arrays that are not pyramids are read into memory up to this size
SAMPLE_CELLS = 2**18            # cells of the level used to guess the colours of a `show` function
BACKGROUND = (0xf4, 0xf4, 0xf4)  # of PNG images, where no layer has data
N_COLOURS = 255          # colours of the palette of a layer; the remaining index of the 256 is transparent
TRANSPARENT = 255        # palette index of the pixels without data (nan)

# slider names of the named `show` options (the stack dimensions otherwise: i, j)
SLIDERS = {'phase': ('image',), 'intf_0': ('image',), 'intf_seq': ('image',), 'intf_all': ('ref', 'sec'),
           'coh': ('ref', 'sec'), 'coh_abs': ('ref', 'sec')}

# the pixels of an image: centre of pixel (0, 0) in data coordinates and the signed step to the next column /
# line (negative sy: north up)
TileGeom = namedtuple('TileGeom', 'x0 y0 sx sy')


def mercator_pixel(z):
    """Screen pixel size in web mercator metres at zoom `z`."""
    return WORLD / TILE / 2 ** z


def grid_geom(z, tx, ty, origin):
    """Pixels of tile (`tx`, `ty`) at zoom `z` of a radar grid map: 2**z screen pixels per data unit, tiles
    counted from `origin` (x, y), y down."""
    s = 2.0 ** -z
    return TileGeom(origin[0] + (tx * TILE + 0.5) * s, origin[1] + (ty * TILE + 0.5) * s, s, s)


def mercator_geom(z, tx, ty):
    """Pixels of web mercator tile (`tx`, `ty`) at zoom `z` (XYZ scheme, north up), in metres."""
    s = mercator_pixel(z)
    return TileGeom(-WORLD / 2 + (tx * TILE + 0.5) * s, WORLD / 2 - (ty * TILE + 0.5) * s, s, -s)


# ---------------------------------------------------------------- helpers

def _ras_post_proc(post_proc):
    """Raster post processing function of `post_proc` (None, a name or a function)."""
    from . import plot
    if callable(post_proc):
        return post_proc
    return {None: plot._default_ras_post_proc, 'phase': plot._ras_phase_post_proc,
            'intf_0': plot._ras_inf_0_post_proc, 'intf_seq': plot._ras_inf_seq_post_proc,
            'intf_all': plot._ras_inf_all_post_proc}[post_proc]


def _pc_post_proc(post_proc):
    """Point cloud post processing function of `post_proc` (None, a name or a function)."""
    from . import plot
    if callable(post_proc):
        return post_proc
    return {None: plot._default_pc_post_proc, 'phase': plot._pc_phase_post_proc,
            'intf_0': plot._pc_inf_0_post_proc, 'intf_seq': plot._pc_inf_seq_post_proc,
            'intf_all': plot._pc_inf_all_post_proc}[post_proc]


def _lut(cmap):
    """(N_COLOURS, 4) uint8 RGBA table of a list of colours or a matplotlib colour map name: the colours of the
    palette of a layer (`_Layer.palette`), from the lowest to the highest value."""
    import matplotlib
    from matplotlib.colors import to_rgba_array
    rgba = matplotlib.colormaps[cmap](np.linspace(0, 1, N_COLOURS)) if isinstance(cmap, str) else to_rgba_array(cmap)
    rgba = rgba[np.linspace(0, len(rgba) - 1, N_COLOURS).round().astype(int)]
    return (rgba * 255).round().astype(np.uint8)


def _hex(lut):
    return ['#%02x%02x%02x' % tuple(c[:3]) for c in lut]


def png(indices, palette):
    """PNG bytes of an (h, w) uint8 image of palette `indices` with the (256, 4) uint8 RGBA `palette`; index
    `TRANSPARENT` is transparent."""
    from PIL import Image
    buf = io.BytesIO()
    # an 8 bit palette image: a quarter of the bytes of RGBA to compress, and the browser decodes it natively
    im = Image.fromarray(np.ascontiguousarray(indices))
    im.putpalette(palette[:, :3].tobytes())
    im.save(buf, format='PNG', compress_level=1, transparency=TRANSPARENT)   # speed over size
    return buf.getvalue()


def _scalar(v):
    v = np.asarray(v).ravel()[0]
    if v.dtype == bool:
        return bool(v)
    return None if np.isnan(v) else float(v)


def _disk(r):
    """Pixel offsets (dy, dx) of a disk of radius `r` pixels."""
    n = int(np.floor(r))
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
    inside = dy ** 2 + dx ** 2 <= r ** 2
    return dy[inside], dx[inside]


def _stamp(img, row, col, indices, r):
    """Draw disks of radius `r` pixels with the palette `indices` (n,) at pixels (`row`, `col`) (n,) of the
    (h, w) index image `img`, in order; transparent indices (nan values) are skipped."""
    h, w = img.shape
    for dy, dx in zip(*_disk(r)):
        rr, cc = row + dy, col + dx
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w) & (indices != TRANSPARENT)
        img[rr[ok], cc[ok]] = indices[ok]
    return img


def _dates(dates):
    """List of date strings from a list or a toml file with ``dates`` (e.g. the output of
    `load_gamma_metadata`)."""
    if dates is None:
        return []
    if isinstance(dates, (str, Path)):
        import toml
        dates = toml.load(dates)['dates']
    return [str(d) for d in dates]


class _Visible:
    """The pixels or points of a stack visible in a tile, given to a `show` function: index it like a numpy
    array of shape (lines, columns, ...) or (points, ...), e.g. ``v[..., k]``; only what is indexed is
    read."""

    def __init__(self, data, sel):
        self._data, self._sel = data, sel
        spatial = tuple(len(range(*s.indices(n))) if isinstance(s, slice) else len(s)
                        for s, n in zip(sel, data.shape))
        self.shape = spatial + tuple(data.shape[len(sel):])
        self.ndim, self.dtype = len(self.shape), data.dtype

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = next(n for n, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (self.ndim - len(key))
        nsp = len(self._sel)
        a = np.asarray(self._data[self._sel + key[nsp:]])     # the stack indices select what is read
        spatial = key[:nsp]
        if any(not (isinstance(k, slice) and k == slice(None)) for k in spatial):
            a = a[spatial]
        return a

    def __array__(self, dtype=None, copy=None):
        a = self[...]
        return a if dtype is None else a.astype(dtype)


def _coh_post_procs(tnet, kind):
    """Raster and point cloud post processing of compressed coherence (..., n_pairs) with image pairs `tnet`
    for the sliders (reference i, secondary j): the phase (`kind` 'phase') or the magnitude ('abs') of
    coherence (i, j), the conjugate for i > j, 0 phase / 1 magnitude on the diagonal, nan for pairs not in
    `tnet`."""
    def pick(read_first, read_pair, i, j):
        i, j = int(i), int(j)
        if i == j:
            first = np.asarray(read_first())
            out = np.full(first.shape, 0.0 if kind == 'phase' else 1.0)
            out[np.isnan(first)] = np.nan
            return out
        ref, sec = min(i, j), max(i, j)
        k = int(tnet.image_pairs_idx(ref=ref, sec=sec))
        if k == -1:
            return np.full(np.asarray(read_first()).shape, np.nan)
        d = np.asarray(read_pair(k))
        if kind == 'abs':
            return np.abs(d)
        return np.angle(d) if i < j else -np.angle(d)

    def ras(data_zarr, xslice, yslice, i, j):
        return pick(lambda: data_zarr[yslice, xslice, 0], lambda k: data_zarr[yslice, xslice, k], i, j)

    def pc(data_zarr, idx_array, i, j):
        return pick(lambda: data_zarr[idx_array, 0], lambda k: data_zarr[idx_array, k], i, j)
    return ras, pc


def _image_pairs(image_pairs, n_pairs):
    """TempNet of `image_pairs` (file, array or None: all pairs of the images) for `n_pairs` pairs."""
    from ..api.tnet import TempNet, nimage_from_npair
    if image_pairs is None:
        return TempNet.from_bandwidth(nimage_from_npair(n_pairs))
    pairs = np.loadtxt(image_pairs, dtype=np.int32, ndmin=2) if isinstance(image_pairs, (str, Path)) \
        else np.asarray(image_pairs, dtype=np.int32)
    if len(pairs) != n_pairs:
        raise ValueError(f'{len(pairs)} image pairs for data with {n_pairs} pairs')
    return TempNet(pairs, check_if_valid=False)


def _series_values(v, v_ref):
    """Time series of a point, relative to the reference point if `v_ref` is given: the phase (of
    v * conj(v_ref)) for complex data, the difference otherwise; nan as None."""
    v = np.asarray(v)
    if np.iscomplexobj(v):
        out = np.angle(v if v_ref is None else v * np.conj(v_ref))
    else:
        out = v.astype(np.float64) - (0 if v_ref is None else np.asarray(v_ref, np.float64))
    return [None if not np.isfinite(a) else float(a) for a in out]


def _open(data):
    """zarr array of a path, the array itself otherwise."""
    return zarr.open(str(data), mode='r') if isinstance(data, (str, Path)) else data


def _crs(x0, y0, xm, ym, res, crs, what):
    """'grid' for non negative integer coordinates, 'web_mercator' otherwise; checks a given `crs`."""
    if crs is not None:
        if crs not in ('grid', 'web_mercator'):
            raise ValueError(f"crs must be 'grid' or 'web_mercator', not {crs!r}")
        return crs
    if all(float(v).is_integer() and v >= 0 for v in (x0, y0, xm, ym)):
        return 'grid'
    if max(abs(x0), abs(xm)) <= 180 and max(abs(y0), abs(ym)) <= 90 and abs(res) < 0.01:
        raise ValueError(f'{what}: the coordinates look like longitude / latitude; the viewer needs web mercator '
                         f'coordinates (EPSG:3857), convert them with `moraine transform`')
    if max(abs(x0), abs(xm), abs(y0), abs(ym)) > WORLD / 2:
        raise ValueError(f'{what}: coordinates outside the web mercator plane (EPSG:3857)')
    return 'web_mercator'


def _show(base, show, image_pairs, sliders, dates, is_pc):
    """(name, raster function, point function, sliders [(name, count)], phase_like, default slider values)
    of a `show` option."""
    complex_data = np.iscomplexobj(np.empty(0, base.dtype))
    stack = tuple(base.shape[2:])
    if show in ('coh', 'coh_abs'):
        if len(stack) != 1:
            raise ValueError(f"show={show!r} needs compressed coherence (..., n_pairs), not shape {base.shape}")
        tnet = _image_pairs(image_pairs, stack[0])
        ras, pc = _coh_post_procs(tnet, 'phase' if show == 'coh' else 'abs')
        n = int(tnet.image_pairs[-1, -1]) + 1
        counts = [n, n]
        return show, ras, pc, list(zip(SLIDERS[show], counts)), show == 'coh', {'sec': min(1, n - 1)}
    if callable(show):
        names = list(inspect.signature(show).parameters)[1:]
        if len(names) > 2:
            raise ValueError(f'a show function takes the visible data and at most 2 sliders, not {names}')
        if sliders is not None:
            missing = [n for n in names if n not in sliders]
            if missing:
                raise ValueError(f'sliders={sliders} misses {missing}')
            counts = [int(sliders[n]) for n in names]
        elif not names or len(names) == len(stack):
            counts = list(stack)
        elif len(stack) == 1 and len(names) == 2:          # e.g. two images of a stack
            counts = [stack[0]] * 2
        elif dates:
            counts = [len(dates)] * len(names)
        else:
            raise ValueError(f'give the slider ranges, e.g. sliders={dict.fromkeys(names, 10)}: they do not '
                             f'follow from the data shape {base.shape}')

        def ras(data, xslice, yslice, *values):
            return np.asarray(show(_Visible(data, (yslice, xslice)), *values))

        def pc(data, idx_array, *values):
            return np.asarray(show(_Visible(data, (idx_array,)), *values))
        return None, ras, pc, list(zip(names, counts)), None, {}
    if show is not None and show not in SLIDERS:
        raise ValueError(f"show must be None, a function or one of {sorted(SLIDERS)}, not {show!r}")
    name, ras, pc, phase_like = _resolve_post_proc(base, show)
    ranges = _kdim_ranges(base.shape, name if isinstance(name, str) else None)
    names = SLIDERS.get(name, ('i', 'j')) if isinstance(name, str) else ('i', 'j')
    counts = [hi + 1 for _, hi in ranges.values()]
    # start on an interferogram, not on the zero phase of an image with itself
    index = {'sec': 1} if name == 'intf_all' and counts[1] > 1 else {}
    return name if isinstance(name, str) else None, _ras_post_proc(ras), \
        (_pc_post_proc(pc) if is_pc else None), list(zip(names, counts)), phase_like, index


def _colours(name, phase_like, stats, label, cmap, clim):
    """(lut, clim, colour bar label): cyclic over (-pi, pi] for phases, viridis over the 1 % - 99 % range of
    `stats()` otherwise, (0, 1) for booleans and 'coh_abs'; `cmap`, `clim` override."""
    if name == 'coh_abs':
        cmap_, clim_, label = 'viridis', (0, 1), 'coherence'
    elif phase_like:
        cmap_, clim_, label = _cyclic_cmap(), (-np.pi, np.pi), 'phase (rad)'
    else:
        st = stats()
        cmap_ = 'viridis'
        if 'true_fraction' in st:
            clim_ = (0, 1)
        elif st.get('p01') is not None and st['p01'] < st['p99']:
            clim_ = (st['p01'], st['p99'])
        else:
            clim_ = (0, 1)
    cmap_ = cmap_ if cmap is None else cmap
    clim_ = clim_ if clim is None else clim
    return _lut(cmap_), tuple(float(c) for c in clim_), label


def _array_stats(levels, max_level):
    """Statistics of the finest level of at most STATS_BYTES of a raster in memory."""
    for level in range(max_level + 1):
        a = levels(level)
        if a.nbytes <= STATS_BYTES or level == max_level:
            return _stats(np.asarray(a).ravel())


# ---------------------------------------------------------------- composition and display

class _Shown:
    """What layers, overlays and layouts share: ``*`` and ``+``, display in a notebook, PNG images and the
    state of the displayed widget."""

    def __mul__(self, other):
        return Overlay(self._layers() + _as_layers(other))

    def __add__(self, other):
        return Layout(self._panels() + _as_panels(other))

    @property
    def widget(self):
        """The `TileView` widget shown in the notebook (made on first use)."""
        if getattr(self, '_widget', None) is None:
            from .viewer import TileView
            self._widget = TileView(self._panels())
        return self._widget

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

    def __repr__(self):
        return describe(self._panels())

    @property
    def selected(self):
        """The clicked pixel or point: dict with ``label`` of the layer, ``key`` (line / column or point index),
        ``x``, ``y`` and ``point``; empty before a click."""
        return dict(self.widget.selected)

    @property
    def reference(self):
        """The reference pixel or point of the time series (double click), like `selected`; empty without."""
        return dict(self.widget.reference)

    @property
    def polygons(self):
        """Polygons drawn on the maps: lists of [x, y] vertices (range / azimuth or longitude / latitude)."""
        return self.widget.polygons

    @polygons.setter
    def polygons(self, value):
        self.widget.polygons = value

    @property
    def index(self):
        """Slider values by name; set it to change the images shown."""
        return dict(self.widget.index)

    @index.setter
    def index(self, value):
        self.widget.index = {**self.widget.index, **value}

    def png(self, path, width:int=1000, index=None, extent:tuple=None)->str:
        """Save a PNG image with axes, colour bars and titles, e.g. to check a result without a browser.

        The finest pyramid level whose cells are at least a pixel is drawn: a smaller `extent` (or a larger
        `width`) shows finer details, down to the data themselves. The title gives the extent and the level.

        Parameters
        ----------
        path : str
            output PNG file
        width : int, default: 1000
            image width in pixels
        index : dict or tuple, optional
            slider values by name, or in the order of the sliders; the first images by default
        extent : tuple, optional
            the part to draw: (west, south, east, north) longitudes and latitudes in degrees on a web mercator
            map, (range_min, azimuth_min, range_max, azimuth_max) pixels on the radar grid; the whole extent
            of the data by default (see `repr` of the view)

        Returns
        -------
        str
            the output path
        """
        return render_png(self._panels(), path, width=width, index=index, extent=extent)


def _as_layers(obj):
    if isinstance(obj, _Layer):
        return [obj]
    if isinstance(obj, Overlay):
        return list(obj.layers)
    raise TypeError(f'cannot overlay {type(obj).__name__}; overlay layers made by `view`')


def _as_panels(obj):
    if isinstance(obj, Layout):
        return [list(p) for p in obj.panels]
    return [_as_layers(obj)]


class Overlay(_Shown):
    """Layers drawn on one map, the first at the bottom (``a * b``)."""

    def __init__(self, layers):
        self.layers = list(layers)

    def _layers(self):
        return list(self.layers)

    def _panels(self):
        return [list(self.layers)]


class Layout(_Shown):
    """Maps side by side with linked zoom and pan and shared sliders (``a + b``); each map is a list of
    layers."""

    def __init__(self, panels):
        self.panels = [list(p) for p in panels]

    def _layers(self):
        raise TypeError('cannot overlay a layout of several maps; overlay the layers of one map')

    def _panels(self):
        return [list(p) for p in self.panels]


def view_sliders(panels):
    """Sliders of layers merged by name (the smallest range), in order: ([{'name', 'max'}], default values)."""
    kdims, index = {}, {}
    for layer in (layer for p in panels for layer in p):
        for k in layer.kdims:
            if k['name'] in kdims:
                kdims[k['name']]['max'] = min(kdims[k['name']]['max'], k['max'])
            else:
                kdims[k['name']] = dict(k)
                index[k['name']] = layer.default_index[k['name']]
    return list(kdims.values()), index


def view_dates(panels):
    """Dates of the first layer that has them."""
    return next((layer.dates for p in panels for layer in p if layer.dates), [])


def describe(panels)->str:
    """Text description of maps of layers, for readers of notebook outputs without a browser."""
    layers = [layer for p in panels for layer in p]
    crs = {layer.crs for layer in layers}
    lines = [f'moraine view: {len(panels)} map(s), '
             + ('radar grid (range, azimuth down)' if crs == {'grid'} else
                'web mercator (north up, over a base map)' if crs == {'web_mercator'} else f'coordinates {crs}')]
    for n, p in enumerate(panels):
        for layer in p:
            lo, hi = layer.clim
            lines.append(f'  map {n + 1}: {layer.describe()}; colours {lo:.4g} .. {hi:.4g} ({layer.bar_label})'
                         + (f'; opacity {layer.opacity}' if layer.opacity != 1 else ''))
            lines.append(f'    extent {_user_extent(layer.extent, layer.crs)}; levels 0..{layer.max_level}, '
                         f'finest cell {_cell_text(layer, layer.cell)} (png(..., extent=...) to zoom in)')
    kdims, index = view_sliders(panels)
    dates = view_dates(panels)
    for k in kdims:
        span = f' ({dates[0]} .. {dates[min(k["max"], len(dates) - 1)]})' if dates else ''
        lines.append(f'  slider {k["name"]}: 0 .. {k["max"]}{span}, now {index[k["name"]]}')
    series = [layer.label for layer in layers if layer.ts is not None]
    if series:
        lines.append(f'  time series of {", ".join(series)}: click a pixel / point; double click: reference')
    files = {layer.polygon_file for layer in layers if layer.polygon_file}
    if files:
        lines.append(f'  polygons saved to {", ".join(sorted(str(f) for f in files))}')
    return '\n'.join(lines)


# ---------------------------------------------------------------- layers

class _Layer(_Shown):
    """Common part of the layers. Cell (i, j) of level 0 is centred at (cx0 + j * rx, cy0 + i * ry) in data
    coordinates; level l keeps one cell of each 2**l x 2**l block."""

    def _layers(self):
        return [self]

    def _panels(self):
        return [[self]]

    def _setup(self, base, levels, max_level, show, image_pairs, sliders, dates, stats, series, default_ts,
               polygons, cmap, clim, opacity, size, is_pc):
        self.shape = tuple(int(n) for n in base.shape[:2])
        self.size = None
        if size is not None:
            size = tuple(int(v) for v in size)
            if len(size) != 2 or min(size) <= 0:
                raise ValueError(f'size must be (width, height) in screen pixels, not {size!r}')
            self.size = size
        self.data_shape, self.dtype = tuple(base.shape), base.dtype
        self.levels, self.max_level = levels, max_level
        self.dates = _dates(dates)
        self.polygon_file = None if polygons is None else str(polygons)
        name, self.post_proc, self.pc_post_proc, counts, phase_like, index = _show(
            base, show, image_pairs, sliders, self.dates, is_pc)
        self.show_name = name if isinstance(name, str) else ('function' if callable(show) else None)
        self.kdims = [{'name': k, 'max': int(n) - 1} for k, n in counts]
        self.default_index = {k['name']: index.get(k['name'], 0) for k in self.kdims}
        if phase_like is None:       # a show function: colours from its values on a coarse level
            phase_like, stats = self._guess(base.dtype)
        self.lut, self.clim, self.bar_label = _colours(name, phase_like, stats, self.label, cmap, clim)
        self.opacity = float(opacity)
        self.title = f'{self.label}  {self.data_shape} {base.dtype}' + \
            (f'  {self.show_name}' if self.show_name else '')
        # time series: `series` or the stack itself, one value per image; none for pairs of coherence
        self.ts = None
        if series is not None:
            self.ts = _open(series)
            if is_pc and (self.ts.ndim != 2 or self.ts.shape[0] != self.n_points):
                raise ValueError(f'series {self.ts.shape} must have shape (n_points, n) like the points '
                                 f'({self.n_points})')
            if not is_pc and (self.ts.ndim != 3 or tuple(self.ts.shape[:2]) != self.shape):
                raise ValueError(f'series {self.ts.shape} must have shape ({self.shape[0]}, {self.shape[1]}, n) '
                                 f'like the raster')
        elif self.show_name not in ('coh', 'coh_abs') and default_ts is not None and \
                default_ts.ndim == (2 if is_pc else 3):
            self.ts = default_ts

    def _guess(self, dtype):
        """(phase_like, stats) of a `show` function from its values on the finest level of at most
        SAMPLE_CELLS cells: phases for complex data shown within (-pi, pi] with negative values (magnitudes
        are not negative)."""
        level = next((lv for lv in range(self.max_level + 1)
                      if np.prod(self.levels(lv).shape[:2]) <= SAMPLE_CELLS), self.max_level)
        data = self.levels(level)
        a = np.asarray(self.post_proc(data, slice(0, data.shape[1]), slice(0, data.shape[0]),
                                      *self.default_index.values()), dtype=np.float64)
        finite = a[np.isfinite(a)]
        phase = np.iscomplexobj(np.empty(0, dtype)) and finite.size > 0 and \
            -np.pi - 1e-6 <= finite.min() < 0 and finite.max() <= np.pi + 1e-6
        return phase, lambda: _stats(a.ravel())

    def describe(self):
        return (f'{self.label}: {self.kind} {self.data_shape} {self.dtype}'
                + (f', show={self.show_name!r}' if self.show_name else '')
                + (' (mean levels)' if self.method == 'mean' else ''))

    # geometry
    @property
    def cell(self):
        """Smallest cell size of level 0 in data units."""
        return min(abs(self.rx), abs(self.ry))

    @property
    def extent(self):
        """(x0, y0, x1, y1): the cells of level 0 in data coordinates, x0 < x1, y0 < y1."""
        ny, nx = self.shape
        xs = (self.cx0 - self.rx / 2, self.cx0 + (nx - 0.5) * self.rx)
        ys = (self.cy0 - self.ry / 2, self.cy0 + (ny - 0.5) * self.ry)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def edge_origin(self):
        """Corner of cell (0, 0) in data coordinates."""
        return self.cx0 - self.rx / 2, self.cy0 - self.ry / 2

    def _slider_values(self, index):
        if isinstance(index, dict):
            return [int(index.get(k['name'], self.default_index[k['name']])) for k in self.kdims]
        return [int(i) for i in index]

    def _cell_of(self, x, y, level=0):
        """Cell (i, j) of `level` at data coordinates, None outside the data."""
        j = math.floor(((x - self.cx0) / self.rx + 0.5) / 2 ** level)
        i = math.floor(((y - self.cy0) / self.ry + 0.5) / 2 ** level)
        ny, nx = self.levels(level).shape[:2]
        return (i, j) if 0 <= i < ny and 0 <= j < nx else None

    # drawing
    def raster_values(self, geom, index=(), size=(TILE, TILE)):
        """(height, width) values of the pixels of image `geom` of `size` (width, height): the finest level
        whose cells are at least a pixel, sampled at the pixel centres; nan outside the data."""
        s = min(abs(geom.sx), abs(geom.sy))
        level = int(min(max(math.ceil(math.log2(s / self.cell) - 1e-9), 0), self.max_level))
        xc = geom.x0 + np.arange(size[0]) * geom.sx
        yc = geom.y0 + np.arange(size[1]) * geom.sy
        j = np.floor(((xc - self.cx0) / self.rx + 0.5) / 2 ** level).astype(np.int64)
        i = np.floor(((yc - self.cy0) / self.ry + 0.5) / 2 ** level).astype(np.int64)
        data = self.levels(level)
        ny, nx = data.shape[:2]
        okj, oki = (j >= 0) & (j < nx), (i >= 0) & (i < ny)
        out = np.full((size[1], size[0]), np.nan)
        if not okj.any() or not oki.any():
            return out
        j0, j1, i0, i1 = j[okj].min(), j[okj].max(), i[oki].min(), i[oki].max()
        a = np.asarray(self.post_proc(data, slice(j0, j1 + 1), slice(i0, i1 + 1), *self._slider_values(index)))
        out[np.ix_(oki, okj)] = a[np.ix_(i[oki] - i0, j[okj] - j0)]
        return out

    def render(self, geom, index=(), size=(TILE, TILE)):
        """(height, width) uint8 palette indices (`indices`, `palette`) of image `geom` of `size` (width,
        height)."""
        return self.indices(self.raster_values(geom, index, size))

    def indices(self, values):
        """Palette indices (uint8, the shape of `values`) of `values` with the colours of the colour bar: 0 to
        N_COLOURS - 1 from the lower to the upper colour limit, `TRANSPARENT` for nan."""
        lo, hi = self.clim
        values = np.asarray(values, dtype=np.float64)
        nan = np.isnan(values)
        t = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
        i = np.clip(np.floor(np.where(nan, 0, t) * N_COLOURS), 0, N_COLOURS - 1).astype(np.uint8)
        i[nan] = TRANSPARENT
        return i

    @property
    def palette(self):
        """(256, 4) uint8 RGBA colours of the palette indices: the colour map, then `TRANSPARENT`."""
        return np.vstack([self.lut, np.zeros((1, 4), np.uint8)])

    def colorize(self, values):
        """RGBA (uint8, last axis 4) of `values` with the colours of the colour bar, transparent for nan."""
        return self.palette[self.indices(values)]

    @property
    def colors(self):
        """Colour bar, hex."""
        return _hex(self.lut[::8]) + _hex(self.lut[-1:])


class RasterLayer(_Layer):
    """Raster layer of a view (see `view`)."""
    kind = 'raster'

    def __init__(self, data, show=None, dates=None, series=None, polygons=None, image_pairs=None, sliders=None,
                 bounds=None, crs=None, cmap=None, clim=None, opacity=1.0, label=None, size=None):
        if isinstance(data, (str, Path)):
            p = Path(data)
            levels = pyramid_levels(p)
            cache = {}

            def level_of(level):
                if level not in cache:
                    cache[level] = zarr.open(str(p / f'{level}.zarr'), mode='r')
                return cache[level]
            max_level, stats = levels[-1], lambda: _pyramid_stats(p, levels, STATS_BYTES)
            self.label = label or p.name
            self.method = (level_of(0).attrs.get('moraine_pyramid') or {}).get('method')
        else:
            a = data
            self.method = None
            max_level = max(int(math.floor(math.log2(max(min(a.shape[:2]), 1)))), 0)

            def level_of(level):
                return a[::2 ** level, ::2 ** level]
            stats = lambda: _array_stats(level_of, max_level)     # noqa: E731
            self.label = label or 'raster'
        base = level_of(0)
        ny, nx = base.shape[:2]
        if bounds is None:
            self.cx0, self.cy0, self.rx, self.ry = 0.0, 0.0, 1.0, 1.0    # pixel (i, j) at range j, azimuth i
            self.crs = _crs(0, 0, 1, 1, 1, crs, self.label)
        else:
            x0, y0, xm, ym = (float(b) for b in bounds)
            self.cx0, self.cy0 = x0, y0
            self.rx = (xm - x0) / (nx - 1) if nx > 1 else 1.0
            self.ry = (ym - y0) / (ny - 1) if ny > 1 else 1.0
            self.crs = _crs(min(x0, xm), min(y0, ym), max(x0, xm), max(y0, ym), self.rx, crs, self.label)
        self.n_points = None
        self._setup(base, level_of, max_level, show, image_pairs, sliders, dates, stats, series, base, polygons,
                    cmap, clim, opacity, size, is_pc=False)

    def locate(self, x, y, s):
        """Pixel under data coordinates (`x`, `y`): dict with its centre ``x``, ``y`` and ``key`` [line,
        column]; None outside. `s` (screen pixel size) is not used."""
        cell = self._cell_of(x, y)
        if cell is None:
            return None
        i, j = cell
        return {'x': self.cx0 + j * self.rx, 'y': self.cy0 + i * self.ry, 'key': [i, j]}

    def value(self, x, y, s, index=()):
        """`locate` plus the ``value`` drawn there for the slider values `index`: of the cell of the pyramid level
        shown with screen pixels of `s` data units (the pixel itself when zoomed in)."""
        found = self.locate(x, y, s)
        if found is None:
            return None
        level = _level_of(self, s)
        i, j = self._cell_of(x, y, level)
        a = self.post_proc(self.levels(level), slice(j, j + 1), slice(i, i + 1), *self._slider_values(index))
        return {**found, 'value': _scalar(a)}

    def _series_at(self, key):
        i, j = key
        return self.ts[int(i), int(j)]

    def series(self, x, y, s, ref=None):
        """`locate` plus the time series ``values`` of the pixel, relative to pixel `ref` (a ``key``) if given:
        the phase for complex data, the value otherwise. None without time series or outside."""
        if self.ts is None:
            return None
        found = self.locate(x, y, s)
        if found is None:
            return None
        v_ref = None if ref is None else self._series_at(ref)
        return {**found, 'values': _series_values(self._series_at(found['key']), v_ref), 'ref': ref}


def _pc_levels_in_memory(x, y, pc, res):
    """Levels of a point cloud rasterized in memory like `pc_pyramid`: (level function, function of the index of
    the point in each cell of a level (-1: empty), max level, cell centre (x0, y0) of cell (0, 0), shape)."""
    from ..api.coord_ import Coord
    from .plot import _next_level_idx_from_raster_of_integer
    yx = np.stack([y, x], axis=-1).astype(np.float64)
    x0, xm, y0, ym = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    nx, ny = math.ceil((xm - x0) / res) + 2, math.ceil((ym - y0) / res) + 2
    gix = Coord(x0, res, nx, y0, res, ny).coords2gixs(yx)
    ny, nx = int(gix[:, 0].max()) + 1, int(gix[:, 1].max()) + 1
    coord = Coord(x0, res, nx, y0, res, ny)
    idx = [coord.rasterize_iidx(gix)]
    for _ in range(coord.maxlevel):
        yi, xi = _next_level_idx_from_raster_of_integer(idx[-1], -1)
        idx.append(idx[-1][yi, xi])
    values = pc if np.iscomplexobj(pc) or np.issubdtype(pc.dtype, np.floating) else pc.astype(np.float64)
    cache = {}

    def level_of(level):
        if level not in cache:
            ras = values[np.maximum(idx[level], 0)]
            ras[idx[level] == -1] = np.nan
            cache[level] = ras
        return cache[level]
    return level_of, idx.__getitem__, coord.maxlevel, (x0, y0), (ny, nx)


class PointLayer(_Layer):
    """Point cloud layer of a view (see `view`)."""
    kind = 'point cloud'

    def __init__(self, data, x=None, y=None, resolution=None, show=None, dates=None, series=None, polygons=None,
                 image_pairs=None, sliders=None, crs=None, cmap=None, clim=None, opacity=1.0, label=None,
                 size=None):
        self._rtree = None
        if isinstance(data, (str, Path)) and pyramid_levels(Path(data)):
            import toml
            p = Path(data)
            levels = pyramid_levels(p)
            cache = {}

            def level_of(level):
                if level not in cache:
                    cache[level] = zarr.open(str(p / f'{level}.zarr'), mode='r')
                return cache[level]

            def idx_of(level):
                if ('idx', level) not in cache:
                    cache['idx', level] = zarr.open(str(p / f'idx_{level}.zarr'), mode='r')
                return cache['idx', level]
            max_level, stats = levels[-1], lambda: _pyramid_stats(p, levels, STATS_BYTES)
            self._x, self._y, self._pc = (zarr.open(str(p / f'{n}.zarr'), mode='r') for n in ('x', 'y', 'pc'))
            self._rtree_dir = p
            self.method = (level_of(0).attrs.get('moraine_pyramid') or {}).get('method')
            x0, y0, xm, ym = (float(v) for v in toml.load(p / 'bounds.toml')['bounds'])
            base = level_of(0)
            ny, nx = base.shape[:2]
            res = (xm - x0) / (nx - 1) if nx > 1 else ((ym - y0) / (ny - 1) if ny > 1 else 1.0)
            self.label = label or p.name
        else:
            self._x, self._y = np.asarray(_open(x)[...], np.float64), np.asarray(_open(y)[...], np.float64)
            self._pc = np.asarray(_open(data)[...])
            if not (self._x.shape == self._y.shape == self._pc.shape[:1]):
                raise ValueError(f'x {self._x.shape}, y {self._y.shape} and data {self._pc.shape} do not match')
            if resolution is None:
                if not (np.all(self._x == np.round(self._x)) and np.all(self._y == np.round(self._y))):
                    raise ValueError('give `resolution` (cell size of the rasterized points): the coordinates '
                                     'are not on an integer grid')
                resolution = 1
            res = float(resolution)
            level_of, idx_of, max_level, (x0, y0), _ = _pc_levels_in_memory(self._x, self._y, self._pc, res)
            stats = lambda: _array_stats(level_of, max_level)     # noqa: E731
            self._rtree_dir = None
            self.method = None
            base = level_of(0)
            xm, ym = x0 + (base.shape[1] - 1) * res, y0 + (base.shape[0] - 1) * res
            self.label = label or 'points'
        self.cx0, self.cy0, self.rx, self.ry = x0, y0, res, res
        self.idx_of = idx_of         # the point drawn in each cell of a level
        self.crs = _crs(x0, y0, xm, ym, res, crs, self.label)
        self.n_points = int(self._pc.shape[0])
        self._setup(base, level_of, max_level, show, image_pairs, sliders, dates, stats, series, self._pc,
                    polygons, cmap, clim, opacity, size, is_pc=True)
        self.data_shape, self.dtype = tuple(self._pc.shape), self._pc.dtype     # the points, not their raster
        self.title = f'{self.label}  {self.data_shape} {self.dtype}' + \
            (f'  {self.show_name}' if self.show_name else '')

    def points_in(self, bounds):
        """Indices of the points within `bounds` (x0, y0, xm, ym) in data coordinates."""
        if self._rtree is None:     # built on the first query: overviews do not read the coordinates
            if self._rtree_dir is not None:
                from .plot import _LazyRtree
                self._rtree = _LazyRtree(self._rtree_dir)
            else:
                from ..api.rtree import HilbertRtree
                self._rtree = HilbertRtree.build(self._x, self._y, page_size=512)
        return self._rtree.bbox_query(bounds, self._x, self._y)

    def point_radius(self, s):
        """Radius in screen pixels of the points drawn with screen pixels of `s` data units."""
        return max(1.0, 0.4 * self.cell / s)

    def render(self, geom, index=(), size=(TILE, TILE)):
        """(height, width) uint8 palette indices of image `geom` of `size` (width, height): the rasterized
        points, or the points as disks when a cell of level 0 is larger than a pixel."""
        s = min(abs(geom.sx), abs(geom.sy))
        if s >= self.cell:
            return self.indices(self.raster_values(geom, index, size))
        img = np.full((size[1], size[0]), TRANSPARENT, np.uint8)
        r = self.point_radius(s)
        # edges of the image, extended by the point radius
        ex = (geom.x0 - geom.sx / 2, geom.x0 + (size[0] - 0.5) * geom.sx)
        ey = (geom.y0 - geom.sy / 2, geom.y0 + (size[1] - 0.5) * geom.sy)
        pad = r * s
        idx = self.points_in((min(ex) - pad, min(ey) - pad, max(ex) + pad, max(ey) + pad))
        if len(idx) == 0:
            return img
        px, py = self._x[idx], self._y[idx]
        col = np.floor((px - ex[0]) / geom.sx).astype(np.int64)
        row = np.floor((py - ey[0]) / geom.sy).astype(np.int64)
        values = self.pc_post_proc(self._pc, idx, *self._slider_values(index))
        return _stamp(img, row, col, self.indices(values), r)

    def locate(self, x, y, s):
        """The point drawn at data coordinates (`x`, `y`) with screen pixels of `s` data units: dict with its
        coordinates ``x``, ``y`` and index ``point`` (also ``key``); None without a point. Where the points are
        rasterized (`s` at least a cell) it is the point of the cell under the cursor at the level shown; where
        they are drawn one by one, the nearest point within a few screen pixels (at least half a cell)."""
        if s >= self.cell:
            level = _level_of(self, s)
            cell = self._cell_of(x, y, level)
            if cell is None:
                return None
            i = int(self.idx_of(level)[cell])
            if i < 0:
                return None
            return {'point': i, 'key': i, 'x': float(self._x[i]), 'y': float(self._y[i])}
        w = max(0.5 * self.cell, PROBE * s)
        idx = self.points_in((x - w, y - w, x + w, y + w))
        if len(idx) == 0:
            return None
        px, py = self._x[idx], self._y[idx]
        k = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
        i = int(idx[k])
        return {'point': i, 'key': i, 'x': float(px[k]), 'y': float(py[k])}

    def value(self, x, y, s, index=()):
        """`locate` plus the ``value`` drawn there for the slider values `index`: of the cell of the pyramid level
        shown where the points are rasterized, of the point itself where they are drawn one by one."""
        found = self.locate(x, y, s)
        if found is None:
            return None
        if s >= self.cell:
            level = _level_of(self, s)
            i, j = self._cell_of(x, y, level)
            a = self.post_proc(self.levels(level), slice(j, j + 1), slice(i, i + 1), *self._slider_values(index))
        else:
            a = self.pc_post_proc(self._pc, np.array([found['point']]), *self._slider_values(index))
        return {**found, 'value': _scalar(a)}

    def series(self, x, y, s, ref=None):
        """`locate` plus the time series ``values`` of the point, relative to point `ref` if given: the phase
        for complex data, the value otherwise. None without time series or point."""
        if self.ts is None:
            return None
        found = self.locate(x, y, s)
        if found is None:
            return None
        v_ref = None if ref is None else self.ts[int(ref)]
        return {**found, 'values': _series_values(self.ts[found['point']], v_ref), 'ref': ref}


# ---------------------------------------------------------------- PNG images

def _panel_geom(extent, crs, width):
    """(geom, size) of an image of `extent` about `width` pixels wide (at most 1:4)."""
    x0, y0, x1, y1 = extent
    wu, hu = x1 - x0, y1 - y0
    s = max(wu / width, hu / (4 * width), wu / (4 * width) if hu > wu else 0)
    size = (max(1, math.ceil(wu / s)), max(1, math.ceil(hu / s)))
    if crs == 'grid':
        return TileGeom(x0 + s / 2, y0 + s / 2, s, s), size
    return TileGeom(x0 + s / 2, y1 - s / 2, s, -s), size


_RADIUS = WORLD / 2 / np.pi


def _lonlat_to_merc(lon, lat):
    return np.radians(lon) * _RADIUS, _RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _merc_to_lonlat(x, y):
    return np.degrees(x / _RADIUS), np.degrees(2 * np.arctan(np.exp(y / _RADIUS)) - np.pi / 2)


def _data_extent(extent, crs):
    """(x0, y0, x1, y1) in data coordinates of a user extent: (west, south, east, north) degrees on web
    mercator, (range_min, azimuth_min, range_max, azimuth_max) pixels on the radar grid."""
    e = [float(v) for v in extent]
    if len(e) != 4 or not (e[0] < e[2] and e[1] < e[3]):
        what = '(west, south, east, north)' if crs == 'web_mercator' else '(range_min, azimuth_min, range_max, azimuth_max)'
        raise ValueError(f'extent must be {what} with the minimum before the maximum, got {tuple(extent)}')
    if crs == 'web_mercator':
        x0, y0 = _lonlat_to_merc(e[0], e[1])
        x1, y1 = _lonlat_to_merc(e[2], e[3])
        return float(x0), float(y0), float(x1), float(y1)
    return tuple(e)


def _user_extent(extent, crs)->str:
    """Text of an extent in data coordinates as the user gives it."""
    x0, y0, x1, y1 = extent
    if crs == 'web_mercator':
        w, s = _merc_to_lonlat(x0, y0)
        e, n = _merc_to_lonlat(x1, y1)
        return f'lon {w:.4f} .. {e:.4f}, lat {s:.4f} .. {n:.4f}'
    return f'range {x0:.6g} .. {x1:.6g}, azimuth {y0:.6g} .. {y1:.6g}'


def _cell_text(layer, cell)->str:
    """Size of a cell of `layer` in data units: metres on the ground for web mercator, pixels on the grid."""
    if layer.crs == 'web_mercator':
        x0, y0, x1, y1 = layer.extent
        lat = _merc_to_lonlat(0, (y0 + y1) / 2)[1]
        return f'{cell * np.cos(np.radians(lat)):.3g} m'
    return f'{cell:.3g} px'


def _level_of(layer, pixel):
    """Pyramid level drawn for pixels of size `pixel` (data units), as in `raster_values`."""
    return int(min(max(math.ceil(math.log2(pixel / layer.cell) - 1e-9), 0), layer.max_level))


def render_png(panels, path, width=1000, index=None, extent=None):
    """Save a PNG of maps of layers (see `_Shown.png`)."""
    from matplotlib.figure import Figure        # no pyplot: works with any backend, also without a display
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    layers = [layer for p in panels for layer in p]
    crs = {layer.crs for layer in layers}
    if len(crs) > 1:
        raise ValueError(f'layers in different coordinates {sorted(crs)} cannot be shown together')
    crs = crs.pop()
    kdims, values = view_sliders(panels)
    if isinstance(index, dict):
        values.update(index)
    elif index:
        values.update(dict(zip([k['name'] for k in kdims], index)))
    dates = view_dates(panels)
    if extent is None:
        ext = np.array([layer.extent for layer in layers])
        extent = (ext[:, 0].min(), ext[:, 1].min(), ext[:, 2].max(), ext[:, 3].max())
    else:
        extent = _data_extent(extent, crs)
    per = max(200, width // len(panels))
    geom, size = _panel_geom(extent, crs, per)
    img_extent = (geom.x0 - geom.sx / 2, geom.x0 + (size[0] - 0.5) * geom.sx,
                  geom.y0 + (size[1] - 0.5) * geom.sy, geom.y0 - geom.sy / 2)   # left, right, bottom, top

    n_bars = max(len(p) for p in panels)
    fig_w = width / 100
    fig_h = max(2.5, fig_w / len(panels) * size[1] / size[0] + 1.2)
    fig = Figure(figsize=(fig_w + 0.6 * n_bars, fig_h), dpi=100)
    axes = fig.subplots(1, len(panels), squeeze=False)
    sliders = ', '.join(f'{k["name"]}={values[k["name"]]}' + (f' ({dates[values[k["name"]]]})'
                        if values[k['name']] < len(dates) else '') for k in kdims)
    for ax, p in zip(axes[0], panels):
        rgb = np.empty((size[1], size[0], 3))
        rgb[:] = np.array(BACKGROUND) / 255
        for layer in p:
            rgba = layer.palette[layer.render(geom, values, size)] / 255
            alpha = rgba[..., 3:] * layer.opacity
            rgb = rgba[..., :3] * alpha + rgb * (1 - alpha)
        ax.imshow(rgb, extent=img_extent, interpolation='nearest', aspect='equal')
        levels = ', '.join(f'level {_level_of(layer, abs(geom.sx))} of 0..{layer.max_level}' for layer in p)
        ax.set_title('  |  '.join(layer.title for layer in p) + (f'\n{sliders}' if sliders else '')
                     + f'\n{_user_extent(extent, crs)}; {levels}', fontsize=9)
        if crs == 'grid':
            ax.set_xlabel('range'); ax.set_ylabel('azimuth')
        else:
            radius = WORLD / 2 / np.pi
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{np.degrees(v / radius):.4f}'))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f'{np.degrees(2 * np.arctan(np.exp(v / radius)) - np.pi / 2):.4f}'))
            ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
            ax.xaxis.set_major_locator(MaxNLocator(4))     # longitudes are long labels
        for layer in p:
            cmap = ListedColormap(layer.lut[:, :3] / 255)
            fig.colorbar(ScalarMappable(Normalize(*layer.clim), cmap), ax=ax, label=layer.bar_label,
                         fraction=0.04, pad=0.02)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    return str(path)


# ---------------------------------------------------------------- the entry point

def view(
    data,
    x=None,
    y=None,
    show=None,
    dates=None,
    series=None,
    polygons:str=None,
    cmap=None,
    clim:tuple=None,
    opacity:float=1.0,
    label:str=None,
    size:tuple=None,
    bounds:tuple=None,
    resolution:float=None,
    image_pairs:str=None,
    sliders:dict=None,
    crs:str=None,
):
    """View a raster or point cloud: an interactive map in a notebook (the last expression of a cell) or a PNG
    (``.png(path)``); ``a * b`` overlays views on one map, ``a + b`` shows maps side by side with linked zoom and
    pan and shared sliders.

    On the map: zoom and pan read only what is on screen; sliders choose the image of a stack; the cursor shows
    the values of all layers; click a pixel or point to plot its time series, double click one to make it the
    reference of the time series; draw polygons with the polygon button. In Python, ``.selected``,
    ``.reference``, ``.polygons`` and ``.index`` (slider values) follow the map. Point clouds are drawn as
    individual points when zoomed in; web mercator coordinates are drawn north up over a base map.

    Parameters
    ----------
    data : str or np.ndarray
        a pyramid directory made by `ras_pyramid` or `pc_pyramid`; a raster, (nlines, width[, n[, m]]), as
        array or zarr path (read into memory, at most 512 MiB); or point data, (n_points[, n[, m]]), with `x`
        and `y`
    x : str or np.ndarray, optional
        x coordinate (range or web mercator x) of point data that is not a pyramid, (n_points,)
    y : str or np.ndarray, optional
        y coordinate (azimuth or web mercator y) of point data that is not a pyramid, (n_points,)
    show : str or callable, optional
        what to show of a stack: 'phase', 'intf_0' (interferograms with the first image), 'intf_seq'
        (sequential interferograms), 'intf_all' (any pair, sliders ref and sec), 'coh' / 'coh_abs' (phase /
        magnitude of compressed coherence (..., n_pairs), sliders ref and sec), or a function
        ``f(v, *sliders)`` of the visible stack ``v`` (index it like a numpy array, e.g. ``v[..., ref]``) whose
        other arguments are sliders named like them, e.g. ``lambda v, ref, sec: np.angle(v[..., ref] *
        np.conj(v[..., sec]))``; the phase for complex data, the values otherwise by default
    dates : list or str, optional
        date of each image, or a toml file with ``dates`` (e.g. of `load_gamma_metadata`), shown with the
        sliders and on the time axis
    series : str or np.ndarray, optional
        time series plotted for a clicked pixel or point: (nlines, width, n) for rasters, (n_points, n) in the
        order of the points; the stack itself by default
    polygons : str, optional
        GeoJSON file of the polygons drawn on the map: read if it exists, written on every change; use it
        with `polygon_mask`
    cmap : str or list, optional
        matplotlib colour map name or list of colours; cyclic for phases, viridis otherwise by default
    clim : tuple, optional
        (min, max) of the colours; (-pi, pi) for phases, the 1 % - 99 % range otherwise by default
    opacity : float, default: 1.0
        opacity of the layer, 0 - 1
    label : str, optional
        name of the layer; the pyramid directory name by default
    size : tuple, optional
        (width, height) of the map in screen pixels; by default the map takes the width of the notebook (at most
        700 pixels high) with the aspect of the scene (at most 1:4). Drag the lower right corner of a map to
        resize it
    bounds : tuple, optional
        raster only: (x0, y0, xm, ym), coordinates of the centres of the first and the last pixel; pixel
        (i, j) at range j, azimuth i by default
    resolution : float, optional
        point data only: cell size of the rasterized points; 1 for integer coordinates
    image_pairs : str, optional
        image pair file (`image_pairs`) of compressed coherence for 'coh' / 'coh_abs'; all pairs by default
    sliders : dict, optional
        number of values of each slider of a `show` function, e.g. {'ref': 17, 'sec': 17}; from the stack
        shape or `dates` by default
    crs : str, optional
        'grid' (radar grid, azimuth down) or 'web_mercator' (metres, north up); from the coordinates by default

    Returns
    -------
    layer
        displayed in a notebook; combine with ``*`` and ``+``; ``.png(path)`` saves an image
    """
    kw = dict(show=show, dates=dates, series=series, polygons=polygons, image_pairs=image_pairs, sliders=sliders,
              crs=crs, cmap=cmap, clim=clim, opacity=opacity, label=label, size=size)
    if x is not None or y is not None:
        if x is None or y is None:
            raise ValueError('give both `x` and `y` for point data')
        if bounds is not None:
            raise ValueError('`bounds` is for rasters; points are placed by `x` and `y`')
        return PointLayer(data, x=x, y=y, resolution=resolution, **kw)
    if resolution is not None:
        raise ValueError('`resolution` is for point data given with `x` and `y`')
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f'{data} does not exist')
        if pyramid_levels(p):
            if (p / 'bounds.toml').exists():
                if bounds is not None:
                    raise ValueError('`bounds` is for rasters; a point cloud pyramid has its coordinates')
                return PointLayer(p, **kw)
            return RasterLayer(p, bounds=bounds, **kw)
        z = zarr.open(str(p), mode='r')
        if not isinstance(z, zarr.Array) or z.ndim < 2:
            raise ValueError(f'{data} is neither a pyramid nor a raster; point data need `x` and `y`')
        if z.nbytes > MEMORY_BYTES:
            raise ValueError(f'{data} ({z.nbytes / 2**30:.1f} GiB) is too large to read; build a pyramid with '
                             f'`moraine ras-pyramid --ras {data} --out_dir <dir>` and view that')
        data = z[...]
        kw['label'] = label or p.name
    a = np.asarray(data)
    if a.ndim < 2:
        raise ValueError(f'data of shape {a.shape}: a raster needs 2 or more dimensions, point data need `x` and `y`')
    return RasterLayer(a, bounds=bounds, **kw)
