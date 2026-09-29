"""Tile viewer of pyramids for notebooks: a Leaflet map whose tiles are rendered by the kernel and sent through
the notebook channel (decision 0015)."""

__all__ = ['TileView', 'tile_view']

import io
from pathlib import Path

import numpy as np
import zarr

from .summary import pyramid_levels, _resolve_post_proc, _colours, _kdim_ranges, _frame_size

try:
    import anywidget
    import traitlets
except ImportError:      # optional dependency: `pip install moraine[view]`
    anywidget = None

TILE = 256               # tile edge in screen pixels
MAX_ZOOM = 4             # at most 2**MAX_ZOOM screen pixels per cell of level 0
PROBE = 4                # the value under the cursor is of the nearest point within this many screen pixels
WORLD = 2 * np.pi * 6378137.0   # extent of the web mercator (EPSG:3857) plane in metres


def _mercator_pixel(z):
    """Screen pixel size in web mercator metres at zoom `z`."""
    return WORLD / TILE / 2 ** z


def _ras_post_proc(post_proc):
    """Raster post processing function of `post_proc` (None, a name or a function)."""
    from ..cli import plot
    if callable(post_proc):
        return post_proc
    return {None: plot._default_ras_post_proc, 'phase': plot._ras_phase_post_proc,
            'intf_0': plot._ras_inf_0_post_proc, 'intf_seq': plot._ras_inf_seq_post_proc,
            'intf_all': plot._ras_inf_all_post_proc}[post_proc]


def _pc_post_proc(post_proc):
    """Point cloud post processing function of `post_proc` (None, a name or a function)."""
    from ..cli import plot
    if callable(post_proc):
        return post_proc
    return {None: plot._default_pc_post_proc, 'phase': plot._pc_phase_post_proc,
            'intf_0': plot._pc_inf_0_post_proc, 'intf_seq': plot._pc_inf_seq_post_proc,
            'intf_all': plot._pc_inf_all_post_proc}[post_proc]


def _lut(cmap):
    """(256, 4) uint8 RGBA table of a list of colours or a matplotlib colour map name."""
    import matplotlib
    from matplotlib.colors import to_rgba_array
    rgba = matplotlib.colormaps[cmap](np.linspace(0, 1, 256)) if isinstance(cmap, str) else to_rgba_array(cmap)
    rgba = rgba[np.linspace(0, len(rgba) - 1, 256).round().astype(int)]
    return (rgba * 255).round().astype(np.uint8)


def _hex(lut):
    return ['#%02x%02x%02x' % tuple(c[:3]) for c in lut]


def _png(rgba):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba)).save(buf, format='PNG', compress_level=1)   # speed over size
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


def _stamp(rgba, row, col, colours, r):
    """Draw disks of radius `r` pixels with `colours` (n, 4) at pixels (`row`, `col`) (n,) of `rgba`, in
    order; transparent colours (nan values) are skipped."""
    h, w = rgba.shape[:2]
    for dy, dx in zip(*_disk(r)):
        rr, cc = row + dy, col + dx
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w) & (colours[:, 3] > 0)
        rgba[rr[ok], cc[ok]] = colours[ok]
    return rgba


if anywidget is not None:
    class TileView(anywidget.AnyWidget):
        """Interactive map of a pyramid for Jupyter / VS Code notebooks: zoom and pan read only the tiles on
        screen, sliders choose the image of a stack, the cursor shows the value under it.

        Point clouds are drawn as their pyramid rasters when zoomed out and as individual points when a cell
        of the finest raster is larger than a screen pixel. Point clouds in web mercator coordinates are
        drawn north up over a base map (satellite images, CARTO or OpenStreetMap, loaded by the browser from
        the internet). No server or port forwarding is needed; the kernel reads the data on the machine that
        holds them.

        Parameters
        ----------
        pyramid : str
            pyramid directory made by `ras_pyramid`, or by `pc_pyramid` from points on the radar grid
            (non negative integer coordinates, e.g. `gix`) or in web mercator coordinates (EPSG:3857, metres)
        post_proc : str, optional
            how to show a stack of images: 'phase', 'intf_0' (interferograms with the first image),
            'intf_seq' (sequential interferograms) or 'intf_all' (any pair); the phase for complex data by
            default
        """
        _esm = Path(__file__).with_name('tile_view.js')
        _css = Path(__file__).with_name('tile_view.css')

        # crs 'grid': the map is in cells of level 0, cell (i, j) covers [j, j + 1) x [i, i + 1), and a map
        # position u is at data coordinate origin + u * res (x: range, y: azimuth down).
        # crs 'web_mercator': Leaflet's EPSG:3857 with standard XYZ tiles, data coordinates in metres.
        title = traitlets.Unicode().tag(sync=True)
        crs = traitlets.Unicode('grid').tag(sync=True)
        shape = traitlets.List().tag(sync=True)          # [ny, nx] cells of level 0
        origin = traitlets.List([-0.5, -0.5]).tag(sync=True)
        res = traitlets.Float(1.0).tag(sync=True)
        extent = traitlets.List().tag(sync=True)         # [x0, y0, x1, y1] of the cells in data coordinates
        frame = traitlets.List().tag(sync=True)          # [width, height] of the map in screen pixels
        max_zoom = traitlets.Int(MAX_ZOOM).tag(sync=True)
        zoom = traitlets.Int().tag(sync=True)            # zoom showing the whole scene in the frame
        colors = traitlets.List().tag(sync=True)         # colour bar, hex
        clim = traitlets.List().tag(sync=True)
        label = traitlets.Unicode().tag(sync=True)
        axis_labels = traitlets.List(['range', 'azimuth']).tag(sync=True)
        kdims = traitlets.List().tag(sync=True)          # sliders [{'name': 'i', 'max': n}, ...]
        index = traitlets.List().tag(sync=True)          # values of the sliders

        def __init__(self, pyramid, post_proc=None, **kwargs):
            p = Path(pyramid)
            levels = pyramid_levels(p)
            if not levels:
                raise ValueError(f'{pyramid} is not a pyramid; build one first with `moraine ras-pyramid` '
                                 f'(rasters) or `moraine pc-pyramid` (point clouds)')
            base = zarr.open(str(p / '0.zarr'), mode='r')
            ny, nx = base.shape[:2]
            self._pc = (p / 'bounds.toml').exists()
            crs, origin, res = 'grid', [-0.5, -0.5], 1.0     # raster: pixel i centred at coordinate i
            axis_labels = ['range', 'azimuth']
            if self._pc:
                import toml
                x0, y0, xm, ym = (float(v) for v in toml.load(p / 'bounds.toml')['bounds'])
                # cell centres of level 0 at x0 + j * res (`pc_pyramid`)
                res = (xm - x0) / (nx - 1) if nx > 1 else ((ym - y0) / (ny - 1) if ny > 1 else 1.0)
                origin = [x0 - res / 2, y0 - res / 2]
                if not all(v.is_integer() and v >= 0 for v in (x0, y0, xm, ym)):
                    if max(abs(x0), abs(xm)) <= 180 and max(abs(y0), abs(ym)) <= 90 and res < 0.01:
                        raise ValueError(f'{pyramid}: the coordinates look like longitude / latitude; the tile '
                                         f'viewer needs web mercator coordinates (EPSG:3857), convert them with '
                                         f'`moraine transform` and rebuild the pyramid')
                    if max(abs(x0), abs(xm), abs(y0), abs(ym)) > WORLD / 2:
                        raise ValueError(f'{pyramid}: coordinates outside the web mercator plane (EPSG:3857)')
                    crs, axis_labels = 'web_mercator', ['longitude', 'latitude']
            extent = [origin[0], origin[1], origin[0] + nx * res, origin[1] + ny * res]
            post_proc, ras_proc, pc_proc, phase_like = _resolve_post_proc(base, post_proc)
            cmap, clim, label = _colours(p, phase_like)
            ranges = _kdim_ranges(base.shape, post_proc if isinstance(post_proc, str) else None)

            self._dir = p
            self._max_level = levels[-1]
            self._post_proc = _ras_post_proc(ras_proc)
            self._pc_post_proc = _pc_post_proc(pc_proc) if self._pc else None
            self._rtree = None
            self._lut = _lut(cmap)
            self._clim = tuple(float(c) for c in clim) if np.all(np.isfinite(clim)) else (0.0, 1.0)
            self._zarrs = {}
            title = f'{p.name}  {tuple(base.shape)} {base.dtype}' + \
                (f'  {post_proc}' if isinstance(post_proc, str) else '')
            frame = _frame_size(nx, ny)
            # zoom showing the whole scene: screen pixels per cell of level 0 at zoom 0, 1 / res for mercator
            per_cell = 1 if crs == 'grid' else res / _mercator_pixel(0)
            zoom = int(np.floor(np.log2(min(frame[0] / nx, frame[1] / ny) / per_cell)))
            max_zoom = int(np.floor(np.log2(2 ** MAX_ZOOM / per_cell)))    # 2**MAX_ZOOM pixels per cell
            super().__init__(title=title, crs=crs, shape=[ny, nx], origin=[float(o) for o in origin],
                             res=float(res), extent=[float(e) for e in extent], axis_labels=axis_labels,
                             frame=list(frame), zoom=min(zoom, max_zoom), max_zoom=max_zoom,
                             colors=_hex(self._lut[::8]) + _hex(self._lut[-1:]), clim=list(self._clim),
                             label=label, kdims=[{'name': k, 'max': int(hi)} for k, (_, hi) in ranges.items()],
                             index=[0] * len(ranges), **kwargs)
            self.on_msg(self._on_msg)

        def _zarr(self, name):
            if name not in self._zarrs:
                self._zarrs[name] = zarr.open(str(self._dir / f'{name}.zarr'), mode='r')
            return self._zarrs[name]

        def _points_in(self, bounds):
            """Indices of the points within `bounds` (x0, y0, xm, ym) in data coordinates."""
            if self._rtree is None:     # built on the first query: overviews do not read the coordinates
                from ..cli.plot import _LazyRtree
                self._rtree = _LazyRtree(self._dir)
            return self._rtree.bbox_query(bounds, self._zarr('x'), self._zarr('y'))

        def tile_values(self, z, tx, ty, index=()):
            """Raster values of tile (`tx`, `ty`) at zoom `z`: 2**z screen pixels per cell of level 0, tiles of
            256 screen pixels counted from the first line and column. nan outside the scene."""
            # the pyramid level with at least one cell per screen pixel, then a stride if the coarsest
            # level is still too fine
            level = min(max(-z, 0), self._max_level)
            m = TILE * 2 ** max(-z - level, 0) // 2 ** max(z, 0)    # tile edge in cells of `level`
            k = max(m // TILE, 1)
            out = np.full((m // k, m // k), np.nan)
            data = self._zarr(level)
            ny, nx = data.shape[:2]
            if tx < 0 or ty < 0 or tx * m >= nx or ty * m >= ny:
                return out
            xs = slice(tx * m, min((tx + 1) * m, nx), k)
            ys = slice(ty * m, min((ty + 1) * m, ny), k)
            a = np.asarray(self._post_proc(data, xs, ys, *index))
            out[:a.shape[0], :a.shape[1]] = a
            return out

        def pixel_size(self, z):
            """Screen pixel size in data coordinates at zoom `z`."""
            return self.res / 2 ** z if self.crs == 'grid' else _mercator_pixel(z)

        def point_radius(self, z):
            """Radius in screen pixels of the points drawn at zoom `z`."""
            return max(1.0, 0.4 * self.res / self.pixel_size(z))

        def mercator_values(self, z, tx, ty, index=()):
            """Raster values of web mercator tile (`tx`, `ty`) at zoom `z` (XYZ scheme, north up), 256 x 256,
            nan outside the scene: the finest level whose cells are at least a screen pixel, sampled at the
            pixel centres."""
            m = WORLD / 2 ** z                                   # tile edge in metres
            s = m / TILE
            xc = -WORLD / 2 + tx * m + (np.arange(TILE) + 0.5) * s
            yc = WORLD / 2 - ty * m - (np.arange(TILE) + 0.5) * s
            level = int(min(max(np.ceil(np.log2(s / self.res) - 1e-9), 0), self._max_level))
            x0, y0 = self.origin[0] + self.res / 2, self.origin[1] + self.res / 2     # centre of cell (0, 0)
            j = np.floor(((xc - x0) / self.res + 0.5) / 2 ** level).astype(np.int64)
            i = np.floor(((yc - y0) / self.res + 0.5) / 2 ** level).astype(np.int64)
            data = self._zarr(level)
            ny, nx = data.shape[:2]
            okj, oki = (j >= 0) & (j < nx), (i >= 0) & (i < ny)
            out = np.full((TILE, TILE), np.nan)
            if not okj.any() or not oki.any():
                return out
            j0, j1, i0, i1 = j[okj].min(), j[okj].max(), i[oki].min(), i[oki].max()
            a = np.asarray(self._post_proc(data, slice(j0, j1 + 1), slice(i0, i1 + 1), *index))
            out[np.ix_(oki, okj)] = a[np.ix_(i[oki] - i0, j[okj] - j0)]
            return out

        def _points_rgba(self, west, top, pixel, r, down, index):
            """256 x 256 RGBA tile with the points as disks of radius `r` pixels; the tile starts at data
            coordinates (`west`, `top`) and has pixels of `pixel` data units, rows going `down` (y grows)
            or up (north up)."""
            rgba = np.zeros((TILE, TILE, 4), np.uint8)
            pad = r * pixel
            span = TILE * pixel
            y0, y1 = (top, top + span) if down else (top - span, top)
            idx = self._points_in((west - pad, y0 - pad, west + span + pad, y1 + pad))
            if len(idx) == 0:
                return rgba
            x, y = self._zarr('x')[idx], self._zarr('y')[idx]
            col = np.floor((x - west) / pixel).astype(np.int64)
            row = np.floor(((y - top) if down else (top - y)) / pixel).astype(np.int64)
            colours = self.colorize(np.asarray(self._pc_post_proc(self._zarr('pc'), idx, *index), np.float64))
            return _stamp(rgba, row, col, colours, r)

        def tile_rgba(self, z, tx, ty, index=()):
            """(h, w, 4) uint8 RGBA image of a tile: the raster (see `tile_values` and `mercator_values`), or
            for point clouds zoomed in until a cell of level 0 is larger than a screen pixel the points as
            disks on a 256 x 256 transparent tile."""
            pixel = self.pixel_size(z)
            if not (self._pc and pixel < self.res):
                values = self.tile_values(z, tx, ty, index) if self.crs == 'grid' else \
                    self.mercator_values(z, tx, ty, index)
                return self.colorize(values)
            r = self.point_radius(z)
            if self.crs == 'grid':
                return self._points_rgba(self.origin[0] + tx * TILE * pixel, self.origin[1] + ty * TILE * pixel,
                                         pixel, r, True, index)
            return self._points_rgba(-WORLD / 2 + tx * TILE * pixel, WORLD / 2 - ty * TILE * pixel,
                                     pixel, r, False, index)

        def colorize(self, values):
            """RGBA (uint8, last axis 4) of `values` with the colours of the colour bar, transparent for nan."""
            lo, hi = self._clim
            t = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
            nan = np.isnan(t)
            i = np.clip(np.floor(np.where(nan, 0, t) * 256), 0, 255).astype(np.intp)
            rgba = self._lut[i]
            rgba[nan] = 0
            return rgba

        def value(self, u, v, z=0, index=()):
            """Value under map position (`u`, `v`) at zoom `z`: cells of level 0 for the radar grid, web
            mercator metres otherwise. A dict with ``value`` and the data coordinates ``x``, ``y``; for point
            clouds the nearest point within a few screen pixels (at least half a cell) and its index
            ``point``. None outside the scene or without a point."""
            if self.crs == 'grid':
                ny, nx = self.shape
                if not (0 <= u < nx and 0 <= v < ny):
                    return None
                if not self._pc:
                    j, i = int(u), int(v)
                    a = self._post_proc(self._zarr(0), slice(j, j + 1), slice(i, i + 1), *index)
                    return {'x': j, 'y': i, 'value': _scalar(a)}
                x, y = self.origin[0] + u * self.res, self.origin[1] + v * self.res
            else:
                x0, y0, x1, y1 = self.extent
                if not (x0 <= u < x1 and y0 <= v < y1):
                    return None
                x, y = u, v
            w = max(0.5 * self.res, PROBE * self.pixel_size(z))    # search half width in data units
            idx = self._points_in((x - w, y - w, x + w, y + w))
            if len(idx) == 0:
                return None
            px, py = self._zarr('x')[idx], self._zarr('y')[idx]
            k = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
            i = int(idx[k])
            a = self._pc_post_proc(self._zarr('pc'), np.array([i]), *index)
            return {'point': i, 'x': float(px[k]), 'y': float(py[k]), 'value': _scalar(a)}

        def _on_msg(self, widget, content, buffers):
            kind, rid = content.get('type'), content.get('id')
            index = [int(i) for i in content.get('index', [])]
            try:
                if kind == 'tile':
                    rgba = self.tile_rgba(int(content['z']), int(content['x']), int(content['y']), index)
                    self.send({'type': 'tile', 'id': rid}, [_png(rgba)])
                elif kind == 'value':
                    found = self.value(float(content['x']), float(content['y']), int(content.get('z', 0)), index)
                    self.send({'type': 'value', 'id': rid, **(found or {'value': None})})
            except Exception as e:     # shown in the widget instead of lost in the kernel log
                self.send({'type': kind, 'id': rid, 'error': f'{type(e).__name__}: {e}'})
else:
    TileView = None


def tile_view(
    pyramid:str,
    post_proc:str=None,
):
    """Interactive tile map of a pyramid for a Jupyter notebook: zoom and pan read only the tiles on screen.

    Colours, axes and sliders are chosen from the data like in `view_pyramid`: the cyclic colorwheel over
    (-pi, pi] for phases, viridis over the 1 % - 99 % range otherwise, range to the right and azimuth down.
    Point clouds are drawn as individual points when zoomed in. Needs the optional dependency anywidget
    (``pip install moraine[view]``).

    Parameters
    ----------
    pyramid : str
        pyramid directory made by `ras_pyramid`, or by `pc_pyramid` from points on the radar grid
        (non negative integer coordinates, e.g. `gix`)
    post_proc : str, optional
        how to show a stack of images: 'phase', 'intf_0' (interferograms with the first image),
        'intf_seq' (sequential interferograms) or 'intf_all' (any pair); the phase for complex data by
        default

    Returns
    -------
    TileView
        the widget; display it as the last expression of a cell
    """
    if TileView is None:
        raise ImportError('the tile viewer needs anywidget: pip install anywidget (or moraine[view])')
    return TileView(pyramid, post_proc=post_proc)
