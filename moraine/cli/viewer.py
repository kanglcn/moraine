"""The notebook widget of views (`moraine.cli.tiles`): Leaflet maps whose tiles are rendered by the kernel and
sent through the notebook channel (decision 0018)."""

__all__ = ['TileView']

from pathlib import Path

import anywidget
import numpy as np
import traitlets

from .plot import _frame_size
from .tiles import grid_geom, mercator_geom, mercator_pixel, png, view_sliders, view_dates, describe

MAX_ZOOM_CELL = 16       # at most this many screen pixels per cell of the finest layer
LAYOUT_FRAME = (560, 480)   # largest map of a layout of several maps, (width, height) in screen pixels


class TileView(anywidget.AnyWidget):
    """Interactive maps of layers for Jupyter / VS Code notebooks, made by displaying `view` results.

    Zoom and pan read only the tiles on screen, sliders choose the image of a stack, the cursor shows the
    values under it. Click a pixel or point to plot its time series, double click one to make it the reference
    of the time series, draw polygons. Several maps are zoomed and panned together. A map takes the width of
    the notebook, or the `size` of its layers; drag its lower right corner to resize it. No server or port
    forwarding is needed; the kernel reads the data on the machine that holds them.

    Parameters
    ----------
    panels : list
        maps, each a list of layers drawn from the bottom up; all in the same coordinates (radar grid or web
        mercator)
    """
    _esm = Path(__file__).with_name('viewer.js')
    _css = Path(__file__).with_name('viewer.css')

    # crs 'grid': map position (lng, lat) is data coordinate (x, y) - view_origin, y down, 2**z screen
    # pixels per data unit. crs 'web_mercator': Leaflet's EPSG:3857 with XYZ tiles, data in metres.
    crs = traitlets.Unicode('grid').tag(sync=True)
    view_origin = traitlets.List([0.0, 0.0]).tag(sync=True)
    extent = traitlets.List().tag(sync=True)         # [x0, y0, x1, y1] of all layers in data coordinates
    frame = traitlets.List().tag(sync=True)          # [width, height] in screen pixels: the aspect of the maps, and
                                                     # with `zoom` the view of a map that has no size yet
    size = traitlets.List().tag(sync=True)           # [width, height] of the maps given by the user, [] for the
                                                     # width of the notebook
    zoom = traitlets.Int().tag(sync=True)            # zoom showing the whole extent in the frame
    max_zoom = traitlets.Int().tag(sync=True)
    axis_labels = traitlets.List(['range', 'azimuth']).tag(sync=True)
    # per map: title, series (a layer has a time series) and layers (label, colours, clim, bar label, opacity,
    # the names of the sliders the layer depends on)
    panels = traitlets.List().tag(sync=True)
    kdims = traitlets.List().tag(sync=True)          # sliders [{'name', 'max'}], merged by name
    index = traitlets.Dict().tag(sync=True)          # slider values by name, set by the map or by python
    dates = traitlets.List().tag(sync=True)
    # polygons drawn on the maps: lists of [x, y] vertices, range / azimuth on the radar grid, longitude /
    # latitude for web mercator
    polygons = traitlets.List().tag(sync=True)
    selected = traitlets.Dict().tag(sync=True)       # clicked pixel / point: panel, layer, label, key, x, y
    reference = traitlets.Dict().tag(sync=True)      # reference of the time series, like selected

    def __init__(self, panels, **kwargs):
        self._panels = [list(p) for p in panels]
        layers = [layer for p in self._panels for layer in p]
        if not layers:
            raise ValueError('no layers to show')
        crs = {layer.crs for layer in layers}
        if len(crs) > 1:
            raise ValueError(f'layers in different coordinates {sorted(crs)} cannot be shown together')
        crs = crs.pop()
        files = {layer.polygon_file for layer in layers if layer.polygon_file}
        if len(files) > 1:
            raise ValueError(f'one polygon file per view, not {sorted(files)}')
        self._polygon_file = files.pop() if files else None
        ext = np.array([layer.extent for layer in layers])
        extent = [ext[:, 0].min(), ext[:, 1].min(), ext[:, 2].max(), ext[:, 3].max()]
        width, height = extent[2] - extent[0], extent[3] - extent[1]
        frame = _frame_size(max(width, 1e-9), max(height, 1e-9))
        if len(self._panels) > 1:
            scale = min(LAYOUT_FRAME[0] / frame[0], LAYOUT_FRAME[1] / frame[1], 1)
            frame = (max(1, round(frame[0] * scale)), max(1, round(frame[1] * scale)))
        # screen pixels per data unit at zoom 0
        per_unit = 1.0 if crs == 'grid' else 1 / mercator_pixel(0)
        zoom = int(np.floor(np.log2(min(frame[0] / width, frame[1] / height) / per_unit)))
        finest = min(layer.cell for layer in layers)
        max_zoom = int(np.floor(np.log2(MAX_ZOOM_CELL / (finest * per_unit))))
        kdims, index = view_sliders(self._panels)
        size = next((layer.size for p in self._panels for layer in p if layer.size), None)
        super().__init__(
            crs=crs, view_origin=[float(v) for v in layers[0].edge_origin] if crs == 'grid' else [0.0, 0.0],
            extent=[float(e) for e in extent], frame=list(frame), zoom=min(zoom, max_zoom), max_zoom=max_zoom,
            axis_labels=['range', 'azimuth'] if crs == 'grid' else ['longitude', 'latitude'],
            panels=[{'title': '  |  '.join(layer.title for layer in p),
                     'series': any(layer.ts is not None for layer in p),
                     'layers': [{'label': layer.label, 'colors': layer.colors, 'clim': list(layer.clim),
                                 'bar_label': layer.bar_label, 'opacity': layer.opacity,
                                 'sliders': [k['name'] for k in layer.kdims]} for layer in p]}
                    for p in self._panels],
            kdims=kdims, index=index, dates=view_dates(self._panels), size=list(size) if size else [], **kwargs)
        if self._polygon_file and Path(self._polygon_file).exists():
            self.load_polygons(self._polygon_file)
        self.observe(self._save_polygons, names='polygons')
        self.on_msg(self._on_msg)

    def __repr__(self):
        return describe(self._panels)

    @property
    def layers(self):
        """The layers of each map."""
        return [list(p) for p in self._panels]

    def tile_geom(self, z, tx, ty):
        """Pixels of tile (`tx`, `ty`) at zoom `z` in data coordinates."""
        if self.crs == 'grid':
            return grid_geom(z, tx, ty, self.view_origin)
        return mercator_geom(z, tx, ty)

    def pixel_size(self, z):
        """Screen pixel size in data units at zoom `z`."""
        return 2.0 ** -z if self.crs == 'grid' else mercator_pixel(z)

    def to_data(self, u, v):
        """Data coordinates of map position (`u`, `v`): lng / lat on the radar grid, metres for web mercator."""
        if self.crs == 'grid':
            return self.view_origin[0] + u, self.view_origin[1] + v
        return u, v

    def _series_layer(self, panel):
        """Index of the top layer of a map with a time series, None without."""
        for k in range(len(self._panels[panel]) - 1, -1, -1):
            if self._panels[panel][k].ts is not None:
                return k
        return None

    @property
    def polygon_coordinates(self):
        """Coordinates of `polygons`: 'radar_grid' (range, azimuth) or 'lonlat'."""
        return 'radar_grid' if self.crs == 'grid' else 'lonlat'

    def save_polygons(self, path):
        """Write the polygons drawn on the maps to a GeoJSON file for `polygon_mask`.

        Parameters
        ----------
        path : str
            output GeoJSON file
        """
        from ..api.polygon import write_polygons
        write_polygons(path, self.polygons, self.polygon_coordinates)

    def load_polygons(self, path):
        """Show the polygons of a GeoJSON file on the maps (replacing the drawn ones).

        Parameters
        ----------
        path : str
            GeoJSON file in the coordinates of the maps: radar grid (range, azimuth) or longitude / latitude for
            web mercator layers
        """
        from ..api.polygon import read_polygons
        polys, coordinates = read_polygons(path)
        if coordinates != self.polygon_coordinates:
            raise ValueError(f'{path} has {coordinates} polygons, the maps need {self.polygon_coordinates}')
        self.polygons = [poly.tolist() for poly in polys]

    def _save_polygons(self, change):
        if self._polygon_file:
            self.save_polygons(self._polygon_file)

    def _on_msg(self, widget, content, buffers):
        kind, rid = content.get('type'), content.get('id')
        index = content.get('index') or {}
        panel = int(content.get('panel', 0))
        try:
            if kind == 'tile':
                layer = self._panels[panel][int(content['layer'])]
                geom = self.tile_geom(int(content['z']), int(content['x']), int(content['y']))
                self.send({'type': 'tile', 'id': rid}, [png(layer.render(geom, index))])
                return
            x, y = self.to_data(float(content['x']), float(content['y']))
            s = self.pixel_size(float(content.get('z', 0)))     # the zoom is continuous
            if kind == 'value':
                values = []
                for layer in self._panels[panel]:
                    found = layer.value(x, y, s, index)
                    if found is not None:
                        values.append({'label': layer.label, 'value': found['value'],
                                       **({'point': found['point']} if 'point' in found else {})})
                self.send({'type': 'value', 'id': rid, 'x': x, 'y': y, 'values': values})
                return
            k = self._series_layer(panel)
            found = None
            if k is not None:
                layer = self._panels[panel][k]
                if kind == 'locate':
                    found = layer.locate(x, y, s)
                elif kind == 'series':
                    ref = content.get('ref') or {}
                    same = ref.get('panel') == panel and ref.get('layer') == k
                    found = layer.series(x, y, s, ref.get('key') if same else None)
            self.send({'type': kind, 'id': rid, **({'layer': k, 'label': self._panels[panel][k].label,
                                                     **found} if found else {})})
        except Exception as e:     # shown in the widget instead of lost in the kernel log
            self.send({'type': kind, 'id': rid, 'error': f'{type(e).__name__}: {e}'})
