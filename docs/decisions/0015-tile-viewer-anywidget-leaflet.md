# 0015 Tile viewer with anywidget and Leaflet, tiles through the notebook channel

## Status

Superseded by 0017

## Date

2026-09-29

## Context

`view_pyramid` (decision 0014) redraws the whole view through holoviews / bokeh on every zoom or pan:
the kernel sends one image of the screen after the interaction ends, most of the time is spent in
holoviews, and nothing is cached in the browser. Map viewers solve this with tiles: fixed 256 x 256
pieces per zoom level, requested in parallel and kept while panning. A tile server is not an option on
the maintainer's cluster (no port forwarding, decision 0014), but the notebook channel of a widget carries
binary buffers.

## Decision

Prototype tile viewer in the CLI layer next to the holoviews plots it replaces (`moraine.cli.tiles`,
`moraine.cli.viewer`; used from notebooks as `mc.ras_layer`, `mc.pc_layer`, `mc.tile_view`, not a command),
built from layers that are combined:

- layers: `ras_layer` (raster pyramid or array in memory, optional `bounds`) and `pc_layer` (point cloud
  pyramid, or point data with coordinates, rasterized in memory like `pc_pyramid`), on the radar grid or in
  web mercator; ``a * b`` overlays layers on one map, ``a + b`` shows maps side by side with linked zoom
  and pan; sliders of the same name are shared; `tile_view(pyramid)` is the one layer shortcut;
- an anywidget widget with Leaflet maps: on the radar grid (`CRS.Simple`, azimuth down) the map position
  is the data coordinate minus the corner of the first layer, 2**z screen pixels per data unit; web
  mercator layers use Leaflet's EPSG:3857 with standard XYZ tiles, north up, longitude / latitude axes and
  a base map (Esri satellite images, CARTO or OpenStreetMap, with their attribution); Leaflet is loaded
  from the jsdelivr CDN in a pinned version;
- the browser asks for the tiles of each layer with custom widget messages; the layer samples its finest
  pyramid level whose cells are at least a screen pixel at the pixel centres, applies the post processing
  and colours of `view_pyramid` and the kernel answers with a PNG buffer; point clouds zoomed in until a
  cell of level 0 is larger than a screen pixel are drawn as disks at the point coordinates, found with a
  lazily built Hilbert R-tree;
- colour bars of each layer, axes, stack sliders (with dates), layer opacity and the values of all layers
  under the cursor (the nearest point for point clouds) are drawn by the widget;
- the interactions of the holoviews plots are kept: click a pixel or point for its time series (the
  stack of the pyramid or another zarr), double click one to make it the reference of the time series
  (not of the map), custom post processing functions and slider counts like `ras_plot` / `pc_plot`,
  and built-in 'coh' / 'coh_abs' for compressed coherence with reference / secondary sliders; the time
  series is drawn as SVG by the widget, without a plotting library;
- polygons are drawn on the map, synchronized with the kernel and saved as GeoJSON (decision 0016);
- anywidget is an optional dependency (`moraine[view]`).

`moraine view` keeps writing holoviews notebooks until the prototype is accepted.

## Consequences

- Panning shows cached tiles at once; only new tiles are read.
- The browser needs internet access to the CDN; the kernel does not.
- Tile requests are answered one by one in order; requests of tiles already scrolled away are still
  rendered (the answers are dropped).
- Point clouds on longitude / latitude are rejected: their cells are not square on the map; they are
  converted to web mercator first (`moraine transform`).
- The base map tiles come from third party servers and need internet access in the browser.
- The first zoom to individual points reads all coordinates to build the R-tree (seconds).
- Layers in different coordinates (radar grid, web mercator) cannot be combined.
- Arrays in memory are decimated on the fly; large data should go through `ras-pyramid` / `pc-pyramid`.

## Do not

- Do not start a tile server or open ports for the viewer.
- Do not send whole levels or arrays to the browser; only tiles on screen.
