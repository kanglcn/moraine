# 0015 Tile viewer with anywidget and Leaflet, tiles through the notebook channel

## Status

Proposed

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

Prototype `moraine.command.tile_view` for raster pyramids and point cloud pyramids on the radar grid:

- an anywidget widget with a Leaflet map (`CRS.Simple`, azimuth down) in cells of level 0, 2**z screen
  pixels per cell; the axes show data coordinates, cell i centred at coordinate i for rasters and at the
  grid coordinates of `bounds.toml` for point clouds; Leaflet is loaded from the jsdelivr CDN in a pinned
  version;
- the browser asks for tiles with custom widget messages; the kernel reads the pyramid level with at
  least one pixel per screen pixel, applies the post processing and colours of `view_pyramid` and
  answers with a PNG buffer; point clouds zoomed in beyond level 0 (z > 0) are drawn as disks at the
  point coordinates, found with the lazily built Hilbert R-tree of `pc_plot`;
- colour bar, axes, stack sliders and the value under the cursor (the nearest point for point clouds)
  are drawn by the widget;
- anywidget is an optional dependency (`moraine[view]`).

`moraine view` keeps writing holoviews notebooks until the prototype is accepted.

## Consequences

- Panning shows cached tiles at once; only new tiles are read.
- The browser needs internet access to the CDN; the kernel does not.
- Tile requests are answered one by one in order; requests of tiles already scrolled away are still
  rendered (the answers are dropped).
- Point clouds on map coordinates are not supported yet (next step: web mercator with a base map).
- The first zoom to individual points reads all coordinates to build the R-tree (seconds).

## Do not

- Do not start a tile server or open ports for the viewer.
- Do not send whole levels or arrays to the browser; only tiles on screen.
