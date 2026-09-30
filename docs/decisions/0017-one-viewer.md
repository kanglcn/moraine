# 0017 One viewer: `moraine.cli.view` replaces the holoviews plots

## Status

Accepted

## Date

2026-09-30

## Context

moraine had two ways to look at results: the holoviews / bokeh plots (`ras_plot`, `pc_plot`, `ts_plot`, used
by `moraine view` and `moraine quicklook`, decisions 0007 and 0014) and the tile viewer prototype
(decision 0015). The holoviews plots redraw the whole view after each zoom or pan, need several large
dependencies and a different call for every kind of data; the tile viewer is faster and does what they did
(stacks, time series, reference, coherence, custom post processing) plus overlays, base maps and polygons.
Users and agents should have one tool to remember.

## Decision

- `moraine.cli.view(data, ...)` is the only viewer. One function for pyramids, rasters in memory and point
  data (`x=`, `y=`), with the same keywords for all (`show`, `dates`, `series`, `polygons`, `cmap`, `clim`,
  `opacity`, `label`); ``a * b`` overlays views on one map, ``a + b`` puts maps side by side with linked zoom
  and pan and sliders shared by name.
- `show` is a name ('phase', 'intf_0', 'intf_seq', 'intf_all', 'coh', 'coh_abs') or a function
  ``f(v, *sliders)`` of the visible stack whose other arguments are sliders named like them; only the
  images the function indexes are read.
- Displayed in a notebook it is an anywidget / Leaflet map whose tiles the kernel renders and sends through
  the notebook channel (no server, no port forwarding): the pyramid level with at least one cell per screen
  pixel is sampled at the pixel centres, point clouds are drawn point by point when zoomed in, web mercator
  data are drawn north up over a base map. The clicked pixel / point, the reference, the slider values and
  the polygons (saved to the `polygons` GeoJSON file, decision 0016) are available in python.
- Every view describes itself in text (`repr`) and saves a PNG (`.png(path)`), so agents can check results
  without a browser; `moraine quicklook` is ``view(pyramid).png(out)`` and `moraine view` writes a notebook of
  `mc.view` calls.
- The rules of decision 0007 stay: views, quicklooks and statistics read pyramids (or small arrays), never
  whole large arrays; `moraine info` reads the finest pyramid level of at most 64 MiB. The colours and axes of
  decision 0014 stay: the cyclic colorwheel over (-pi, pi] for phases, viridis over the 1 % - 99 % range
  otherwise, range to the right and azimuth down on the radar grid, maps of the scene's aspect (at most 1:4).
- holoviews, bokeh and jupyter_bokeh are no longer dependencies; anywidget is.

## Consequences

- `ras_plot`, `pc_plot`, `ts_plot`, `bg_alpha` and `view_pyramid` are removed; old notebooks that call them
  need `mc.view`.
- The browser needs internet access for Leaflet (jsdelivr CDN) and the base maps; the kernel does not.
- Layers in different coordinates (radar grid, web mercator) cannot be combined; longitude / latitude point
  clouds are converted to web mercator first (`moraine transform`).
- Tile requests are answered one by one; tiles already scrolled away are still rendered (their answers are
  dropped). The first zoom to individual points reads all coordinates to build an R-tree.

## Do not

- Do not add plotting libraries or write plotting code for checking results; extend `moraine.cli.view`.
- Do not start plot or tile servers or open ports for users on remote machines.
- Do not send whole levels or arrays to the browser; only the tiles on screen.
