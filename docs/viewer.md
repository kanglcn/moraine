# The viewer: how it works, what was learnt, what is open

Implementation notes of the visualization system (`moraine.cli.view`, decisions 0007, 0018, 0035, 0036) for
whoever changes it next. The user documentation is the docstring of `view` and `AGENTS.md`; the formats are
in `docs/contracts/pyramid.md`. Updated 2026-10-10.

## Parts

| part | role |
|---|---|
| `moraine/cli/plot.py` | pyramids: `ras_pyramid` (bands of `rows` lines, levels by decimation or mean), `pc_pyramid` (points rasterized on a web mercator / radar grid of `ras_resolution`, `idx_l.zarr` per level, `rtree.zarr`), statistics of all the data at build time (`stats` in the marker, `stats.zarr`), `_pyramid_stats`, `_channel_warnings`, `_LazyRtree` |
| `moraine/cli/tiles.py` | the layers: `RasterLayer`, `PointLayer` (pyramid or in memory), `_setup` (colours, sliders, time series, `size`, `terrain`), `raster_values` (the level with at least a cell per pixel sampled at the pixel centres), `render` -> uint8 palette indices, `png` (8 bit palette PNG, index 255 transparent), probes `value` / `locate` / `series` (the cell drawn at the zoom shown, points through `idx_l`, R-tree only where points are drawn one by one), `Overlay` (`*`), `Layout` (`+`), `describe` (the text `repr`), `render_png` (`.png()`, 2D only), `view()` |
| `moraine/cli/viewer.py` | `TileView` (anywidget): traits `crs`, `view_origin`, `extent`, `frame`, `size`, `zoom`, `max_zoom`, `panels` (layers: label, colours, clim, opacity, sliders), `kdims`, `index`, `dates`, `polygons`, `selected`, `reference`, `terrain`; `_on_msg` answers the browser's messages |
| `moraine/cli/viewer.js` | the front end: one module, two map backends behind one small interface, `makeLeafletPanel` (2D) and `makeDeckPanel` (3D) |
| `moraine/cli/viewer.css` | the styles (`.moraine-tv-*`) |
| `moraine/command/summary.py` | `info` (marker statistics), `quicklook` (PNG through `view`), `view` (a notebook of `mc.view` calls, `--terrain`) |
| `tests/test_tile_view.py`, `tests/test_command.py` | the python side: geometry, colours, probes, messages, widget state, PNGs, the notebook |
| `tests/browser/` | the JavaScript: `smoke.mjs` (jsdom, real Leaflet, fake deck.gl) and `e2e/e2e.mjs` (headless Chromium, real deck.gl); run by hand, see its README |

## Messages between the browser and the kernel

The browser sends `{type, id, index: {slider: value}, ...}` with `model.send`; the kernel answers
`{type, id, ...}` (`model.on("msg:custom")`), binary buffers for tiles. One message type per need:

| type | request | answer |
|---|---|---|
| `tile` | `panel`, `layer`, `z`, `x`, `y` (XYZ tile, Leaflet zoom) | a palette PNG of 256 x 256 pixels as the only buffer |
| `value` | `panel`, `x`, `y` (data coordinates: metres for web mercator, lng / lat = x / y - view_origin on the grid), `z` (float zoom) | `x`, `y` of the cell / point, `values: [{label, value, point?}]` of all layers |
| `locate` | like `value` | the pixel / point: `layer`, `label`, `key`, `x`, `y`, `point?` |
| `series` | like `value` plus `ref: {panel, layer, key}` or null | like `locate` plus `values` (null for nan), `ref` |

Errors come back as `{type, id, error}` and are shown in the status line. The zoom `z` is the Leaflet zoom
(256 pixel tiles); the kernel turns it into a pixel size (`pixel_size`) to choose the level drawn. The browser
keeps `pending` (id -> callback) and drops the answers of tiles unloaded meanwhile (Leaflet `tileunload`,
deck.gl `signal`). The kernel answers one message at a time (about 13 ms per tile of a point cloud: 7 ms
rendering, 1.5 ms encoding, 66 kB), so a view change costs the number of tiles on screen; the browser never
cancels a request that already left (a later `cancel` message could not overtake it).

## 2D maps (Leaflet 1.9.4 from jsdelivr)

- `crs` grid: `L.CRS.Simple` with the identity transformation, map position = data - `view_origin`, y down,
  `2**z` screen pixels per data unit; web mercator: `L.CRS.EPSG3857`, base maps Esri satellite (default),
  CARTO light, OpenStreetMap, none (`<img>` tiles, no CORS needed).
- `Tiles` (GridLayer) asks the kernel in `createTile`; `zoomSnap: 0` (continuous zoom, tiles at the nearest
  integer), `updateWhenZooming: false`.
- A slider change swaps only the tile layers that depend on it (`panels[].layers[].sliders`): the new layer is
  added on top and the old one removed on `load`, so the map never blanks.
- Maps take the width of the notebook (at most 700 px high, the aspect of the scene) or `size=(w, h)`; the
  corner handle resizes them (pointer capture); `ResizeObserver` fits the scene when the element gets its size
  (deferred to the next frame).
- Markers (clicked point white, reference red), polygons (drawn here only, saved as GeoJSON), axes with ticks.

## 3D views (deck.gl 9.1.12)

Entered when `TileView.terrain` is not empty (`view(..., terrain=True)`, web mercator only).

- Loading: the UMD bundle `dist.min.js` is fetched and imported as a blob module in which `define`, `exports`
  and `module` are shadowed (`loadDeck`): imported as it is, it registers with the RequireJS `define` of the
  notebook front end instead of defining `globalThis.deck`. Loaded once per page.
- Terrain: `TerrainLayer` in tiled mode with `elevationData: "moraine-terrain://{z}/{x}/{y}"` and a custom
  `fetch` (`terrainFetch`): up to `max_zoom` (15) the public AWS Terrain Tiles (Terrarium, `TERRARIUM`
  decoder); beyond, `overzoomedTile` cuts the tile from its parent at zoom 15 (heights decoded on a canvas,
  cached in `parents`, bilinear, encoded again, given to the loader as a blob URL). Needed because deck.gl
  sizes the texture draped over a terrain tile when the tile appears and keeps it (cap 2048 px): with terrain
  tiles stuck at zoom 15 the draped data blurred when zooming in. `meshMaxError` 8 m, `maxRequests` 10.
- Draping: `_TerrainExtension`; the satellite `TileLayer` (`SATELLITE`, Esri; CARTO and OpenStreetMap refuse
  the fetch + GPU texture path), one `TileLayer` per data layer (`getTileData` asks the kernel, the PNG becomes
  an `ImageBitmap`, `BitmapLayer` with nearest filtering, `extent` = the data, `debounceTime` 80 ms so that a
  zoom through several levels asks only for the last), markers (`ScatterplotLayer`, drape mode: offset mode
  would make deck.gl render a 2048 px height map at every move) and polygons (`PathLayer`, drape, read only).
  A slider change makes a new data `TileLayer` (`data-<l>-<generation>`); the old one stays in `fading`
  until the new one's `onViewportLoad` (checked against the generation).
- Zoom: deck.gl's zoom is Leaflet's minus one (a world of 512 px); `EXTRA_ZOOM_3D = 2` more levels than the
  2D map (64 instead of 16 screen pixels per cell) because deck.gl's tile selection never refines the tiles
  near the camera and the perspective magnifies them.
- The point under the cursor (`pick`): no GPU picking. deck.gl 9.1 cannot pick a terrain that is draped over
  (its picking colour has no layer index: "Picked non-existent layer"), its depth unprojection (`unproject3D`)
  was 100 m off on a flat test terrain, `_pickable: false` would also disable `pickObject`, and a picking pass
  waits for the GPU. Instead `onTileLoad` of the terrain layer keeps the mesh and bounding box of every
  loaded tile (`terrainTiles`), and the point is the first intersection of the pixel's ray with those meshes
  (ray against the boxes, then Moeller-Trumbore on the triangles), in deck.gl's common space: the ray from
  `vp.projectPosition(vp.unproject([x, y, -1 | 1]))`, mesh z (metres) times `distanceScales.unitsPerMeter[2]`,
  the hit back through `unprojectFlat`. It agrees with `vp.unproject([x, y], {targetZ})` to the metre and
  costs well under a millisecond, so the hover is probed once per animation frame (never while a button is
  down); the kernel request is one in flight as in 2D. deck.gl's own hover picking is unhooked in `onLoad`
  (`eventManager.off("pointermove" | "pointerleave", deck._onPointerMove)`).
- Controls: layer checkboxes and "top view" over the map, the attribution below; right mouse button or ctrl
  + drag tilts and rotates (`maxPitch` 85), `doubleClickZoom` off (double click sets the reference); a DOM
  `click` within 3 px of the `pointerdown` is a click, otherwise a drag; several maps share the view state.
- `mapEl.moraine = {deck, pick}` exists for the tests.
- Not in 3D: polygon drawing (shown only), `.png()` (always the 2D image), a choice of base maps, vertical
  exaggeration (removed on request).

## Measured (2026-10-09 / 10, Campi Flegrei pyramids, env work2)

- One 256 x 256 tile of a complex point cloud / stack: render about 7 ms, palette PNG 1.5 ms, 66 kB (RGBA
  before: 8.9 ms, 206 kB). A screen is 15 - 20 tiles in 2D, 10 - 40 in 3D (far tiles at coarser levels).
- Public terrain tiles over Campi: 60 - 120 kB, about 200 ms from the cluster, CORS open, sea 0 m, zoom 16
  is a 404; a 256 x 256 Terrarium PNG encodes in 6.5 ms (would matter for a kernel served DEM).
- A Campi DEM (`hgt.zarr`, 981 x 4160, 59 % finite, the rest sea) mapped to 20 m web mercator cells: 4.1
  pixels per cell, holes inside the footprint 0.8 % of the cells, all of them cells without pixels (mapping gaps
  on slopes); cells with pixels but nan heights are sea. A kernel served DEM pyramid would fill the first from
  coarser levels and the second with 0 m.
- Headless Chromium (SwiftShader): the CPU pick < 1 ms; the former GPU pick 16 ms including its warm up.

## Testing without a browser in the kernel

- `pytest tests/test_tile_view.py tests/test_command.py`: everything python, including the messages
  (`_on_msg` with a captured `send`).
- `node tests/browser/smoke.mjs`: the module in jsdom with the real Leaflet and a fake deck.gl and model;
  fast, catches wiring mistakes (ids, props, events, the bundle loading, the interpolated tiles).
- `node tests/browser/e2e/e2e.mjs`: the module in headless Chromium with the real deck.gl; the page makes the
  tiles and answers the messages; checks hover (position to the metre), click, double click, drag, slider,
  zoom, and that the page log is clean. It found the three defects of the 3D view that nothing else could
  (the AMD loader, the unpickable terrain, the imprecise depth). On the cluster the Chromium build of an
  earlier playwright is in `~/.cache/ms-playwright/`; point `MORAINE_CHROMIUM` to its `chrome`.
- The user checks the result in VS Code; what they reported in this order: the bundle did not load (AMD),
  CARTO / OSM refused, 3D too slow (picking at every move, offset markers, covers), points coarse when zoomed
  in (zoom limit), clicks found nothing (unpickable terrain), hover and time series slow (GPU picking).

## Open items

Listed in `docs/roadmap.md` (Visualization) so that they are not lost; the details and the reasons:

- 3D: polygon drawing on the terrain; points as 3D sprites standing on the terrain (a `points` message with
  positions and palette indices, offset mode, a threshold of about 500 k points in view) instead of draped
  disks; points at their own heights (`z=`, e.g. after a DEM error estimate); a kernel served DEM
  (`pc-pyramid` of `hgt` with `x=e`, `y=n`: needs 2D inputs flattened, and the fill rule above) for offline
  use or the processing DEM; `overzoomedTile` in a Worker if zooming in stalls (about 10 ms per tile on the
  main thread); Google photorealistic 3D Tiles with the user's key (`Tile3DLayer`).
- Both: one message for all the tiles of a screen (the kernel answers one by one now); the time axis of the
  series chart by date; colour bar ticks, a scale bar, keyboard stepping of the sliders; Leaflet and deck.gl
  shipped with the package for offline use and for China (jsdelivr is unreliable there, npmmirror has no
  deck.gl); a Tianditu base map with a key; quicklooks of `moraine run` with `show` / `index` per step.
- Pyramids: level 0 as a symlink to the source array (disk); the duplicated post processing functions of
  `plot.py` and `tiles.py`; colorcet and pillow as dependencies.
