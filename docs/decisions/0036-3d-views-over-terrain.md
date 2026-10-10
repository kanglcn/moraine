# 0036 3D views over the terrain with deck.gl and public elevation tiles

## Status

Accepted

## Date

2026-10-10

## Context

The maps of `moraine.cli.view` (decision 0018) are 2D Leaflet maps. Deformation and coherence are easier to
judge over the relief (slopes, valleys, the flank of a volcano), and the maintainer wanted a Google Earth like
view of the results without preparing elevation data for every site. Options: CesiumJS (a globe; needs an ion
token or self made quantized mesh terrain), deck.gl (a site level view in web mercator over heightmap terrain
tiles, the same tile scheme and base maps as the 2D maps), a DEM pyramid served by the kernel (the `hgt` of the
GAMMA geocoding mapped to web mercator), or the public Terrarium elevation tiles of AWS Open Data (Mapzen /
Tilezen: SRTM, EU-DEM, 3DEP, ..., about 30 m, zoom levels 0 to 15, no key, open CORS). Measured over Campi
Flegrei: a public tile is 60 to 120 kB and arrives in about 200 ms, the sea is 0 m.

## Decision

- A view with `terrain` is a 3D view: deck.gl (its UMD bundle from the jsdelivr CDN, loaded only for 3D views)
  renders the terrain from elevation tiles with its TerrainLayer; the base map and the kernel's tiles are
  draped over it with the TerrainExtension, the markers stand on it. The kernel takes no part in the terrain:
  the browser fetches the elevation tiles like the base map, and the tile and probe messages of decision 0018
  are unchanged (the cursor position is the point of the terrain under it; deck's zoom plus one is Leaflet's).
- `terrain=True` uses the public AWS Terrain Tiles; the URL template of another Terrarium encoded service can
  be given. `exaggeration` scales the heights (a slider too). Web mercator layers only.
- Polygons are shown on the terrain but drawn on 2D maps; `.png` stays the 2D image (no browser in the kernel).
- The 2D maps keep Leaflet: one viewer module with two map backends behind a small interface (tiles, markers,
  polygons, events); sliders, time series, reference and polygons are shared.

## Consequences

- The browser needs internet access for deck.gl, the base map and the elevation tiles; a 3D view depends on
  the public tile service as the 2D maps depend on the base maps. The URL is a parameter.
- The terrain is 30 m data up to zoom 15; the kernel's tiles stay sharp at any zoom (draped at screen
  resolution). The heights are those of the public DEM, not of the DEM used in processing.
- A DEM pyramid served by the kernel (offline use, the processing DEM) and points at their own heights can come
  later behind the same `terrain` parameter.

## Do not

- Do not build terrain in the kernel or send elevation data through the notebook channel while the public tiles
  do; do not add a second viewer for 3D (decision 0018: one viewer).
- Do not compose textures in the browser (canvas) to drape data: the TerrainExtension does it.
