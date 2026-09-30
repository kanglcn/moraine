# 0017 Polygons are GeoJSON files in longitude / latitude or radar grid coordinates

## Status

Accepted

## Date

2026-09-29

## Context

Users draw polygons on results to keep or remove areas (e.g. water, layover, an area of interest). The
polygons must be saved, reused for other results of the same scene and exchanged with GIS software. Results
are on the radar grid or have longitude / latitude (and web mercator coordinates for map views).

## Decision

- Polygons are GeoJSON FeatureCollections of Polygon features (`moraine.polygon`), as specified in
  `docs/contracts/data.md`.
- Vertices are longitude / latitude (standard GeoJSON), or range / azimuth pixel coordinates marked by the
  top level member `moraine_coordinates = "radar_grid"`. `moraine.cli.view` saves radar grid polygons for
  radar grid pyramids and longitude / latitude for web mercator pyramids.
- `polygon-mask` turns them into a bool mask (True to keep) of a raster or point cloud, keeping the inside
  or the outside; applying the mask is left to existing commands (`math`, `pc-logic-pc`).

## Consequences

- Longitude / latitude polygons open in QGIS and other GIS software and apply to any result with
  longitude / latitude (`load-gamma-lat-lon-hgt`, point clouds made with `ras2pc`).
- Radar grid polygons apply to every raster and point cloud of the same radar geometry.

## Do not

- Do not store polygons in web mercator or other projected coordinates.
- Do not write masking into each processing command; make a mask and combine it.
