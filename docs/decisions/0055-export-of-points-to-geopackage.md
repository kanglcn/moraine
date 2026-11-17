# 0055 Export of points to GeoPackage, written without GDAL, with the field names of MintPy

## Status

Accepted

## Date

2026-11-17

## Context

Decisions 0040 and 0048 planned, as their last step, an export of the products: points to CSV / GeoPackage, rasters
to GeoTIFF. The results are zarr arrays that GIS software does not read. `save_qgis` of MintPy writes the points as an
ESRI Shapefile through GDAL (EPSG:4326 point geometry; the fields CODE, HEIGHT, H_STDEV, VEL, V_STDEV, COHERENCE,
EFF_AREA and one field D followed by YYYYMMDD per date, in millimeters), which the QGIS plugin InSAR Explorer reads:
its documentation names Shapefile and GeoPackage layers, the velocity field VEL (or velocity, mean_velocity) and the
date fields DYYYYMMDD (or YYYYMMDD, D_YYYYMMDD).

A Shapefile holds at most 2 GB per file, 255 fields (one per image: about 250 images at most) and field names of 10
characters. GeoPackage is one SQLite file (OGC standard) without these limits, which QGIS and GDAL open directly.
Python's `sqlite3` has the R-tree module that the spatial index of GeoPackage uses.

## Decision

- `pc-export` (CLI only, `moraine/cli/export.py`) writes points to a GeoPackage file (`.gpkg`; no other format):
  one layer named after the file, a point geometry at the longitude and latitude (WGS84, EPSG:4326), the field
  `point` (int64, the index of the point in the arrays, the `ref_point` of the maps), a field of every array given as
  an extra keyword argument (`[step.kw]` / `--kw NAME=path`, as for `math`; (n_points,) bool, integer or floating
  point; the name of the argument is the name of the field) and, with `ts` (n_points, nimages) and `dates`, a field D
  followed by YYYYMMDD per image. `mask` selects points.
- No unit conversion: the values are written as they are in the arrays (millimeters for the displacement), nan as
  NULL. float32 values are written as the shortest decimal number of the same float32 value (1.6, not
  1.600000023841858) in fields declared FLOAT; integer fields are declared with their size (TINYINT, SMALLINT,
  MEDIUMINT, INTEGER), bool ones BOOLEAN.
- Written with Python's `sqlite3`, without GDAL: the tables of GeoPackage 1.2 as GDAL writes them (application id,
  user_version 10200, `gpkg_spatial_ref_sys` with EPSG:4326, `gpkg_contents` with the extent, `gpkg_geometry_columns`,
  the R-tree spatial index with its triggers and `gpkg_extensions`, and GDAL's `gpkg_ogr_contents` with the number of
  points), with pages of 64 KiB (GDAL: 4 KiB), which make files of about 100 fields 10 % smaller. The points are read
  in blocks (`chunks` points with all their fields) and written in their order; the file is written as `<out>.part`
  and renamed when complete, so a failed run leaves an existing file as it was.
- Field names: letters, digits and _, starting with a letter; not fid, geom, point or a date field of `ts`, ignoring
  case (SQLite does). Example 06 writes `disp/points.gpkg` with the names of MintPy: VEL, V_STDEV (mm per year),
  HEIGHT (m, the DEM height), COHERENCE (temporal coherence of 04), DEM_ERROR (m), RMS (residual RMS, mm) and the
  displacement without the DEM error (mm) per date. CODE, H_STDEV and EFF_AREA of MintPy are not written (an index,
  and the constants 0 and 1).
- Rasters to GeoTIFF stay a later change.

## Consequences

- Validation: the file read back with `sqlite3` (tables, extent, R-tree, every value, NULL) and, where GDAL is
  installed, with OGR (fields, values, spatial filter) and GDAL's `validate_gpkg` (no error or warning); the
  shortest decimals equal those of numpy for float32 values of every magnitude.
- QGIS opens the file by drag and drop. InSAR Explorer's documentation names GeoPackage layers with the fields VEL
  and D followed by YYYYMMDD; this was not tried in QGIS.
- SQLite writes from one thread: 10 million points with 6 fields and 92 dates take 306 s and 1.65 GB of memory
  (chunks of 1 million points) and 9.7 GB on disk (about 10 bytes per value); the shortest decimals and the R-tree take
  about a third of the time each. `mask` or no `ts` make smaller files.
- The R-tree triggers of the file call GDAL's ST_ functions: the file can be edited with GDAL / QGIS, not with a plain
  SQLite client.

## Do not

- Do not add GDAL (or fiona / geopandas) as a dependency for writing GeoPackage; GDAL is used by the tests only when
  installed.
- Do not convert units or rename fields in `pc-export`: the units are those of the arrays, the names those given.
- Do not write Shapefiles (size, field count and name limits above).
