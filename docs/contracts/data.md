# Contract: data conventions

Shapes, orders and meanings of the arrays moraine functions and commands exchange. Changing them needs a
decision record, since every stored result depends on them.

## Rasters

- Shape `(nlines, width[, n])`: azimuth (lines) first, range second, stacks last.
- rslc stack: `(nlines, width, nimages)` complex64, images in date order.
- nan marks missing data.

## Point clouds

- Arrays of shape `(n_points, ...)`, the first axis indexes the points.
- `gix`: grid index `(n_points, 2)` int32, (azimuth, range) of each point in its raster.
- `hix`: hilbert index `(n_points,)` int64 of the grid index, computed with the raster shape.
- Point clouds are kept in hilbert order (`pc-sort`), so points close in the array are close on the
  ground. `pc-union`, `pc-intersect`, `pc-diff` and `pc-select-data` require sorted indices.
- Commands working per raster chunk (`ras2pc-ras-chunk`, `emperical-co-pc`,
  `emperical-co-emi-temp-coh-pc`) write a directory with one zarr per chunk (`0.zarr`, `1.zarr`, ...), in
  raster chunk order. `pc-concat` merges it back, sorted with the key of `ras2pc-ras-chunk` and, for
  hilbert order, the key of `pc-sort`.

## Interferometric data

- Image pairs `(n_pairs, 2)` int: (reference, secondary) image index; the interferogram is
  `ref * conj(sec)`. Image pair files have these two columns.
- Interferograms and phase histories are complex; the phase is `angle(...)`. Filtered interferograms and
  phase histories have unit amplitude.
- Coherence of point clouds is compressed: the upper triangle of the coherence matrix,
  `(n_points, n_pairs)` in the order of `numpy.triu_indices(nimages, 1)` unless image pairs are given;
  `moraine.uncompress_coh` restores full matrices.
- Unwrapped phase: float32 radians, `(n_points, n_pairs)`.

## Polygons

- GeoJSON files (decision 0016): a FeatureCollection of Polygon (or MultiPolygon) features; only the outer
  rings are used, several polygons are united.
- The top level member `moraine_coordinates` gives the vertex coordinates: `lonlat` (longitude, latitude
  in degrees, the GeoJSON default when the member is missing) or `radar_grid` (range, azimuth pixel
  coordinates, pixel (i, j) at range j, azimuth i).
- Masks made from them (`polygon-mask`) are bool arrays of the shape of the data, True for the data to
  keep.
