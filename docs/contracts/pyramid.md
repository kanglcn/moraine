# Contract: pyramids

Version 1. Multi resolution copies of rasters and point clouds made by the `ras-pyramid` and
`pc-pyramid` commands, read by `moraine.cli.view`, `moraine info` and `moraine quicklook`
(decision 0007).

## Marker

`0.zarr` carries the attribute `moraine_pyramid = {"version": 1, "kind": "raster" | "point cloud", "method":
..., "stats": {...}}`. Readers reject versions newer than they support. Pyramids made before the marker existed
are recognized by level 1 being half the size of level 0.

`method` (optional, absent in pyramids made before it existed) says how each level is made from the finer
one: `"decimate"` (the default of rasters: every 2nd pixel) or `"mean"` (the mean of the 2 x 2 blocks over the
finite pixels) for rasters, `"first"` (the default: the first cell with a point of each 2 x 2 block) or `"mean"`
(the mean of the cells with points) for point clouds. A pyramid without `method` is `decimate` / `first`.

`stats` (optional, written since the statistics are computed when the pyramid is built) are the statistics
of all the data (the points of a point cloud pyramid): `nan_fraction`, `min`, `max`, `mean`, `std` (exact),
`p01`, `p50`, `p99` (of a regular sample of at most 2**23 finite values) and `warnings`; for complex data
the value fields describe the amplitude and are named `amplitude_min`, ...; for boolean data only
`true_fraction`. The optional `stats.zarr`, shape `(*channels, 8)` float64 with the attribute `columns`
(nan_fraction, min, max, mean, std, p01, p50, p99), has the same statistics of every channel (image, pair)
of a stack, nan where a channel has no finite value.

## Raster pyramid (`ras-pyramid`)

```
<dir>/
├── 0.zarr        the raster itself, shape (nlines, width[, n])
├── 1.zarr        every 2nd pixel of 0
├── ...
├── L.zarr        L = floor(log2(min(nlines, width))), at least 2 pixels per side
└── stats.zarr    per channel statistics (see the marker above)
```

- Level `l` has shape `(ceil(nlines / 2**l), ceil(width / 2**l)[, n])`. With `method` decimate it is
  `ras[::2**l, ::2**l]`, same dtype; with `method` mean every cell is the mean of its `2**l x 2**l` block of the
  data over the finite pixels (nan where all are nan; complex data averaged as complex numbers), same dtype for
  floating point and complex data, float32 for integer and boolean data.
- Chunks are `(chunks[0], chunks[1], 1, ...)` (default `(256, 256)`).

## Point cloud pyramid (`pc-pyramid`)

```
<dir>/
├── bounds.toml   bounds = [x0, y0, xm, ym] of the rendered grid
├── x.zarr, y.zarr   coordinates of the points, (n_points,)
├── pc.zarr       the point cloud data, (n_points[, n])
├── 0.zarr ...    the points rasterized on a grid of `ras_resolution`, level l with cell size ras_resolution * 2**l
├── idx_0.zarr ...   per level, the index of the point shown in each cell, -1 for empty cells
├── rtree.zarr    bounding box tree of the points (`HilbertRtree`): (n_nodes, 4) float64, [x0, y0, xm, ym] of
│                 each node, `page_size` (points per leaf) and `n_points` in its attributes; optional, readers
│                 build it from x.zarr / y.zarr when a pyramid has none
└── stats.zarr    per channel statistics of the points (see the marker above)
```

- Rows of the grid are y, columns x, like the rasters.
- Cell (i, j) of level 0 is centred at (x0 + j * ras_resolution, y0 + i * ras_resolution) of `bounds`; every
  point is in the cell of its nearest centre and the grid reaches the cells of the largest coordinates, so
  `bounds` = [min x, min y, x0 + (width - 1) * ras_resolution, y0 + (nlines - 1) * ras_resolution] and points
  in different cells never share one. Pyramids made before this was fixed may have the points of the last
  line or column merged into the previous one.
- Cells without points are nan in `l.zarr` and -1 in `idx_l.zarr`; the cells have the dtype of the points for
  floating point and complex data and are float32 for integer and boolean data.
- `idx_l` keeps the first non empty cell of each 2 x 2 block for the next level. With `method` first, `l.zarr`
  is the value of that cell; with `method` mean, the mean of the non empty cells of the block (over the points
  of the finer levels: the mean of the points of the `2**l x 2**l` block of level 0 cells).
