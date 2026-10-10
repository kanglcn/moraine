# Contract: pyramids

Version 1. Multi resolution copies of rasters and point clouds made by the `ras-pyramid` and
`pc-pyramid` commands, read by `moraine.cli.view`, `moraine info` and `moraine quicklook`
(decision 0007).

## Marker

`0.zarr` carries the attribute `moraine_pyramid = {"version": 1, "kind": "raster" | "point cloud"}`.
Readers reject versions newer than they support. Pyramids made before the marker existed are recognized
by level 1 being half the size of level 0.

## Raster pyramid (`ras-pyramid`)

```
<dir>/
├── 0.zarr        the raster itself, shape (nlines, width[, n])
├── 1.zarr        every 2nd pixel of 0
├── ...
└── L.zarr        L = floor(log2(min(nlines, width))), at least 2 pixels per side
```

- Level `l` is `ras[::2**l, ::2**l]` (decimation, not averaging), shape
  `(ceil(nlines / 2**l), ceil(width / 2**l)[, n])`, same dtype.
- Chunks are `(chunks[0], chunks[1], 1, ...)` (default `(256, 256)`).

## Point cloud pyramid (`pc-pyramid`)

```
<dir>/
├── bounds.toml   bounds = [x0, y0, xm, ym] of the rendered grid
├── x.zarr, y.zarr   coordinates of the points, (n_points,)
├── pc.zarr       the point cloud data, (n_points[, n])
├── 0.zarr ...    the points rasterized on a grid of `ras_resolution`, level l with cell size ras_resolution * 2**l
├── idx_0.zarr ...   per level, the index of the point shown in each cell, -1 for empty cells
└── rtree.zarr    bounding box tree of the points (`HilbertRtree`): (n_nodes, 4) float64, [x0, y0, xm, ym] of
                  each node, `page_size` (points per leaf) and `n_points` in its attributes; optional, readers
                  build it from x.zarr / y.zarr when a pyramid has none
```

- Rows of the grid are y, columns x, like the rasters.
- Cell (i, j) of level 0 is centred at (x0 + j * ras_resolution, y0 + i * ras_resolution) of `bounds`; every
  point is in the cell of its nearest centre and the grid reaches the cells of the largest coordinates, so
  `bounds` = [min x, min y, x0 + (width - 1) * ras_resolution, y0 + (nlines - 1) * ras_resolution] and points
  in different cells never share one. Pyramids made before this was fixed may have the points of the last
  line or column merged into the previous one.
- Cells without points are nan in `l.zarr` and -1 in `idx_l.zarr`.
- In each 2 x 2 block the first non empty cell is kept for the next level.
