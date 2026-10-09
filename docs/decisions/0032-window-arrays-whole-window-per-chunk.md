# 0032 Window arrays keep the whole window of a pixel in one chunk
## Status
Accepted
## Date
2026-10-09
## Context
The SHP test writes the p values of every pixel against the pixels of its window as a raster
`(nlines, width, az_win, r_win)`, `select-shp` writes the SHP flags in the same shape, and
`ras2pc-ras-chunk` turns the flags of the DS candidates into point arrays `(n_points, az_win, r_win)`.
Following the rule for stacks (decision 0019) they were chunked with one window cell per chunk,
`(lines_block, width_block, 1, 1)` and `(points_block, 1, 1)`: 121 chunks per block for an 11 x 11
window. Every writer and reader of these arrays works per block of pixels and needs the whole window,
so each block was written and read through 121 strided slices and 121 chunk operations; on the sample
data `select-shp` spent 1.4 s of its 1.5 s per block on them and `ras2pc-ras-chunk` 1 s of 1.2 s, and
threads or processes did not help, because the cost is per chunk operation.
## Decision
- Window arrays are chunked in space only, with the whole window of a pixel in one chunk:
  `(lines_block, width_block, az_win, r_win)` for rasters, `(points_block, az_win, r_win)` for point
  clouds. The spatial block stays the processing chunk of the command.
- Stacks keep one image per chunk (0019): `ras2pc-ras-chunk` tells a window array (4 dimensions) from a
  stack (3 dimensions) by its number of dimensions.
- The rule is part of the data conventions contract (`docs/contracts/data.md`, section Chunks).
## Consequences
- Writing a block of SHP flags takes 0.3 instead of 1.4 s, a block of point windows 0.2 instead of 1 s.
- A block of p values is one chunk of 484 MB for 1000 x 1000 pixels and an 11 x 11 window, compressed and
  decompressed by one thread (0.8 s to read, against 0.3 s for 121 chunks read in threads): the SHP test
  and `select-shp` gain little or lose a little on small data until the p values are no longer written.
- Readers use whole blocks and read old arrays with one cell per chunk as before.
## Do not
- Do not chunk a window array along the window axes.
