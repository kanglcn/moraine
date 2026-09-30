# 0007 Visualization and result statistics go through pyramids

## Status

Superseded by 0017

## Date

2026-09-28

## Context

moraine is made for data larger than memory. zarr reads whole chunks, so strided sampling of a large
array still reads every chunk. moraine already has pyramids (`ras-pyramid`, `pc-pyramid`) and plots that
read the level matching the view (`ras_plot`, `pc_plot`, holoviews).

## Decision

- `moraine info` shows metadata only for plain arrays (no data read).
- For pyramids it adds statistics from the finest level of at most 64 MiB (a regular decimation of the
  scene) and warnings for all-nan, infinite and constant values; point cloud pyramids skip empty cells.
- `moraine quicklook` only accepts pyramids and draws them with `ras_plot` / `pc_plot`, saved with the
  holoviews matplotlib backend (no browser needed).
- `pc_plot` builds its rtree lazily so overviews do not read all coordinates.

## Consequences

- Results worth checking get a pyramid step in the pipelines.
- Checking a result costs the same for any data size.

## Do not

- Do not write custom sampling, statistics over full arrays or custom plotting code for checks.
- Do not read whole arrays to inspect results.
