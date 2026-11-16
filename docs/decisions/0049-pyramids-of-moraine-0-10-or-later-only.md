# 0049 Only complete pyramids of moraine 0.10.0 or later are read; their statistics are the stored ones

## Status

Accepted

## Date

2026-11-15

## Context

Since decision 0035, `ras-pyramid` and `pc-pyramid` store the statistics of all the data in the marker of a pyramid
(`moraine_pyramid` of `0.zarr`). The readers kept reading the pyramids made before:

- without a marker (0.9.0 and the development versions before 2026-09-28), recognized by level 1 being half the size
  of level 0;
- with a marker without `stats` (development versions of 2026-09-28 to 2026-10-09), summarized from their finest
  level of at most 64 MiB (`summarize(max_bytes=)`, `stats_level` > 0 in the output of `moraine info`);
- point cloud pyramids without `rtree.zarr`, whose tree the views built from all the coordinates.

Every pyramid of 0.10.0 (2026-10-10) and later has the marker with `method` and `stats`, `stats.zarr` and, for
point clouds, `rtree.zarr`. The fallbacks served the pyramids of 0.9.0 and of eleven days of development, at the cost
of code, tests and contract text, and they hid a failure: a `ras-pyramid` that stopped before writing its marker (a
killed process, a full disk) has all its levels and passed as an old pyramid, with the statistics of a coarse level.

## Decision

- A directory is read as a pyramid only with the marker of moraine 0.10.0 or later: `version` (at most the version
  of the reader), `method` and `stats`, and for a point cloud pyramid `rtree.zarr`. The marker is written after
  everything else, so a pyramid with it is complete.
- `pyramid_levels`, which every reader calls first (`moraine info`, `quicklook`, `view`, `moraine.cli.view`), raises
  a ValueError naming the path and asking to rebuild it with `moraine ras-pyramid` / `moraine pc-pyramid` for levels
  halving in size without the marker, for a marker without `method` or `stats` and for a point cloud pyramid without
  `rtree.zarr`. Without the marker and levels halving in size (e.g. the per-chunk arrays `0.zarr`, `1.zarr`, ...),
  a directory is not a pyramid, as before.
- The statistics of a pyramid are those of its marker; none are computed from its levels, and `summarize` has no
  `max_bytes`. The JSON field `stats_level` of pyramid summaries stays, always 0, so that readers of version 1 of
  the JSON output keep working; it is removed with the next version of that contract.
- The bounding box tree of the points is built only for points in memory; that of a point cloud pyramid is read
  from its `rtree.zarr`.
- Decision 0035 stands otherwise: levels by decimation or by mean, the statistics of all the data computed when a
  pyramid is built, the probes.

## Consequences

- Pyramids made before 0.10.0, and pyramids whose command did not finish, are rebuilt with `moraine ras-pyramid` /
  `pc-pyramid` from their data, or with `moraine run FILE --only STEP --force` for a step of a pipeline (a pipeline
  does not rerun a step whose arguments and inputs did not change). The Campi Flegrei demo pyramid `raw/rslc_pyramid`
  of 2026-10-08 is one.
- The pyramid contract stays version 1: the pyramids of 0.10.0 and later are read as before, and no release wrote a
  version 1 pyramid without `method` and `stats`. Its `method`, `stats`, `stats.zarr` and `rtree.zarr` are no longer
  optional, and pyramid summaries always have `method`.
- The JSON output stays version 1: `stats_level` is 0 for every pyramid that is summarized, as it was for every
  pyramid of 0.10.0 and later.

## Do not

- Do not compute statistics from the levels of a pyramid, nor build the tree of a point cloud pyramid from its
  coordinates: rebuild the pyramid.
- Do not suggest rebuilding a raster pyramid from its own `0.zarr` into the same directory: `ras-pyramid` clears
  its output directory first.
- Do not remove `stats_level` without a new version of the JSON output contract.
