# 0035 Pyramid levels by decimation or by mean; statistics of the data computed when a pyramid is built

## Status

Accepted

## Date

2026-10-09

## Context

The levels of the pyramids (decision 0007 / 0018) were made by decimation: level `l` is every `2**l`-th
pixel. A decimated overview of a phase history or of interferograms is speckle, so the whole scene of a
complex pyramid showed nothing and the check "fringes should be continuous" could only be made zoomed in;
decimated coherences and counts are noisy too. The statistics of `moraine info` and the colour range of the
views were computed at every call from the finest level of at most 64 MiB: a sample, 0.1-0.3 s per call, and
they would be biased by averaged levels.

Averaging is a multilook: valid for real values and for phases relative to a reference image (the phase
histories of phase linking, filtered interferograms, points), but not for the pixels of an rslc stack, whose
scatterer phases differ from pixel to pixel, so that the mean of SLC pixels has a random phase and the
interferogram formed from averaged SLCs is not the multilooked interferogram. A pyramid cannot tell an SLC
stack from a phase history.

## Decision

- `ras-pyramid` and `pc-pyramid` take `method`: `decimate` (rasters) / `first` (point clouds), the default
  and the only way until now, or `mean`: every level is the mean of the `2 x 2` blocks of the finer level over
  its finite pixels (cells with points), exact (weighted by the number of finite data pixels of each cell),
  complex data averaged as complex numbers, integer and boolean data as float32 levels. The marker of the
  pyramid records the method.
- The default stays decimation because the tool cannot know whether complex data may be averaged; the
  pipelines choose: the example pipelines build the rslc pyramid decimated and every other pyramid with
  `mean`, and a mean pyramid of the n2f interferograms for the whole-scene check of the fringes.
- The statistics of a pyramid (nan fraction, minimum, maximum, mean, standard deviation, percentiles of a
  regular sample of 2**23 values) are computed from all the data while the pyramid is built and stored in its
  marker, with a per channel table (`stats.zarr`); `moraine info` and the views read them. Pyramids made
  before are still summarized from a coarse level.
- The probes of the views give the value of the cell drawn at the zoom shown (the mean, for mean levels);
  the time series stay those of the data.

## Consequences

- The whole scene of a mean pyramid is readable (fringes, smooth coherence); `moraine info` is exact and
  instant and names the images of a stack that are all nan or constant.
- The levels of integer and boolean data are float32 with `mean`; the cells of integer and boolean point
  clouds are float32 (nan where empty) with both methods.
- `ras-pyramid` reads the raster in bands of `rows` lines (bounded memory), `pc-pyramid` holds the rasterized
  channel (three rasters with `mean`).

## Do not

- Do not build a mean pyramid of an rslc stack (SLC pixels); make interferograms or phase histories first.
- Do not compute statistics from the levels of a mean pyramid; use the stored ones (or level 0).
