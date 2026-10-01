# 0012 Redundant networks, EMCF unwrapping and time series inversion

## Status

Accepted

## Date

2026-09-28

## Context

moraine stops at unwrapping. `mcf_pc` unwraps every interferogram separately (2D) on a Delaunay
triangulation of the grid indices, with unit costs, the first point as implicit reference and no
quality output. The example unwraps sequential pairs only (bandwidth 1): n - 1 interferograms for
n - 1 unknowns, so unwrapping errors cannot be detected.

After phase linking, the wrapped phase history is consistent in time (zero wrapped closure), so a
redundant network adds information only against unwrapping errors. Redundancy serves two purposes:
temporal unwrapping (EMCF) and a least squares inversion whose residuals measure unwrapping quality.

Reference implementations (isce-framework, reviewed 2026-09-28):

- spurt (`workflows/emcf`): EMCF on sparse points. Temporal step: for each spatial Delaunay edge, the
  wrapped gradients of all interferograms are summed around the triangles of the interferogram network
  in time - perpendicular baseline space and an MCF on that temporal network corrects them. Spatial
  step: per interferogram, MCF on the spatial triangulation with the corrected gradients, then flood
  fill. Costs: constant, distance or centroid.
- dolphin (`timeseries.py`): incidence matrix without the first date; L2 by batched least squares,
  weighted by the Cramer-Rao bound `(1 - g^2) / (2 L g^2)`; L1 by ADMM (20 iterations, cached
  Cholesky of A^T A); missing / unreliable data masked per pixel; residuals per date as quality;
  linear velocity by weighted polyfit. Reference point: centroid of the largest high quality region.

moraine's `mcf_pc` already is the spatial step of EMCF (Delaunay, dual graph MCF with OR-Tools,
integration). `meta.toml` has dates, perpendicular baselines and pixel spacings.

## Decision

Extend the chain after phase linking in five stages, each a complete change (API function, CLI
command, docstring, tests, example and guide):

1. **Network**: `TempNet.from_delaunay(dates, bperp, ...)`: Delaunay triangulation of the images in
   (time, perpendicular baseline) space, normalized by user scales, optionally limited by maximum
   temporal / perpendicular baseline; it also returns the triangles (temporal cycles). The
   `image-pairs` command gets a `meta` input and a `delaunay` method.
2. **Spatial MCF improvements** (`mcf_pc`, output unchanged): coordinates scaled by the pixel spacings
   (or map coordinates), and edge costs `constant` (current behaviour, default) or `distance`.
3. **EMCF** (`emcf_pc`, command `emcf-pc`; its network and the split into steps are changed by decision 0020): inputs point coordinates, the wrapped phase history
   (n_points, nimages) and the network; output unwrapped phase (n_points, n_image_pairs) float32, same
   layout as `mcf_pc`. Temporal MCFs are many and small: batched over spatial edges, parallel.
4. **Inversion** (`invert_pc`, command `ts-inversion-pc`): unwrapped phase (n_points, n_image_pairs)
   and image pairs -> phase time series (n_points, nimages) float32 radians relative to the first
   image, plus the residual per point (n_points,) as quality. Methods `l2` (optionally weighted) and
   `l1` (ADMM as in dolphin: cached Cholesky of A^T A, fixed iterations, batched over points; robust
   to single unwrapping errors; chosen over IRLS, whose per point weights prevent batching). Points processed in chunks, numpy / cupy.
   A single reference network (no redundancy) is copied, not inverted.
5. **Products** (separate record when started): reference point, velocity (linear, optional annual
   terms) with uncertainty, DEM error, stratified atmosphere and ramps, LOS decomposition, export
   (CSV / GeoPackage / GeoTIFF).

The residual of stage 4 gets a pyramid in the example, so `moraine info` reports its statistics and
agents can detect unwrapping problems. New array layouts are added to `docs/contracts/data.md` when
their stage is implemented.

Validation per stage: synthetic data with known deformation and injected 2 pi errors (the network /
EMCF / L1 must detect or remove them), the sample data (`examples/05_unwrap.toml` extended or a new
`06_*.toml`), and rewrapping the result must give the input phase.

## Consequences

- `mcf_pc` and `gamma_mcf_pt` stay; EMCF is an alternative for redundant networks.
- Stages 1-4 need no new dependencies (scipy Delaunay, OR-Tools, numba / cupy).
- Stages are done in order; stage 2 is a separate change, reused by the spatial step of EMCF.
- `examples/05_unwrap.toml` switches from sequential pairs + `mcf-pc` to a Delaunay network, EMCF and
  the inversion once stages 3 and 4 work (one standard chain, no separate example).

## Do not

- Do not invert a network without redundancy and report its residual as a quality measure: it is
  zero by construction.
- Do not change the output layout of `mcf_pc` / `gamma_mcf_pt`.
- Do not add external data dependencies (weather models, TEC maps) in stages 1-4.
