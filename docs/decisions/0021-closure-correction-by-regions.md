# 0021 Unwrapping errors are corrected by phase closure per region, as in MintPy

## Status

Accepted

## Date

2026-09-30

## Context

With a phase history (consistent image phases), the unwrapped interferograms of every loop of image pairs
must add up to zero; at a point i, unw_k = phi_a - phi_b + 2 pi c_k with integer cycles c_k, and the loops
close if c = A N for integer cycles N of the images (A the incidence matrix of the pairs). Spatial
unwrapping of every interferogram (`mcf_pc`, and the spatial step of `emcf_pc`) breaks this where it places
phase jumps differently in different interferograms.

MintPy (Yunjun et al. 2019, Computers & Geosciences 133, 104331, `unwrap_error_phase_closure.py`) estimates
the integer ambiguities from the triplet closure of a few sampled pixels per common connected component of
SNAPHU (L1 regularized least squares, rounded), takes their median per component and adds it to the whole
component; pixels outside large components are not corrected (and are masked later). It assumes that most
interferograms of a component are right.

Measured on synthetic point clouds with known truth (clusters with bridges, uniform noise, islands separated
by water with point x image decorrelation; Hop-3 networks; after `mcf_pc` and `emcf_pc`):

- a per point correction (every point by its own loops) fixes whole shifted clusters, but creates salt and
  pepper errors where single points decorrelate in some images (isolated wrong points x1.5 to x2.5, edge
  errors x2 to x3 on the islands);
- regions made of the edges where the unwrapping placed phase jumps do not enclose the errors: only a third
  of the boundary edges of a wrongly unwrapped area carry such a jump, the rest are jumps of the truth that
  the unwrapper missed;
- regions made of the reliable edges (as SNAPHU's components) enclose them: without the longest edges of the
  point network, 98 % of the errors are constant per region;
- per region, the correction fixes a region wrong in a few interferograms and makes worse one wrong in many,
  correlated in time (as MintPy states);
- weighting the pairs by their time span (the former EMCF repair) was worse than unit weights everywhere.

## Decision

- New API function `unwrap_correct_closure_pc(pc_x, pc_y, ph, unw, image_pairs, max_edge_factor=4.0,
  min_region_points=30) -> (unw_cor, misclosure_fraction, region)`, separate from the unwrappers.
- Per point, the correction d = A N - c closest to c in L1 with unit weights (`_l1_fit`, exact integer
  solution, any network, no triplets).
- Regions: connected components of the point triangulation without the edges longer than `max_edge_factor`
  times the median edge length. Every region of at least `min_region_points` points gets, per
  interferogram, the median of d of its points.
- Unlike MintPy, points of smaller regions (e.g. isolated points between clusters) are corrected one by one
  (they are kept in moraine's results; leaving them uncorrected doubled the edge errors on the clusters with
  bridges), and the regions are returned (-1 for small regions) for masking.
- `misclosure_fraction`: per point, the fraction of the interferograms that do not fit the loops before the
  correction.
- Units for the CLI: per block of points (d, int8), per interferogram (median per region); the triangulation
  and the region of every point stay in memory.

## Consequences

- The correction works after any unwrapper and needs a network with loops (e.g. Hop-3); sequential pairs
  are returned unchanged with a warning.
- Measured on 10 million points and 84 pairs: 24 s (18 s of them for the triangulation), within the memory
  of the unwrapping before it.
- Regions from other reliabilities (point quality, coherence) and the bridging method of MintPy are not
  part of this decision.

## Do not

- Do not correct every point by its own loops alone where it belongs to a large region (salt and pepper).
- Do not make regions from the phase jumps of the unwrapping.
- Do not weight the pairs of the correction by their time span without a new measurement.
