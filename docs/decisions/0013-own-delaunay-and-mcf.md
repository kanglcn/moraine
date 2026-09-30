# 0013 Own Delaunay triangulation and successive shortest path MCF

## Status

Accepted

## Date

2026-09-28

## Context

`mcf_pc` used scipy's Qhull triangulation and OR-Tools' min cost flow. Measured on 10 million points
(synthetic, known truth): Qhull 101 s, edge extraction (`np.unique`) 9 s, OR-Tools 73-170 s depending
on the edge order, integration 6 s, peak memory about 3.5 GB above the data; GAMMA `mcf_pt` takes
35 s. Qhull breaks cocircular ties (frequent on the integer grid of radar coordinates) by input order,
so results depended on the point order; OR-Tools' solution among equal cost optima too.

Residues usually pair with close neighbours, which suits successive shortest paths (SSP, also used by
GAMMA): each Dijkstra stops at the nearest sink and stays local.

## Decision

- `moraine/api/unwrap/delaunay_.py`: own sweep-hull triangulation (following Delaunator, ISC license, notice in
  the file) with exact integer predicates and coordinate based tie breaking, returning half-edges.
- `moraine/api/unwrap/mcf.py`: residues, SSP min cost flow and integration work directly on the half-edges (the
  dual graph is read from them, no derived arrays). Arc cost 1 between triangles and `earth_cost`
  (default 1, the previous behaviour) across the convex hull.
- OR-Tools is removed from the dependencies.

Measured on 10 million points: 15.9 s in total (triangulation 10 s, SSP 2.2 s, integration 3.2 s),
about 0.6 GB above the data. The flow is exactly optimal (same cost as OR-Tools, no negative cycle in
the residual network), results are identical for any point order, errors against the truth equal
GAMMA's and OR-Tools'.

## Consequences

- Correctness is tested independently of other programs (`tests/test_unwrap.py`: optimality
  certificate, brute force on tiny sets, flow conservation, known truth, order independence;
  `tests/test_delaunay.py`: identical to Qhull where unique, Delaunay property where cocircular).
- Results differ from GAMMA `mcf_pt` and from the previous OR-Tools version by 2 pi at some points in
  low coherence areas: equal cost optima and different valid triangulations. GAMMA behaves like
  `earth_cost=3`; on the same network moraine's flow is never more expensive.
- The half-edge indices are int32: at most about 3.5e8 points.

## Do not

- Do not test the unwrapping by point by point equality with GAMMA or another solver.
- Do not make the result depend on the point order again (e.g. index based tie breaking).
- Do not add `triangle` (Shewchuk) as a dependency: its license forbids commercial redistribution,
  which conflicts with GPL-3.0.
