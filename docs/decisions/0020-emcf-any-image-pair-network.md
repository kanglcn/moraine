# 0020 EMCF on any network of image pairs, in steps of their own units

## Status

Accepted

## Date

2026-09-30

## Context

Decision 0012 (stage 3) made `emcf_pc` build its own network: the Delaunay triangulation of the images
in the plane of time and perpendicular baseline. The temporal step and a final per point repair of the
loops were min cost flows on the dual of that triangulation (residues on the triangles of images), so they
needed a planar triangulation and could not take the image pairs chosen by the user (e.g. a network without some
images, or the Hop-3 network of spurt, whose pairs only follow the time order).

On a spatial edge (p, q), with wrapped gradients of the images d_i and wrapped interferogram gradients
g_k, the corrected gradients g_k + 2 pi z_k are differences of image gradients exactly when
z = m + n_a - n_b for integer cycles n of the images, m_k = round((d_a - d_b - g_k) / 2 pi). The
temporal step is therefore min over n of sum_k c_k |m_k + n_a - n_b| (an L1 potential problem on the
graph of image pairs); its dual is a min cost circulation with capacities c_k. On a planar
triangulation this is the dual min cost flow of 0012 (the earth node is the outer face): same optimal
cost on every spatial edge of the synthetic tests.

The API must also be split into units for data larger than memory (`docs/development.md`).

## Decision

- `emcf_pc(pc_x, pc_y, ph, t=None, image_pairs=None, ...)`: the interferograms are the given image
  pairs, in their order and direction; by default every image with the next three (Hop-3). The
  perpendicular baseline is not used; `bperp`, `t_scale`, `bperp_scale` and `exclude` are removed
  (images are left out by leaving them out of `image_pairs`). `temporal_cost` 'length' uses the time
  span of the pairs only. `earth_cost` applies to the network of points only.
- Costs only where they have a counterpart in the literature: the temporal step corrects an
  interferogram at cost 'constant' (1) or 'length' (shorter time spans cost more, like the distance costs
  of spurt), default 'constant'; the spatial step places phase jumps first on long edges ('length', like
  spurt) and at points of low quality ('weight', like coherence weights of GAMMA or Costantini),
  default 'length'. The former adaptive costs are removed: the temporal 'gradient' (|wrapped gradient|
  close to pi cheaper) and the spatial 'gradient' and 'correction' (edges corrected by the temporal step
  cheaper). Measured on the synthetic data of `tests/unwrap_benchmark.py` (median share of wrong (point,
  interferogram) after `emcf_pc` and the closure correction): clusters with bridges 0.0010 with them,
  0.0041 without (the spatial pair together made the difference); islands 0.0178 and 0.0169; uniform
  noise 0.0072 and 0.0058. They are removed nevertheless, without a basis and with a benefit on one kind
  of synthetic data only; better costs (e.g. the temporal coherence of the edges, as in PSI networks) are
  to be measured on real data.
- The temporal step solves the L1 potential problem on any graph of image pairs (one numba solver,
  `_l1_fit`: spanning forest, then negative cycle cancelling of the dual circulation). A network
  without loops is accepted with a warning (every interferogram is then unwrapped alone).
- The repair, its options (`repair`, `repair_cost`) and the `misclosure` output are removed from
  `emcf_pc`, which returns `(unw, image_pairs)`: loops that do not close are corrected by a separate
  step for any unwrapper (decision 0021).
- Two steps on their own units, chained in memory by `emcf_pc` and later mapped by the CLI over zarr
  chunks (one command, intermediate results in a temporary zarr):
  temporal step per block of spatial edges (`_emcf_temporal`, rows of the phase history and the two
  end points of every edge; the edges are sorted by their smaller and larger point index so that a
  block of edges reads a block of rows, and keep the direction of a half-edge of the triangulation so
  that ties between equal cost corrections do not depend on the point order), spatial step per
  interferogram (`_emcf_spatial`). Only the coordinates and the triangulation of the points stay in
  memory for the whole run.
- The number of interferograms unwrapped at the same time is by default bounded by the available
  cores and half of the available memory (cgroup limits included), with about 200 bytes per point
  each.

## Consequences

- Results with the Delaunay network of 0012 can still be obtained by passing its pairs; with the same
  pair costs they equal the previous results up to equal cost optima.
- The default network changes from the (time, baseline) Delaunay triangulation to Hop-3: results of
  `emcf_pc` and `emcf-pc` change.
- The Delaunay network in time and baseline is to be offered by the `image-pairs` command
  (0012 stage 1), not inside `emcf_pc`.

## Do not

- Do not require a triangulation of the image pairs again in the temporal step.
- Do not load the whole phase history in a step that works per unit once the CLI maps them.
