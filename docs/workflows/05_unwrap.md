# 05 Phase unwrapping

Pipeline: `examples/05_unwrap.toml`. Tutorial: `nbs/Tutorials/CLI/phase_unwrapping.ipynb`.
Needs `04_refine`.

```bash
moraine run examples/05_unwrap.toml --workdir WORK
```

Set `[vars] shape` to the (nlines, width) of `raw/rslc.zarr`.

## Steps

1. `pc-gix`: grid index of the refined points (from their hilbert index).
2. `image-pairs`: sequential pairs (bandwidth 1).
3. `mcf-pc`: minimum cost flow unwrapping on a Delaunay network of the points, one interferogram at a
   time (moraine's own implementation: exactly optimal, independent of the point order; `earth_cost`
   sets the cost of phase jumps across the border of the point cloud, default 1). `gamma-mcf-pt` uses
   GAMMA's `mcf_pt` instead and takes e/n; its results differ by 2 pi at some points in low coherence
   areas, where several unwrappings are equally good.
4. `pc-pyramid` of the unwrapped phase on the map.

Output: `WORK/unw/pc_unw.zarr` (n_points, n_image_pairs) float32, unwrapped phase in radians.

## Expected results (sample data)

| result | sample value | sane range |
|---|---|---|
| `unw/pc_unw_pyramid` | p01 -5.8, p50 -0.07, p99 8.6, min -14.7, max 19.9 | a few multiples of 2 pi |

Run time: about 12 s for 157 189 points and 16 interferograms (CPU, one worker).

Correctness check (in python): rewrapping the result must give the input phase,
`np.angle(np.exp(1j*unw) * np.conj(intf))` near 0 (sample: max 1.1e-6 rad).

## Checks

- `moraine info unw/pc_unw_pyramid`: no warnings; values within a few tens of radians.
- `moraine quicklook unw/pc_unw_pyramid --index I -o unw.png`: smooth, no isolated 2 pi jumps between
  neighbouring areas. The Delaunay network connects all points, so point clusters separated by large
  gaps (water, vegetation) are linked by long edges, where 2 pi errors between clusters are most likely.

## Problems

- `mcf-pc` needs unique points (unique gix); the refined points from `04_refine` are unique.
- Unwrapping errors along gaps: more points (lower thresholds in 02-04) make the network denser.
