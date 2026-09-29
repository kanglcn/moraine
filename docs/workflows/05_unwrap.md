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

## Alternative: EMCF

`emcf-pc` unwraps a redundant network instead of sequential pairs: the interferograms of the Delaunay
triangulation of the images in time and perpendicular baseline (from `raw/meta.toml`), unwrapped
together. It writes the unwrapped phase and the image pairs it used (a file like the one of
`image-pairs`):

```toml
[[step]]
name = "unwrap_emcf"
run = "emcf-pc"
gix = "unw/pc_gix.zarr"
ph = "pc/pc_ph.zarr"
meta = "raw/meta.toml"
unw_ph = "unw/pc_unw_emcf.zarr"
pairs = "unw/pairs_emcf.txt"
```

Sample data: 42 interferograms, about 17 s for the refined points (20 s for the 293 814 DS points). At
every point the interferograms of every triangle of images close: for three images a < b < c,
unw(a, b) + unw(b, c) = unw(a, c). The optional output `misclosure` is, per point, the fraction of image
triangles that did not close before the network was made consistent (sample: mean 21 % for the refined
points, 26 % for the DS points); where it is high the result relies on the majority of the interferograms.
Coordinates are converted to meters with the pixel spacings of `raw/meta.toml`; `weight` (e.g. the temporal
coherence) can make phase jumps prefer low quality points (`spatial_cost` with `weight`).

How the defaults were chosen (`tests/unwrap_benchmark.py`, synthetic data with known truth: clusters of
points linked by sparse points, a winter gap, seasonal deformation, DEM error, atmosphere, noise; 8
realisations): with the defaults the median share of wrong cycles is 0.10 % (worst realisation 10 %,
wrong neighbour differences 0.088 %), against 1.08 % (43 %, 0.174 %) when every interferogram is unwrapped
alone with `mcf-pc`, and 0.22 % (26 %, 0.106 %) for the EMCF of spurt with distance costs, which takes
more than 50 times longer (9 s against 0.1 s for 29 000 points). On the sample data the defaults differ from `spatial_cost = "constant"` in 20 - 30 %
of the values; without a truth there it is unknown which is right, so look at the quicklooks when the
result matters.

## Checks

- `moraine info unw/pc_unw_pyramid`: no warnings; values within a few tens of radians.
- `moraine quicklook unw/pc_unw_pyramid --index I -o unw.png`: smooth, no isolated 2 pi jumps between
  neighbouring areas. The Delaunay network connects all points, so point clusters separated by large
  gaps (water, vegetation) are linked by long edges, where 2 pi errors between clusters are most likely.

## Problems

- `mcf-pc` needs unique points (unique gix); the refined points from `04_refine` are unique.
- Unwrapping errors along gaps: more points (lower thresholds in 02-04) make the network denser.
