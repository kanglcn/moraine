# 05 Phase unwrapping

Pipeline: `examples/05_unwrap.toml`. Tutorial: `nbs/Tutorials/CLI/phase_unwrapping.ipynb`.
Needs `04_refine`.

```bash
moraine run examples/05_unwrap.toml --workdir WORK
```

Set `[vars] shape` to the (nlines, width) of `raw/rslc.zarr` and the pixel spacings to the ones of the
data (`range_pixel_spacing`, `azimuth_pixel_spacing` in meters, from a `*.rslc.par`).

## Steps

1. `pc-gix`: grid index of the refined points (from their hilbert index).
2. `image-pairs`: sequential pairs (bandwidth 1).
3. `mcf-pc`: minimum cost flow unwrapping on a Delaunay network of the points (in meters from the pixel
   spacings), every interferogram alone, several at the same time (moraine's own implementation: exactly
   optimal, independent of the point order; `earth_cost` sets the cost of phase jumps across the border of the
   point cloud, default 1; `spatial_cost = "length"` places them first on long edges). `gamma-mcf-pt` uses
   GAMMA's `mcf_pt` instead and takes e/n; its results differ by 2 pi at some points in low coherence areas,
   where several unwrappings are equally good.
4. `pc-pyramid` of the unwrapped phase on the map.

Output: `WORK/unw/pc_unw.zarr` (n_points, n_image_pairs) float32, unwrapped phase in radians.

## Expected results (sample data)

| result | sample value | sane range |
|---|---|---|
| `unw/pc_unw_pyramid` | p01 -6.1, p50 -0.07, p99 8.6, min -19.9, max 19.9 | a few multiples of 2 pi |

Run time: 0.8 s for 157 193 points and 16 interferograms (CPU, 16 at the same time).

Correctness check (in python): rewrapping the result must give the input phase,
`np.angle(np.exp(1j*unw) * np.conj(intf))` near 0 (sample: max 1.1e-6 rad).

## Alternative: EMCF

`emcf-pc` unwraps a redundant network instead of sequential pairs: the interferograms of a network of
image pairs that closes loops, e.g. every image with the next three (`image-pairs` with `bandwidth = 3`),
unwrapped together. The pixel spacings convert the grid index to meters (from the `*.rslc.par` of the
data; 4.29 m in range and 3.74 m in azimuth for the sample):

```toml
[[step]]
name = "pairs_hop3"
run = "image-pairs"
rslc = "pc/pc_ph.zarr"
bandwidth = 3
out = "unw/pairs_hop3.txt"

[[step]]
name = "unwrap_emcf"
run = "emcf-pc"
gix = "unw/pc_gix.zarr"
ph = "pc/pc_ph.zarr"
image_pairs = "unw/pairs_hop3.txt"
unw_ph = "unw/pc_unw_emcf.zarr"
range_pixel_spacing = 4.290037
azimuth_pixel_spacing = 3.740175
```

Sample data (2026-09-30): 45 interferograms of 17 images, 6.3 s for the 157 193 refined points; rewrapping
gives the input phase (max 2.6e-6 rad). The loops of image pairs do not close at every point after EMCF:
on the sample, 12.8 % of the interferograms of a point (mean) do not fit them. The phase closure correction
is a separate step, after `emcf-pc` or `mcf-pc` on a network with loops:

```toml
[[step]]
name = "closure"
run = "unwrap-correct-closure-pc"
gix = "unw/pc_gix.zarr"
ph = "pc/pc_ph.zarr"
unw_ph = "unw/pc_unw_emcf.zarr"
image_pairs = "unw/pairs_hop3.txt"
unw_cor = "unw/pc_unw_emcf_cor.zarr"
misclosure_fraction = "unw/pc_misclosure_fraction.zarr"
region = "unw/pc_region.zarr"
range_pixel_spacing = 4.290037
azimuth_pixel_spacing = 3.740175
```

Where the loops do not add up to zero, every region (points connected without edges longer than 4 times
the median edge length, e.g. a cluster or an island) is shifted by whole cycles in the interferograms that
disagree with the majority (MintPy's phase closure correction, decision 0021); points of regions of fewer
than 30 points are corrected one by one and marked -1 in `region`. Sample data: 0.8 s, 144 regions (3.0 %
of the points in smaller ones), 12.6 % of the values changed, mostly in the interferograms across the
winter gap (October 2021 to June 2022, with the snow image 2021-10-25; three of them at almost every point)
and in those of September and October 2022. Without a truth it is unknown
whether every change is right: it relies on most interferograms of a region being right, so look at the
quicklooks of `unw_cor` and `misclosure_fraction`. 10 million points, 294 interferograms: 108 s, 3.3 GB
peak memory. `ph` must be chunked one image per chunk
(`docs/contracts/data.md`). Large data: 10 million points, 100 images, 294 interferograms take 13 min with
4 interferograms at the same time (8.2 GB peak memory) and 6 min with 32 (51 GB, the default on a
32 core machine is bounded by the cores and half of the available memory).

`temporal_cost = "length"` (with `dates`, the acquisition dates) corrects the longest interferograms first
where loops disagree; `spatial_cost = "weight"` (with `weight`, e.g. the temporal coherence) places phase
jumps first at points of low quality, `spatial_cost = "length"` on long edges. Both costs are
`constant` by default; on synthetic data with known truth (`tests/unwrap_benchmark.py`) the median share
of wrong (point, interferogram) is 0.83 % against 1.54 % when every interferogram is unwrapped alone with
`mcf-pc` (decision 0020).

## Checks

- `moraine info unw/pc_unw_pyramid`: no warnings; values within a few tens of radians.
- `moraine quicklook unw/pc_unw_pyramid --index I -o unw.png`: smooth, no isolated 2 pi jumps between
  neighbouring areas. The Delaunay network connects all points, so point clusters separated by large
  gaps (water, vegetation) are linked by long edges, where 2 pi errors between clusters are most likely.

## Problems

- `mcf-pc` needs unique points (unique gix); the refined points from `04_refine` are unique.
- Unwrapping errors along gaps: more points (lower thresholds in 02-04) make the network denser.
