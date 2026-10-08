# 05 Phase unwrapping

Pipeline: `examples/05_unwrap.toml`. Tutorials: `nbs/Tutorials/CLI/Xinpu/05_unwrap.ipynb`, `nbs/Tutorials/CLI/CampiFlegrei/05_unwrap.ipynb`.
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
| `unw/pc_unw_pyramid` | p01 -6.0, p50 0.08, p99 10.4, min -20.7, max 18.5 | a few multiples of 2 pi |

Run time: 2.3 s for 352 530 points and 16 interferograms (CPU, 16 at the same time).

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
is a separate step, after `emcf-pc` or `mcf-pc`; it also gives the unwrapped phase of every image:

```toml
[[step]]
name = "closure"
run = "unwrap-correct-closure-pc"
gix = "unw/pc_gix.zarr"
ph = "pc/pc_ph.zarr"
unw_ph = "unw/pc_unw_emcf.zarr"
image_pairs = "unw/pairs_hop3.txt"
ts = "unw/pc_ts.zarr"
misclosure_fraction = "unw/pc_misclosure_fraction.zarr"
change_fraction = "unw/pc_change_fraction.zarr"
region = "unw/pc_region.zarr"
range_pixel_spacing = 4.290037
azimuth_pixel_spacing = 3.740175
```

Where the loops do not add up to zero, every region (points connected without edges longer than 4 times
the median edge length, e.g. a cluster or an island) is shifted by whole cycles in the interferograms that
disagree with the majority (MintPy's phase closure correction, decision 0021); points of regions of fewer
than 30 points are corrected one by one and marked -1 in `region`. Where the correction of its region does
not close every loop of a point, the interferograms fitting most of its loops are kept (decision 0022). Sample data: 0.8 s, 144 regions (3.0 % of the points in smaller ones), 12.6 % of the values
changed, mostly in the interferograms across the winter gap (October 2021 to June 2022, with the snow image
2021-10-25; three of them at almost every point) and in those of September and October 2022.

Output: `ts` (n_points, nimages) float32, the unwrapped phase of every image in radians relative to image
`ref` (default 0, its column is 0); interferogram (a, b) is `ts[:, a] - ts[:, b]`, so every loop closes.
Rewrapping `ts[:, j]` gives the phase of `ph[:, j] * conj(ph[:, ref])`. `ts` has no spatial reference yet:
compare points only relative to each other. On the tutorial data (157 365 points, 45 interferograms of
17 images, 2026-10-02): 0.6 s, rewrapping max 2.0e-6 rad, 1.5 % of the values differ from the region
correction alone; `change_fraction` > 0 at 78 % of the points (mean 0.039, p99 0.13, max 0.24). Without a
truth it is unknown whether every change is right: it relies on most interferograms of a region being right,
so look at the quicklooks of `ts`, `misclosure_fraction` and `change_fraction`. 10 million points,
100 images, 294 interferograms: 145 s, 3.3 GB peak memory.

`change_fraction`, the share of the interferograms of a point that the correction changed, is the point
quality of this step for masking: on synthetic data with known truth it separated points wrong in at least
one interferogram from right ones better than `misclosure_fraction` (area under the ROC curve 0.55-0.96).
The threshold depends on the data; there is no default. No measure of this step sees errors that close every
loop, e.g. the same whole cycles in every interferogram of one image (0.4-9 % of the points without any
misclosure were wrong after `emcf-pc` on the synthetic data): combine it with the phase quality (temporal
coherence) of the points. `ph` must be chunked one image per chunk
(`docs/contracts/data.md`). Large data: 10 million points, 100 images, 294 interferograms take 13 min with
4 interferograms at the same time (8.2 GB peak memory) and 6 min with 32 (51 GB, the default on a
32 core machine is bounded by the cores and half of the available memory).

`temporal_cost = "length"` (with `dates`, the acquisition dates) corrects the longest interferograms first
where loops disagree; `spatial_cost = "weight"` (with `weight`, e.g. the temporal coherence) places phase
jumps first at points of low quality, `spatial_cost = "length"` on long edges. Both costs are
`constant` by default; on synthetic data with known truth (`tests/unwrap_benchmark.py`) the median share
of wrong (point, interferogram) is 0.83 % against 1.54 % when every interferogram is unwrapped alone with
`mcf-pc` (decision 0020).

## Figures

Quicklooks of the sample data run of 2026-10-08 (the numbers above are from the same run):

![Unwrapped sequential interferogram 5](../assets/workflows/sample_05_unw_5.webp)

*Unwrapped sequential interferogram 5 (`unw/pc_unw_pyramid --index 5`): smooth, no isolated 2π jumps.*

## Checks

- `moraine info unw/pc_unw_pyramid`: no warnings; values within a few tens of radians.
- `moraine quicklook unw/pc_unw_pyramid --index I -o unw.png`: smooth, no isolated 2 pi jumps between
  neighbouring areas. The Delaunay network connects all points, so point clusters separated by large
  gaps (water, vegetation) are linked by long edges, where 2 pi errors between clusters are most likely.

## Problems

- `mcf-pc` needs unique points (unique gix); the refined points from `04_refine` are unique.
- Unwrapping errors along gaps: more points (lower thresholds in 02-04) make the network denser.
