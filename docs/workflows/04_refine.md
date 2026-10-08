# 04 Merge PS and DS, refine with n2ft

Pipeline: `examples/04_refine.toml`. Tutorials: `nbs/Tutorials/CLI/Xinpu/04_refine.ipynb`, `nbs/Tutorials/CLI/CampiFlegrei/04_refine.ipynb`.
Needs `02_ps` and `03_ds`.

```bash
moraine run examples/04_refine.toml --workdir WORK
```

## Steps

1. `pc-union` of the PS candidates (their rslc) and the refined DS (their phase history), with e/n;
   `ras2pc` adds lon/lat.
2. `image-pairs` (sequential) and `n2ft`: Noise2Fringe Transformer filtering of the point cloud
   interferograms (chunks of 20 000 points with k nearest neighbour halos).
3. `temp-coh` between the filtered and the raw phase; points with temporal coherence > 0.8 are kept
   (`pc-logic-pc`, `pc-select-data`).

Outputs (WORK/pc/): merged candidates `pc_can_*`; refined points `pc_hix.zarr`, `pc_ph.zarr`
(n_points, nimages), `pc_e.zarr`, `pc_n.zarr`; pyramids `pc_can_temp_coh_pyramid`, `pc_ph_pyramid`.

## Parameters

- Temporal coherence threshold: 0.7-0.85; lower keeps more points with more noise.
- `n2ft` `chunks`: points per chunk (20 000); smaller uses less GPU memory.

## Expected results (sample data)

| result | sample value | sane range |
|---|---|---|
| merged candidates | 984 239 | about PS + DS minus overlap |
| `pc_can_temp_coh_pyramid` | p50 0.53, p99 0.99 | 0..1 |
| refined points (`pc_hix`) | 352 530 (36 % of the candidates) | |

Run time: about 1 minute with an A100 (n2ft 38 s).

## Figures

Quicklooks of the sample data run of 2026-10-08 (the numbers above are from the same run):

![Temporal coherence of the merged candidates after the Noise2Fringe Transformer](../assets/workflows/sample_04_temp_coh.webp)

*Temporal coherence of the merged candidates after the Noise2Fringe Transformer (`pc/pc_can_temp_coh_pyramid`).*

![Sequential interferogram 5 of the refined points](../assets/workflows/sample_04_pc_intf_seq_5.webp)

*Sequential interferogram 5 of the refined points (`pc/pc_ph_pyramid --show intf_seq --index 5`): continuous fringes, no noise.*

## Checks

- `moraine quicklook pc/pc_ph_pyramid --show intf_seq --index I -o pc_intf.png`: interferograms of
  the refined points should show smooth, spatially continuous phase (fringes), not noise.
- Compare with the raw interferogram: `moraine quicklook raw/rslc_pyramid --show intf_seq --index I`.

## Problems

- `n2ft` is slow on CPU; use a GPU.
- Too few points: lower the temporal coherence threshold or revisit PS / DS thresholds.
