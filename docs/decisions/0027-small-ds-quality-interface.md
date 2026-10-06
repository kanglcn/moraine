# 0027 Weighted temporal coherence through `ds_temp_coh`; no EMI quality output; speckle correlation internal

## Status

Accepted

## Date

2026-10-05

## Context

Decisions 0023 to 0026 exposed, besides their algorithms, the EMI quality (outputs of `emi` and of the
fused command), `ds_temp_coh_weighted`, `slc_correlation` with the command `slc-correlation`,
`shp_n_looks`, and the options `oversampling` and `rho2`. To get the weighted temporal coherence a user
had to run one more command and pass its output on, and to choose between two estimates of the effective
number of looks of which the exact one (decision 0025) is never worse. The EMI quality is not comparable
between regularized points (decision 0026) and is not used for the selection any more. The weighted
temporal coherence existed only in the fused command, not in the separate steps `emperical-co-pc`, `emi`,
`ds-temp-coh`.

## Decision

- `emi(coh, ref=0, regularize=True)` returns the phase history only; the commands `emi` and
  `emperical-co-emi-temp-coh-pc` have no EMI quality output.
- `ds_temp_coh(coh, ph, image_pairs=None, block_size=128, n_looks=None)`: with `n_looks` (a number or one
  per point) it returns `(t_coh, t_coh_w, eff_n_pairs)` of decision 0024 from one pass, `t_coh` identical
  to the one without `n_looks`. The command `ds-temp-coh` gets the input `n_looks` and the outputs
  `t_coh_w`, `eff_n_pairs`.
- The effective number of looks of a point is always the exact one of its SHP set (decision 0025), with
  |ρ|² measured from the rslc block being processed (median of 3 evenly spaced images; per raster chunk
  in the commands); the number of SHPs where the block is too small to measure it. `emperical_co_pc(...,
  return_n_looks=True)` returns it (command: `n_looks_dir`); `emperical_co_emi_temp_coh_pc(...,
  weighted=True)` (command: `t_coh_w_dir`, `eff_n_pairs_dir`) computes it the same way, so the fused
  command and the separate steps give the same results (identical on the CPU).
- `slc_correlation`, `shp_n_looks` and `ds_temp_coh_weighted` become internal; the command
  `slc-correlation` and the options `oversampling` and `rho2` are removed.

## Consequences

- Breaking: `ph = emi(coh)` instead of `ph, quality = emi(coh)` (`emi(coh)[0]` still runs and gives the
  phase history of the first point); `ph, t_coh = emperical_co_emi_temp_coh_pc(...)` (four outputs with
  `weighted=True`); pipeline files with `emi_quality`, `emi_quality_dir`, `oversampling` or `rho2` stop
  with an unknown argument error.
- The EMI quality stays available to the tests (`moraine.api.pl._emi`). Without the regularization, a
  temporal coherence near 0 shows the points whose coherence matrix is not positive definite.
- The effective number of looks depends a little on the raster chunk it is measured in (a few percent
  for chunks of a few hundred pixels); measuring |ρ|² costs about 0.2 s per raster chunk of 1000 x 1000
  pixels, only when the weighted outputs are asked for.
- `ds_temp_coh` uses one kernel per device with and without `n_looks`, so that `t_coh` is the same bit for
  bit (two GPU kernels round differently); on an A100 100 000 points x 4186 image pairs take 3.9 ms
  instead of 3.3 ms without `n_looks` (4.1 ms with it), on the CPU 85 instead of 102 ms.
- On the GPU the separate steps and the fused command differ by the rounding of the batched eigen
  solvers of EMI (different batch sizes), as before; on the CPU they are identical.

## Do not

- Do not add an option to choose how the effective number of looks is estimated: the exact one is never
  worse than number of SHPs / oversampling.
- Do not offer the weighted temporal coherence in the fused command only: the separate steps must give
  the same results.
- Do not bring back the EMI quality as a selection criterion while the regularization is on.
