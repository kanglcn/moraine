# 0024 Weighted DS temporal coherence with noise-free squared coherence weights

## Status

Accepted

## Date

2026-10-05

## Context

The DS temporal coherence `ds_temp_coh` is |Σₖ exp(jψₖ)| / P over all P image pairs, ψ the residual
phase of the phase history. Every pair counts the same, so the incoherent pairs of a point that is coherent
only in part of the pairs (short time spans in vegetation, seasons) add random phases and lower it: the
point is rejected although its coherent pairs fit. This is the dilution that ISBAS avoids with a hard mask
of the pairs (coherence > 0.25).

Under phases that close, ψₖ is estimation noise with variance (1 − |γₖ|²) / (2n|γₖ|²) for n looks
(Cramér–Rao bound of the interferometric phase), so the likelihood of independent pairs is
Σ κₖ cos ψₖ with κₖ ∝ |γₖ|² / (1 − |γₖ|²); n cancels in the normalization. With the sample coherence
the weights are biased: incoherent pairs have E|γ̂|² = 1/n, and at n = 50-121 SHPs thousands of such
pairs still carry about half of the weight. Measured with the phases of regularized EMI (decision 0023),
2026-10-04, Xinpu (60 images, 287 000 candidates) and Campi Flegrei (92 images, 400 000):

- weights |γ̂|²: Xinpu t ≥ 0.8 for 28 % instead of 14 %, but correlation 0.97 with the uniform t, i.e.
  a shift, not a new ranking;
- noise-free weights max(0, (|γ̂|² − 1/nₑ) / (1 − 1/nₑ)), nₑ = SHPs / 2: Xinpu 41 %, and a new
  ranking: among points with uniform t < 0.6, the spatial consistency with reference points 11-20 pixels
  away (mean cos of the phase difference over the 12-day interferograms) is 0.51 for weighted t ≥ 0.8,
  0.28 for 0.6-0.8 and 0.15 below 0.6; selecting the same number of points, the points chosen by the
  weighted instead of the uniform t have 0.52 instead of 0.35 at the top 40 % (0.71 instead of 0.76 at
  the top 14 %). Campi Flegrei hardly changes (its coherence is flat in time span).

## Decision

- New `ds_temp_coh_weighted(coh, ph, n_looks, image_pairs=None)` returns the weighted temporal
  coherence |Σ wₖ exp(jψₖ)| / Σ wₖ with wₖ = max(0, (|γ̂ₖ|² − 1/n_looks) / (1 − 1/n_looks)) (the
  squared coherence without the noise bias; exact in mean at γ = 0 and γ = 1; close to the Fisher weight
  for small coherence and without its divergence at |γ| → 1) and the effective number of image pairs
  (Σw)² / Σw². NaN and 0 where no pair has a positive weight or n_looks ≤ 1. `ds_temp_coh` is unchanged.
  (Changed by decision 0027: `ds_temp_coh(..., n_looks=)` returns them, `ds_temp_coh_weighted` is internal.)
- `emperical_co_emi_temp_coh_pc` (API `weighted`, command `t_coh_w_dir`, `eff_n_pairs_dir`) computes both
  with n_looks = number of SHPs / `oversampling` (pixels per independent look, default 1.0: no assumption,
  the weaker noise correction). (Changed by decision 0027: always the effective number of looks of the
  SHP set of decision 0025, no parameter.)
- The CPU version is numba (one point per iteration); the GPU version is a numba.cuda kernel with one
  warp per point (coalesced reads of the image pairs, shuffle reduction), as asked by the maintainer: on an
  A100 100 000 points x 4186 pairs take 5 ms (`ds_temp_coh` with a thread per point: 22 ms); CPU with 128
  threads as fast as `ds_temp_coh`. numba.cuda kernels take cupy arrays without a copy. The GPU version
  of `ds_temp_coh` uses the same kernel structure instead of a cupy kernel with a thread per point
  (uncoalesced reads): 3.6 ms instead of 22 ms for the same case.

## Consequences

- One parameter, the oversampling of the SLCs; the results depend on it (Xinpu t ≥ 0.8: 34 % for 1,
  41 % for 2). It can be measured once per data set (equivalent number of looks of multilooked intensity
  in homogeneous areas against the number of pixels) instead of guessed; not part of this change.
- A weighted temporal coherence from few effective pairs is optimistic: select with both outputs.
- The weighted temporal coherence is not a likelihood test (the pairs of one point are correlated); its
  expected value for phases that close still depends on the information of the point, so thresholds are
  empirical.

## Do not

- Do not weight by |γ̂|² without the noise correction: on Xinpu it only shifts the uniform temporal
  coherence.
- Do not use the Fisher weights |γ̂|² / (1 − |γ̂|²) without a limit: a few pairs close to 1 dominate.
- Do not select by the weighted temporal coherence without the effective number of image pairs (Changed by decision 0028: without `n_components`, not `eff_n_pairs`.)
