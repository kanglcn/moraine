# 0025 The oversampling of the SLCs is measured from the speckle correlation

## Status

Accepted

## Date

2026-10-05

## Context

Neighbouring pixels of an SLC are correlated (oversampling of the focused data, interpolation of the
coregistration), so n SHPs are fewer independent looks. Every noise level of a coherence estimate depends
on the effective number of looks nₑ, not on n: incoherent image pairs have E|γ̂|² = 1/nₑ (the noise floor
removed by the weights of decision 0024), the EMI coherence magnitude matrix needs nₑ ≳ M/2 to be positive
definite (decision 0023), precision bounds scale as 1/√nₑ.

For a circular Gaussian speckle with complex correlation coefficient ρ(Δ) between pixels at lag Δ, the
variance of a second order estimate over a set S of n pixels is proportional to Σ_{p,q∈S} |ρ(p − q)|² / n²,
so nₑ = n² / Σ_{p,q∈S} |ρ(p − q)|² ≈ n / f for a compact S, with the oversampling
f = Σ_Δ |ρ(Δ)|² (sampling rate over the equivalent noise bandwidth of the spectrum, a product of an
azimuth and a range factor). ρ is the response of the system and of the resampling, the same for every
scatterer, so f is one number per data set.

Measured 2026-10-05 (central 1000 x 1000 pixels, 10 images): Campi Flegrei f = 2.59 (azimuth 1.9 x range
1.4, |ρ| 0.64 at the azimuth and 0.43 at the range neighbour), Xinpu 2.75; between images ± 3 %. With 121
SHPs at most, nₑ ≤ 47 for Campi Flegrei: M / (2nₑ) ≥ 1 for its 92 images, which is why 97 % of its DS
candidates had a coherence magnitude matrix that is not positive definite.

## Decision

- `slc_correlation(slc, max_lag=(4, 6))` (API) returns |ρ(Δ)|² of one SLC, (2·max_lag + 1) lags per axis:
  - each pixel divided by the square root of its local mean power (15 x 15 pixels), so that bright extended
    structures, whose neighbours are more correlated than speckle, do not dominate (without it Campi
    Flegrei gave 2.87);
  - per azimuth line pair |Σ a b*|² / (Σ|a|² Σ|b|²) − 1/m, averaged over the line pairs: an azimuth phase
    ramp (TOPS) is constant along a line pair, while an average over a block of lines cancels the azimuth
    correlation (it gave f_az = 1.00);
  - the remaining bias of the correlated pixels of a line (about (f_range − 1)/m per lag, 4 % of f in
    total) is removed with the mean of the outermost lags, where the correlation is 0.
- The command `slc-correlation` writes the median table over evenly spaced images of a central block,
  with the sum as attribute `oversampling`, the value for `oversampling` of
  `emperical-co-emi-temp-coh-pc` (nₑ = number of SHPs / f; decision 0024). (Removed by decision 0027:
  the commands measure the table themselves, per raster chunk.)
- `shp_n_looks(pc_is_shp, rho2)` returns the exact nₑ of each SHP set, n² / Σ_{p,q∈S} |ρ(p − q)|² (negative
  table values as 0, lags outside the table as 0); `emperical-co-emi-temp-coh-pc --rho2` uses it for the
  weighted temporal coherence instead of n / `oversampling` (decision 0027: always, internally) (CPU numba, GPU numba.cuda with a warp per
  point: 200 000 points in 0.18 s on 128 cores, 27 ms on an A100).

## Consequences

- For compact SHP sets n / f is 7-11 % below the exact nₑ (full window, compact blobs, strips), for
  scattered SHPs up to 40 % below (50 scattered pixels: 19.6 against 32.3), i.e. the noise floor of the
  weights is then set a little too high; `rho2` avoids it.
- Xinpu with the measured f = 2.75 instead of 2 (decision 0024): weighted temporal coherence >= 0.8 for
  46 % instead of 41 % of the candidates, the points it adds to the uniform one (uniform < 0.6) 49 000
  instead of 33 000 with about the same spatial consistency (0.48 against 0.51).
- On synthetic speckle with a known kernel the estimate is 1.5 % below the truth; texture, holes and
  azimuth phase ramps do not change it. 0.06 s per image of 1000 x 1000 pixels (numba).
- A window that keeps EMI without regularization for M images needs about 2 M f SHPs (490 for 92 images
  and f = 2.6, a 23 x 23 window).

## Do not

- Do not estimate the azimuth correlation of TOPS SLCs over blocks of lines.
- Do not use the number of SHPs as the number of independent looks.
- Do not use the intensity (power weighted) correlation without normalizing the local power: urban
  structures bias it upwards.
