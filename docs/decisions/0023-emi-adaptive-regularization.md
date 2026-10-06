# 0023 EMI regularizes coherence matrices that are not positive definite, adaptively per point

## Status

Accepted

## Date

2026-10-04

## Context

EMI (Ansari et al. 2018, `emi`) is the maximum likelihood phase estimator for the covariance
Σ = Θ A Θᴴ, Θ = diag(exp(jθ)), with a real positive definite coherence magnitude A. It replaces A by
the estimate |Γ| (elementwise modulus of the sample coherence Γ) and takes the phases from the
eigenvector of the smallest eigenvalue (the EMI quality) of |Γ|⁻¹ ∘ Γ.

|Γ| is not always positive definite. If it is, |Γ|⁻¹ ∘ Γ is positive semidefinite (Schur product
theorem), so a negative EMI quality proves that it is not. The problem is then not a likelihood:
|Γ|⁻¹ ∘ Γ = Σₖ (1/aₖ) diag(uₖ) Γ diag(uₖ) over the eigenpairs (aₖ, uₖ) of |Γ|, the term of a negative
eigenvalue close to 0 dominates, and the phase history follows that eigenvector, i.e. the estimation
noise (temporal coherence close to that of random phases).

For any phase vector θ, |Γ| = Re(Θᴴ Γ Θ) + T with Re(Θᴴ Γ Θ) positive semidefinite and
Tᵢⱼ = |γᵢⱼ| (1 − cos ψᵢⱼ) ≥ 0, ψ the phase residual: |Γ| is positive semidefinite when the phases close,
and loses it when the phase noise (sampling, or physical non-closure) outweighs the smallest eigenvalue
of the real part, which falls to 0 as the number of independent looks approaches half the number of
images (Marchenko–Pastur). Campi Flegrei (92 images, 11 × 11 window, at most 121 SHPs, oversampled
Sentinel-1): 97 % of 1.79 million DS candidates had a negative EMI quality although their mean coherence
magnitude was 0.53 at 12 days and 0.35 at 960 days. The sample data (17 images) are hardly affected.

## Decision

- `emi` and `emperical_co_emi_temp_coh_pc` (API and commands) get `regularize` (bool, default False).
  With it the coherence matrix of a point becomes Γ_β = (1 − β) Γ + β I, data and weights together
  (|Γ_β| = (1 − β) |Γ| + β I), with the smallest β ≥ 0 such that
  - λ_min(|Γ_β|) ≥ (1 − β) δ, δ = max(0, −λ_min(|Γ|)): the true matrix is positive definite, so δ is a
    lower bound of the norm of the estimation error of |Γ| (and (1 − β) δ of the regularized one);
    solution β = 2δ / (1 + 2δ);
  - λ_min(|Γ_β|) ≥ λ_max(|Γ_β|) / 10⁵, so that the float32 inverse keeps about two digits (an internal
    constant, not a parameter).

  Points with a positive definite |Γ| of condition number at most 10⁵ get β = 0 and the results of
  `regularize=False`, bit for bit on the CPU.
- Why this family and this β: for every β a coherence matrix whose phases close is solved exactly with
  quality 1 (Fiedler: λ_min(A ∘ A⁻¹) = 1 for A = |Γ_β|); β = 0 is EMI and β → 1 tends to the estimator
  that maximizes Σ |γᵢⱼ|² cos ψᵢⱼ without any inverse. β therefore does not bias phases that close; it
  trades the efficiency of EMI (which needs an accurate |Γ|⁻¹) for robustness. The rule takes the
  smallest β that the data of the point prove necessary; a rule from the number of SHPs would need the
  effective number of independent looks, unknown for oversampled and resampled SLCs.
- β is not an output. The EMI quality of a regularized point is the smallest eigenvalue of the
  regularized problem (at least β, Schur's inequality; no longer an output since decision 0027). The temporal coherence is computed with Γ as
  estimated; it depends only on the phases of Γ, which the regularization does not change.

## Consequences

- The EMI quality is not comparable between points with different β: for the same misfit q − 1
  decreases with β (as (1 − β)² for β close to 1). With `regularize`, DS are selected by the temporal
  coherence, not by a fixed EMI quality range.
- Campi Flegrei (1.79 million candidates, 92 images, `emperical-co-emi-temp-coh-pc` on one A100,
  2026-10-04): β > 0 for 97.7 % of the candidates (median 0.27, 99 % below 0.62 in a sample of 400 000);
  temporal coherence ≥ 0.8 for 1 187 702 candidates instead of 49 089 (the old refinement by EMI quality
  and temporal coherence kept 18 126), and their phase linked interferograms show continuous fringes of
  the caldera uplift. Points that keep β = 0 have identical results.
- Cost: one more eigenvalue decomposition of a real nimages × nimages matrix per point. Campi Flegrei:
  19 min 03 s instead of 18 min 15 s for the fused command (the coherence estimation dominates); `emi`
  alone: GPU +5 % time and 412 instead of 398 kB GPU memory per point, CPU (16 threads) 177 instead of
  137 ms per point and thread.
- The default stays False so that the verified examples do not change. Making it the default and
  selecting the DS of `examples/03_ds.toml` by temporal coherence is a separate change with a rerun of
  the examples (decision 0009); done by decision 0026.

## Do not

- Do not regularize only |Γ|⁻¹ (the weights): a coherence matrix whose phases close then no longer gets
  quality 1 (0.39 instead of 1 for β = 0.2 on a synthetic case) and the qualities lose their scale.
- Do not taper the coherence matrix with a hard 0/1 band of image pairs: the band matrix is not positive
  semidefinite (Dirichlet kernel), so the tapered matrix can lose positive semidefiniteness. If tapering
  is added, use positive definite functions of the time span (triangle, exponential, Gaussian).
- Do not select regularized DS by a fixed EMI quality range.
