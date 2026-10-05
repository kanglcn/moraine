# 0026 EMI is regularized by default; DS are selected by temporal coherence

## Status

Accepted

## Date

2026-10-05

## Context

Decision 0023 added the adaptive regularization of EMI with `regularize` defaulting to False, so that the
verified examples did not change. Without it EMI silently returns a phase history that follows the
estimation noise wherever the coherence magnitude matrix is not positive definite, and that is the common
case for real stacks: 97 % of the DS candidates of Campi Flegrei (92 images) and 98 % of Xinpu (60 images)
with 11 x 11 windows, because 121 SHPs are only about 45 independent looks (decision 0025). With the
regularization the EMI quality of a point depends on its β and cannot be compared between points, so the
example's selection by EMI quality (1.0 to 1.05) and temporal coherence no longer fits.

## Decision

- `regularize` defaults to True in `emi` and `emperical_co_emi_temp_coh_pc` (API and commands). Points
  with a positive definite, well conditioned coherence magnitude matrix are unchanged.
- `examples/03_ds.toml` selects DS by temporal coherence alone (0.8 to 1.0); the EMI quality range and the
  intersection are removed. The EMI quality pyramid stays as a diagnostic.

## Consequences

- The sample data (17 images) change little: see `docs/workflows/03_ds.md` to `05_unwrap.md` for the
  numbers of the rerun examples.
- Code that used negative EMI qualities to find points with a matrix that is not positive definite needs
  `regularize=False`.
- Decision 0023 stays accepted; its default is changed by this record.

## Do not

- Do not select regularized DS by a fixed range of the EMI quality.
- Do not switch the regularization off for stacks of many images without checking the fraction of
  negative EMI qualities.
