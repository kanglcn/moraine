# 0022 The phase closure correction outputs the phase time series; no least squares inversion

## Status

Accepted

## Date

2026-10-01

## Context

Decision 0012 (stage 4) planned a time series inversion after unwrapping (`invert_pc`, `l2` / `l1` by ADMM)
that turns the unwrapped interferograms (n_points, n_pairs) into image phases (n_points, nimages) and
reports its residual per point as unwrapping quality.

In moraine the wrapped phase history is consistent: after phase linking (and for PS, whose interferograms
are differences of image phases) the wrapped closure is zero. At a point, unw_k = phi_a - phi_b + 2 pi c_k
with integer cycles c_k, so the unwrapped closure of a loop is 2 pi times an integer, and the only unknowns
left are integer cycles N of the images (c = A N where every loop closes). Hence:

- where every loop closes, the network is consistent: L2, L1 and integration along any spanning tree of the
  image pairs give the same image phases, with zero residual; a least squares solver adds nothing;
- where loops do not close, L2 spreads a 2 pi error over all images as a fraction of a cycle (harder to
  detect than the jump itself); L1 picks the majority, which is what `_l1_fit` of the closure correction
  (decision 0021) already does exactly, in integers;
- a residual of the inversion only counts the interferograms that do not fit the loops, which the closure
  correction can report directly.

The inversion of SBAS (averaging independent noise of multilooked interferograms whose wrapped closure is
not zero) does not apply to moraine's points.

## Decision

- Stage 4 of decision 0012 (`invert_pc`, `ts-inversion-pc`) is not implemented. The image phases are an
  output of `unwrap_correct_closure_pc`, which gets the reference image `ref` (int, default 0) and returns
  `(ts, misclosure_fraction, change_fraction, region)`; `unw_cor` is removed (interferogram (a, b) is
  `ts[:, a] - ts[:, b]`).
- Steps (1 and 2 as in decision 0021, 3 new):
  1. per point: `_l1_fit` of the cycles, corrections d (n_points, n_pairs) and `misclosure_fraction`;
  2. per interferogram: points of large regions get the median of d of their region, points of small
     regions their own d;
  3. per point: `_l1_fit` on the result of step 2 gives the integer cycles N of the images, relative to
     image `ref`; ts = angle(ph * conj(ph[:, ref])) + 2 pi (N - N_ref).
  The median is taken over d, not over N: d is constant in a region where spatial unwrapping shifted it,
  N is not (it holds the spatial unwrapping of every point).
- Outputs:
  - `ts`: unwrapped phase of every image relative to image `ref` in radians, (n_points, nimages), float32;
    `ts[:, ref] = 0`; rewrapped, `ts[:, j]` is the phase of `ph[:, j] * conj(ph[:, ref])`;
  - `misclosure_fraction`: (n_points,) float32, 0..1, interferograms not fitting the loops before the
    correction (unchanged);
  - `change_fraction`: (n_points,) float32, 0..1, interferograms whose whole cycles in `ts` differ from the
    input `unw` (steps 2 and 3 together), as point quality for masking;
  - `region`: unchanged.
- A network without loops (e.g. sequential pairs) is accepted with the warning of 0021: N follows from
  the spanning tree, nothing is corrected, both fractions are 0.
- Point quality: measured on synthetic data with known truth (3 kinds of `tests/unwrap_benchmark.py` and
  dense points with errors inside the point cloud, after `emcf_pc` and after `mcf_pc` of every Hop-3
  interferogram, 3 realisations each, 2026-10-02), the area under the ROC curve for points wrong in at least
  one interferogram of `ts` was 0.55-0.96 for `change_fraction`, 0.55-0.88 for `misclosure_fraction`,
  0.50-0.97 for the changes of step 3 alone (useful on dense points only, 0.50-0.60 otherwise) and 0.50 for
  being in a small region. `change_fraction` is returned; the changes of step 3 alone are not. No point
  measure of this step sees errors that close every loop (`c = A N`, e.g. the same cycles in every
  interferogram of one image): 0.4-9 % of the points without any misclosure were wrong after `emcf_pc`.
- CLI units: step 3 is a third pass per block of points over the result of step 2 (temporary zarr) and the
  input; `ts` is chunked by blocks of points and one image (decision 0019).
- The spatial reference (reference point or area) is not part of this step: it subtracts a constant per
  image that is not a whole cycle, so `ts` would no longer rewrap to `ph`, and changing it should not rerun
  the correction. It is the first product of stage 5 of decision 0012, a separate change.

## Consequences

- One unwrapped result per point, consistent by construction: every loop of `ts` closes exactly.
- The output layout of `unwrap-correct-closure-pc` changes from (n_points, n_pairs) to (n_points, nimages):
  `docs/contracts/data.md`, `docs/workflows/05_unwrap.md`, the tutorial and the tests change; when
  `examples/05_unwrap.toml` switches to EMCF (decision 0012), `change_fraction` gets a pyramid there so
  that `moraine info` reports it.
- Masking needs more than this step: phase quality (e.g. temporal coherence of phase linking) and checks of
  temporal and spatial consistency of `ts` for the errors that close every loop are separate changes.
- `mcf-pc` and `gamma-mcf-pt` results of sequential pairs also get image phases through this step.
- Step 3 is one more pass over the points (reading the result of step 2 and the input): 10 million points,
  100 images, 294 pairs take 145 s instead of 94 s, with the same peak memory (3.3 GB).
- Decisions 0012 (stage 4) and 0021 (outputs) stay accepted, with a note that this record changes them.

## Do not

- Do not add a least squares (L2 / L1 / ADMM) inversion of unwrapped interferograms of a consistent phase
  history without a new measurement showing a benefit.
- Do not take the median of the image cycles N over a region.
- Do not subtract a reference point inside the closure correction.
- Do not report the changes of step 3 alone (or a flag of small regions) as point quality: on synthetic data they
  do not predict wrong points.
