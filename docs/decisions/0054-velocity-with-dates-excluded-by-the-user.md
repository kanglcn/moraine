# 0054 Velocity of points by least squares, with the dates to exclude given by the user

## Status

Accepted

## Date

2026-11-16

## Context

Decision 0048 planned two last steps of the products: a residual RMS of every image over the points of the region of
the reference, from which the reference date (the lowest RMS, with a change of the time reference) and the dates to
exclude (RMS above 3 times the median) were chosen automatically (step 6, after the `residual_RMS` and
`reference_date` steps of MintPy), then the velocity of every point over the dates left (step 7).

StaMPS does not choose dates itself: `ps_calc_ifg_std` prints the standard deviation of every interferogram, and the
user drops interferograms by setting `drop_ifg_index`; the velocity (`ps_plot` with `v`) is the least squares of a
line over the interferograms left. A threshold on the RMS of an image depends on the data (scene size, season,
deformation that the line does not describe, which raises the RMS of the first and last images when the deformation
accelerates), and a change of the reference date changes every displacement of the products, which moraine already
fixes at the reference image of 05 and the reference area of 06.

## Decision

- `pc-velocity` (API `moraine.pc_velocity`, in `moraine.api.ts` and `moraine.cli.ts`): least squares, with equal
  weights, of the displacement of every point (mm, `(n_points, nimages)`, positive towards the satellite) by
  d(t) = a + v t + sum_j (c_j cos(2 pi t / P_j) + s_j sin(2 pi t / P_j)), t in years (365.25 days) from the first
  date; a straight line by default, the periods P_j in years optional (`periods`, e.g. [1] or [1, 0.5]).
- The dates to exclude are given by the user (`exclude`, YYYYMMDD of `dates`); none by default. Nothing is excluded
  automatically and the reference date does not change: the displacement products are not changed by the fit.
- Outputs: the velocity v (mm per year, `(n_points,)` float32), its standard deviation for uncorrelated noise from
  the residuals, sqrt(RSS / (n - p) (G^T G)^-1_vv) with n the fitted dates of the point and p the coefficients,
  optionally the residual RMS of every point over its fitted dates (mm), the coefficients `(n_points, p)` and the
  residual RMS of every image over the points of a mask (mm, `(nimages,)`, all points by default). The image RMS
  includes the excluded dates and is a diagnostic only: it shows the images to look at before excluding them.
- Points with values missing at some dates are fitted over the dates where they have values; fewer than p + 2 such
  dates, or dates at which the coefficients are not determined (e.g. periodic terms of points left with dates one
  period apart), give nan. Fewer than p + 2 dates fitted in all, or periods that cannot be told apart at the fitted
  dates, are argument errors (decision 0052).
- One task per block of points with all its images; the normal equations of the points are solved by a Cholesky
  factorization in array operations, without BLAS and LAPACK, on the CPU or the GPU (cupy) for the API.
- Example 06: a `velocity` step on the displacement without the DEM error, with the variable `exclude` (empty) and
  pyramids of the velocity, its standard deviation and the residual RMS of the points.
- This replaces steps 6 and 7 of decision 0048; its other steps stand.

## Consequences

- Validation: synthetic point clouds (the model exact without noise; the least squares of numpy; the standard
  deviation equal to the spread of the velocity errors of 20 000 points with white noise within 3 %; the fit with a
  date excluded equal to the fit without that date; nan); Campi Flegrei below.
- Campi Flegrei (336 830 points, 92 images): 0.0 mm per year in the reference area, 85.7 at Pozzuoli (median),
  standard deviation 0.69 mm per year (median). In the reference area the velocities of the odd and of the even dates
  differ as their standard deviations predict (0.56 against 0.60 mm per year); at Pozzuoli the residual holds the
  acceleration of the uplift (+13, -13, +13 mm at the first, middle and last image), and the standard deviation is
  larger than that difference (2.4 against 0.36). The means of the difference over 1 km cells vary by 0.62 mm per
  year: the atmosphere left is correlated in space, and the velocities of areas kilometers apart are uncertain by
  about 0.3 mm per year beyond the standard deviation of the points.
- The two images of the largest RMS on Campi Flegrei (2021-09-12 and 2021-07-14, 3.4 and 2.7 times the median over
  all points, 4.4 and 4.0 times over the points 8 km or more from the uplift) are planes of about 1.5 mm per km left
  after ERA5, not unwrapping errors; excluding them changes the velocities by 0.10 mm per year (median). An automatic
  threshold of 3 times the median would have excluded only the first one over all points and both over the stable
  points.
- The standard deviation assumes uncorrelated noise: atmosphere correlated in space and time makes it too small for
  areas, deformation that the model does not describe makes it too large for points (decision 0048).
- The RMS of the images over a deforming area also holds the deformation the model does not describe (on Campi
  Flegrei it ranks the last images higher); `mask` takes it over other points.
- 10 million points and 92 images take 12.6 s with 4 threads and 2.5 GB, reading and writing the chunks included
  (16 threads: 10.7 s, 8.0 GB).

## Do not

- Do not exclude dates or change the reference date automatically without a new decision.
- Do not change the displacement products by the velocity fit (excluded dates stay in them).
- Do not report the standard deviation of the velocity as its uncertainty without saying that it assumes
  uncorrelated noise.
