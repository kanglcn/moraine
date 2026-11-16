# 0053 The troposphere stratified with height, fitted to the displacement as a power law of the height

## Status

Accepted

## Date

2026-11-16

## Context

ERA5 (decision 0051) gives the troposphere of every image from a weather model: independent of the deformation, but
with the errors of the model, whose grid of 25 km does not hold the local stratification of the lower troposphere.
On the Campi Flegrei it removed about 40 % of the variance of the slope of the 12 day differences of the displacement
with height. The stratified troposphere can also be fitted to the displacement itself. Bekaert et al. (2015, JGR,
doi:10.1002/2014JB011558; the TRAIN toolbox) write the difference of the delays of two dates as a power law of the
height, K (h0 - h)^alpha below the height h0 above which the delays do not differ, with one K per interferogram
(a linear function of the height for alpha 1), and fit it on band filtered data in overlapping windows, rejecting
the bands that deformation spoils (defaults h0 10 km, alpha 1.6, 50 windows, bands of 2 to 128 km). Deformation that
varies with the height gets into K unless it is kept out of the fit.

## Decision

- `pc-tropo-powerlaw` (API `pc_tropo_powerlaw`, module `tropo`): in every image the displacement of the points of
  `mask` (all points by default) is fitted by least squares with c + K (h0 - h)^alpha, h the height above the geoid
  (`hgt - geoid`, as `pc-tropo-era5`); the troposphere of every point is K ((h0 - h)^alpha - (h0 - h_m)^alpha), h_m
  the median height of the points of the fit, in millimeters with the sign of `pc-disp`; the corrected displacement is
  `disp - out`. One command for both relations: alpha 1 is the linear one (h0 not used). Defaults of TRAIN: alpha 1.6,
  h0 10 000 m. Outputs the troposphere and K with its standard deviation per image.
- One K for the whole scene per image, no band filtering, no windows (this record); the band filtering with its
  consistency test, windows with a K varying over the scene and h0 and alpha from the profiles of ERA5 are later steps
  (`docs/roadmap.md`).
- The mask is the way to keep deformation out of the fit: the points of an area without deformation that varies with
  the height.
- Not in the default chain of `examples/06_products.toml`: the guide shows it as an optional step, on the
  displacement before or after the correction with ERA5.

## Consequences

- One task per image reads its column of the displacement; the heights and the mask stay in memory (about 15 bytes
  per point), a task holds about 50 bytes per point: 10 million points and 92 images in 24 s and 2.5 GB with four
  threads.
- Without a good mask, deformation that grows or falls with the height (volcanoes, slopes) is taken for troposphere
  and removed.
- Campi Flegrei (336 830 points, 92 images, mask of the points 8 km or more from Pozzuoli): the slope of the 12 day
  differences with height drops from 2.64 to 0.93 mm per 100 m (ERA5 2.04), 80 % of the pairs improve (ERA5 60 %),
  also outside the mask; linear and power law give the same with 400 m of relief. ERA5 alone improves the agreement
  of the DEM error from the odd and the even dates (0.42 instead of 0.27, the power law 0.28): it holds the horizontal
  part. Without a mask the uplift of Pozzuoli gets into K (slope of the median image 0.3 instead of 5.1 mm per
  100 m).

## Do not

- Do not put it into the default chain of example 06 without a new decision record.
- Do not fit it without a mask on data whose deformation varies with the height.
