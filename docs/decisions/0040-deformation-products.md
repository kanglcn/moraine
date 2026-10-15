# 0040 Deformation products: displacement time series, DEM error and velocity

## Status

Accepted

## Date

2026-10-15

## Context

Stage 5 of decision 0012. The processing ends with the unwrapped phase of every image, `ts` of
`unwrap-correct-closure-pc` (decision 0022): (n_points, nimages) float32 radians relative to a reference image,
with an unknown offset per image that is common to all points (no spatial reference). `examples/05_unwrap.toml`
still makes sequential interferograms with `mcf-pc` only; the EMCF chain that gives `ts` is shown in the guide.
Users need line of sight displacements in millimeters relative to a stable area, their velocity with an uncertainty,
the quality of every point and files for GIS.

MintPy (`smallbaselineApp`) references all interferograms to one pixel before the network inversion, then
subtracts per image corrections from the time series (solid earth tides, ionosphere, troposphere, ramps), then
estimates the DEM error with a quadratic deformation model, takes the residual RMS of every image to choose the
reference date and to exclude noisy dates, and fits the velocity last. It references that early because
interferograms unwrapped one by one carry arbitrary constants that do not close in loops, which the closure
correction and the inversion cannot handle. In moraine every interferogram is the difference of the phase history
of the points and is unwrapped congruent to its wrapped phase, so the loops close up to whole cycles without a
spatial reference. A spatial reference subtracts one value per image from all points; it commutes with the per
image corrections (fields from models or external data) and with the conversion to meters.

Sign of `ts`, checked on the Campi Flegrei run of the 05 guide (EMCF, every image with the next three, closure
correction, 92 images from 2019-01-02 to 2021-12-29): the median of `ts` of the points within 500 m of Rione
Terra (Pozzuoli) minus that of the easternmost 2 % of the points grows steadily to +57.9 rad in the last image,
25.6 cm with lambda / (4 pi). Pozzuoli rose during these years, so a positive `ts` is a shorter range, a motion
towards the satellite.

## Decision

- The products start from `ts`. The standard chain of `examples/05_unwrap.toml` becomes EMCF on the network of
  every image with the next three plus the closure correction, as decision 0012 planned.
- Steps on point clouds (n_points, n) chunked (points_block, 1), in this order:
  1. No new selection of points: the points are those of 04, selected by the n2ft temporal coherence before the
     unwrapping. What the unwrapping adds, the change fraction and the region of every point from the closure
     correction, is kept as flags of the points, not used to drop them.
  2. `pc-disp`: d = 1000 lambda / (4 pi) * ts in millimeters (lambda in meters), positive towards the satellite.
     The only place of the sign and of the unit.
  3. Per image corrections from models or external data (solid earth tides, troposphere from ERA5 or GACOS,
     ionosphere): optional, not in the first version; each subtracts a field per image, interpolated at the
     points.
  4. `pc-reference`, spatial reference: a reference value is subtracted from every image, by default the median
     of the reference points (a bool mask, e.g. from `polygon-mask`, within one region of the closure correction),
     optionally their mean or the value of one given point. After the per image corrections, so that it is done
     once.
  5. DEM error per point: least squares of d(t) = a0 + a1 t + a2 t^2 (+ steps at given dates) + c(t) dh with
     c(t) = 1000 B_perp(t) / (R sin(incidence)), t in years; outputs the displacement without c(t) dh (mm), dh (m)
     and the residual time series (mm).
  6. Residual RMS of every image over the points of the region of the reference: the reference date (lowest RMS, optional change of the
     time reference) and the dates to exclude (RMS > 3 times the median).
  7. Velocity per point: least squares of a time function (linear by default, an annual cosine and sine
     optional) over the dates not excluded; velocity (mm / year), its standard deviation from the residuals and
     the residual RMS (mm).
  8. Turbulent atmosphere: optional and later, a spatial low pass of the temporal high pass of the residuals of
     step 5, subtracted, then steps 4 to 7 again.
  9. Export (points to CSV / GeoPackage, rasters to GeoTIFF): a later change.
- Units: displacements and residuals in millimeters, velocities in millimeters per year, the DEM error in meters,
  time in years (days / 365.25 from the reference image), phases in radians. Displacements and velocities are
  positive towards the satellite (as in MintPy).
- The commands take the values they need as arguments and never read `meta.toml`: the radar wavelength (m), the
  dates (YYYYMMDD) and the perpendicular baselines (m) of the images as lists, whose values are in the metadata
  written by `load-gamma-metadata`; the slant range and the look vector elevation angle of every point from the 04
  outputs (`pc_range`, `pc_theta`; incidence = 90 degrees - elevation).
- Quality per point kept with the products: temporal coherence (n2ft), misclosure and change fractions (closure
  correction), residual RMS (DEM error and velocity fits), velocity standard deviation.
- API in a new module `moraine.api.ts`, commands in `moraine.cli.ts`.
- Validation: synthetic point clouds with known velocity, acceleration, DEM error, annual term and noise must be
  recovered within the noise; on Campi Flegrei the sign (uplift of Pozzuoli towards the satellite) and the
  consistency of the steps. A comparison with GNSS is a later step.

## Consequences

- A new example pipeline 06 (products) and its guide follow 05; `docs/contracts/data.md` gets the displacement
  arrays.
- The DEM error and the velocity are fitted separately: the quadratic model of step 5 keeps accelerating
  deformation (Campi Flegrei 2019-2021) out of the DEM error; the velocity of step 7 is the summary a user
  chooses.
- One reference for all points: points of regions of the closure correction other than the one of the reference
  (separated by water or decorrelated areas) can be off by whole cycles in some images, which the loops do not
  detect. Campi Flegrei: 87 regions, the largest with 79 % of the points, the second 13 %, 1 346 points in regions
  too small to be corrected together; the closure correction changed almost nothing (change fraction p99 0.4 %).
  The region stays a flag of every point; checking or fixing the offsets between regions (e.g. bridging as in
  MintPy) is a later change.
- The DEM error is estimated after the unwrapping. A DEM error large enough to disturb the unwrapping or the
  spatial n2ft filter (points on roofs next to ground points) would have to be estimated before n2ft; to be
  checked on the points of buildings.
- A troposphere estimated from the correlation of the phase with the height also removes deformation that is
  correlated with the height (Campi Flegrei: uplift at the coast, rim higher); external data are preferred there.
- The velocity uncertainty from the residuals assumes uncorrelated noise; atmosphere left in the time series
  makes it too small.

## Do not

- Do not take `meta.toml` as an input of the product commands.
- Do not convert phases to displacements outside `pc-disp`, and do not use another sign: displacements are
  positive towards the satellite everywhere.
- Do not apply a spatial correction after the reference without referencing again.
