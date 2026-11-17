# 0048 Deformation products, with the DEM error in two parts

## Status

Accepted

## Date

2026-11-15

## Context

Decision 0040 set the deformation products of the image phases `ts` (stage 5 of decision 0012); its context (the
order of MintPy, why moraine can reference late, the sign of `ts`) still holds. Its step 5 fitted the displacement of
every point by a quadratic deformation model (plus steps) and the DEM error term. A model of the shape of the
deformation leaves in the DEM error whatever the model does not describe (seasonal, episodic or other non-quadratic
motion), and the model has to be chosen for every area.

StaMPS estimates the DEM (look angle) error in two parts. The spatially uncorrelated part is fitted per point in its
temporal coherence, on the wrapped phase minus the filtered phase of its neighbourhood, and removed from the phase
before the unwrapping. The spatially correlated part is fitted on the unwrapped phase (`ps_calc_scla`, single master):
least squares with equal weights of the differences between neighbouring dates by a constant, the time step (a
constant velocity) and the DEM error term, then clamped to the range of its neighbours (`ps_smooth_scla`). Least
squares of the differences with equal weights is generalized least squares of the time series with the covariance
of a random walk: a deformation whose velocity changes freely, without a model of its shape. The deformation enters
the DEM error only through the changes of its velocity between neighbouring dates (summation by parts); white noise
of the time series (atmosphere, noise of the points) weighs 1.5 times as much as in least squares of the time series.

In moraine the temporal coherence of 04 compares the phase of every candidate with its interferograms filtered by
n2ft, the phase of its neighbourhood. A point dh above its neighbourhood (a roof among ground points) differs from
them by 4 pi / lambda (B_perp,a - B_perp,b) dh / (R cos(theta)) in the interferogram of the images a and b, which the
temporal coherence takes for noise: Sentinel-1 baselines of neighbouring dates differ by about 70 m, a roof 20 m
above the ground keeps 0.81 of its coherence, 30 m 0.62. The phase that 05 unwraps holds this term. The perpendicular
baseline of every point is known (decision 0044).

## Decision

- 04 (refinement): the height, look vector, slant range and perpendicular baselines are taken for the candidates.
  `pc-su-dem-error` fits the DEM error of every candidate relative to its neighbourhood in the filtered
  interferograms: the dh within -max_dh .. max_dh (20 m by default, as StaMPS; 100 m in the Campi Flegrei example for
  the buildings of Naples) that maximizes its temporal coherence with the term removed, and that coherence, which
  selects the points (> 0.8 as before). The phase of the points without this DEM error is the phase 05 unwraps.
- The products start from `ts`. The standard chain of `examples/05_unwrap.toml` is EMCF on the network of every image
  with the next three plus the closure correction.
- Steps on point clouds (n_points, n) chunked (points_block, 1), in this order:
  1. No new selection of points: the points are those of 04. The change fraction and the region of every point from
     the closure correction are flags of the points, not used to drop them.
  2. `pc-disp`: d = 1000 lambda / (4 pi) * ts in millimeters (lambda in meters), positive towards the satellite. The
     only place of the sign and of the unit.
  3. Per image corrections from models or external data (solid earth tides, troposphere from ERA5 or GACOS,
     ionosphere): optional, not yet; each subtracts a field per image, interpolated at the points.
  4. `pc-reference`, spatial reference: a reference value is subtracted from every image, by default the median of
     the reference points (a bool mask, e.g. from `polygon-mask`, within one region of the closure correction),
     optionally their mean or the value of one given point. After the per image corrections, so that it is done
     once.
  5. `pc-sc-dem-error`, the spatially correlated DEM error: least squares with equal weights of the differences of the
     displacement between neighbouring dates by a constant, the time step and (c_k - c_k-1) dh, with
     c_k = 1000 B_perp,k / (R cos(theta)) millimeters per meter (B_perp,k of the point, R its slant range, theta the
     elevation of its look vector); outputs dh (m) and the displacement without (c_k - c_ref) dh (mm). The whole DEM
     error is the sum of the two parts (a `math` step of the example). Unlike StaMPS: the B_perp of every point
     instead of the mean of the scene in the fit; no clamping of the spatially correlated part to the range of its
     neighbours, as the part relative to the neighbourhood from n2ft is smaller than the local DEM error (see below)
     and the rest is in this part; no constant of the reference image (its noise is a matter of step 6).
  6. (Replaced by decision 0054: no automatic choice of dates.) Residual RMS of every image over the points of the
     region of the reference, from a residual defined with this step: the reference date (lowest RMS, optional
     change of the time reference) and the dates to exclude (RMS > 3 times the median).
  7. (Done as decision 0054, with the dates to exclude given by the user.) Velocity per point: least squares of a
     time function (linear by default, an annual cosine and sine optional) over the dates not excluded; velocity
     (mm / year), its standard deviation from the residuals and the residual RMS (mm).
  8. Turbulent atmosphere: optional and later, a spatial low pass of the temporal high pass of the displacement,
     subtracted, then steps 4 to 7 again.
  9. Export (points to CSV / GeoPackage, rasters to GeoTIFF): later. (Points done as decision 0055, to GeoPackage
     only; rasters to GeoTIFF later.)
- Units: displacements in millimeters, velocities in millimeters per year, DEM errors in meters, phases in radians.
  Displacements and velocities are positive towards the satellite.
- The commands take the values they need as arguments and never read `meta.toml`: the radar wavelength (m) and the
  dates (YYYYMMDD) as a list, whose values are in the metadata written by `load-gamma-metadata` (pipeline files read
  them with variables from a key of `meta.toml`); the perpendicular baselines, the slant range and the look vector
  elevation angle of every point from the 04 outputs (`pc_bperp`, `pc_range`, `pc_theta`; incidence = 90 degrees -
  elevation).
- Quality per point kept with the products: temporal coherence (with the DEM error relative to the neighbourhood),
  misclosure and change fractions (closure correction), residual RMS and velocity standard deviation (velocity fit).
- API `pc_su_dem_error` in `moraine.api.pqm` (with the temporal coherence), `pc_sc_dem_error` and the other products in
  `moraine.api.ts`; commands in `moraine.cli.pqm` and `moraine.cli.ts`.
- Validation: synthetic point clouds (both parts exact without noise and with the error of their least squares with
  noise; the coherence of random phases with the fit); the sign and size of c against GAMMA's `phase_sim_orb` of a
  height 500 m above and below the DEM; Campi Flegrei.

## Consequences

- The part relative to the neighbourhood is relative to the neighbourhood in n2ft; the spatially correlated part is
  relative to the reference area: `pc-reference` subtracts the displacement of the reference points, their DEM error
  term included (StaMPS also subtracts its reference before the fit).
- c agrees with `phase_sim_orb` to 6e-5 (median) on the ALOS-2 test data; a wrong sign would give -2.
- Campi Flegrei (675 687 candidates, 92 images, B_perp -73 to 142 m): with the DEM error relative to the neighbourhood
  336 830 points are kept instead of 330 754 (+1.8 %, none lost; all PS candidates, their DEM error relative to the
  neighbourhood p05 -14.3, p50 2.6, p95 16.4 m); random phases reach 0.35 (p99.99) with the fit over +-80 m and 91
  image pairs. On the refined points that part is 0.0 m (median, 1.0 m median absolute deviation); it correlates
  with the deviation of the whole DEM error from its 1 km neighbourhood by 0.72, but is about 15 % smaller (slope
  1.19): where many points lie on one building, the filtered phase holds part of its height. The spatially
  correlated part is p01 -5.9, p50 7.2, p99 18.7 m, 0.2 m in the reference area. Both parts add up to the whole DEM
  error estimated without the split (correlation 0.998, difference 0.04 m median).
- The large scale of the spatially correlated part is atmosphere: estimated from the odd and from the even dates,
  its 1 km means correlate by 0.27 only (atmosphere that correlates by chance with the baselines, larger with the
  distance from the reference), its deviations from the 1 km means by 0.50 (the rest of the local DEM error). It
  needs the atmosphere steps 3 or 8 first. A smooth uplift of Pozzuoli (polynomials of degree 2 to 5 fitted to its
  median displacement) puts less than 0.3 m into it; the correction changes the displacements by a few millimeters
  (Pozzuoli at most 3.3 mm).
- Clamped to its Delaunay neighbours (19 m apart, median), the whole DEM error would change at 22.8 % of the points,
  whose deviations from their neighbours repeat between the odd and the even dates (correlation 0.76; 0.92 and 99 %
  of the signs for the 2 566 points clamped by more than 5 m): real DEM errors. The spatially correlated part holds
  the rest of them, so it is not clamped either.
- StaMPS's constant of the reference image is the intercept at its date of a straight line in time. On Campi Flegrei
  it is 0.1 mm (1.4 mm median absolute deviation) in the reference area, the noise it is meant for, but -15.7 mm at
  Pozzuoli (+19.3 mm with 2020-07-07 as reference): the curvature of the uplift. It does not change dh (the
  differences do not hold the reference image).
- n2ft filters the candidates with their DEM errors relative to the neighbourhood: a building with many points can
  still shift the filtered phase of the ground around it.
- The PS candidates of 02 are selected by the n2f temporal coherence of the rasters without the DEM error relative to
  the neighbourhood (later).
- One reference for all points: points of regions of the closure correction other than the one of the reference
  can be off by whole cycles in some images, which the loops do not detect (Campi Flegrei: 86 regions, the largest
  79 %); checking or fixing the offsets between regions is a later change.
- A troposphere estimated from the correlation of the phase with the height also removes deformation that is
  correlated with the height (Campi Flegrei: uplift at the coast, rim higher); external data are preferred there.
- The velocity uncertainty from the residuals assumes uncorrelated noise; atmosphere left in the time series makes
  it too small.

## Do not

- Do not take `meta.toml` as an input of the product commands.
- Do not convert phases to displacements outside `pc-disp`, and do not use another sign: displacements are positive
  towards the satellite everywhere.
- Do not apply a spatial correction after the reference without referencing again.
- Do not fit the DEM error with a model of the shape of the deformation, or with baselines of the scene centre,
  without a new decision.
- Do not select the refined points by a temporal coherence without the DEM error relative to the neighbourhood.
