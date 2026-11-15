# 0044 Perpendicular baselines per pixel from the orbits

## Status

Accepted

## Date

2026-11-14

## Context

The DEM error of decision 0040, and any other model of the topographic phase, needs the perpendicular baseline B⊥ of
every image at every point: a height error dh shifts the phase of image j by 4 pi / lambda B⊥ dh / (rho_j up.n). B⊥
varies over the scene: B⊥ = B cos(theta - alpha), dB⊥/dtheta = -B∥. Over a Sentinel-1 IW frame (look angles over 0.26
rad) it changes by about a quarter of the parallel baseline, e.g. 25 m for B∥ = 100 m, 0.5 mm of line of sight per 10 m
of DEM error and different in every image. The scene centre values of GAMMA's `base_calc` in `meta.toml` do not have
this. The orbits of the images are in their `*.rslc.par` files (state vectors), the positions of the pixels in the
lat / lon / hgt rasters of the 01 step.

## Decision

- B⊥ is computed at every pixel from the orbits, by the geometry: the satellite of every image at its own zero-Doppler
  time of the pixel, the baseline to the satellite of the reference projected on the direction perpendicular to the
  line of sight and to the flight direction of the reference, positive upwards (the sign of `base_calc`).
- The reference is the reference of the co-registration (its B⊥ is 0); B⊥ relative to another image is computed
  again, not as a difference.
- One command in the 01 step, `load-gamma-bperp`, writes a raster (nlines, width, nimages) float32 meters; points get
  it with `ras2pc`. Inside, a reader of the GAMMA state vectors and a function independent of the processor
  (`moraine.api.utils_.orbit.bperp`), so that another processor needs only a reader.
- The state vectors are resampled every second by Lagrange interpolation over 8 vectors, then interpolated by cubic
  Hermite polynomials (cubic Hermite polynomials on vectors 60 s apart err by up to 0.3 m).
- Images with different numbers of state vectors are an error.

## Consequences

- B⊥ is exact to the geometry: the height sensitivity of the range difference computed from it equals the finite
  difference of the exact ranges over -+ 500 m to 3e-6 (the error of the finite difference), whatever the
  interpolation of the orbits.
- GAMMA's `phase_sim_orb` places a pixel at the range and time of the reference par at zero Doppler; at these
  positions its height sensitivity is that of B⊥ to 3e-6 (an ALOS-2 sample with state vectors 60 s apart; Campi
  Flegrei 0.01 mm). The 01 step places the pixels, where the points are, with the offsets of the geocoding
  `diff_par` (Campi Flegrei 4.7 m in range, the ALOS-2 sample 13.5 m and 2 lines); there B⊥ differs from that of
  `phase_sim_orb` by up to 3 mm on Campi Flegrei (-1.8e-5 B⊥ - 7e-6 B∥) and by 16 mm on the ALOS-2 sample (9e-5 at
  B⊥ = -184 m), 2e-5 and 9e-5 of a DEM error. At the scene centre B⊥ is within 6 mm of `base_calc` (Campi
  Flegrei), mostly 5e-5 B∥ of its scene centre model.
- Campi Flegrei (981 x 4160 x 92): 5 s, 0.4 GB on disk; 12 M pixels: 17 s and 1 GB of memory on 128 cores.
- `perpendicular_baseline` of `meta.toml` stays, as a check of the scene centre.

## Do not

- Do not use scene centre baselines in a model per pixel or point (DEM error, ...).
- Do not interpolate state vectors tens of seconds apart by cubic polynomials alone.
