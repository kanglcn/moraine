# 0056 Solid earth tides of the IERS Conventions, own code, one time per image

## Status

Accepted

## Date

2026-11-17

## Context

Step 3 of decision 0048: corrections of every image from models. The Sun and the Moon deform the solid Earth, up to
about 30 cm up and a few cm horizontally, mostly with periods of half a day and a day. Between two images the change is
nearly constant over a scene, which the spatial reference removes (`pc-reference`), but its gradient is of the order of
0.05 mm per km (degree 2 tide over the radius of the Earth): about a millimeter over the Campi Flegrei, about a
centimeter over a whole Sentinel-1 frame. MintPy corrects it with PySolid (Yunjun et al., 2022), a wrapper of the
Fortran code of D. Milbert (`solid.for`), which follows the IERS Conventions 2003 with the Sun and the Moon of low
precision series, on a grid at the centre time of every image.

The IERS Conventions (2010), section 7.1.1 (Mathews et al., 1997), give the displacement of a station in two steps:
step 1 in the time domain (degree 2 and 3 tides with nominal Love and Shida numbers and their latitude dependence, the
out of phase parts in the diurnal and semidiurnal band, the latitude dependence of l), step 2 the frequency dependence
of the Love numbers in the diurnal (31 terms) and long period band (5 terms); their routine DEHANTTIDEINEL (version
of 2017) takes the positions of the Sun and the Moon from the user and has test cases.

## Decision

- `pc-solid-tide` (API `pc_solid_tide`, module `tide`; `solid_tide` gives the east, north and up displacement at one
  time): the displacement of every point at the time of every image, projected on its look vector
  l = (cos theta cos phi, cos theta sin phi, sin theta) (east, north, up; `pc_theta`, `pc_phi` of GAMMA's
  `look_vector`, from the ground to the satellite), in millimeters relative to the image `ref`, positive towards the
  satellite (the sign of `pc-disp`); the corrected displacement is `disp - out`.
- Own code, numba, no PySolid dependency: the displacement exactly as DEHANTTIDEINEL of the IERS Conventions 2010
  (its tables 7.3a and 7.3b, the time counted from J2000.0 as corrected by the IERS in 2013, the hours of the day in
  UTC); the Sun and the Moon from the low precision series of Montenbruck and Gill (2000, Satellite Orbits, section
  3.3.2) in TT, rotated into the Earth fixed frame by the Greenwich mean sidereal time (UT1 taken as UTC, nutation and
  polar motion neglected), as PySolid; TT from UTC with a table of leap seconds (1972 to 2017). The terms of step 2 and
  the Sun and the Moon are computed once per image, the rest per point and image.
- The permanent tide is not restored (conventional tide free, as DEHANTTIDEINEL): it cancels in the difference of two
  images.
- Points on the WGS84 ellipsoid (the height changes the tide by about 0.05 % per 3 km), at one time per image, the
  `times` of `load-gamma-metadata` (UTC of the centre line). The points of a scene are imaged within seconds of it; the
  tide changes by at most about 0.04 mm per second in the line of sight, so within 12 s of the centre (about one
  Sentinel-1 frame) the error is at most 0.5 mm, mostly a ramp in azimuth. Times of every point (zero Doppler time
  from the orbits, as `load-gamma-bperp`) are left for later.
- Optional: not in the default chain of `examples/06_products.toml`; the guide shows the two steps (the tides, then
  `math` `disp - tide`) after the troposphere of ERA5 and before the spatial reference.
- The helpers that read the times of images move from `moraine.api.tropo` to `moraine.api.utils_.times` (a stage does
  not import another stage, decision 0041).

## Consequences

- Validation: the three consistent test cases of DEHANTTIDEINEL to 1e-15 m (its fourth has a Sun 1.4e10 m from the
  Earth and the expected output of the third); against PySolid 0.3.4 at 10 places and 6 days every 5 minutes (17 340
  samples) the differences are at most 0.30 mm up and 0.04 mm horizontally (RMS 0.10 mm up), all from step 2: with the
  table of 2003 and the time origin of PySolid (half a day before J2000.0, hours of the day in TT) instead of those of
  the IERS code, they are below 0.0001 mm.
- Low precision Sun and Moon: errors of their positions change the absolute tide by less than a millimeter and its
  gradient over a scene by far less.
- 10 million points and 92 images in 15 s with the default four threads (about 1.6 ns per point and image for the
  computation, the rest writing), peak 2.5 GB; a task holds about 4 bytes per point and image.
- Campi Flegrei (330 754 points, 17 by 14 km, 92 images, relative to 2019-01-02): the mean of the tides over the points
  between -73 and +201 mm, the spread over the points (p99 - p1) 0.6 mm in the median image, 2.3 mm at most.
- A new leap second needs a new row in the table of `moraine/api/tide.py`.

## Do not

- Do not put it into the default chain of example 06 without a new decision record.
- Do not take the tables of PySolid (IERS 2003) or its time origin: the IERS code of 2010 is the reference.
