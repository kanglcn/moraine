# 0057 Ionosphere from the global ionosphere maps of JPL, at the pierce point of every point

## Status

Accepted

## Date

2026-11-17

## Context

Step 3 of decision 0048: corrections of every image from models or external data. The ionosphere advances the phase
of the radar signal: the phase path is shorter by K STEC / f^2 (K = 40.31 m^3/s^2, STEC the electrons per square
meter along the line of sight, f the radar frequency), 13.80 mm per TEC unit (1e16 electrons per square meter) at the
wavelength of Sentinel-1, about 240 mm at L band. The vertical TEC changes by several TEC units between images and
varies over hundreds of kilometers: the change between two images is nearly constant over a small scene, which the
spatial reference removes, but over a Sentinel-1 frame it leaves ramps of centimeters, from the horizontal gradient of
the TEC and from the incidence angle, which changes the slant path across the swath.

Global ionosphere maps of the IGS analysis centres (IONEX files, Schaer et al., 1998) give the vertical TEC of a thin
layer at 450 km: e.g. the final maps of JPL, 2.5 by 5 degrees, every 2 hours. MintPy's `iono_tec` (Yunjun et al.,
2022) takes the vertical TEC at the pierce point of the centre of the scene only and maps it to every pixel with its
incidence angle, after a "refraction" angle from n = 1 + K VTEC / f^2 (their equation 9). Wang et al. (2026, IEEE TGRS)
take the TEC at the pierce point of every pixel; with Sentinel-1 the maps reduce the error of the velocity relative to
GNSS from 1.7-3.8 to 0.9-2.9 mm per year on descending tracks and mid-latitude ascending tracks, as much as the split
spectrum method, but not at low latitudes at dusk (ascending) or at L band; the gradients of the maps are accurate to
0.2-0.5 TEC units per 100 km when the TEC is low.

The maps are in NASA's archive CDDIS (Earthdata login), the final maps of JPL also on JPL's own server
(`sideshow.jpl.nasa.gov/pub/iono_daily/IONEX_final`, without an account, gzip, about 110 kB per day, from 2008-11-19 on,
about 6 weeks after the day; 2026-10-02 on 2026-11-17). Its maps of higher resolution (JPLH, 2 by 2 degrees, every
hour, an extended slab model) start on 2023-12-11; the 15 minute maps of UPC are compressed with the Unix `compress`
(LZW), which Python does not read.

## Decision

- `download-ionex` (`moraine.cli.iono`): the final maps of JPL (`JPLGddd0.yyI.gz`) of every UTC day of the images
  from JPL's server, one file per day, written through a temporary file and checked (IONEX with the maps of the whole
  day); files that hold their day are kept, so a download continues. Threads; urllib and gzip of the standard
  library, no new dependency.
- `pc-iono-tec` (API `pc_iono_tec`, with `read_ionex`, `pierce_point` and `ionex_vtec` in `moraine.api.iono`):
  - the pierce point of every point: the line from the point on the WGS84 ellipsoid along its look vector (`pc_theta`,
    `pc_phi`, relative to the ellipsoid normal) intersected with the sphere of the layer (BASE RADIUS + HGT1 of the
    file, 6371 + 450 km for JPL); mapping factor 1 / cos z', z' the zenith angle of the line at the pierce point;
  - the vertical TEC there by the interpolation of the IONEX format: bilinear in the maps, linear in time between the
    two maps around the time of the image, each turned in longitude with the Sun (360 degrees per day); the maps from
    the file of the day of the image;
  - iono_k = 1000 ratio K / f^2 (STEC_k - STEC_ref) in millimeters, STEC = mapping factor x vertical TEC, f = c /
    `wavelength`: the apparent displacement, positive towards the satellite (the shorter phase path) like `pc-disp`;
    the corrected displacement is `disp - iono`;
  - `ratio` (0 < ratio <= 1, default 1): the part of the TEC of the maps below the satellite. The maps hold the
    electrons up to the GNSS orbits (20 200 km), a SAR satellite flies at 500 to 800 km; studies give 0.69 to 0.90 for
    Sentinel-1, but the maps smooth the gradients, and MintPy (its default) and Wang et al. (2026) take all of the TEC.
- No refraction factor: equation 9 of Yunjun et al. (2022), used by MintPy and Wang et al. (2026), takes K VTEC / f^2,
  a length in meters (the vertical delay), for n - 1. The refractive index is 1 + K Ne / f^2 with Ne the electron
  density (electrons per cubic meter): with Ne = 1e12 at the peak, n - 1 is 1.4e-6 at C band and 2.6e-5 at L band,
  and the bending of the line is negligible.
- One time per image (the `times` of `load-gamma-metadata`): the maps change over hours, the images over seconds.
- Optional: not in the default chain of `examples/06_products.toml`; the guide shows the three steps (the download,
  the ionosphere, then `math` `disp - iono`) before the spatial reference.

## Consequences

- Validation: the vertical TEC of `ionex_vtec` agrees with MintPy's reader and interpolation (`read_ionex`,
  `interp_3d_maps` with rotated maps, transcribed from its source) to 4.7e-9 TEC units at 20 000 random places and
  times of 10 days from 2015 to 2025, storms included (up to 131 TEC units); the pierce points and mapping factors
  agree with a bisection along the line and with equations 11 and 12 of Yunjun et al. (2022) given the geocentric
  latitude, the radial vertical and the radius of the point to 1e-13 degrees and 1e-15. MintPy's convention (the
  geodetic latitude and the ellipsoid normal taken as geocentric and radial, a radius of 6371 km for the point) moves
  the pierce points by up to 0.27 degrees (a tenth of a cell of the maps) and the mapping factor by up to 0.9 % at
  incidence angles of 20 to 70 degrees. Its refraction factor makes its delays smaller by 4, 7, 11 and 16 % at 5, 10,
  20 and 50 TEC units (C band, 42 degrees).
- Campi Flegrei (330 754 points, 17 by 14 km, 92 images, relative to 2019-01-02): the vertical TEC at the scene at
  05:12 UTC between 5.7 and 14.8 TEC units (median 9.1); the mean over the points between -5.7 and +142 mm, removed by
  the spatial reference; the spread over the points (p99 - p1) 0.50 mm in the median image, 1.41 mm at most, of the
  order of the error of the gradients of the maps over that distance (0.6-1.4 mm). Over an area of a Sentinel-1
  frame (3 by 2 degrees, incidence 30 to 46 degrees, the same times): 12 mm in the median image, 42 mm at most.
- 10 million points and 92 images in 16 s with the default four threads, peak 0.8 GB; a task holds about 4 bytes per
  point and image, the maps around the images about 41 kB per image. The 92 days of the Campi Flegrei download in 8 s.
- Only the long wavelengths (hundreds of kilometers) of the ionosphere: smaller structures, strong ionosphere (low
  latitudes at dusk, storms) and L band need the split spectrum method, later.
- The maps of the last weeks are not on the server yet; other maps (JPLH, CODE, rapid maps) are left for later.

## Do not

- Do not put it into the default chain of example 06 without a new decision record.
- Do not apply the refraction factor of n = 1 + K VTEC / f^2.
