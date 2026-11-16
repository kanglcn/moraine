# 06 Deformation products

Pipeline: `examples/06_products.toml`. Tutorial: `nbs/Tutorials/CLI/CampiFlegrei/06_products.ipynb`.
Needs `04_refine` and `05_unwrap`. The steps and conventions are decision 0048.

```bash
moraine run examples/06_products.toml --workdir WORK
```

Set `[vars] ref_area` to a reference area of the data (an expression of `lon` and `lat` in the syntax of `math`, the longitude and
latitude of the points); the default is the one of the sample data set (a box east of the caldera). The radar
wavelength in meters (`wavelength`, 0.0554658 m for Sentinel-1) and the dates of the images (`dates`) are read from
`raw/meta.toml` of `01_load`.

## Steps

1. `math`: the reference points, a bool mask of the points inside `ref_area`. A polygon drawn on a map
   (`mc.view(..., polygons='reference.geojson')`, or written with `moraine.write_polygons`) works too:

   ```toml
   [[step]]
   name = "ref_mask"
   run = "polygon-mask"
   polygons = "reference.geojson"
   x = "pc/pc_lon.zarr"
   y = "pc/pc_lat.zarr"
   mask = "disp/pc_ref_mask.zarr"
   ```

2. `pc-reference`: from every image the median of the reference points is subtracted, so that it is 0 in every
   image; `method = "mean"` takes their mean, `ref_point` (instead of `ref_mask`) one point, e.g. the point of a
   double click on a map. The median is not moved by a few reference points with unwrapping errors.
3. `pc-disp`: line of sight displacement in millimeters, `1000 * wavelength / (4 pi)` per radian (4.41 mm for
   Sentinel-1), positive towards the satellite (a shorter range).
4. `pc-sc-dem-error`: the spatially correlated DEM error of every point, from the perpendicular baselines of the
   point (`pc/pc_bperp.zarr` of `04_refine`), and the displacement without it; `math` adds the DEM error relative to
   the neighbourhood of `04_refine`: the whole DEM error, the height of the scatterer above the DEM of the processing
   (see [The DEM error](#the-dem-error)).
5. `pc-pyramid` of the displacement, the two DEM errors and the displacement without the DEM error on the map.

Outputs in `WORK/disp/`:

- `pc_ref_mask.zarr` (n_points,) bool: the reference points.
- `pc_ts_ref.zarr` (n_points, nimages) float32: the unwrapped phase of every image relative to the first image
  and to the reference area, in radians.
- `ref_value.zarr` (nimages,) float32: the value subtracted from every image, in radians.
- `pc_disp.zarr` (n_points, nimages) float32: the line of sight displacement in millimeters relative to the first
  image and to the reference area, positive towards the satellite. It is along the line of sight, not vertical: a
  vertical motion u gives about u cos(incidence) (incidence about 35 degrees on the sample data), plus a share of
  the horizontal motion.
- `pc_sc_dem_error.zarr` (n_points,) float32: the spatially correlated DEM error in meters, positive above the DEM,
  relative to the reference area.
- `pc_dem_error.zarr` (n_points,) float32: the whole DEM error in meters, `pc/pc_su_dem_error.zarr` of `04_refine`
  plus `pc_sc_dem_error.zarr`.
- `pc_disp_demcor.zarr` (n_points, nimages) float32: `pc_disp.zarr` without the spatially correlated DEM error term,
  in millimeters (the part relative to the neighbourhood was removed before the unwrapping): the displacement
  without the DEM error.

## The reference area

The displacements are relative to the reference area, so its own motion is in every point. Take an area that is
stable (outside the deforming area), coherent (many refined points), not affected by strong local atmosphere, close
to the area of interest and at a similar height (MintPy's guidelines), and inside one region of the closure
correction (`unw/pc_region.zarr`): points of other regions can be off by whole cycles in some images, which the
loops do not detect. Check the regions of the reference points in python:

```python
import numpy as np, zarr
region = zarr.open('unw/pc_region.zarr', mode='r')[:]
ref = zarr.open('disp/pc_ref_mask.zarr', mode='r')[:]
np.unique(region[ref], return_counts=True)
```

The sample scene is small: every part of it is within about 10 km of Pozzuoli, so no area is far from the uplift.
The default box east of the caldera (Naples, 8-13 km from Pozzuoli) has 11 829 points, 99.7 % of them in the
region of Pozzuoli, and is flat inside (its points 9-10, 10-11 and 11-13 km from Pozzuoli differ by less than
1 mm in the last image). Areas in the north-west and the north of the scene move by 12-15 mm less than it over the
three years, and only 49 % (north-west) and 76 % (north) of their points are in that region; with them as reference
all displacements shift by that much. Which area is stable is not known from the data alone (GNSS stations would
tell).

## The DEM error

A scatterer dh meters above the DEM of the processing (a roof on a terrain model, a wall above the ground) shifts the
displacement of image k by c_k dh, c_k = 1000 B_perp,k / (R cos(theta)) mm per meter (B_perp,k the perpendicular
baseline of the point, R its slant range, theta the elevation of its look vector): on the sample data about
0.2 mm per meter for 100 m of baseline. The DEM error is estimated in two parts (decision 0048). The part relative to
the neighbourhood in the filtered interferograms (a roof above the ground around it, `pc/pc_su_dem_error.zarr`)
was fitted in the temporal coherence of `04_refine` and removed from the phase before the unwrapping.
`pc-sc-dem-error` fits the rest, the spatially correlated part, from the differences of the displacement between
neighbouring dates, with a constant velocity besides: a deformation that changes slowly from one date to the next,
as the accelerating uplift of Pozzuoli, does not get into it (its smooth part, polynomials of degree 2 to 5, less
than 0.3 m). The dates come from `raw/meta.toml`; they must be those of the images, in order. The `math` step adds
the two parts; they agree with the DEM error estimated in one part from the displacement (correlation 0.998).

The part relative to the neighbourhood is about 0 on average everywhere (median 0.0 m, 1.0 m median absolute
deviation) and is the one to trust for single buildings; it is about 15 % smaller than the local DEM error where
many points lie on one building, and the rest is in the spatially correlated part. The spatially correlated part
is relative to the reference area (the reference subtracted the displacement of its points, their DEM error
included): 0.2 m there. Estimated from the odd and from the even dates of the sample data separately, its 1 km
means do not agree (correlation 0.27): atmosphere that correlates by chance with the baselines of the images gives
patterns of several meters over kilometers, here 0 m east (the reference) to 11 m (west) and 16 m (Pozzuoli).
Correcting the atmosphere first (decision 0048, steps 3 and 8, later) reduces them. With the small baselines of
Sentinel-1 the correction changes the displacements by a few millimeters.

## Expected results (sample data, Campi Flegrei, 336 830 points, 92 images)

| result | sample value | sane range |
|---|---|---|
| log of `ref_mask` | 11 829 reference points | thousands of points |
| log of `reference` | reference values min -3.5, median 0.21, max 4.4 (radians) | the level of every image, a few radians |
| `disp/pc_disp_pyramid` | p01 -29.2, p50 1.2, p99 186.2, min -113, max 370 (mm); warning "constant values: image 0" | image 0 is 0 everywhere; the uplift of Pozzuoli positive, about 0 in the reference area |
| Pozzuoli (the 662 points within 500 m of Rione Terra, 40.8197 N 14.1203 E), median | 0, 24, 73, 89, 139, 201, 250 mm on 2019-01-02, 2019-07-01, 2019-12-28, 2020-06-25, 2020-12-22, 2021-06-20, 2021-12-29 | growing, towards the satellite |
| reference points in the last image | median 0, median absolute deviation 2.7 mm, 90 % within 7.8 mm | a few mm |
| log of `sc_dem_error` | 92 images from 20190102 to 20211229 (1092 days); dh of 336 830 points (0 nan): percentiles 1, 50, 99 -5.91, 7.16, 18.68 m | all dates; no nan; meters to tens of meters |
| `disp/pc_sc_dem_error_pyramid` | p01 -5.9, p50 7.2, p99 18.7, min -56, max 73 (m) | the large scale atmosphere (see above); hundreds of meters: a wrong `bperp` or wrong dates |
| spatially correlated DEM error of the reference points | median 0.2 m, median absolute deviation 3.0 m | about 0 |
| `disp/pc_dem_error_pyramid` (the whole DEM error) | p01 -9.9, p50 7.3, p99 21.6, min -67, max 79 (m) | buildings and the large scale atmosphere |
| `disp/pc_disp_demcor_pyramid` | p01 -29.0, p50 1.2, p99 186.5, min -113, max 369 (mm); warning "constant values: image 0" | that of `pc_disp_pyramid` within a few mm |
| Pozzuoli without the DEM error | 0, 22, 70, 89, 140, 201, 252 mm: the median changes by at most 3.3 mm | a few mm |

Run time: `pc-reference` 1.1 s, `pc-disp` 0.9 s, `pc-sc-dem-error` 1.2 s, `math` 0.1 s, the four pyramids 7.6, 1.3, 1.8
and 5.5 s; 24 s and 1.5 GB peak memory for the pipeline. Large data (10 million points, 92 images, 4 threads):
`pc-reference` 10.3 s and 0.9 GB, `pc-disp` 15.5 s and 0.97 GB, `pc-sc-dem-error` 27 s and 3.4 GB (16 threads: 28 s,
9.4 GB; the reading and writing of the chunks limit it).

## Figures

Quicklooks of the sample data runs of 2026-10-19 (the displacement, with 1.8 % fewer points; the map looks the same)
and 2026-11-15 (the DEM error; the numbers above are from this run):

![Displacement of the last image](../assets/campi/06_disp_91.webp)

*Line of sight displacement on 2021-12-29 relative to 2019-01-02 and to the box east of the caldera,
`disp/pc_disp_pyramid --index 91`: the uplift of Pozzuoli, up to 360 mm towards the satellite about 1 km east of
Rione Terra; the colours span the 1 % - 99 % range of all images, so the centre is saturated.*

![Spatially correlated DEM error](../assets/campi/06_sc_dem_error.webp)

*Spatially correlated DEM error relative to the box east of the caldera, `disp/pc_sc_dem_error_pyramid`: about 0 in
the box, rising to 10 - 20 m towards Pozzuoli and the west. These kilometer scale patterns do not repeat between the
odd and the even dates (atmosphere through the baselines); the buildings are in the part relative to the
neighbourhood of `04_refine`.*

## Checks

- `moraine info disp/pc_disp_pyramid`: only the warning for image 0; values within a few hundred millimeters.
- `moraine quicklook disp/pc_disp_pyramid --index I -o disp.png`: about 0 in the reference area, smooth; the
  sign of a known motion right (uplift: positive).
- `moraine view disp/pc_disp_pyramid --dates raw/meta.toml -o view.ipynb`: click points for their displacement
  time series.
- `moraine quicklook disp/pc_sc_dem_error_pyramid -o dh.png`: about 0 in the reference area; smooth patterns of
  several meters over kilometers are atmosphere. `pc/pc_su_dem_error_pyramid` of `04_refine`: buildings stand out
  from their surroundings.

## Problems

- `pc-reference` fails with "no value (all nan)": the reference points have no data in an image; take another
  area.
- `pc-disp` fails with "between 0 and 1": the wavelength is not in meters (e.g. 5.55 in centimeters).
- Jumps of about 27.7 mm (half a wavelength for Sentinel-1) between areas or dates are whole cycles of unwrapping
  errors: check the regions and the change fraction of `05_unwrap`.
- `pc-sc-dem-error` fails with "N dates for M images" or "must increase": the dates are not those of the images of
  the stack; with "bperp must have shape": `pc/pc_bperp.zarr` is not of these points (rerun `04_refine`);
  `pc/pc_bperp.zarr` missing: the working directory was made before the `bperp` step of `01_load`.
- The velocity and the residual of every image are not computed yet (decision 0048, later steps).
