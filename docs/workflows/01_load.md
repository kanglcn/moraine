# 01 Load GAMMA results

Pipeline: `examples/01_load.toml`. Tutorial: `nbs/Tutorials/CLI/CampiFlegrei/01_load.ipynb`.

## Input

A GAMMA processing directory with co-registered SLCs and geocoding products:

```
gamma/
├── rslc/       YYYYMMDD.rslc, YYYYMMDD.rslc.par   (co-registered SLCs, one per date)
├── geocoding/  <geo>.hgt, <geo>.diff_par, <geo>.lt_fine, <geo>.lv_theta, <geo>.lv_phi
└── DEM/        dem_seg_par
```

Ask the user for the directory, the reference date (`reference`, the date the rslc stack is flattened to)
and the date prefix of the geocoding files (`geo`) if they are not obvious from the file names. The defaults
of the file are the ones of the sample data set, Campi Flegrei (`data/CampiFlegrei/gamma`, decision 0033).

```bash
moraine run examples/01_load.toml --workdir WORK --var gamma=/path/to/gamma --var reference=20200707 --var geo=20200707
```

GAMMA must be on PATH (`which base_calc phase_sim_orb geocode create_offset`).

## Steps and outputs (WORK/raw/)

| step | command | output |
|---|---|---|
| rslc | load-gamma-flatten-rslc | `rslc.zarr` (nlines, width, nimages) complex64, flat earth and topography removed |
| lat_lon_hgt | load-gamma-lat-lon-hgt | `lat.zarr`, `lon.zarr` float64, `hgt.zarr` float32, (nlines, width); nan where 0 in GAMMA |
| bperp | load-gamma-bperp | `bperp.zarr` (nlines, width, nimages) float32, meters: perpendicular baseline of every image relative to the reference image at every pixel, from the orbits of the `*.rslc.par` files; 0 for the reference, nan where lat / lon / hgt are nan |
| look_vector | load-gamma-look-vector | `theta.zarr`, `phi.zarr` (elevation / orientation angle) |
| range | load-gamma-range | `range.zarr` slant range distance |
| metadata | load-gamma-metadata | `meta.toml`: `dates` (YYYYMMDD), `times` (UTC time of the centre line of every image, to the microsecond), radar wavelength, pixel spacings, extent of the DEM, perpendicular baselines at the scene centre (`base_calc`) |
| web_mercator | transform | `e.zarr`, `n.zarr` web mercator coordinates (EPSG:3857) for map plots |
| rslc_pyramid | ras-pyramid | `rslc_pyramid/` to check the interferograms, decimated: SLC pixels cannot be averaged (their scatterer phases differ), so the whole scene is speckle; zoom in, or look at the mean pyramid of the n2f interferograms of 02 |

Sample data (Campi Flegrei, 981 x 4160 x 92, 2026-11-13): 148 s, 118 s of it `phase_sim_orb` for each date
(64 threads); the perpendicular baselines take 5 s (0.4 GB on disk), the pyramid 5 s.

## Expected results (sample data)

| result | sample value | sane range |
|---|---|---|
| `rslc.zarr` | complex64 (981, 4160, 92), chunks (1000, 1000, 1) | the number of dates in `rslc/` |
| `rslc_pyramid` amplitude | p01 0.023, p50 0.29, p99 1.56 | p50 roughly 0.1-1 |
| `rslc_pyramid` nan_fraction | 0.40 (the Gulf of Pozzuoli, zero in the coregistered SLCs) | small, unless water or the footprint is masked |
| `meta.toml` dates | 92, 12 days apart | all dates of `rslc/`, in order |
| `meta.toml` times (2026-11-15) | 05:12:13.78 to 05:12:31.89 UTC, on the days of `dates` | one per date, on its day (except an acquisition across 00:00 UTC); about the same time of day for one track (here within 22 s) |
| `bperp.zarr` | -73 to 142 m; varies by 0.6 m (median) to 1.6 m over the scene in an image; at the scene centre within 6 mm of `perpendicular_baseline` of `meta.toml` | within the orbit tube of the satellite (Sentinel-1: a few hundred m); 0 for the reference date |

## Figures

Quicklooks of the sample data run of 2026-10-08:

![Sequential interferogram 0 of the raw stack](../assets/campi/01_intf_seq_0.webp)

*Sequential interferogram 0 of the raw stack (`quicklook raw/rslc_pyramid --show intf_seq --index 0`): noisy single
look phase with structure on land, nan on the gulf.*

## Checks

- `moraine info raw/rslc.zarr`: shape (nlines, width, number of dates in `rslc/`).
- `moraine info raw/rslc_pyramid`: amplitude p50 around 0.1-1. A nan_fraction near 1 means the GAMMA files did
  not match; on the sample data 0.40 is the sea.
- `moraine quicklook raw/rslc_pyramid --show intf_seq --index I -o intf.png`: sequential
  interferograms. Expect noisy phase with visible fringes / structure in coherent areas; pure uniform
  noise everywhere suggests a wrong reference date or unflattened data.
- `raw/meta.toml`: `dates` must list all dates in order; `times` one time per date, on that day and at about the
  same time of day.
- `raw/bperp.zarr`: at the scene centre (logged by the step) close to `perpendicular_baseline` of `raw/meta.toml`,
  which GAMMA's `base_calc` gives for the scene centre: a few cm apart at most, more where the centre pixel is high
  above the ellipsoid. A different sign or order means wrong `reference` or files.

## Problems

- `sh: base_calc: command not found`: GAMMA is not on PATH.
- A step fails on a missing `*.par` / geocoding file: check `geo` / `reference`.
- The scratch directory (`raw/scratch`) keeps GAMMA intermediates; it can be deleted after the run.
