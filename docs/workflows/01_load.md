# 01 Load GAMMA results

Pipeline: `examples/01_load.toml`. Tutorial: `nbs/Tutorials/CLI/load_data.ipynb`.

## Input

A GAMMA processing directory with co-registered SLCs and geocoding products:

```
gamma/
├── rslc/       YYYYMMDD.rslc, YYYYMMDD.rslc.par   (co-registered SLCs, one per date)
├── geocoding/  <geo>.hgt, <geo>.diff_par, <geo>.lt_fine, <geo>.lv_theta, <geo>.lv_phi
└── DEM/        dem_seg_par
```

Ask the user for the directory, the reference date (`reference`, the date the rslc stack is flattened to)
and the date prefix of the geocoding files (`geo`) if they are not obvious from the file names.

```bash
moraine run examples/01_load.toml --workdir WORK --var gamma=/path/to/gamma --var reference=20220620 --var geo=20210802
```

GAMMA must be on PATH (`which base_calc phase_sim_orb geocode create_offset`).

## Steps and outputs (WORK/raw/)

| step | command | output |
|---|---|---|
| rslc | load-gamma-flatten-rslc | `rslc.zarr` (nlines, width, nimages) complex64, flat earth and topography removed |
| lat_lon_hgt | load-gamma-lat-lon-hgt | `lat.zarr`, `lon.zarr` float64, `hgt.zarr` float32, (nlines, width); nan where 0 in GAMMA |
| look_vector | load-gamma-look-vector | `theta.zarr`, `phi.zarr` (elevation / orientation angle) |
| range | load-gamma-range | `range.zarr` slant range distance |
| metadata | load-gamma-metadata | `meta.toml` (dates, wavelength, heading, pixel spacing, perpendicular baselines) |
| web_mercator | transform | `e.zarr`, `n.zarr` web mercator coordinates (EPSG:3857) for map plots |
| rslc_pyramid | ras-pyramid | `rslc_pyramid/` to check the interferograms |

Sample data (2500 x 1834 x 17): about 3 minutes, most of it `phase_sim_orb` for each date.

## Checks

- `moraine info raw/rslc.zarr`: shape (nlines, width, number of dates in `rslc/`).
- `moraine info raw/rslc_pyramid`: amplitude p50 around 0.1-1 and nan_fraction small (sample: 0.03,
  the nan are outside the image footprint). A nan_fraction near 1 means the GAMMA files did not match.
- `moraine quicklook raw/rslc_pyramid --show intf_seq --index I -o intf.png`: sequential
  interferograms. Expect noisy phase with visible fringes / structure in coherent areas; pure uniform
  noise everywhere suggests a wrong reference date or unflattened data.
- `raw/meta.toml`: `dates` must list all dates in order.

## Problems

- `sh: base_calc: command not found`: GAMMA is not on PATH.
- A step fails on a missing `*.par` / geocoding file: check `geo` / `reference`.
- The scratch directory (`raw/scratch`) keeps GAMMA intermediates; it can be deleted after the run.
