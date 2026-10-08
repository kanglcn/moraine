# Checking results

Results are large; never load one to look at it. moraine gives three ways to check a result without a plot
script, and the [workflow guides](../workflows/index.md) say what to expect from each step.

## Pyramids

A pyramid is a multi resolution copy of a raster (`ras-pyramid`) or of a point cloud rendered on a grid
(`pc-pyramid`), stored next to the result ([contract](../contracts/pyramid.md)). The example pipelines build
pyramids for the results worth checking, and `moraine run` saves a PNG of every pyramid a step makes in
`WORK/.moraine/<file name>/quicklook/`.

```bash
moraine ras-pyramid --ras ps/ras_adi.zarr --out_dir ps/ras_adi_pyramid
moraine pc-pyramid --pc pc/pc_ph.zarr --out_dir pc/pc_ph_pyramid --x pc/pc_e.zarr --y pc/pc_n.zarr --ras_resolution 20
```

## `moraine info`: shape, chunks and statistics

```bash
moraine info ps/ras_adi.zarr ps/ras_adi_pyramid
```

```
  ps/ras_adi.zarr: float32 (981, 4160) chunks (1000, 1000)
  ps/ras_adi_pyramid: float32 (981, 4160) raster pyramid, 10 levels
    stats_level=0, nan_fraction=0.40655, min=0.0335447, max=4.39345, mean=0.537522, std=0.0957743, p01=0.255305, p50=0.541975, p99=0.786118
```

For a plain array nothing is read but the metadata. For a pyramid the statistics come from a coarse level
(`nan_fraction`, `min`, `max`, `mean`, `std`, `p01`, `p50`, `p99`; the amplitude for complex data, `true_fraction`
for booleans) and `warnings` name anomalies: all values nan, infinite or constant values. With `--json` the same
numbers are one JSON object ([contract](../contracts/json-output.md)).

## `moraine quicklook`: a picture

```bash
moraine quicklook ps/ras_temp_coh_pyramid -o tcoh.png
moraine quicklook raw/rslc_pyramid --show intf_seq --index 5 -o intf_5.png
moraine quicklook raw/rslc_pyramid --show intf_all --index 0 91 --extent 1400,0,2600,600 -o zoom.png
```

`--show` chooses what to draw of a stack: `phase` (default for complex data), `intf_0` (interferogram with
the first image), `intf_seq` (the I-th sequential interferogram), `intf_all` (any pair `ref sec`), `coh`,
`coh_abs`. Phases use a cyclic colour map over (-π, π]; other values `viridis` over the 1-99 % range. `--extent`
draws a part of the scene (`west,south,east,north` degrees on a map, `range_min,azimuth_min,range_max,azimuth_max`
pixels on the radar grid): a smaller part is drawn from a finer level, down to single pixels or points; the
title gives the extent and the level. Look at the whole scene first, then zoom into what looks wrong.

What to look for: fringes should be continuous, noise should sit where the coherence is low, selected points
should cover the stable areas (towns, rock), unwrapped phase should have no isolated 2π jumps between
neighbouring areas.

## `moraine view`: interactive maps in a notebook

```bash
moraine view ps/ras_adi_pyramid pc/pc_ph_pyramid -o view.ipynb --show intf_seq --dates raw/meta.toml
```

writes a notebook with one map per pyramid: zoom and pan load details from the pyramid levels, sliders choose
the image, a click plots the time series of a pixel or point, a double click makes it the reference, polygons
drawn on the map are saved for `polygon-mask`. The notebook holds only paths and runs on the data machine, so it
works over any Jupyter or VS Code connection without a server or port forwarding (decision 0018).

The same views are available in Python: `mc.view(data)` takes a pyramid, a raster array or point data (`x=`,
`y=`); `show=` chooses what to show of a stack (`'intf_seq'`, ... or a function `lambda v, ref, sec: ...`), `a * b`
overlays views, `a + b` puts them side by side, `repr(v)` describes a view in text and `v.png('out.png', ...)`
saves an image. See [moraine.cli in Python](../api/cli.md).

## Expected ranges

Each workflow guide has a table of the values of the sample data and the sane ranges, e.g. for the PS
candidates: amplitude dispersion `p50` roughly 0.3 to 1, temporal coherence between 0 and 1, a few percent to
about 30 % of the pixels as candidates. Compare `moraine info` with the table after every run and look at the
quicklooks before continuing; a value outside the range usually means a wrong parameter or input, not a
different site.
