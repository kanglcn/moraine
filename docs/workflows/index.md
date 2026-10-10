# Workflows

The processing chain of moraine as five verified pipeline files (decisions 0009, 0033): each one is run end to
end on the sample data set before it changes, and the numbers and figures in its guide come from that run. The
sample data set is Campi Flegrei: Sentinel-1 descending track 22, 92 dates from 2019-01-02 to 2021-12-29, 981 x
4160 pixels at full resolution, the uplifting caldera west of Naples (`data/CampiFlegrei/README.md` in the
repository says how the GAMMA results were made). The `[vars]` defaults of the files are its values; for other
data give yours with `--var`. The files share one working directory and read each other's outputs, so run them in
order:

```bash
moraine run examples/01_load.toml --workdir WORK --var gamma=/path/to/gamma --var reference=YYYYMMDD --var geo=YYYYMMDD
moraine run examples/02_ps.toml --workdir WORK --var shape=NLINES,WIDTH
moraine run examples/03_ds.toml --workdir WORK --var shape=NLINES,WIDTH
moraine run examples/04_refine.toml --workdir WORK
moraine run examples/05_unwrap.toml --workdir WORK --var shape=NLINES,WIDTH --var range_pixel_spacing=... --var azimuth_pixel_spacing=...
```

| pipeline | guide | what it does | outputs |
|---|---|---|---|
| `examples/01_load.toml` | [01 Load GAMMA results](01_load.md) | GAMMA results to zarr, coordinates, web mercator | `WORK/raw/` |
| `examples/02_ps.toml` | [02 PS candidates](02_ps.md) | amplitude dispersion + Noise2Fringe temporal coherence | `WORK/ps/` |
| `examples/03_ds.toml` | [03 DS processing](03_ds.md) | SHP, DS candidates, coherence, phase linking, DS selection | `WORK/ds/` |
| `examples/04_refine.toml` | [04 Merge and refine](04_refine.md) | PS + DS, Noise2Fringe Transformer, temporal coherence | `WORK/pc/` |
| `examples/05_unwrap.toml` | [05 Phase unwrapping](05_unwrap.md) | minimum cost flow on a Delaunay network | `WORK/unw/` |

To change a parameter, copy the file into the working directory and edit the copy, or override a `[vars]` value
with `--var name=value`. A rerun skips the steps whose arguments and inputs did not change and reruns everything
downstream of a change; `moraine run FILE --dry-run` shows the plan, `moraine status FILE` the state. A failing
step prints its error and log; fix the cause and run the file again. The file format is the
[pipeline files contract](../contracts/pipeline-file.md); `moraine run FILE --json` prints one JSON object with the
plan, the step records and the summaries of the outputs ([JSON output](../contracts/json-output.md)).

Each guide lists the steps, the parameters to tune, the expected results on the sample data set with the figures
of that run, and what to do when they are off. The guides are the same files the [AI agents](../agent/index.md)
read; the [recorded session](../agent/index.md) ran exactly these five files on the sample data set.
