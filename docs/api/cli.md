# `moraine.cli` in Python

`import moraine.cli as mc` gives the file based functions: the same operations as the [commands](../cli/index.md),
called from Python with the same arguments (`mc.amp_disp('raw/rslc.zarr', 'ps/ras_adi.zarr', cuda=True)`), plus the
viewer and a few helpers that are not commands.

```python
import moraine.cli as mc
logger = mc.get_logger()                       # the log lines the command line prints
mc.amp_disp('raw/rslc.zarr', 'ps/ras_adi.zarr', cuda=True)
mc.view('ps/ras_adi_pyramid', label='amplitude dispersion index', clim=(0, 1))
```

## view

::: moraine.cli.tiles.view
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

The map widget behind `view` is `moraine.cli.TileView` (anywidget and Leaflet; the tiles are rendered by the kernel
and sent through the notebook channel, so no server or port forwarding is needed, decision 0018). `moraine view
PYRAMID ... -o view.ipynb` writes a notebook of such views.

## get_logger

::: moraine.cli.logging.get_logger
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## zarr and dask helpers

`moraine.cli.dask_` has the parallel zarr readers and writers the commands use: `dask_from_zarr(path, chunks)` and
`dask_from_zarr_overlap(...)` make dask arrays of a zarr array (optionally with halos), `dask_to_zarr(darr, path,
chunks)` writes one, `parallel_read_zarr` / `parallel_write_zarr` read and write whole arrays with threads, and
`ZarrDir` iterates a directory of per chunk zarrs (`docs/contracts/data.md`). Errors of the parallel reads and
writes are raised.
