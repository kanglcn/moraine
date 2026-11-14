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

## zarr helpers and workers

`moraine.cli.zarr_` has the zarr readers and writers the commands use: `parallel_read_zarr(z, slices)` and
`parallel_write_zarr(data, z, slices)` read and write a part of a zarr array chunk by chunk with threads (errors of
the reads and writes are raised), and `ZarrDir` takes a directory of per chunk zarrs as one array
(`docs/contracts/data.md`). `moraine.cli.executor` runs the tasks of a command (decisions 0033, 0034): `Executor`
starts the workers (threads of the process on the CPU, one process per GPU with `cuda`) and `chunk_task` reads the
`Chunk` inputs of a task, calls an API function and writes its outputs.
