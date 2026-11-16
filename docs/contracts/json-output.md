# Contract: JSON output

Version 1. With `--json`, every `moraine` command prints exactly one JSON object on stdout (logs and
progress bars go to stderr) and exits with 0 on success, 1 on a processing error and 2 on a usage error. A usage
error is a wrong command line or pipeline file, or arguments that the command does not accept, wherever moraine finds
them (decision 0052): values outside their allowed values, arguments missing or excluding each other, inputs whose
shape, dtype or number of points do not fit. Problems found in the values of the data are processing errors.

`tests/test_contracts.py` compares the fields listed here with the real output: required fields must be
present, fields marked *optional* may be missing, other fields must not appear.

## Every output

| field | type | description |
|---|---|---|
| `version` | integer | version of this contract, 1 |
| `ok` | boolean | false if the command failed |

### error

Printed instead of the normal output when the command fails (`ok` is false).

| field | type | description |
|---|---|---|
| `error` | string | the error message |

### processing command

Any command of `moraine list` (e.g. `moraine amp-disp ... --json`).

| field | type | description |
|---|---|---|
| `command` | string | command name |
| `seconds` | number | run time |
| `outputs` | list of strings | argument paths created or modified by the command |
| `inputs` | object | argument paths read by the command -> their modification time (ns) |
| `summaries` | list of summaries | one summary (see below) per output |

### list

| field | type | description |
|---|---|---|
| `commands` | list of objects | `name`, `module` (moraine.cli module) and `summary` (first docstring line) of every command |

### info

| field | type | description |
|---|---|---|
| `summaries` | list of summaries | one summary per path argument |

### quicklook

| field | type | description |
|---|---|---|
| `png` | string | the PNG written |

### view

| field | type | description |
|---|---|---|
| `notebook` | string | the notebook written |
| `pyramids` | list of strings | the pyramids it shows, as given |

### run

| field | type | description |
|---|---|---|
| `pipeline` | string | the pipeline file |
| `workdir` | string | the working directory |
| `plan` | list of objects | per step: `name`, `command`, `action` (`run (reason)` or `skip (reason)`) |
| `steps` | list of step records | the steps that ran (see below); with `--dry-run` empty |

A failing step stops the run: `ok` is false, the exit code is 1 and the last step record has
`status = "failed"` and the `error`; there is no top level `error` field.

### status

| field | type | description |
|---|---|---|
| `pipeline` | string | the pipeline file |
| `workdir` | string | the working directory |
| `steps` | list of objects | per step: `name`, `command`, `status` (`pending`, `done`, `failed`, `outdated, will run (reason)`), and when it ran before `seconds`, `finished`, `error`, `log`, `outputs`, `quicklooks` |

### step record

Items of `steps` in the `run` output.

| field | type | description |
|---|---|---|
| `name` | string | step name |
| `command` | string | command name |
| `hash` | string | hash of the command and its arguments, used to detect changes |
| `args` | object | the arguments as written in the pipeline file (after `${var}` substitution) |
| `started` | string | start time, `YYYY-MM-DD HH:MM:SS` |
| `finished` | string | end time |
| `log` | string | log file, relative to the working directory |
| `status` | string | `done` or `failed` |
| `seconds` | number | *optional*, run time (`done` only) |
| `outputs` | list of strings | *optional*, paths created or modified (`done` only) |
| `inputs` | object | *optional*, paths read -> modification time (`done` only) |
| `summaries` | list of summaries | *optional*, summaries of the outputs (`done` only) |
| `quicklooks` | list of strings | *optional*, PNGs of the pyramid outputs |
| `error` | string | *optional*, the error (`failed` only) |

## Summaries

Objects describing a path (`moraine info`, outputs of commands and steps). `kind` tells which one.

### array summary

| field | type | description |
|---|---|---|
| `path` | string | the path |
| `kind` | string | `array` |
| `shape` | list of integers | array shape |
| `dtype` | string | numpy dtype |
| `chunks` | list of integers | zarr chunk shape |

### pyramid summary

`kind` is `raster pyramid` or `point cloud pyramid` (see the pyramid contract). The statistics are those
of all the data (the points of a point cloud pyramid), computed when the pyramid was built. A pyramid without
them, made before moraine 0.10.0, or one whose `ras-pyramid` / `pc-pyramid` did not finish is not summarized:
`moraine info` fails with a usage error (exit code 2) asking to rebuild it, and the summary of such an output is a
failed summary (decision 0049). For complex data
the value fields are named `amplitude_min`, `amplitude_max`, ... and describe the amplitude; for boolean
data only `true_fraction` is given.

| field | type | description |
|---|---|---|
| `path` | string | the path |
| `kind` | string | `raster pyramid` or `point cloud pyramid` |
| `shape` | list of integers | shape of level 0 |
| `dtype` | string | numpy dtype of the data (of the points for a point cloud pyramid) |
| `levels` | integer | number of levels |
| `method` | string | how the levels are made: `decimate` or `mean` (rasters), `first` or `mean` (point clouds) |
| `stats_level` | integer | always 0: the statistics are of all the data (the level they came from in pyramids made before 0.10.0, which are not summarized any more); to be removed with the next version of this contract |
| `nan_fraction` | number | *optional*, fraction of nan values |
| `min` | number | *optional*, minimum (finite values) |
| `max` | number | *optional*, maximum |
| `mean` | number | *optional*, mean |
| `std` | number | *optional*, standard deviation |
| `p01` | number | *optional*, 1st percentile |
| `p50` | number | *optional*, median |
| `p99` | number | *optional*, 99th percentile |
| `amplitude_min` | number | *optional*, as `min`, for complex data |
| `amplitude_max` | number | *optional* |
| `amplitude_mean` | number | *optional* |
| `amplitude_std` | number | *optional* |
| `amplitude_p01` | number | *optional* |
| `amplitude_p50` | number | *optional* |
| `amplitude_p99` | number | *optional* |
| `true_fraction` | number | *optional*, fraction of true values, for boolean data |
| `warnings` | list of strings | *optional*, anomalies: all values nan, infinite values, constant values; the images or channels of a stack that are all nan or constant |

### directory summary

| field | type | description |
|---|---|---|
| `path` | string | the path |
| `kind` | string | `directory` |
| `n_zarr` | integer | number of zarr arrays below it |
| `zarr` | list of strings | their paths relative to it, at most 20 |

### group summary

| field | type | description |
|---|---|---|
| `path` | string | the path |
| `kind` | string | `group` |
| `members` | list of strings | member names |

### file summary

| field | type | description |
|---|---|---|
| `path` | string | the path |
| `kind` | string | `file` |
| `bytes` | integer | file size |

### failed summary

A summary that could not be made (the command itself succeeded).

| field | type | description |
|---|---|---|
| `path` | string | the path |
| `error` | string | why the summary failed |
