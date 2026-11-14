# Command line

Every processing function of `moraine.cli` is a sub command of the `moraine` executable. The options, their
types and the help text are generated from the function signature and its numpy docstring (decision 0005), so
the pages of this section are the same text as `moraine COMMAND --help`, one page per module of `moraine.cli`:

```bash
moraine list                        # all processing commands, by module
moraine amp-disp --help             # the arguments of one command
moraine amp-disp --rslc raw/rslc.zarr --adi ps/ras_adi.zarr --cuda
```

## Conventions

- **Arguments** are `--name VALUE`; `--a-b` is accepted for `--a_b`. Booleans are `--cuda` / `--no-cuda`. Lists
  of paths take several values (`--ras a.zarr b.zarr`). Tuples take their integers (`--chunks 1000 1000`, also
  `1000,1000`). Image pairs are a file made by `image-pairs` (two integer columns) or a literal `[[0,1],[1,2]]`.
- **Unknown argument names are errors**, with a suggestion; nothing is passed silently into `**kwargs`. Extra
  keyword arguments (the input arrays of `math`) are explicit: `--kw a=a.zarr --kw b=b.zarr`.
- Every argument's help names **input** or **output**, the shape and dtype of the array, the unit and the
  default. Paths are relative to the current directory; outputs are overwritten.
- **Global options** of every command: `--json` (one JSON object on stdout, logs on stderr, see the
  [contract](../contracts/json-output.md)), `--traceback`, `--log FILE`, `-q` / `--quiet`.
- `--cuda` runs on the GPUs listed in `CUDA_VISIBLE_DEVICES`, one worker process per GPU (`--n_workers` for
  fewer); on the CPU, `--n_workers`, `--threads_per_worker` and `--processes` set how many tasks run at the same
  time and whether in threads or in processes (the defaults are in the help of each command).

## Built-in commands

`list`, `info`, `quicklook`, `view`, `run` and `status` are not processing commands: they inspect results, make
pictures and notebooks and run pipeline files. Their help is on the [built-in commands](builtin.md) page;
[Checking results](../start/checking.md) and [Workflows](../workflows/index.md) explain how to use them.

## In Python

The same functions are called as `moraine.cli.<name>(...)` with the same arguments (`import moraine.cli as mc`);
`mc.get_logger()` turns on the logging that the command line shows. See [moraine.cli in Python](../api/cli.md).
