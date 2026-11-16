# 0051 Errors in the arguments of a command are usage errors, raised as ArgumentError in any layer

## Status

Accepted

## Date

2026-11-16

## Context

The `moraine` command exits with 2 on a usage error and with 1 on a processing error (contract `json-output.md`), so
that a script or an agent can tell a wrong call, to be fixed on the command line, from a processing that failed. The
usage errors were those found by the command layer (unknown or missing arguments, values that are not numbers, image
pairs against the data conventions) and the `ArgumentError` of `moraine/cli/utils_.py`, raised by one check of
`pc-select-data`. The other checks of arguments raised ValueError, in the CLI functions (`rows` and `method` of
`ras-pyramid`, `method` of `pc-pyramid`, "exactly one of" of `polygon-mask`) and in the API functions they call (the
`shape` needed with an empty grid index of the set operations and of `pc_sort`): the same kind of mistake exited with
2 or with 1, depending on the layer that found it. The CLI layer cannot raise the command layer's `UsageError`, and the
API, which finds many argument errors, imports neither the CLI nor the command layer (decision 0011).

## Decision

- An argument error is a call that the parameters of a command or function do not accept, found from the arguments
  and from the metadata of the inputs they name (number of dimensions, shape, dtype, chunks), not from the values of
  the data:
  - a value outside its allowed values or range: a `method`, `rows` a power of 2, a wavelength in meters, the index of
    a reference image or point, dates YYYYMMDD in increasing order, an `operation` that is not a supported expression,
    image pairs against the data conventions or not linking the images;
  - an argument left out that the others need, or arguments that exclude each other: `shape` for an empty grid index,
    exactly one of `x` and `y`, `gix` or `shape`;
  - inputs whose number of dimensions, shape, dtype or chunks do not fit the help or each other, inputs without data
    where data are needed, and inputs beyond the sizes a function supports (e.g. SHP windows wider than 64 pixels).
- Argument errors raise `ArgumentError`, a ValueError defined in `moraine/api/utils_/errors.py`, the lowest layer:
  the API, the CLI and the command layer raise and recognize the same class. It is importable as
  `moraine.api.utils_.ArgumentError` and, as before, `moraine.cli.utils_.ArgumentError`; it is not a name of
  `moraine.*` or `moraine.cli.*`. The documented errors of the API stay ValueErrors, and `except ValueError` catches it.
- `moraine COMMAND` reports an ArgumentError as a usage error, wherever it was raised (the CLI function, the API
  function it calls, a worker of the executor, whose errors reach the command as they are): exit code 2, the message
  without the name of the exception and without the `--traceback` hint. The command layer raises its `UsageError` for
  what it checks itself (the command line, pipeline files, the inputs of `info`, `quicklook` and `view`).
- A command checks its arguments, and the metadata of its inputs, before it cleans or writes its outputs, where these
  suffice.
- Not argument errors (processing errors, exit code 1), as they are:
  - problems found in the values of the data: a reference mask without points or reference points without values,
    indices that are not sorted or unique, collinear or duplicated points of a triangulation, unwrapped phases that do
    not rewrap, no convergence;
  - files of other programs that are incomplete or malformed (GAMMA parameter files without a date or with incomplete
    orbits), missing files, the environment (no GPU, GAMMA or PyTorch) and internal errors of moraine (slices made by
    moraine outside an array, a task that does not cover whole chunks);
  - errors that are not ValueErrors keep their type (TypeError, NotImplementedError, FileExistsError).
- In a pipeline a step that fails with an ArgumentError fails as any step: `moraine run` exits with 1 and the error of
  the step names `ArgumentError` (contract `json-output.md`).
- The viewer `moraine.cli.view` is not a command: its argument checks stay ValueErrors, which `moraine quicklook` and
  `moraine view` turn into usage errors. Pyramids to rebuild stay ValueErrors of `pyramid_levels`, usage errors of
  `info`, `quicklook` and `view` (decision 0049).

## Consequences

- A wrong command line exits with 2 whichever layer finds it; a processing error (1) means the data or the
  environment, not the call.
- New argument checks raise ArgumentError (`from ..utils_.errors import ArgumentError` in the API, `from .utils_
  import ArgumentError` in the CLI); their tests check the class and the message, those of the command line the exit
  code 2 and the JSON error.
- An ArgumentError found by an API function inside a task, after the command started its outputs, is still a usage
  error; the command keeps what it wrote, as for a processing error. Checks that the arguments and the metadata of
  the inputs allow belong before the outputs.

## Do not

- Do not raise a plain ValueError for an argument error in a command or in an API function, and do not raise
  ArgumentError for a problem in the values of the data.
- Do not import `moraine.command` in the CLI or `moraine.cli` in the API to make a usage error; raise ArgumentError.
- Do not make ArgumentError anything but a ValueError: the API documents its errors as ValueErrors.
- Do not clean or write the outputs of a command before the checks that need only the arguments and the metadata of
  its inputs.
