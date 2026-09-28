# 0005 The command line is generated from moraine.cli signatures and docstrings

## Status

Accepted

## Date

2026-09-28

## Context

AI agents and scripts need a command line with complete help and machine readable output. Writing the
commands by hand would duplicate every function signature and drift from the code.

## Decision

Every `@mc_logger` function in `moraine/cli/` is a `moraine` sub command (`moraine/command/`). Options,
types and help come from the signature and the numpy docstring, so the docstrings are the specification:
shapes, dtypes, input or output and real defaults must be exact. Unknown argument names are errors with
a suggestion; extra keyword arguments must be explicit (`--kw` / `[step.kw]`). Tuple arguments are
validated (`tuple[int, int]` needs two integers). `--json` prints one JSON object on stdout, logs go to
stderr.

## Consequences

- Adding a CLI function adds a command; `tests/test_command.py` checks that every argument is documented.
- A wrong docstring is a user-facing bug.

## Do not

- Do not hand-write argument parsing for individual commands.
- Do not let unknown arguments pass silently into `**kwargs`.
- Do not print anything but the JSON result on stdout with `--json`.
