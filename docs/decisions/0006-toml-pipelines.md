# 0006 Processing chains are TOML pipelines that resume

## Status

Accepted

## Date

2026-09-28

## Context

A processing chain has tens of steps, takes hours on real data and is tuned by rerunning parts of it.
Agents need to change parameters without writing code and resume after failures.

## Decision

Processing chains are TOML files run by `moraine run` (`moraine/command/pipeline.py`): `[vars]` with
`${name}` (overridable with `--var`), `[defaults]`, `[[step]]` with a command and its arguments. A step is
skipped when its argument hash and the modification times of its inputs are unchanged; steps downstream
of a changed output rerun. State, logs and quicklooks are kept per pipeline file in
`<workdir>/.moraine/<file name>/`.

## Consequences

- Several pipeline files can share a working directory and read each other's outputs.
- `moraine status` / `--dry-run` explain what will run and why.

## Do not

- Do not keep processing state elsewhere than `.moraine/<file name>/`.
- Do not make a step depend on anything but its arguments and input files.
