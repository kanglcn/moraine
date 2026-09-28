# 0010 Formats others depend on are versioned contracts

## Status

Accepted

## Date

2026-09-28

## Context

Agents and scripts parse the `--json` output, users keep pipeline files and pyramids for a long time, and
every stored result follows the data conventions. When these change silently, old files and scripts break
without a clear error, and nobody can tell which behaviour is intended.

## Decision

The `--json` output, pipeline files, pyramids and data conventions are specified in `docs/contracts/`.
The first three carry a version (`JSON_VERSION`, `PIPELINE_VERSION`, `PYRAMID_VERSION`) written into the
output or file; readers reject newer versions with a clear error. `tests/test_contracts.py` compares the
documented fields and layouts with the real outputs.

## Consequences

- A change to these formats changes the contract and its tests in the same commit.
- Writing the contract tests found two bugs: outputs named without `/` or `.` were not detected, and a
  failed `moraine run` raised `SystemExit` from `main()`.

## Do not

- Do not add, rename or remove a field of the `--json` output without updating
  `docs/contracts/json-output.md`.
- Do not make a breaking change without bumping the version and recording the decision.
