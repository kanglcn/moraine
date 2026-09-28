# Contracts

Formats and behaviour that users, scripts, agents or other moraine versions depend on. A contract is a
promise: when the code and a contract disagree, the code is wrong, unless a new contract version is
released on purpose. `tests/test_contracts.py` checks the code against them.

| contract | version | where the version is |
|---|---|---|
| [JSON output](json-output.md) of `moraine ... --json` | 1 | `version` field of every output; `moraine.command.JSON_VERSION` |
| [Pipeline files](pipeline-file.md) run by `moraine run` | 1 | optional `[pipeline] version`; `moraine.command.pipeline.PIPELINE_VERSION` |
| [Pyramids](pyramid.md) made by `ras-pyramid` / `pc-pyramid` | 1 | `moraine_pyramid` attribute of `0.zarr`; `moraine.cli.plot.PYRAMID_VERSION` |
| [Data conventions](data.md) of arrays and zarr datasets | - | changes need a decision record |

## Changing a contract

- Compatible additions (a new optional field, a new optional table key) keep the version; document
  them here and in the tests in the same change.
- Anything that breaks existing files, scripts or readers (renamed or removed fields, changed meaning,
  changed layout) bumps the version, keeps reading the old version where possible, and needs a decision
  record in `docs/decisions/`.
- Update `CHANGELOG.md` for every contract change.
