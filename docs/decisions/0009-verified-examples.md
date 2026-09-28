# 0009 Example pipelines are verified on real data

## Status

Accepted

## Date

2026-09-28

## Context

Agents start from examples. An example that was never run spreads its errors; running the chain
end to end also finds bugs that unit tests miss.

## Decision

`examples/*.toml` cover the whole chain (load, PS, DS, refinement, unwrapping) and are run end to end on
the sample data (`data/gamma`, 2500 x 1834 x 17) before they change. The numbers in `docs/workflows/`
(counts, ranges, run times) come from such a run.

## Consequences

- The first full run (commit 85c2139) found four bugs (missing parent directories in `load-gamma-*`,
  image pair files read too early, stdout pollution of `moraine run --json`, per-chunk directories taken
  for pyramids).

## Do not

- Do not add or change an example without running it, and update the guide numbers from that run.
- Do not copy parameters into the guides that the examples do not use.
