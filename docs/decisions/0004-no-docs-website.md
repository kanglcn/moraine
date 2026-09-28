# 0004 No documentation website for now

## Status

Accepted

## Date

2026-09-28

## Context

The Quarto website was built by nbdev from the API notebooks, which no longer exist (0003).

## Decision

No documentation website is maintained for now. Documentation is: docstrings (`help()`,
`moraine COMMAND --help`), `README.md`, `AGENTS.md`, `docs/workflows/`, `docs/decisions/` and the
tutorial notebooks.

## Consequences

- The deploy workflow and the Quarto configuration were removed.

## Do not

- Do not add a documentation generator or website without a new decision record.
