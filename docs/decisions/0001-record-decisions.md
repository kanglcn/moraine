# 0001 Record design decisions as ADRs in docs/decisions

## Status

Accepted

## Date

2026-09-28

## Context

moraine is developed more and more with AI agents (Claude Code, Codex, ...). Decisions taken in a
conversation or kept in one tool's private memory are invisible to the next session, another tool or a
new contributor, which then "improves" the code back into a rejected design. moraine is small now but is
expected to grow.

## Decision

Every decision that shapes how moraine is built or used is written as a record in `docs/decisions/`,
listed in `docs/decisions/README.md`. `AGENTS.md` tells agents to read the index before design changes.
The format and the index are checked by `tests/test_decisions.py`.

## Consequences

- Decisions travel with the code and are versioned with it.
- A design change starts with a new record (or a superseding one), not with code.

## Do not

- Do not keep project decisions only in chat history, commit messages or tool-specific memory.
- Do not delete superseded records; mark them `Superseded by NNNN`.
