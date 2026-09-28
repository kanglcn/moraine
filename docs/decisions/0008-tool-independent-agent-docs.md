# 0008 Agent documentation is tool independent and tested

## Status

Accepted

## Date

2026-09-28

## Context

Users drive moraine with different AI tools (Claude Code, Codex, DeepSeek through Cursor / Cline).
Documentation that drifts from the code misleads agents more than no documentation.

## Decision

`AGENTS.md` is the entry point for all agents; `docs/workflows/` has one guide per example pipeline with
expected result ranges. Tool specific files only point to them: `CLAUDE.md` is `@AGENTS.md`,
`.claude/skills/` are thin wrappers. `tests/test_docs.py` checks that documented commands and files exist
and that the example pipelines parse.

## Consequences

- One set of instructions for every tool.

## Do not

- Do not put instructions only in a tool specific file (`CLAUDE.md`, skills, `.cursor/`, ...).
- Do not document commands, files or numbers that were not checked against the code or a real run.
