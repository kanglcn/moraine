---
name: moraine-development
description: Change the moraine code itself (new or changed functions, commands, pipelines, tests, dependencies, documentation of the code). Use when the user asks to implement, fix, refactor, test or document moraine, not when they ask to process data with it.
---

# moraine development

The rules are tool independent and live in the repository. Before changing anything:

1. Read `docs/development.md` (how to make, validate, commit and report a change; docstring rules;
   what must change together).
2. Read `ARCHITECTURE.md` for where the change goes and the layer rules.
3. Read the index `docs/decisions/README.md` and the contracts in `docs/contracts/` the change touches.
   Do not change code against an accepted decision or contract; propose a new record and ask.

Checklist before reporting the change as done:

- the smallest complete change: code, numpy docstring (no implementation details), tests, and the
  docs / examples / changelog listed in the "What changes together" table;
- a new test fails on the old code and passes on the new one;
- `ruff check`, the consistency tests and `pytest -m "not slow"` pass (slow and GPU tests when the
  change touches `moraine/cli/`, `moraine/command/` or GPU code); skipped tests are reported;
- bugs found on the way are fixed in their own commit or reported;
- nothing is committed, pushed or merged without the maintainer's go.
