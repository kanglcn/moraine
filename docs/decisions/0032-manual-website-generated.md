# 0032 The manual is a website generated from the docstrings, the command registry and docs/

## Status

Accepted

## Date

2026-10-08

## Context

Decision 0004 left moraine without a website after the nbdev notebooks were removed; the stale Quarto site
built from them was still served at kanglcn.github.io/moraine and documented an old API. Users, not only
agents, need a manual: the public functions and the commands with their arguments and usage examples, the
workflows, and how an AI agent runs the processing. A reference written by hand drifts from the code, as the
old site did.

## Decision

`mkdocs.yml` and MkDocs Material build the manual (`mkdocs build --strict`) from what the repository already
has:

- the numpy docstrings of the public names (`__all__`) of `moraine.api` and `moraine.cli`, through
  mkdocstrings (static analysis: no import of GPU code);
- the command registry `moraine.command.commands()` for the command line reference, one page per `moraine.cli`
  module, with the real steps of `examples/*.toml` as usage examples, and the built-in commands from their
  `--help`;
- the markdown of `docs/` (workflows, contracts, decisions, development, roadmap) as pages, and
  `ARCHITECTURE.md`, `CHANGELOG.md`, `AGENTS.md` included as snippets;
- the tutorial notebooks of `nbs/Tutorials/` converted to pages (text and code; the repository holds no outputs);
- short examples in `docs/api/examples/` that run at build time (markdown-exec, CPU, synthetic data).

The generator is `docs/_scripts/refgen.py`; the MkDocs hook `docs/_scripts/mkdocs_hooks.py` writes its pages into
`docs/` before every build (ignored by git). The site is built in English and Chinese (mkdocs-static-i18n:
`page.zh.md` next to `page.md`; pages without a translation show the English text). Figures are
quicklook PNGs of real runs, committed under `docs/assets/`. The theme is Material with the overrides in
`docs/overrides/` and `docs/assets/`. The CI builds the site on every push and deploys `main` to the `gh-pages`
branch. The dependencies are the optional group `docs` of `pyproject.toml`.

## Consequences

- Decision 0004 is superseded. Nothing in the manual is written twice: a docstring, an example pipeline or a
  guide changes the manual.
- A wrong docstring, a dead link, a command without documentation or a failing example breaks the build.
- `tests/test_docs.py` checks the generator without building the site; the full build is a slow test.

## Do not

- Do not write argument lists, signatures or command tables by hand in the manual; fix the docstring.
- Do not commit generated pages (`docs/cli/<module>.md`, `docs/api/<module>.md`, `docs/tutorials/`) or
  notebook outputs to get content into the site.
- Do not add pages that `mkdocs build --strict` does not build.
- Do not deploy from a branch other than `main`.
