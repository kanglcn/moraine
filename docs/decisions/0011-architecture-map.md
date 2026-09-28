# 0011 The module map and layer rules are tested

## Status

Accepted

## Date

2026-09-28

## Context

moraine will grow. Agents and new contributors need to know where code goes before changing it, and a
hand written map goes stale unless something checks it.

## Decision

`ARCHITECTURE.md` lists every module of `moraine/` with its responsibility and states the dependency
rules between the API, CLI and command layers. `tests/test_architecture.py` fails when a module is
missing from the map, the map names a module that does not exist, a layer imports a higher one, or
`import moraine` imports torch. `docs/development.md` describes how changes are made and validated.

## Consequences

- Adding, moving or removing a module needs an `ARCHITECTURE.md` update in the same change.

## Do not

- Do not import `moraine.cli` or `moraine.command` from the API layer, or `moraine.command` from the CLI
  layer.
- Do not work around the architecture test by adding exceptions; propose a new record instead.
