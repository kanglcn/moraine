# Developing moraine

How to change moraine, for contributors and coding agents. The map of the code is `ARCHITECTURE.md`,
the design decisions are in `docs/decisions/`, the promised formats in `docs/contracts/`, planned features in
`docs/roadmap.md`.

## Before changing code

1. Read `ARCHITECTURE.md` for where the change goes, the index of `docs/decisions/` and the contracts
   the change touches.
2. Make sure the task is clear. If a request is ambiguous, restate what you understood and ask; do not
   guess on behaviour that users or other code depend on.
3. For a non-trivial change, state the goal, what must not change and what is out of scope.
4. Do not change code against an accepted decision or contract: propose a new decision record (or a new
   contract version) and ask the maintainer.

## Making the change

- Work on a branch, one topic per branch; never commit directly to `main`.
- Make the smallest complete change: the code, its tests and everything listed below that depends on it.
  Do not refactor, rename or reformat code that the task does not need; propose it separately.
- Bugs found on the way are fixed in their own commit (or reported), not mixed into the feature.
- Stop when the maintainer says so.

### Parallel work in git worktrees

Several agents (or people) work at the same time in separate worktrees, one topic each:

```bash
git worktree add ../moraine-<topic> -b <topic> main    # a new directory and branch per topic
git worktree list
git worktree remove ../moraine-<topic>                 # after the branch is merged
```

- Work only inside your worktree; do not edit files of the main checkout or of other worktrees.
- The development environment has moraine installed in editable mode from the main checkout. Inside a
  worktree, `python -m pytest` tests the worktree's code, but the `moraine` executable runs the main
  checkout's code: use `python -m moraine ...` (or `PYTHONPATH=$PWD`) to run your version.
- The pre-commit hook and the git configuration are shared by all worktrees.
- GPUs: use only the GPUs given to you (`CUDA_VISIBLE_DEVICES`); only one GPU pipeline per GPU.
- Processing runs use a working directory of your own, never one shared with another worktree.
- Changes to shared files (`CHANGELOG.md`, `ARCHITECTURE.md`, the decision index) are merged by hand when
  the branches come together; keep them to the lines your topic needs.

### What changes together

| when you change | also change, in the same commit |
|---|---|
| a function in `moraine/cli/` (arguments, behaviour) | its numpy docstring (it is the command help, decision 0005) and its tests; the examples and guides that use it |
| any public function | its numpy docstring (shapes, dtypes, defaults) and its tests |
| modules (add, remove, move, rename) | `ARCHITECTURE.md` (checked by `tests/test_architecture.py`) |
| the `--json` output, pipeline files, pyramids or data conventions | the contract in `docs/contracts/` and its tests; bump the contract version if old files / scripts break |
| an example pipeline | rerun it on the sample data and update the numbers in its guide (decision 0009) |
| a design choice (dependency, interface, approach) | a new record in `docs/decisions/` |
| dependencies | `pyproject.toml`; a decision record for a major one |
| user visible behaviour | the `Unreleased` section of `CHANGELOG.md` |

### Docstrings

Numpy style docstrings are the user documentation: the ones in `moraine/cli/` are the `moraine COMMAND
--help` text (decision 0005). Write what the function does and what a user needs to call it correctly:

- one summary line, then `Parameters` and `Returns`;
- every parameter: input or output (CLI), shape, dtype, unit, default, meaning of the value;
- constraints the caller must meet (e.g. unique coordinates), and the meaning of the result (e.g. the
  first point keeps its wrapped phase).

Do not write implementation details: algorithms and solvers used internally, performance tricks, caching,
optimality claims, comparisons with other software or with earlier versions. They go into code comments
next to the code, and design choices into `docs/decisions/`.

## Validating

Run the layers that the change can affect, from cheap to expensive:

```bash
ruff check                    # syntax errors and undefined names (rules in pyproject.toml)
pytest tests/test_architecture.py tests/test_docs.py tests/test_decisions.py tests/test_contracts.py  # seconds
pytest -m "not slow"          # about 2 min; GPU tests run when a GPU is visible
pytest -m slow                # CLI processing chain and GAMMA loading, about 3 min (15 min more without data/gamma/sim_orb)
git diff --check              # whitespace errors
```

- Changes in `moraine/cli/` or `moraine/command/`: run the slow tests too.
- GPU code: run the tests on a GPU node (`CUDA_VISIBLE_DEVICES` set); skipped GPU tests prove nothing.
- Commands used by `examples/`: run the affected examples end to end on `data/gamma`.
- Tests that need the sample data are skipped without it; say so when you report.

### Pre-commit hook

`.githooks/pre-commit` runs the cheap checks on every commit: whitespace errors, notebooks without
outputs, `ruff check` of the staged python files and the four consistency test files above. Install it
once per clone and point it to the python of the development environment:

```bash
git config core.hooksPath .githooks
export MORAINE_PYTHON=/path/to/env/bin/python   # optional, default: python
```

A failing hook blocks the commit: fix the cause. `git commit --no-verify` only in an emergency, said in
the commit message. The CI runs `ruff check` and `pytest -m "not slow"` on every push.

## Committing and releasing

- Commit messages say what changed and why; mention bugs found and fixed.
- Do not commit data, processing outputs, credentials or executed notebook outputs.
- `*.toml` is ignored by `.gitignore`; add an exception for new TOML files that belong in the repository.
- Pushing, merging into `main`, tagging and releasing need the maintainer's explicit go. Never force-push
  shared branches.

## Reporting

Report what was verified and what was not (skipped tests, missing data, no GPU). Report failures with
their output instead of working around them silently.
