# Developing moraine

How to change moraine, for contributors and coding agents. The map of the code is `ARCHITECTURE.md`,
the design decisions are in `docs/decisions/`, the promised formats in `docs/contracts/`, planned features in
`docs/roadmap.md`.

## Before changing code

1. Read `ARCHITECTURE.md` for where the change goes, the index of `docs/decisions/` and the contracts
   the change touches.
2. Make sure the task is clear. If a request is ambiguous, restate what you understood and ask; do not
   guess on behaviour that users or other code depend on.
3. One change at a time: a single smallest new feature or a single smallest improvement. Split a larger
   task into such changes and do them one after another, each discussed, implemented and reported on its
   own.
4. Discuss every change with the maintainer before implementing it, until each step is clear: the goal,
   what must not change and what is out of scope, the functions and files touched, the tests, and how it
   is validated (e.g. memory and run time measured). Then ask for the go; do not start the implementation
   before it.
5. When you propose an interface (a new or changed function or command), give every input and output
   with its shape, dtype, unit and meaning (what one element is, e.g. "unwrapped phase in radians of
   pair k at point i"), the defaults and the allowed values, before any implementation. Functions and
   commands take exactly the values they need (e.g. pixel spacings as numbers, dates as a list), never a
   metadata file such as `meta.toml`.
6. Do not change code against an accepted decision or contract: propose a new decision record (or a new
   contract version) and ask the maintainer.

## Making the change

- Work on a branch, one topic per branch; never commit directly to `main`.
- Make the smallest complete change: the code, its tests and everything listed below that depends on it.
  Do not refactor, rename or reformat code that the task does not need; propose it separately.
- Bugs found on the way are fixed in their own commit (or reported), not mixed into the feature.
- Stop when the maintainer says so.

- numba CPU functions use `mjit`, `ngjit` or `ngpjit` of `moraine/api/utils_.py` and CUDA kernels `mcuda_jit`, never
  `cache=True` (decision 0029). Their compiled code is cached in `~/.cache/moraine/numba/<hash of the sources>`; a
  change of any moraine file compiles everything once again.

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

### How a command runs its work

A command of `moraine/cli/` lists its tasks and hands them to `moraine.cli.executor.Executor` (decision 0033):

```python
tasks = [([Chunk(rslc, (*sl, slice(0, nimages)))], [Chunk(adi, sl)]) for sl in all_chunk_slices((nlines, width), chunks)]
with Executor(cuda=cuda, n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes) as ex:
    ex.map_chunks(mr.amp_disp, tasks, desc='amplitude dispersion')
```

A task is a plain function with picklable arguments: `chunk_task` reads the `Chunk` inputs (cupy arrays with
`cuda`), calls the API function and writes the results to the `Chunk` outputs, which must cover whole chunks of
the output array (create the output zarr before, with the processing chunks as a multiple of its chunks). Other
tasks go through `ex.map(fn, [args, ...])`; an object every task needs (an index array) is shared with `ex.put`;
per worker setup (something every worker loads once) with `ex.run_on_workers`. A worker process reads the `Chunk`
inputs of the next task while it runs the current ones, so it holds the inputs of `threads_per_worker + 1` tasks. Do
not start threads or processes of your own in a command (decision 0034).

### Large data and memory

moraine is made for data larger than memory (tens of millions of points, stacks of hundreds of images);
every change is designed for that size, not for the sample data.

- Memory, not only time, decides a design. Estimate the peak memory of a change as
  `shared inputs + number of parallel workers x memory per task` and keep it bounded: the number of
  workers (and chunks) must be a parameter, and its default must not multiply a large per task memory by
  the number of cores.
- Threads share the inputs (one copy) but every thread has its own working arrays; worker processes
  (`processes=True`, the GPU workers) load their inputs themselves (`Chunk`) or once per worker (`put`). Choose by memory: threads with
  numba `nogil` functions for work on shared arrays in memory, processes only where the work holds the
  GIL, and in both cases few workers when the per task memory is large.
- Split the work into units that fit in memory. An API function (`moraine/api/`) processes one unit: one
  image (or image pair) of the whole scene, or one block of pixels / points with its whole time series
  (plus a halo where neighbours are needed). The CLI function (`moraine/cli/`) cuts the zarr data into
  these units, maps the API function over them with the executor (bounded number of workers) and writes the
  results to zarr. The CLI never loads a whole stack; an API function that chains several steps on a
  whole stack in memory is fine for small data and tests, but the CLI uses the per unit functions.
- An algorithm whose steps need different units (e.g. per block of points, then per image, then per
  block again) passes its intermediate results between the steps through zarr, chunked so that every
  step reads and writes whole chunks (e.g. `(n_points_block, 1)` chunks are written by a per image step
  and read by a per block step without rechunking). Only small global structures (coordinates, the
  network of a point cloud for unwrapping) are held in memory for the whole run; say so in the docstring
  of the command, with their size per point.
- Keep working arrays compact: int32 instead of int64 indices where the size allows, int8 / bool for
  small values, sparse storage for mostly empty results, float32 outputs.
- Measure: for a change of a processing step, report run time and peak memory (e.g. `/usr/bin/time -f
  %M`, `resource.getrusage`) on a large synthetic case (about 10 million points) as well as on the sample
  data, and the memory per worker.

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
