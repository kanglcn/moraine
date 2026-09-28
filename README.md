# Moraine

> Modern Radar Interferometry Environment; A simple, stupid InSAR
> postprocessing tool in big data era

This package provide functions for InSAR post-processing which refers as
processing after SAR images co-registration and geocoding. The functions
include PS/DS identification, coherence matrix estimation, phase linking
etc.

<div>

> **Warning**
>
> Due to the heavy dependence on CuPy and CUDA, this package only works
> on device with Nivida GPU.

</div>

<div>

> **Warning**
>
> This package is under intensive development. API is subjected to
> change without any noticement.

</div>

## Principle[^1]

### Simplicity

There is no perfect workflow that always generate satisfactory InSAR
result, mostly due to decorrelation, strong atmospheric artifact and
high deformation gradient. Different methods implemented in different
packages need to be tried which is frustrating especially when the
packages are over-encapsulated and no detailed documentations are
provided. Furthermore, it brings more dirty work when users need to save
intermediate data from one software and prepared them in a designated
format and structure required by another software.

Moraine defines simplicity as *without complex data structure and
over-encapsulation*. Most data in Moraine is just the basic
multi-dimentional array, i.e., [NumPy](https://numpy.org/) array or
[CuPy](https://cupy.dev/) array in memory, or
[Zarr](https://zarr.readthedocs.io/en/stable/) on disk. Instead of
providing a standard workflow as
[StamPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/) and
[MintPy](https://mintpy.readthedocs.io/en/latest/), Moraine is designed
as a collection of functions that implement specific InSAR processing
techniques and data manipulation infrastructure.

### Modernity

Moraine strives to implement state-of-art InSAR techniques, including
advanced PS/DS identification, phase linking and deep-learning-based
methods.

Moraine also emphasizes performance, especially in this big data era.
Most Moraine functions are implemented with well-optimized GPU code or
OpenMP code. Furthermore, with the support of
[Dask](https://docs.dask.org/en/stable/), Moraine can be runed on
multi-GPUs to further accelerate the processing, get rid of the
limitation of memory and achieve asynchronous IO.

### Pragmatism

Moraine is a pragmatic library rather than an ideological workflow. The
large number of functions in MOraine offer free and open source
implementation for many InSAR techniques. Ultimately, workflow designs
are made on a case-by-case basis by the user. We provide the necessary
infrastructure and your role is to be innovative!

### User centrality

Whereas many InSAR packages attempt to be more user-friendly, Moraine
has always been, and shall always remain user-centric. This package is
intended to fill the needs of those contributing to it, rather than
trying to appeal to as many users as possible. It is targeted at the
proficient InSAR user, or anyone with a do-it-yourself attitude who is
willing to read the documentation, and solve their own problems.

All users are encouraged to participate and contribute to this package.
Reporting bugs and requesting new features by raising a Github
[issue](https://github.com/kanglcn/moraine/issues) is highly valued and
bugs fixing, documentation improving and features implementation by make
a Github [pull request](https://github.com/kanglcn/moraine/pulls) are
very appreciated. Users can also freely ask questions, provide technical
assistance to others or just exchange opinions in the
[discussions](https://github.com/kanglcn/moraine/discussions).

## Install

Because of GPU driver and CUDA Version Compatibility, there is no simple
solution for CUDA related packages installation. Users need to
successfully install
[cupy](https://docs.cupy.dev/en/stable/install.html#installation) and
[dask_cuda](https://docs.rapids.ai/api/dask-cuda/stable/) first.

Here is some tips for installing them. Generally, the cuda driver is
alrealy installed and maintained by the system administrator. Users only
need to determine the right cudatoolkit version. Frist run

``` bash
nvidia-smi
```

It will prints something like:

    ...
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    ...

The `CUDA Version` is the maxminum cudatoolkit version that supported by
the current CUDA driver. Here we use version 11.8 as an example. Then
you can install the needed `cudatoolkit`, `cupy`, `dask_cuda` by:

``` bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c conda-forge cupy cuda-version=11.8
conda install -c rapidsai -c conda-forge -c nvidia dask-cuda rmm cuda-version=11.8
```

Then

With conda:

``` bash
conda install -c conda-forge moraine
```

Or with pip:

``` bash
pip install moraine
```

The deep learning filters (`n2f`, `n2fs3d`, `n2ft`) additionally need
[PyTorch](https://pytorch.org/get-started/locally/) and the trained
models:

``` bash
pip install 'moraine[dl]'   # or install torch following the PyTorch guide for your CUDA version
python -c "import moraine; moraine.download_dl_model()"
```

In development mode:

``` bash
git clone git@github.com:kanglcn/moraine.git ./moraine
cd ./moraine
pip install -e '.[dev,dl]'
pytest                      # or `pytest -m "not slow"` for a quick run
```

Tests that need the sample data set, a GPU, GAMMA or the deep learning
models are skipped when these are not available. Point
`MORAINE_TEST_DATA` to the sample data directory (default: `./data`,
containing `rslc.zarr` and the GAMMA output `gamma/`; `gamma/sim_orb/` with the
precomputed simulated orbital phases makes the GAMMA loading test take seconds
instead of ~15 min) to run them.

## How to use

Read the [software
architecture](./nbs/Introduction/software_architecture.ipynb) for an
overview of the software design. Refer to [Tutorials](./nbs/Tutorials)
for the examples. Every function is documented by its docstring, e.g.
`help(moraine.emi)` or `help(moraine.cli.emi)`.

## Command line and pipelines

Every processing function of `moraine.cli` is also a `moraine` command,
with its options and help generated from the function docstring:

``` bash
moraine list                                   # all commands
moraine amp-disp --help
moraine amp-disp --rslc raw/rslc.zarr --adi ps/adi.zarr --cuda
moraine info ps/adi.zarr                       # shape, dtype and chunks of a result
moraine ras-pyramid --ras ps/adi.zarr --out_dir ps/adi_pyramid
moraine info ps/adi_pyramid                    # + statistics and anomaly warnings, from a coarse level
moraine quicklook ps/adi_pyramid -o adi.png    # PNG of the whole scene, drawn from the pyramid
moraine image-pairs --rslc raw/rslc.zarr --bandwidth 1 --out pairs.txt
```

A whole processing chain can be written in a TOML file and run with
`moraine run pipeline.toml`. Finished steps are skipped when their
arguments and inputs did not change, so a failed or modified pipeline is
resumed where needed; `moraine status pipeline.toml` shows the state,
and logs, output metadata and PNGs of the pyramids made by `ras-pyramid`
/ `pc-pyramid` steps are kept in `.moraine/`:

``` toml
[defaults]
cuda = true

[[step]]
name = "adi"
run = "amp-disp"
rslc = "raw/rslc.zarr"
adi = "ps/adi.zarr"

[[step]]
name = "ps_can"
run = "pc-logic-ras"
ras = "ps/adi.zarr"
gix = "ps/ps_can_gix.zarr"
operation = "(ras>=0)&(ras<=0.3)"
```

Add `--json` to any command for machine readable output (logs go to
stderr), which makes moraine easy to drive from scripts and AI agents.

The whole processing chain is available as verified example pipelines
in [examples](./examples) (load GAMMA data, PS, DS, PS+DS refinement,
unwrapping), each with a guide in [docs/workflows](./docs/workflows):
parameters, expected result ranges and checks.

## Processing with an AI agent

[AGENTS.md](./AGENTS.md) tells coding agents (Claude Code, Codex,
Cursor, ...) how to run and check moraine processing; `CLAUDE.md` and
`.claude/skills/` point Claude Code to it. Prepare the input data, then
ask the agent in plain words, e.g. "process the GAMMA data in
/data/site_a (reference 20220620) up to unwrapping in /work/site_a".

## Contact us

- Most discussion happens on
  [GitHub](https://github.com/kanglcn/moraine). Feel free to [open an
  issue](https://github.com/kanglcn/moraine/issues/new) or comment on
  any open issue or pull request.
- use github
  [discussions](https://github.com/kanglcn/moraine/discussions) to ask
  questions or leave comments.

## License

- This package is opened under
  [GPL-v3](https://www.gnu.org/licenses/gpl-3.0.en.html).
- The deep learning models under `moraine/dl_model/*` are opened under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## General Guidelines for Making a Pull Request (PR)[^2]

We follow the [git pull request
workflow](https://www.asmeurer.com/git-workflow/) to make changes to our
codebase.

Before you make a PR, have a quick look at the titles of all the
existing issues first. If there is already an issue that matches your
PR, leave a comment there to let us know what you plan to do. Otherwise,
open an issue describing what you want to do.

What should be included in a PR

- Each pull request should consist of a small and logical collection of
  changes; larger changes should be broken down into smaller parts and
  integrated separately.

- Bug fixes should be submitted in separate PRs.

How to write and submit a PR

- Read [docs/development.md](./docs/development.md) (how to change,
  validate and report), [ARCHITECTURE.md](./ARCHITECTURE.md) (module map),
  [docs/decisions/](./docs/decisions/README.md) and
  [docs/contracts/](./docs/contracts/README.md) (formats others depend on).

- The source code is the `.py` files in `moraine/`. Document new
  functions with [numpy style
  docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) and
  add tests in `tests/`; run `pytest` before submitting.

- The CI runs on machines without GPU, so GPU related packages (`cupy`,
  `dask_cuda`, `rmm`) must only be imported behind
  `moraine.utils_.is_cuda_available()`, and GPU tests are marked with
  `@pytest.mark.gpu`.

- Describe what your PR changes and why this is a good thing. Be as
  specific as you can. The PR description is how we keep track of the
  changes made to the project over time.

- Do not commit changes to files that are irrelevant to your feature or
  bugfix (e.g.: .gitignore, IDE project files, etc).

- Write descriptive commit messages. Chris Beams has written a
  [guide](https://cbea.ms/git-commit/) on how to write good commit
  messages.

PR review

Be willing to accept criticism and work on improving your code; we don’t
want to break other users’ code, so care must be taken not to introduce
bugs.

Be aware that the pull request review process is not immediate, and is
generally proportional to the size of the pull request.

[^1]: The pronciples are modified from the
    [principle](https://wiki.archlinux.org/title/Arch_Linux) of Arch
    Linux.

[^2]: this is modified from the [Contributers
    Guide](https://www.pygmt.org/latest/contributing.html) of
    [PyGMT](https://www.pygmt.org/latest/index.html).
