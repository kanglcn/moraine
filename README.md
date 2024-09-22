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

In development mode:

``` bash
git clone git@github.com:kanglcn/moraine.git ./moraine
cd ./moraine
pip install -e '.[dev]'
```

## How to use

Read the [software
architecture](./Introduction/software_architecture.ipynb) for an
overview of the software design. Refer to [Tutorials](./Tutorials) for
the examples. Refer to [API](./API/) and [CLI](./CLI) for the detailed
usage of every functions.

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

- This package is developed with the [nbdev](https://nbdev.fast.ai/), a
  notebook-driven development platform. Developers should write or edit
  the notebooks rather than the `.py` files. After than, run
  `nbdev_export` to export the code in the notebooks to `.py` files and
  run `nbdev_clean` to clean the notebooks.

- The github bot to generate docs do not support GPU, so all GPU related
  packages (`dask_cuda`) should be prevented in import cells and export
  cells. `cupy` v13 now allow `import cupy` on a cpu-only machine. So it
  can be used now.

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
