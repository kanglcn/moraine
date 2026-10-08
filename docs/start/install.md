# Install

moraine is a Python package (>= 3.11). Everything runs on the CPU with numba; the GPU packages are optional
and make the heavy steps (phase linking, deep learning filters, coherence) many times faster.

## CPU only

```bash
pip install moraine
```

or, with conda:

```bash
conda install -c conda-forge moraine
```

## GPU (CUDA)

The GPU code needs cupy, numba-cuda, dask-cuda and rmm, which have to match the CUDA driver of the machine.
Find the highest CUDA version the driver supports:

```bash
nvidia-smi
```

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
+-----------------------------------------------------------------------------+
```

Install the packages for that version (here 11.8 as an example) with conda, then moraine:

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c conda-forge cupy numba-cuda cuda-version=11.8
conda install -c rapidsai -c conda-forge -c nvidia dask-cuda rmm cuda-version=11.8
pip install moraine
```

!!! note "How moraine finds the GPU"
    moraine uses a GPU only when `CUDA_VISIBLE_DEVICES` is set to a non-empty value; without it every
    command runs on the CPU (`cuda = false`). GPU commands start one dask worker per listed GPU; list fewer
    GPUs to leave some free. numba-cuda finds the CUDA libraries through the activated conda environment
    (`CONDA_PREFIX`) or `CUDA_HOME`; when a script calls the python of an environment without activating it,
    moraine sets `CUDA_HOME` to that environment.

## Deep learning filters

`n2f`, `n2fs3d` and `n2ft` need [PyTorch](https://pytorch.org/get-started/locally/) and the trained models:

```bash
pip install 'moraine[dl]'        # or install torch for your CUDA version following the PyTorch guide
python -c "import moraine; moraine.download_dl_model()"
```

The models are downloaded once into the package directory (`moraine/dl_model/`). They are published under
CC BY-NC-SA 4.0; the code is GPL-3.0.

## GAMMA

Loading GAMMA results (`load-gamma-*`) runs GAMMA programs (`phase_sim_orb`, `create_offset`, `geocode`,
`base_calc`), which must be on `PATH`. Nothing else needs GAMMA: the unwrapping of `mcf-pc` and `emcf-pc` is
moraine's own.

## Development mode

```bash
git clone git@github.com:kanglcn/moraine.git
cd moraine
pip install -e '.[dev,dl,docs]'
pytest -m "not slow"             # tests needing the sample data, a GPU, GAMMA or the models are skipped
mkdocs serve                     # this manual, at http://127.0.0.1:8000/
```

See [Developing moraine](../development.md) for how changes are made and validated.

## Check

```bash
moraine list                     # the processing commands
moraine emi --help               # the arguments of one command
python -c "import moraine; print(moraine.__version__)"
```
