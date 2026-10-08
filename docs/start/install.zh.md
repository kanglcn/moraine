# 安装

moraine 是一个 Python 包（>= 3.11）。所有功能都能用 numba 在 CPU 上运行；GPU 包是可选的，能让重的步骤
（相位连接、深度学习滤波、相干性估计）快很多倍。

## 仅 CPU

```bash
pip install moraine
```

或者用 conda：

```bash
conda install -c conda-forge moraine
```

## GPU（CUDA）

GPU 代码需要 cupy、numba-cuda、dask-cuda 和 rmm，它们必须与机器的 CUDA 驱动匹配。先看驱动支持的最高 CUDA 版本：

```bash
nvidia-smi
```

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
+-----------------------------------------------------------------------------+
```

用 conda 安装对应版本的包（这里以 11.8 为例），再装 moraine：

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c conda-forge cupy numba-cuda cuda-version=11.8
conda install -c rapidsai -c conda-forge -c nvidia dask-cuda rmm cuda-version=11.8
pip install moraine
```

!!! note "moraine 怎样找到 GPU"
    只有当 `CUDA_VISIBLE_DEVICES` 设为非空值时 moraine 才使用 GPU；否则每条命令都在 CPU 上运行（`cuda = false`）。
    GPU 命令为列出的每块 GPU 启动一个 dask worker；少列几块就能留出空闲的 GPU。numba-cuda 通过激活的 conda 环境
    （`CONDA_PREFIX`）或 `CUDA_HOME` 找 CUDA 库；脚本直接调用某个环境的 python 而没有激活它时，moraine 会把
    `CUDA_HOME` 设为该环境。

## 深度学习滤波

`n2f`、`n2fs3d` 和 `n2ft` 需要 [PyTorch](https://pytorch.org/get-started/locally/) 和训练好的模型：

```bash
pip install 'moraine[dl]'        # 或按 PyTorch 指南为你的 CUDA 版本安装 torch
python -c "import moraine; moraine.download_dl_model()"
```

模型只下载一次，放在包目录（`moraine/dl_model/`）里。模型的许可是 CC BY-NC-SA 4.0；代码是 GPL-3.0。

## GAMMA

载入 GAMMA 结果（`load-gamma-*`）要运行 GAMMA 程序（`phase_sim_orb`、`create_offset`、`geocode`、`base_calc`），
它们必须在 `PATH` 上。其余功能都不需要 GAMMA：`mcf-pc` 和 `emcf-pc` 的解缠是 moraine 自己的实现。

## 开发模式

```bash
git clone git@github.com:kanglcn/moraine.git
cd moraine
pip install -e '.[dev,dl,docs]'
pytest -m "not slow"             # 需要样例数据、GPU、GAMMA 或模型的测试会被跳过
mkdocs serve                     # 本手册，在 http://127.0.0.1:8000/
```

改动代码的方法和验证步骤见[开发 moraine](../development.md)（英文）。

## 检查

```bash
moraine list                     # 处理命令
moraine emi --help               # 某条命令的参数
python -c "import moraine; print(moraine.__version__)"
```
