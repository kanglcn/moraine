# 命令行

`moraine.cli` 的每个处理函数都是 `moraine` 可执行程序的一条子命令。选项、类型和帮助文本由函数签名和 numpy
docstring 生成（决策 0005），所以本节的页面就是 `moraine COMMAND --help` 的内容，`moraine.cli` 的每个模块一页
（docstring 是英文的）：

```bash
moraine list                        # 所有处理命令，按模块
moraine amp-disp --help             # 某条命令的参数
moraine amp-disp --rslc raw/rslc.zarr --adi ps/ras_adi.zarr --cuda
```

## 约定

- **参数**写成 `--name VALUE`；`--a-b` 等同于 `--a_b`。布尔值是 `--cuda` / `--no-cuda`。路径列表接受多个值
  （`--ras a.zarr b.zarr`）。元组给出它的整数（`--chunks 1000 1000`，也可以 `1000,1000`）。像对是 `image-pairs`
  生成的文件（两列整数）或字面量 `[[0,1],[1,2]]`。
- **未知的参数名是错误**，并给出建议；什么都不会悄悄传进 `**kwargs`。额外的关键字参数（例如 dask 集群选项）要
  显式给出：`--kw memory_limit=20GB`。
- 每个参数的帮助都写明 **input** 或 **output**、数组的形状和类型、单位和默认值。路径相对于当前目录；输出会被覆盖。
- 每条命令的**全局选项**：`--json`（stdout 上一个 JSON 对象，日志走 stderr，见[约定](../contracts/json-output.md)）、
  `--traceback`、`--log FILE`、`-q` / `--quiet`。
- `--cuda` 在 `CUDA_VISIBLE_DEVICES` 列出的 GPU 上运行，每块一个 dask worker；`--n_workers`、`--threads_per_worker`
  和 `--processes` 决定 CPU 集群的大小。

## 内置命令

`list`、`info`、`quicklook`、`view`、`run` 和 `status` 不是处理命令：它们查看结果、出图、写 notebook、运行 pipeline
文件。它们的帮助在[内置命令](builtin.md)页；用法见[检查结果](../start/checking.md)和[工作流](../workflows/index.md)。

## 在 Python 里

同样的函数在 Python 里以 `moraine.cli.<name>(...)` 调用，参数相同（`import moraine.cli as mc`）；`mc.get_logger()`
打开命令行显示的那些日志。见 [Python 里的 moraine.cli](../api/cli.md)。
