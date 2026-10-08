# 概念

moraine 由哪些部分组成、它们怎样配合。这里的十分钟能省下翻参考手册的几小时。

## 一组函数，而不是一条固定的工作流

没有哪条工作流总能给出满意的 InSAR 结果：失相干、大气和强形变梯度总会在某处让固定的处理链失效。所以 moraine
是一组函数，每个函数实现一种技术（幅度离差指数、同质像元的统计检验、EMI 相位连接、深度学习滤波、最小费用流解缠……），
再加上围绕它们的数据基础设施。处理链由你自己组合；[经过验证的示例 pipeline](../workflows/index.md) 是起点，不是唯一的路。

数据刻意保持简单：**内存中是 numpy 或 cupy 数组，磁盘上是 zarr 数组**。没有容器对象，没有工程文件；每个结果都是
任何工具都能打开的数组。

## 三层

| 层 | 命名空间 | 数据 | 一次调用处理 |
|---|---|---|---|
| API | `moraine.*`（`import moraine as mr`） | 内存中的 numpy（CPU，numba）或 cupy（GPU）数组；输出与输入同类 | 一个单元：整幅场景的一景影像或一个像对，或一块像元 / 点及其完整时间序列 |
| CLI | `moraine.cli.*`（`import moraine.cli as mc`） | 磁盘上的 zarr 数据集，可大于内存；用 dask 分块，在 CPU 或多块 GPU 上 | 整个数据集 |
| 命令 | `moraine COMMAND ...`、`moraine run FILE` | 同样的 zarr 数据集 | 整个数据集，从 shell 或 pipeline 文件运行 |

CLI 函数不是 API 函数的简单封装：它把数据切成 API 函数能处理的单元，在数量受限的 dask worker 里映射这些单元，
再写回 zarr。每个 `moraine.cli` 函数都是 `moraine` 可执行程序的一条命令，帮助由 docstring 生成（决策 0005），
所以三层永远一致。`help(mr.emi)`、`help(mc.emi)` 和 `moraine emi --help` 说的是同一件事。

```python
import moraine as mr
import moraine.cli as mc

ph = mr.emi(coh)                                   # 内存中的一块点
mc.emi('ds/ds_can_coh.zarr', 'ds/ds_can_ph.zarr', cuda=True)   # 整个数据集，在 GPU 上
```

```bash
moraine emi --coh ds/ds_can_coh.zarr --ph ds/ds_can_ph.zarr --cuda
```

## 栅格与点云

处理链里流动的数据集有两种（完整约定见[数据约定](../contracts/data.md)，英文）：

- **雷达网格上的栅格**：形状 `(nlines, width[, n])`，方位向在前、距离向在后、堆栈维在最后。rslc 堆栈是
  `(nlines, width, nimages)` 的 complex64；`nan` 表示缺失数据。
- **选出像元（PS、DS）的点云**：形状 `(n_points, ...)` 的数组。点用网格索引 `gix` `(n_points, 2)`（方位，距离）
  或 hilbert 索引 `hix` `(n_points,)` 定位，点云**按 hilbert 顺序**存放，数组里相邻的点在地面上也相邻。`pc-union`、
  `pc-intersect`、`pc-diff` 和 `pc-select-data` 是排好序的索引上的集合运算；`ras2pc` 和 `pc2ras` 在两种数据之间搬运。

干涉数据是复数：像对 `(ref, sec)` 的干涉图是 `ref * conj(sec)`，相位连接后的相位历史和滤波后的干涉图幅度为 1，
点云的相干性压缩存放（相干矩阵的上三角，`(n_points, n_pairs)`），解缠相位是 float32 弧度 `(n_points, n_pairs)`。

## 分块

zarr 数组按块存放，dask 并行处理各块；块的布局决定内存和速度。moraine 的约定（决策 0019）是：**空间上分块，
每景影像（或每个像对）一个块**，例如栅格堆栈 `(lines_block, width_block, 1)`、点云堆栈 `(points_block, 1)`。
按影像处理的步骤读整块，按空间块处理的步骤读每景影像的一块，都不需要重新分块。命令有 `chunks`（处理）和
`out_chunks`（存储）参数；块太小时间都花在调度上，太大则内存不够。栅格沿方位向而不是距离向切分。

## 处理链

示例 pipeline 实现了一条常见的处理链；每一步都是一条命令，都可以替换：

1. **载入** GAMMA 结果到 zarr：去平去地形的 rslc 堆栈、坐标、视向量、元数据（[01](../workflows/01_load.md)）。
2. **PS 候选点**：幅度离差指数，以及 Noise2Fringe 滤波后干涉图的时间相干性（[02](../workflows/02_ps.md)）。
3. **DS**：统计同质像元（KS 检验）、DS 候选点、相干矩阵、带自适应正则化的 EMI 相位连接，按加权时间相干性和
   相干像对的连通性筛选 DS（[03](../workflows/03_ds.md)）。
4. **合并与精化**：PS 和 DS 合在一起，用 Noise2Fringe Transformer 滤波，按时间相干性保留（[04](../workflows/04_refine.md)）。
5. **解缠**：点的 Delaunay 网络上的最小费用流，逐幅干涉图（`mcf-pc`）或冗余网络整体（`emcf-pc`）解缠，
   按相位闭合校正解缠误差（[05](../workflows/05_unwrap.md)）。

## Pipeline、续跑与 JSON

一条处理链就是一个 TOML 文件，由 `moraine run FILE` 运行（[Pipeline 文件](../contracts/pipeline-file.md)，英文）。
参数相同且输入没变的步骤会被跳过，所以出错或改了参数之后再运行一次文件即可。`moraine run FILE --dry-run` 显示
将要运行什么，`moraine status FILE` 显示状态。加 `--json` 后每条命令在 stdout 打印一个 JSON 对象，日志走 stderr
（[JSON 输出](../contracts/json-output.md)）：脚本和 [AI agent](../agent/index.md) 读的就是它。

## GPU 还是 CPU

API 函数接受 numpy 或 cupy 数组，按类型分派（`moraine.api.utils_.get_array_module`）；CLI 函数和命令用 `cuda`
参数（每块 GPU 一个 dask worker，一个 rmm 内存池）。只有 `CUDA_VISIBLE_DEVICES` 列出了 GPU 才会用它。影像很多
（超过约 100 景）时，相位连接在多核 CPU 上可能比单块 GPU 更快；指南里有数字。
