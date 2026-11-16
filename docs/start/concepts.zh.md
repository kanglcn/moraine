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
| CLI | `moraine.cli.*`（`import moraine.cli as mc`） | 磁盘上的 zarr 数据集，可大于内存；由 moraine 自己的 worker 分块处理，CPU 上用线程，每块 GPU 一个进程 | 整个数据集 |
| 命令 | `moraine COMMAND ...`、`moraine run FILE` | 同样的 zarr 数据集 | 整个数据集，从 shell 或 pipeline 文件运行 |

CLI 函数不是 API 函数的简单封装：它把数据切成 API 函数能处理的单元，作为执行器的任务来运行（CPU 上是线程，每块 GPU 一个进程；决策 0033、0034），
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

每层有哪些模块、哪一层可以调用哪一层，见开发者页面[架构](../dev/architecture.md)（英文）。

## 栅格与点云

处理链里流动的数据集有两种（完整约定见[数据约定](../contracts/data.md)，英文）：

- **雷达网格上的栅格**：形状 `(nlines, width[, n])`，方位向在前、距离向在后、堆栈维在最后。rslc 堆栈是
  `(nlines, width, nimages)` 的 complex64；`nan` 表示缺失数据。
- **选出像元（PS、DS）的点云**：形状 `(n_points, ...)` 的数组。点用网格索引 `gix` `(n_points, 2)`（方位，距离）
  或 hilbert 索引 `hix` `(n_points,)` 定位，点云**按 hilbert 顺序**存放，数组里相邻的点在地面上也相邻。`pc-union`、
  `pc-intersect`、`pc-diff` 和 `pc-select-data` 是排好序的索引上的集合运算；`ras2pc` 和 `pc2ras` 在两种数据之间搬运。

干涉数据是复数：像对 `(ref, sec)` 的干涉图是 `ref * conj(sec)`，相位连接后的相位历史和滤波后的干涉图幅度为 1，
点云的相干性压缩存放（相干矩阵的上三角，`(n_points, n_pairs)`），解缠相位是 float32 弧度 `(n_points, n_pairs)`。

## 磁盘上的数据

磁盘上的数据是 zarr 数组，而一个 zarr 数组是一个目录：`zarr.json` 记录形状、数据类型和分块，每个块存成一个文件，
用 zstd 压缩（决策 0039）。复制、移动或删除时把整个目录当作一个文件。样例数据集的 rslc 堆栈：

```text
raw/rslc.zarr/       complex64 (981, 4160, 92)，分块 (1000, 1000, 1)
├── zarr.json        形状、数据类型、分块、压缩方式
└── c/               460 个块文件：c/<方位向块>/<距离向块>/<影像>
    ├── 0/0/0        方位向 0-980，距离向 0-999，影像 0
    ├── 0/0/1        方位向 0-980，距离向 0-999，影像 1
    ├── ...
    └── 0/4/91       方位向 0-980，距离向 4000-4159，影像 91
```

`moraine info PATH` 不读数据，只打印形状、数据类型和分块。任何 zarr 读取工具都能打开它，读出来的是 numpy 数组；
只读需要的部分，不要读整个堆栈：

```python
import zarr

rslc = zarr.open('raw/rslc.zarr', mode='r')   # 只读元数据
block = rslc[:500, 1000:2000, 0]              # numpy 数组 (500, 1000) complex64
```

## 分块

zarr 数组按块存放，worker 并行处理各块；块的布局决定内存和速度。moraine 的约定（决策 0019）是：**空间上分块，
每景影像（或每个像对）一个块**，例如栅格堆栈 `(lines_block, width_block, 1)`、点云堆栈 `(points_block, 1)`。
按影像处理的步骤读整块，按空间块处理的步骤读每景影像的一块，都不需要重新分块。命令有 `chunks`（处理）和
`out_chunks`（存储）参数；块太小时间都花在调度上，太大则内存不够。栅格沿方位向而不是距离向切分。

## 命令怎样运行

```text
moraine run FILE            pipeline：按顺序运行各步，跳过没有变化的步骤
└── moraine COMMAND ...     命令：帮助和参数检查来自 docstring
    └── moraine.cli.<name>  打开输入，创建输出，列出任务
        └── executor        运行任务：CPU 上用线程，每块 GPU 一个进程
            └── task        zarr ──► 内存中 numpy / cupy 数组上的 API 函数 ──► zarr
```

一条命令从不把整个数据集放进内存。它在 `moraine.cli` 里的函数（Python 里是 `mc.<name>(...)`）把工作切成任务，
每个任务是 API 函数处理的一个单元：一块像元或点（`chunks`），或者一景影像、一个像对。任务读入自己的输入块，在内存中
对它们调用 API 函数，再写出自己的输出块，这些块不会被别的任务写。moraine 的执行器用自己的 worker 运行任务（决策
0033、0034）：CPU 上是命令所在进程的线程，numba 内核把每个任务分摊到各个核上；用 `cuda` 时是 `CUDA_VISIBLE_DEVICES`
里每块 GPU 一个进程，它在计算当前任务的同时读入下一个任务的块。`n_workers`、`threads_per_worker` 和 `processes`
决定同时运行多少个任务、在哪里运行（[命令行](../cli/index.md)）。所以一条命令需要的是同时运行的那些任务的内存，
加上少数命令在整个运行期间保留的小数组（帮助里给出每个点的字节数），而不是整个数据集的内存：内存不够时，把
`chunks` 调小，或者减少同时运行的任务。

能放进内存的数据（一小块裁剪、一次测试）直接用 API 函数更快：不用读写 zarr，也不用启动 worker。有些命令没有自己的
API 函数：载入 GAMMA 结果、金字塔，以及对内存中的数组一行 numpy 就能完成的数据操作（`math`、`pc-logic-ras`、
`pc-select-data`、`pc-concat` 等）。

## 处理链

示例 pipeline 实现了一条常见的处理链；每一步都是一条命令，都可以替换：

1. **载入** GAMMA 结果到 zarr：去平去地形的 rslc 堆栈、坐标、视向量、元数据（[01](../workflows/01_load.md)）。
2. **PS 候选点**：幅度离差指数，以及 Noise2Fringe 滤波后干涉图的时间相干性（[02](../workflows/02_ps.md)）。
3. **DS**：统计同质像元（KS 检验）、DS 候选点、相干矩阵、带自适应正则化的 EMI 相位连接，按加权时间相干性和
   相干像对的连通性筛选 DS（[03](../workflows/03_ds.md)）。
4. **合并与精化**：PS 和 DS 合在一起，用 Noise2Fringe Transformer 滤波，拟合每个点相对邻域的 DEM 误差后按时间相干性保留，并把这部分 DEM 误差从相位中去掉（[04](../workflows/04_refine.md)）。
5. **解缠**：点的 Delaunay 网络上的最小费用流，冗余像对网络整体（`emcf-pc`）或逐幅干涉图（`mcf-pc`）解缠，
   按相位闭合校正解缠误差，得到每景的解缠相位（[05](../workflows/05_unwrap.md)）。
6. **产品**：视线向位移（毫米），用 ERA5 改正对流层后相对参考区，以及每个点的 DEM 误差（[06](../workflows/06_products.md)）；速度随后加入（决策 0048、0051）。

## Pipeline、续跑与 JSON

一条处理链就是一个 TOML 文件，由 `moraine run FILE` 运行（[Pipeline 文件](../contracts/pipeline-file.md)，英文）。
参数相同且输入没变的步骤会被跳过，所以出错或改了参数之后再运行一次文件即可。`moraine run FILE --dry-run` 显示
将要运行什么，`moraine status FILE` 显示状态。加 `--json` 后每条命令在 stdout 打印一个 JSON 对象，日志走 stderr
（[JSON 输出](../contracts/json-output.md)）：脚本和 [AI agent](../agent/index.md) 读的就是它。

## GPU 还是 CPU

API 函数接受 numpy 或 cupy 数组，按类型分派（`moraine.api.utils_.get_array_module`）；CLI 函数和命令用 `cuda`
参数（每块 GPU 一个 worker 进程，从 cupy 的内存池分配）。只有 `CUDA_VISIBLE_DEVICES` 列出了 GPU 才会用它。影像很多
（超过约 100 景）时，相位连接在多核 CPU 上可能比单块 GPU 更快；指南里有数字。
