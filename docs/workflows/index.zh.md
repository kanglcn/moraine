# 工作流

moraine 的处理链是五个经过验证的 pipeline 文件（决策 0009、0033）：每个文件改动前都在样例数据集上完整跑一遍，
指南里的数字和图都来自那次运行。样例数据集是 Campi Flegrei：Sentinel-1 降轨 track 22，2019-01-02 到 2021-12-29
共 92 景，全分辨率 981 x 4160 像元，那不勒斯西边正在隆升的火山口（仓库里的 `data/CampiFlegrei/README.md` 说明
GAMMA 结果是怎样做出来的）。文件的 `[vars]` 默认值就是它的值；其他数据用 `--var` 给出你的值。五个文件共用一个
工作目录、互相读取输出，所以要按顺序运行：

```bash
moraine run examples/01_load.toml --workdir WORK --var gamma=/path/to/gamma --var reference=YYYYMMDD --var geo=YYYYMMDD
moraine run examples/02_ps.toml --workdir WORK --var shape=NLINES,WIDTH
moraine run examples/03_ds.toml --workdir WORK --var shape=NLINES,WIDTH
moraine run examples/04_refine.toml --workdir WORK
moraine run examples/05_unwrap.toml --workdir WORK --var shape=NLINES,WIDTH --var range_pixel_spacing=... --var azimuth_pixel_spacing=...
```

| pipeline | 指南 | 做什么 | 输出 |
|---|---|---|---|
| `examples/01_load.toml` | [01 载入 GAMMA 结果](01_load.md) | GAMMA 结果转 zarr，坐标，web mercator | `WORK/raw/` |
| `examples/02_ps.toml` | [02 PS 候选点](02_ps.md) | 幅度离差 + Noise2Fringe 时间相干性 | `WORK/ps/` |
| `examples/03_ds.toml` | [03 DS 处理](03_ds.md) | SHP、DS 候选点、相干性、相位连接、DS 筛选 | `WORK/ds/` |
| `examples/04_refine.toml` | [04 合并与精化](04_refine.md) | PS + DS，Noise2Fringe Transformer，时间相干性 | `WORK/pc/` |
| `examples/05_unwrap.toml` | [05 相位解缠](05_unwrap.md) | Delaunay 网络上的最小费用流 | `WORK/unw/` |

要改参数，把文件复制到工作目录再改副本，或者用 `--var name=value` 覆盖 `[vars]` 里的值。再次运行时，参数和输入都没变
的步骤会被跳过，改动下游的步骤重跑；`moraine run FILE --dry-run` 显示计划，`moraine status FILE` 显示状态。失败的
步骤会打印错误和日志；修好原因后再运行一次文件。文件格式见 [Pipeline 文件约定](../contracts/pipeline-file.md)（英文）；
`moraine run FILE --json` 在 stdout 打印一个 JSON 对象，含计划、各步记录和输出摘要（[JSON 输出](../contracts/json-output.md)）。

五份指南目前是英文的。每份都列出步骤、要调的参数、样例数据集上的预期结果和那次运行的图，以及数字不对时怎么办。
指南就是 [AI agent](../agent/index.md) 读的文件；[那次会话记录](../agent/index.md)在样例数据集上运行的正是这五个文件。
