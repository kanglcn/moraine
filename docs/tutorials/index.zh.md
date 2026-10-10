# 教程

仓库里 `nbs/Tutorials/CLI/CampiFlegrei/` 的五个 notebook：与[工作流](../workflows/index.md)相同的步骤，写成
`moraine.cli` 的 Python 调用，每个结果都有交互视图（`mc.view`），数据是 Campi Flegrei（Sentinel-1 降轨 track 22，
2019-2021 共 92 景，那不勒斯附近正在隆升的火山口：城市，PS 和 DS 都多；数据的制作过程见 `data/CampiFlegrei/README.md`）。
notebook 的内容是英文的。

| notebook | 做什么 |
|---|---|
| [01 载入数据](CampiFlegrei/01_load.md) | GAMMA 结果转 zarr，坐标，元数据 |
| [02 PS 处理](CampiFlegrei/02_ps.md) | 幅度离差、Noise2Fringe 时间相干性、PS 候选点 |
| [03 DS 处理](CampiFlegrei/03_ds.md) | SHP、DS 候选点、相干矩阵、相位连接、DS 筛选 |
| [04 像元精化](CampiFlegrei/04_refine.md) | 合并 PS 和 DS，Noise2Fringe Transformer，时间相干性 |
| [05 相位解缠](CampiFlegrei/05_unwrap.md) | EMCF 解缠与相位闭合校正 |

!!! info "只有文字和代码"
    notebook 提交时不带输出，所以这些页面只有文字和代码，没有图。在 Jupyter 或 VS Code 里运行 notebook 可以看到
    交互地图，或者看[工作流指南](../workflows/index.md)和 [agent 会话](../agent/index.md)里真实运行的图。每页都链接到
    GitHub 上的 notebook。

按顺序运行；每个 notebook 读取前面各步在它自己文件夹里的结果。
