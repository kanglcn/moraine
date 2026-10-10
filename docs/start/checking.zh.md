# 检查结果

结果很大，绝不要把它整个加载进来看。moraine 提供三种不用写绘图脚本的检查方式，[工作流指南](../workflows/index.md)
写明每一步该期待什么。

## 金字塔

金字塔是栅格（`ras-pyramid`）或渲染到网格上的点云（`pc-pyramid`）的多分辨率副本，存放在结果旁边
（[约定](../contracts/pyramid.md)，英文）。示例 pipeline 为值得检查的结果建了金字塔，`moraine run` 还把每一步生成的
金字塔存成 PNG，放在 `WORK/.moraine/<文件名>/quicklook/`。

```bash
moraine ras-pyramid --ras ps/ras_adi.zarr --out_dir ps/ras_adi_pyramid
moraine pc-pyramid --pc pc/pc_ph.zarr --out_dir pc/pc_ph_pyramid --x pc/pc_e.zarr --y pc/pc_n.zarr --ras_resolution 20
```

## `moraine info`：形状、分块和统计量

```bash
moraine info ps/ras_adi.zarr ps/ras_adi_pyramid
```

```
  ps/ras_adi.zarr: float32 (981, 4160) chunks (1000, 1000)
  ps/ras_adi_pyramid: float32 (981, 4160) raster pyramid, 10 levels
    stats_level=0, nan_fraction=0.40655, min=0.0335447, max=4.39345, mean=0.537522, std=0.0957743, p01=0.255305, p50=0.541975, p99=0.786118
```

普通数组只读元数据。金字塔的统计量来自某个粗层级（`nan_fraction`、`min`、`max`、`mean`、`std`、`p01`、`p50`、
`p99`；复数数据给幅度，布尔数据给 `true_fraction`），`warnings` 指出异常：全是 nan、有无穷值或常数值。加 `--json`
后同样的数字是一个 JSON 对象（[约定](../contracts/json-output.md)）。

## `moraine quicklook`：一张图

```bash
moraine quicklook ps/ras_temp_coh_pyramid -o tcoh.png
moraine quicklook raw/rslc_pyramid --show intf_seq --index 5 -o intf_5.png
moraine quicklook raw/rslc_pyramid --show intf_all --index 0 91 --extent 1400,0,2600,600 -o zoom.png
```

`--show` 选择堆栈要画什么：`phase`（复数数据的默认值）、`intf_0`（与第一景的干涉图）、`intf_seq`（第 I 幅序列
干涉图）、`intf_all`（任意像对 `ref sec`）、`coh`、`coh_abs`。相位用 (-π, π] 上的循环色标，其他值用 1-99 % 范围内的
`viridis`。`--extent` 画场景的一部分（地图上是 `west,south,east,north` 度，雷达网格上是
`range_min,azimuth_min,range_max,azimuth_max` 像元）：范围越小用的层级越细，直到单个像元 / 点；标题给出范围和层级。
先看整幅场景，再放大看起来不对的地方。

看什么：条纹应当连续，噪声应当出现在相干性低的地方，选出的点应当覆盖稳定区域（城镇、岩石），解缠相位在相邻区域
之间不应有孤立的 2π 跳变。

## `moraine view`：notebook 里的交互地图

```bash
moraine view ps/ras_adi_pyramid pc/pc_ph_pyramid -o view.ipynb --show intf_seq --dates raw/meta.toml
```

生成一个 notebook，每个金字塔一张地图：缩放平移时从金字塔层级读取细节，滑块选择影像，点击一个像元或点绘出
它的时间序列，双击把它设为参考，地图上画的多边形会保存下来供 `polygon-mask` 使用。notebook 里只有路径，在存放
数据的机器上运行，所以任何 Jupyter 或 VS Code 连接都能用，不需要服务器或端口转发（决策 0018）。

同样的视图在 Python 里也有：`mc.view(data)` 接受金字塔、栅格数组或点数据（`x=`、`y=`）；`show=` 选择堆栈显示的
内容（`'intf_seq'` 等，或函数 `lambda v, ref, sec: ...`），`a * b` 叠加视图，`a + b` 并排显示，`repr(v)` 用文字描述
视图，`v.png('out.png', ...)` 保存图片。见 [Python 里的 moraine.cli](../api/cli.md)。

## 预期范围

每份工作流指南都有一张样例数据的数值和合理范围的表，例如 PS 候选点：幅度离差指数 `p50` 大约在 0.3 到 1 之间，
时间相干性在 0 和 1 之间，候选点占像元的百分之几到 30 % 左右。每次运行后把 `moraine info` 的结果和表比对，看过
quicklook 再继续；超出范围通常意味着参数或输入有误，而不是区域不同。
