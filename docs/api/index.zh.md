# Python API

`import moraine as mr` 给出数组函数：作用于内存中 numpy 数组（CPU，numba）或 cupy 数组（GPU）的算法，按主题
分在 `moraine.api` 的各模块里（决策 0015），并以 `mr.<name>` 重新导出。大多数函数两种数组都接受，返回同类数组。

```python
import numpy as np
import moraine as mr
import moraine.api.utils_

ph = mr.emi(coh)                          # numpy 进，numpy 出
if moraine.api.utils_.is_cuda_available():
    import cupy as cp
    ph_gpu = mr.emi(cp.asarray(coh))      # cupy 进，cupy 出
```

一次调用处理一个单元：整幅场景的一景影像或一个像对，或一块像元 / 点及其完整时间序列（需要邻域时再加一圈
halo）。数据大于内存时，[命令行](../cli/index.md)用 dask 把这些函数映射到 zarr 的各块上。

| 模块 | 主题 | 主要名字 |
|---|---|---|
| [calamp](calamp.md) | SLC 幅度与幅度定标 | `rslc2amp`、`calamp` |
| [pc](pc.md) | 点云索引（网格 / hilbert）、排序、集合运算、栅格与点云互转 | `pc_hix`、`pc_sort`、`pc_union`、`ras2pc`、`pc2ras` |
| [rtree](rtree.md) | 点云的 hilbert R 树，包围盒查询 | `HilbertRtree` |
| [tnet](tnet.md) | 时间网络：像对 | `TempNet` |
| [polygon](polygon.md) | 多边形 GeoJSON 文件，点在多边形内 | `read_polygons`、`polygons_contain` |
| [ps](ps.md) | PS 筛选用的幅度离差指数 | `amp_disp` |
| [shp](shp.md) | 统计同质像元（KS 检验） | `ks_test`、`select_shp` |
| [co](co.md) | 干涉图、相干矩阵、正则化 | `emperical_co_pc`、`uncompress_coh`、`nearestPD` |
| [pl](pl.md) | 相位连接（EMI）与 DS 时间相干性 | `emi`、`ds_temp_coh`、`emperical_co_emi_temp_coh_pc` |
| [dl](dl.md) | 深度学习滤波与模型下载 | `n2f`、`n2fs3d`、`n2ft`、`download_dl_model` |
| [pqm](pqm.md) | 像元质量：时间相干性 | `temp_coh` |
| [unwrap](unwrap.md) | Delaunay 三角网、最小费用流、EMCF、相位闭合校正、GAMMA `mcf_pt` | `delaunay`、`mcf_pc`、`emcf_pc`、`unwrap_correct_closure_pc` |
| [utils](utils.md) | numba 装饰器、GPU 检测、numpy / cupy 分派（`moraine.api.utils_`，未重新导出） | `is_cuda_available`、`get_array_module` |

每页列出每个公开函数的签名和 docstring（形状、类型、单位、默认值；英文），以及在合成数据上的小示例。示例在构建
本手册时真实运行，所以它们和文档对应的版本一定是匹配的。所有函数共用的数据约定见[数据约定](../contracts/data.md)（英文）。
