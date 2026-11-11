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
| [pc](pc.md) | 点云：索引（网格 / hilbert）、排序、集合运算、栅格与点云互转、包围盒查询的 hilbert R 树 | `pc_hix`、`pc_sort`、`pc_union`、`ras2pc`、`pc2ras`、`HilbertRtree` |
| [tnet](tnet.md) | 时间网络：像对 | `TempNet` |
| [polygon](polygon.md) | 多边形 GeoJSON 文件，点在多边形内 | `read_polygons`、`polygons_contain` |
| [slc](slc.md) | SLC 幅度、幅度定标、干涉图、多视 | `rslc2amp`、`calamp`、`intf` |
| [ps](ps.md) | PS 筛选用的幅度离差指数 | `amp_disp` |
| [ds](ds.md) | 分布式散射体：统计同质像元（KS 检验）、相干矩阵、正则化、相位连接（EMI）、DS 时间相干性 | `select_shp`、`emperical_co_pc`、`emi`、`ds_temp_coh`、`emperical_co_emi_temp_coh_pc` |
| [dl](dl.md) | 深度学习滤波与模型下载 | `n2f`、`n2fs3d`、`n2ft`、`download_dl_model` |
| [pqm](pqm.md) | 像元质量：时间相干性 | `temp_coh` |
| [unwrap](unwrap.md) | Delaunay 三角网、最小费用流、EMCF、相位闭合校正、GAMMA `mcf_pt` | `delaunay`、`mcf_pc`、`emcf_pc`、`unwrap_correct_closure_pc` |
| [utils](utils.md) | numba 装饰器、GPU 检测、numpy / cupy 分派（`moraine.api.utils_`，未重新导出） | `is_cuda_available`、`get_array_module` |

每页列出每个公开函数的签名和 docstring（形状、类型、单位、默认值；英文），以及在合成数据上的小示例。示例在构建
本手册时真实运行，所以它们和文档对应的版本一定是匹配的。所有函数共用的数据约定见[数据约定](../contracts/data.md)（英文）。
