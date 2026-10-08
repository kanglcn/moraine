# Python 里的 `moraine.cli`

`import moraine.cli as mc` 给出基于文件的函数：与[命令](../cli/index.md)相同的操作，在 Python 里以相同的参数调用
（`mc.amp_disp('raw/rslc.zarr', 'ps/ras_adi.zarr', cuda=True)`），外加查看器和几个不是命令的辅助函数。

```python
import moraine.cli as mc
logger = mc.get_logger()                       # 命令行打印的那些日志
mc.amp_disp('raw/rslc.zarr', 'ps/ras_adi.zarr', cuda=True)
mc.view('ps/ras_adi_pyramid', label='amplitude dispersion index', clim=(0, 1))
```

## view

::: moraine.cli.tiles.view
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

`view` 背后的地图部件是 `moraine.cli.TileView`（anywidget 加 Leaflet；瓦片由内核渲染，通过 notebook 通道发送，
所以不需要服务器或端口转发，决策 0018）。`moraine view PYRAMID ... -o view.ipynb` 生成一个由这种视图组成的 notebook。

## get_logger

::: moraine.cli.logging.get_logger
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## zarr 与 dask 辅助函数

`moraine.cli.dask_` 里是命令使用的并行 zarr 读写：`dask_from_zarr(path, chunks)` 和 `dask_from_zarr_overlap(...)`
把 zarr 数组变成 dask 数组（后者带 halo），`dask_to_zarr(darr, path, chunks)` 写回，`parallel_read_zarr` /
`parallel_write_zarr` 用线程读写整个数组，`ZarrDir` 遍历按块存放的 zarr 目录（`docs/contracts/data.md`）。并行读写
的错误会被抛出。
