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

## zarr 辅助函数与 worker

`moraine.cli.zarr_` 里是命令使用的 zarr 读写：`parallel_read_zarr(z, slices)` 和 `parallel_write_zarr(data, z, slices)`
用线程按块读写 zarr 数组的一部分（读写的错误会被抛出），`ZarrDir` 把按块存放的 zarr 目录当作一个数组
（`docs/contracts/data.md`）。`moraine.cli.executor` 运行一条命令的任务（决策 0033、0034）：`Executor` 启动 worker
（CPU 上是本进程的线程，`cuda` 时每块 GPU 一个进程），`chunk_task` 读取任务的 `Chunk` 输入、调用 API 函数并写出结果。
