The filters need PyTorch and the trained models (`moraine.download_dl_model()`), so this example is not run when
the manual is built.

```python
import numpy as np
import moraine as mr

mr.download_dl_model()                                   # once; the .pth files go to moraine/dl_model/

# rasters: Noise2Fringe on an interferogram stack (nlines, width, n_pairs), chunked with halos (depths)
intf_filtered = mr.n2f(intf, chunks=(256, 256), depths=(32, 32))

# with the amplitude dispersion index as a second input
intf_filtered = mr.n2fs3d(adi, intf, chunks=(256, 256), depths=(32, 32))

# point clouds: the Noise2Fringe Transformer on (n_points, n_pairs) with the coordinates of the points
intf_filtered = mr.n2ft(lon, lat, intf, chunks=20000, k=128, cuda=True)
```

The outputs have unit amplitude; the temporal coherence between the filtered and the raw interferograms
(`temp_coh`) is the quality of a pixel or point.
