```python exec="true" source="above" result="text" session="calamp"
import numpy as np
import moraine as mr

rng = np.random.default_rng(0)
rslc = (rng.standard_normal((4, 5, 3)) + 1j * rng.standard_normal((4, 5, 3))).astype(np.complex64)   # (nlines, width, nimages)
amp = mr.rslc2amp(rslc)                       # amplitude stack, float32
print(amp.shape, amp.dtype, np.allclose(amp, np.abs(rslc)))
cal = mr.calamp(np.ascontiguousarray(amp[..., 0]))   # calibrated amplitude of one image (a contiguous 2D array)
print(cal.shape, cal.dtype)
```
