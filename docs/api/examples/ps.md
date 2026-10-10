```python exec="true" source="above" result="text" session="ps"
import numpy as np
import moraine as mr

rng = np.random.default_rng(0)
nimages = 20
# a stable scatterer (constant amplitude, small noise) next to noisy pixels
stable = 5 + 0.2 * rng.standard_normal((3, 3, nimages))
noisy = np.abs(rng.standard_normal((3, 3, nimages))) + 0.1
amp = np.concatenate([stable, noisy], axis=1)
rslc = (amp * np.exp(1j * rng.uniform(-np.pi, np.pi, amp.shape))).astype(np.complex64)   # (nlines, width, nimages)

adi = mr.amp_disp(rslc)                       # amplitude dispersion index (std / mean over time), (nlines, width)
print(adi.shape, adi.dtype)
print('stable pixels', adi[:, :3].mean().round(3), ' noisy pixels', adi[:, 3:].mean().round(3))
```

PS candidates are the pixels with a low index, e.g. `adi < 0.4` (`pc-logic-ras` in the pipelines).
