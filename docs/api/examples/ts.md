```python exec="true" source="above" result="text" session="ts"
import numpy as np
import moraine as mr

n_points, nimages = 6, 4
offset = np.array([0.0, 0.5, -0.3, 1.2], dtype=np.float32)     # unknown value of every image, common to all points
motion = np.zeros((n_points, nimages), dtype=np.float32)
motion[4:] = np.arange(nimages) * 0.4                            # the last two points move, the first four are stable
ts = motion + offset                                             # e.g. image phases of the closure correction
ts[3, 2] += 2 * np.pi                                            # one stable point with an unwrapping error

stable = np.zeros(n_points, dtype=bool); stable[:4] = True      # the reference area
out, ref_value = mr.pc_reference(ts, stable)                     # median of the reference points per image
print(ref_value)
print(out.round(2))
out_mean = mr.pc_reference(ts, stable, method='mean')[0]         # the unwrapping error shifts the mean
print(out_mean[:, 2].round(2))
out_point = mr.pc_reference(ts, 0)[0]                            # one reference point
print(out_point[0])
```

The median of the reference area removes the value common to all points in every image; one point with an
unwrapping error in the area does not change it, while it shifts the mean of image 2.

```python exec="true" source="above" result="text" session="ts"
disp = mr.pc_disp(out, wavelength=0.0554658)                    # Sentinel-1: 1 rad = 4.41 mm
print(disp.dtype)
print(disp.round(2))
```

The referenced phases in millimeters along the line of sight: the moving points come towards the satellite by
1.77 mm per image (0.4 rad); a whole cycle of unwrapping error is half a wavelength, 27.7 mm.
