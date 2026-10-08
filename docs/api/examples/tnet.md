```python exec="true" source="above" result="text" session="tnet"
import moraine as mr

tnet = mr.TempNet.from_bandwidth(6, bandwidth=2)   # 6 images, every image paired with the next two
print(tnet.image_pairs.shape, tnet.image_pairs.dtype)
print(tnet.image_pairs.tolist())
print(mr.nimage_from_npair(15))                    # 15 pairs of all images: 6 images
```

The image pairs file of the `image-pairs` command has the same two columns (reference, secondary); the
interferogram of a pair is `ref * conj(sec)`.
