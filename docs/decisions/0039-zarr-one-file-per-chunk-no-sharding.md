# 0039 Zarr arrays are written one file per chunk, without sharding

## Status

Accepted

## Date

2026-10-15

## Context

zarr v3 can store a grid of chunks in one file (a shard, with an index of the byte ranges of its chunks). A run of
the example pipelines on Campi Flegrei (92 images, 981 x 4160 pixels) writes 14 GB in 40 019 files, 80 % of them
pyramid tiles, so sharding was considered to speed up writing and reading. Measured on the Lustre file system of the
development cluster with zarr 3.4.1 (2026-10-15), for the access patterns of moraine:

| (600 000, 92) complex64, chunks (200 000, 1) | one file per chunk | shards (200 000, 92) |
|---|---|---|
| files | 277 | 4 |
| write per block (phase linking, n2ft) | 0.6-0.9 s | 1.0-1.1 s |
| read per block, warm / cold | 0.4-0.7 / 0.5-0.6 s | 0.7-1.4 / 1.5-2.2 s |
| read per image | 0.7-0.8 s | 1.0-1.2 s |
| write per image (loading, unwrapping, point cloud shuffles) | 1.4-1.7 s | 19-21 s |

A pyramid level (981 x 4160 x 92 float32, 256 x 256 tiles) with one shard per image has 93 files instead of 6 257;
it is written in 7.8-8.5 s instead of 5.6-7.7 s, and 300 tile reads of the viewer take 1.1-1.4 s instead of 0.5 s
(4 s instead of 0.75-1.1 s cold).

- The chunks of a block are read by parallel threads from separate files; a shard is read through its index and then
  byte ranges of one file.
- Writing part of a shard reads and rewrites the whole shard, and threads writing different chunks of one shard at
  the same time corrupt it (a checksum error at the next read). The parallel writes of moraine (one thread per chunk)
  are safe only without shards.
- A small file read cold from the servers costs 15-20 ms on this file system, the only case where fewer files could
  pay; the cold reads measured above were still slower with shards.
- zarr 3.4.1 writes chunks about 2 times faster than 3.1 to 3.4.0 in its own code (RAM disk: 0.5-0.7 s instead of
  1.1-1.7 s for the block writes of the table, reads unchanged); on Lustre the gain is within the run-to-run variation
  (the Campi Flegrei pipelines 02-05: 125 s instead of 132 s of steps). zarr 3.2 and later need Python 3.12.
- Bigger pyramid tiles (512 / 1024) cut the files 3.8 / 13.6 times and slow the tile reads 1.2-1.7 / 2.6 times. zstd,
  the default compressor of zarr, saves 8 % on phases and 45-50 % on rslc stacks with sea and on rasters, at
  450-600 MB/s per thread.

## Decision

- moraine writes zarr arrays without shards, one file per chunk, with the default compressor of zarr (zstd). The
  chunks follow decisions 0019 and 0032; pyramid tiles stay 256 x 256.
- moraine requires zarr >= 3.4.1 on Python >= 3.12 (any zarr on Python 3.11).
- The rule is part of the data conventions contract (`docs/contracts/data.md`, section Chunks).

## Consequences

- A run writes one file per chunk: tens of thousands of files for a scene like Campi Flegrei, mostly pyramid tiles,
  growing with the area and the number of images.
- To revisit when the number of files becomes the limit (file count quotas, object stores charging per request,
  much larger areas): shard the pyramids first (one shard per image of a level). A shard must then be written whole by
  one task, so its shape follows the task of the writer, and the parallel writes must split a region by shard instead
  of by chunk.

## Do not

- Do not create arrays with `shards` while their writes are split by chunk in threads: concurrent writes into one
  shard corrupt it.
- Do not shard a stack across images when it is written image by image: every image written rewrites every shard.
