# 0019 Data are chunked by blocks in space and one image per chunk

## Status

Accepted

## Date

2026-09-30

## Context

moraine processes stacks larger than memory in steps that need different units (`docs/development.md`):
per image or image pair (e.g. spatial unwrapping of one interferogram) and per block of points or pixels
with its whole time series (e.g. phase linking, the temporal step of EMCF). The data conventions
(`docs/contracts/data.md`) fixed shapes and orders but not chunks, so a command could not know whether a
column of a stack is cheap to read; a stack chunked with all images in one chunk is read entirely for
every image.

## Decision

- Stacks are chunked by blocks in space and with one image (or image pair) per chunk along the image
  axis: `(points_block, 1)` for point clouds, `(lines_block, width_block, 1)` for rasters. Arrays without
  an image axis are chunked by blocks in space only.
- Temporary zarrs between the steps of a command follow the same layout.
- Commands expect this layout and report other chunks; they do not rechunk.
- The rule is part of the data conventions contract (`docs/contracts/data.md`, section Chunks).

## Consequences

- Steps per image and steps per block both read and write whole chunks.
- Commands that write another layout today are fixed when they are next changed.

## Do not

- Do not add rechunking for other layouts inside commands.
- Do not write stacks with several images per chunk.
