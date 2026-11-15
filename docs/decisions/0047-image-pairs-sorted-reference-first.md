# 0047 Image pairs are sorted without duplicates, the reference index smaller than the secondary

## Status

Accepted

## Date

2026-11-15

## Context

Image pairs `(n_pairs, 2)`, the reference and secondary image index of each pair, give the interferograms and, in
their order, the columns of compressed coherence, interferograms and unwrapped phase. The data conventions did not
fix their order and direction, and the readers of image pairs did not agree:

- `moraine image-pairs` writes the pairs of `numpy.triu_indices`: sorted by reference and then secondary index,
  every reference index smaller than its secondary index.
- The viewer (`moraine.cli.view`, `show='coh'`) finds the column of the pair of its sliders by a binary search in
  the codes `ref * n_images + sec` of the image pair file, which needs them sorted. With the file (0, 2), (0, 1),
  (1, 2) the sliders (0, 1) showed the coherence of (0, 2) and (0, 2) that of (1, 2), without an error; a pair
  written (2, 0) was shown as missing.
- `TempNet` requires sorted pairs without duplicates, the reference index smaller. `emcf_pc` takes the pairs in
  their order and direction (decision 0020), and so does `emperical_co_pc`; `emcf_pc` and
  `unwrap_correct_closure_pc` reject a pair of an image with itself.
- The commands took any image pairs of shape (n, 2).

A pair of an image with itself is no interferogram (phase 0, coherence 1); a duplicate pair is the same
interferogram twice.

## Decision

- Image pairs are sorted by reference and then secondary index without duplicates, and every reference index is
  smaller than its secondary index (`docs/contracts/data.md`).
- `moraine.check_image_pairs` checks it; the error names the first pair that breaks it. The image pairs moraine
  reads are checked: the `image_pairs` arguments of the commands and pipeline files (text files, `.npy` files and
  lists), the image pairs of the viewer, `TempNet` (`check_if_valid`, which also checks that the pairs link all the
  images) and `TempNet.load` (`ds-temp-coh --tnet`).
- The API functions keep the image pairs they take: `emcf_pc` and `emperical_co_pc` take any order and direction.
  Finding a pair by its images (the viewer, `TempNet.image_pairs_idx`) relies on the order.

## Consequences

- The viewer shows the pair of its sliders for every image pair file it accepts.
- Image pair files in another order, with duplicate pairs or with the secondary image first, which the commands
  took before, are rejected. Results whose columns follow such a file are computed again with sorted pairs to be
  viewed. The files of `moraine image-pairs`, and so the example pipelines, are unchanged.
- The commands `emcf-pc` and `emperical-co-pc` take sorted pairs only; their API functions keep the other orders
  and directions (decision 0020).

## Do not

- Do not sort the image pairs of a command or of the viewer silently: the columns of the data follow the order of
  the file.
- Do not make a connected network part of the conventions: a network may leave out images (decision 0020);
  `TempNet` checks the links of its images for its own use.
