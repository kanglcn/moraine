# 0028 DS are selected by the connectivity of the coherent image pairs, not by the number of effective pairs

## Status

Accepted

## Date

2026-10-06

## Context

Decision 0024 asks to select by the weighted temporal coherence only together with the effective number of
image pairs `eff_n_pairs`, because a high weighted value that rests on few pairs is not reliable. But the
minimum of `eff_n_pairs` has no meaning of its own and scales with the number of image pairs P = N (N - 1) / 2
(about 0.45 P gave the same quality as the plain temporal coherence >= 0.8 on the 17 images of the sample
data and on 92 images of Campi Flegrei, 2026-10-06). What makes a weighted temporal coherence unreliable is
that the coherent pairs do not link all images: the phase between two groups of images that no coherent pair
connects is not constrained by the data, whatever the weighted value is.

The graph with the images as nodes and the image pairs with a positive weight as edges is useless as it is:
a pair of pure noise has |coh|^2 > 1/n_looks with probability 0.37, so the graph of every point, noise
included, is connected (sample 99.6 %, Campi Flegrei 100 %).

## Decision

- `ds_temp_coh(coh, ph, image_pairs=None, block_size=128, n_looks=None, alpha=1e-3)` returns with `n_looks`
  also `n_components`, int16, shape (n_points,): the number of connected components of the graph of the
  images whose edges are the image pairs with |coh_k|^2 > 1 - p^(1/(n_looks-1)). For |coh| of a pair without
  coherence the probability of |coh|^2 > x is (1-x)^(n_looks-1), so an incoherent pair is an edge with
  probability p. p is chosen so that the random graph on N images with edge probability p is connected with
  probability `alpha` (Gilbert's recursion for the probability that G(N, p) is connected): a point without
  any coherent pair has `n_components` 1 with probability `alpha`, whatever N is. 1e-6 <= alpha <= 0.5
  (below, the recursion loses its digits). Points with n_looks <= 1 get N.
- `t_coh`, `t_coh_w` and `eff_n_pairs` are unchanged (`t_coh` bit for bit); `eff_n_pairs` stays an output
  but is not used for the selection. The weighted API function `emperical_co_emi_temp_coh_pc` returns
  `n_components` as fifth output; the commands `ds-temp-coh` and `emperical-co-emi-temp-coh-pc` get the output
  `n_components` / `n_components_dir` and the input `alpha`.
- The examples select the DS with `n_components == 1` and `t_coh_w >= 0.9` (`alpha` 1e-3).
- Limits of the GPU kernel: `block_size` x nimages at most 393216 (4 bytes of shared memory per image and
  warp, 48 KB per block; 3072 images at the default block size); the CPU version has no limit.

## Consequences

- Breaking: the weighted outputs of `ds_temp_coh` and `emperical_co_emi_temp_coh_pc` have one more element;
  code that unpacks them fails.
- Measured (A100, 100 000 points x 4186 image pairs, random coherence, 92 images): `ds_temp_coh` without
  `n_looks` 3.9 ms before and 4.3 ms now, with `n_looks` 4.2 ms before and 15.2 ms now; the phase linking step
  of the sample (733 000 points) takes 21 s with and without the change. Per point 2 more bytes of output.
- On the sample data (17 images, 732 727 candidates) 70.1 % of the candidates are connected; connected and
  `t_coh_w >= 0.9` keeps 58.4 % with a mean cosine of the phase difference of the sequential interferograms
  to reference points 11-20 pixels away of 0.739 and 9.7 % of the points below 0.5, the plain
  `t_coh >= 0.8` 53.2 %, 0.753 and 6.9 %: a little looser. On an 800 x 800 block of Campi Flegrei (92 images)
  74.0 % with 0.927 against 69.7 % with 0.933.
- Pure noise simulations (17 and 92 images, 4000 trials) give the connected fraction `alpha` within the
  sampling error although the edges of the speckle noise are not independent (tests check 17 images).
- `alpha` only decides how a noise point is treated, not the quality of partly coherent points: for these the
  weighted temporal coherence decides, with its empirical threshold.

## Do not

- Do not use the pairs with a positive weight as edges: every point is connected then.
- Do not select by the weighted temporal coherence without `n_components` (this replaces the last rule of
  decision 0024, which asked for `eff_n_pairs`).
- Do not make the minimum of `eff_n_pairs` a selection parameter again: it depends on the number of images.
