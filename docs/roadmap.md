# Roadmap

Planned features that are not started yet, with the reason for them. When one is started it gets a branch
and, if it is a design choice, a record in `docs/decisions/`; remove it here when it is merged.

## Detection of decorrelated images

Unwrapping (`emcf-pc`, `mcf-pc`) fails where images are decorrelated, e.g. by snow; such images can be
left out of the image pairs, but the user has to find them. On the sample data (2026-09-29) the interferograms
using 2021-10-25 and the images from 2022-09-12 to 2022-10-24 differed in 35 - 70 % of the points between
EMCF settings, even 14 day interferograms, against 1 - 15 % for the other images; leaving out 2021-10-25
reduced the triangles of images that do not close from 13 % to 5 % (refined points, constant spatial cost).

Idea: a quality score per image from the data before unwrapping, e.g. how often the wrapped phase
difference between neighbouring points exceeds a threshold in the short interferograms using the image, or
the phase linking quality around its date; report it (`moraine info`, a pyramid of per image scores) so that
users and agents choose the images to leave out of the image pairs, and possibly lower the weight of
interferograms using bad images instead of leaving the images out. Validate on `tests/unwrap_benchmark.py`
with images made decorrelated on purpose.

## Vegetated areas: short temporal baselines and intermittent points (Xinpu)

Not started; recorded 2026-10-04 so that it can be picked up later. Starting point: the Xinpu tutorial
(`nbs/Tutorials/CLI/Xinpu/`, Sentinel-1 ascending track 84, 60 dates, 12 days, Three Gorges Reservoir
landslide complex, vegetated) selects almost no DS and poor PS with the parameters of the sample data.

Measured on the Xinpu results of the tutorial (5130 x 1225 pixels, 60 images, window 11 x 11):

- PS: ADI p50 0.55, only 39 349 points with ADI < 0.4 (0.6 %); the n2f temporal coherence (sequential
  pairs) has p50 0.42, p90 0.59, p99 0.77, so `> 0.58` selects 699 003 points (11 %), mostly noise.
- DS: 3 394 016 candidates (>= 50 SHP, 54 % of the pixels), 98.8 % of them with EMI quality < 1 (coherence
  matrix not positive definite), temporal coherence p50 0.016, 19 321 refined DS.
- Mean |coherence| of the DS candidates by time lag (12 day steps): 1: 0.32, 2: 0.28, 3 (36 days): 0.26,
  5: 0.24, 10 and more: 0.21 (a floor). Only pairs of up to about 36 days carry signal.

Why: a 60 x 60 coherence matrix is estimated from 50 - 121 samples (rank deficient, ill conditioned), and
the long pairs are noise. The SHP test compares amplitude distributions only, so decorrelated vegetation
passes it. Loosening the refinement thresholds only lets noise in.

What others do (checked 2026-10-04, abstracts and search results only, not the full texts):

- Zheng et al. (2023, Geomatics, Natural Hazards & Risk, doi 10.1080/19475705.2023.2289835) on the same
  landslide: 112 interferograms from the 60 scenes with a temporal baseline <= 36 days and a spatial
  baseline <= 150 m; intermittent multi-temporal InSAR assisted by ICA: intermittently coherent points are
  connected in a Delaunay network, the deformation of every arc comes from spatial integration, ICA gives
  the reliable deformation signals. The details are in their supporting information, not read.
- ISBAS (intermittent SBAS, Sowter et al.): pixels that are coherent only in part of the interferograms are
  kept, every pixel is inverted with its own valid pairs; coverage 39 - 99 % of the land pixels against
  4 - 12 % for SBAS. Threshold and inversion details not read.
- Sequential phase linking (mini-stacks, compressed SLCs; e.g. arXiv 2502.09248): lowers the cost of long
  stacks. With coherence at the noise floor beyond 36 days, compressed SLCs would not link the mini-stacks.

How time segments connect: not through the phase of one point (a point may be coherent in one summer only)
but through space (in one interferogram the valid points are connected by a triangulation and unwrapped
across the arcs) and through the network of interferograms (a connected network of short pairs and one
spatial reference for all of them give the image phases by inversion). A time span in which all pairs are
decorrelated (snow) breaks the network.

What moraine has and lacks: `n2f` / `temp-coh` take `image_pairs` (a bandwidth 3 network can be used now);
`emcf-pc` unwraps any network of image pairs (decision 0020) and `unwrap-correct-closure-pc` corrects by
loops. Missing: phase linking limited to short pairs (`emi` uses the whole matrix); a validity mask per
interferogram (the points are one fixed set with a phase in every image); the spatial reference (see below).

Steps, each its own change, in this order:

1. No code: PS selection with a bandwidth 3 network (`image-pairs --bandwidth 3`, `n2f`, `temp-coh`,
   n2f threshold from 0.7); count the points that stay coherent within 36 days.
2. DS phase linking on short pairs (mini-stacks of 10 - 15 images, or a coherence matrix limited to a
   bandwidth); validate on Xinpu and CampiFlegrei.
3. Intermittent points: a validity mask of points per interferogram through selection, unwrapping, closure
   correction and inversion, with the spatial reference. Largest change, needs a decision record.

## Next stages of decision 0012

Deformation products (stage 5), starting with the spatial reference (reference point or area) of the
image phases of `unwrap-correct-closure-pc`: see `docs/decisions/0012-network-emcf-inversion.md`. Stage 4
(time series inversion) is replaced by those image phases (decision 0022).

## Small issues

- The generated help shows `PATH` as placeholder for every list of strings, also for dates
  (`emcf-pc --dates`).
