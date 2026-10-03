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

## Next stages of decision 0012

Deformation products (stage 5), starting with the spatial reference (reference point or area) of the
image phases of `unwrap-correct-closure-pc`: see `docs/decisions/0012-network-emcf-inversion.md`. Stage 4
(time series inversion) is replaced by those image phases (decision 0022).

## Small issues

- The generated help shows `PATH` as placeholder for every list of strings, also for dates
  (`emcf-pc --dates`).
