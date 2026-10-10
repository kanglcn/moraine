# Release notes

<!-- do not remove -->

## Unreleased

The 04 example merges the DS before the PS candidates (`pc-union` keeps the data of its first input for a point in both): a point that is a DS and a PS candidate now goes into the refinement with its phase linked history instead of the raw phase of its pixel. On the sample data 29 % of the DS are PS candidates too; with the raw phase 10 % of them passed the n2ft refinement, with the linked phase 91 % (Campi Flegrei: 23 % of the DS, 1 % -> 99.9 %), so the refined points are 477 440 instead of 352 482 (Campi Flegrei 241 064 instead of 223 847) with the same share of good points among the others. The tutorials of Campi Flegrei and Xinpu merge in the same order

The 03 example selects the DS with a weighted temporal coherence of at least 0.85 instead of 0.9 (`t_coh_w_min`): measured against the n2ft refinement, the quality of the DS rises with the weighted temporal coherence all the way to 1 (sample data: 69 % of the points between 0.85 and 0.9 pass the refinement, 83 % between 0.9 and 0.95, 87 % above), so the lower threshold adds points of a quality the refinement sorts out; the refined points of the sample data are 549 341 instead of 477 440. `alpha` of `shp-test` stays 0.05: a larger value loses points without a cleaner phase history (0.2: 82 % instead of 88 % pass the refinement). With about 90 images the SHP window should be larger: the Campi Flegrei tutorial uses `az_half_win = 3`, `r_half_win = 13` (7 x 27 pixels, about 100 m x 60 m on the ground) with at least 80 SHPs, which doubles the DS at the same quality (167 007 instead of 77 295 with `t_coh_w_min` 0.85)

`moraine.cli.view(..., terrain=True)` shows web mercator layers in 3D over the terrain in a notebook (decision 0036): the layers and the satellite base map are draped over the terrain of public elevation tiles (AWS Terrain Tiles, heights of about 30 m up to zoom level 15, fetched by the browser like the base map, no key), the right mouse button (or ctrl + drag) tilts and rotates the view, which zooms two levels further than the 2D map, the cursor shows the height of the terrain with the values, and the clicked point, the reference and the polygons are shown on the terrain (polygons are drawn on 2D maps). Another service of Terrarium encoded tiles can be given as a URL template; beyond its last zoom level the terrain of a tile is interpolated from its parent tile, so that the meshes and the textures draped over them follow the zoom of the view. The base map of the 3D views is Esri's satellite imagery: its tiles may be read by the GPU, which CARTO and OpenStreetMap refuse. `moraine view --terrain` writes notebooks of such maps. The 3D view loads deck.gl from the jsdelivr CDN, for 3D views only; the 2D maps are unchanged and `.png` stays the 2D image

The 04 example keeps, for the refined points, their longitude, latitude and temporal coherence (`pc/pc_lon`, `pc_lat`, `pc_temp_coh`) and takes their height, look vector and slant range from the rasters of 01 (`pc/pc_hgt`, `pc_theta`, `pc_phi`, `pc_range`, new step `geometry`): the inputs of the deformation products

## 0.10.0

`ras-pyramid` and `pc-pyramid` have a `method`: the levels of a pyramid are the mean of the blocks of the data (`method = "mean"`: nan left out, complex data averaged as complex numbers, integer and boolean data as float32 levels) or, as before and by default, every 2nd pixel (`decimate`) / the first point of each block (`first`). The mean is a multilook, so the whole scene of a mean pyramid shows the fringes of a phase history or of filtered interferograms and the smooth values of coherences, counts and unwrapped phases, where a decimated one shows speckle. It is right for real values and for phases relative to a reference image, not for the SLC pixels of an rslc stack (the scatterer phases of neighbouring pixels differ), so the rslc pyramid stays decimated while the other pyramids of the example pipelines are made with `method = "mean"`, and the 02 example builds a mean pyramid of the n2f interferograms to check the fringes of the whole scene (decision 0035). The marker of a pyramid records the `method`, `moraine info` reports it. Bug fixed on the way: the cells without points of the pyramid of an integer point cloud (e.g. `n_components`) held 0 instead of nan and were drawn as the value 0; the cells of integer and boolean point clouds are float32 now, nan where empty

`ras-pyramid` reads the raster in bands of `rows` lines (a power of 2, default 1024) instead of a whole channel per process, so that its memory is bounded for any raster size: a process needs about 1 GB plus 3 x rows x width x itemsize bytes (an 8000 x 8000 complex64 raster: 1.4 GB with the default, 2.9 GB for a whole channel before). The pyramid is the same bit for bit

`ras-pyramid` and `pc-pyramid` compute the statistics of all the data while they build the pyramid and store them in it (the marker of `0.zarr`, and per image or channel in `stats.zarr`): `moraine info` and the colour range of `moraine.cli.view` are exact (nan fraction, minimum, maximum, mean and standard deviation of all the values, percentiles of a regular sample of 8 million values) and instant instead of being computed from a coarse level at every call (0.3 s for the Campi Flegrei rslc pyramid); `info` also names the images of a stack that are all nan or constant. The statistics of pyramids made before still come from a coarse level
The probes of the maps of `moraine.cli.view` (the values under the cursor, the clicked pixel or point) give what is drawn at the zoom shown: for rasters the value of the cell of the pyramid level on screen (before: the full resolution pixel under the cursor, which is not what the colour shows at coarse levels), for point clouds the point of that cell and its value (before: the nearest point within 4 screen pixels, found with a bounding box tree built from all coordinates at the first probe; the tree is now used only where the points are drawn one by one)

`pc-pyramid` saves the bounding box tree of the points (`rtree.zarr`) and the views read it: the first zoom to single points, or the first probe of a point, no longer builds the tree from all coordinates (0.2 s per million points, several seconds more when the numba cache is cold). Pyramids made before get the tree built as before
The maps of `moraine.cli.view` take the width of the notebook (at most 700 pixels high) with the aspect of the scene instead of a fixed 900 x 700 frame, `view(..., size=(width, height))` sets their size, and the lower right corner of a map can be dragged to resize it. Zooming is continuous: the scene fills the map at the start (before, the zoom was a power of 2 and the scene took 50 - 100 % of the frame)

Changing the image of a stack with a slider of `moraine.cli.view` no longer blanks the map until the new tiles arrive: the new tiles are drawn over the old ones, which stay until all are in; and only the layers that depend on the moved slider are redrawn (an amplitude background under a stack, or a point cloud without that slider, is not requested again)
The tiles of `moraine.cli.view` are sent to the browser as 8 bit palette PNGs (255 colours and a transparent index) instead of RGBA PNGs: a 256 x 256 tile of an interferogram is 66 instead of 206 kB and is encoded in 1.5 instead of 8.9 ms (Campi Flegrei rslc pyramid, one of about 15 tiles on screen), so panning and changing the image are faster over a slow connection. The colours have 255 instead of 256 levels (a change of at most one level); `render` of a layer gives the palette indices (uint8) instead of RGBA, `colorize` is unchanged. nan values stay transparent when the colour limits are equal
Wrong arguments are reported with the argument, the expected shape and the actual one (`rslc must have 3 dimensions (nlines, width, nimages), got shape (4, 5)`, `give one path or a list of paths for all of ras, pc: ras is a list, pc is a path`, `idx must be sorted and unique`, ...) instead of a bare `AssertionError`; the checks run with `python -O` too, which drops asserts. The commands check their inputs before they create outputs. Spelling in messages and help texts fixed (`dimentation`, `hypothetic test`, `implented`, `Homogenious`)

`bool2gix` wrote the grid index as a bool array (every index 1): it writes the integer index now; the command has a test

The help of a list argument that is not a list of paths shows `STR` instead of `PATH` (`emcf-pc --dates STR [STR ...]`)

`n2f` runs its network in half precision (float16) on the GPU (the command with `cuda`, the API with a cupy input), weights and activations in the channels last layout of the tensor core kernels; the output is normalized again in float32, so the amplitude stays 1. An A100 filters a (1000, 1000) processing chunk in 12 instead of 21 ms, the activations take half the memory; the `n2f` command on Campi Flegrei (981 x 4160 pixels, 91 image pairs) takes 18 instead of 22 s from the start of the workers to the end (the rest reads the rslc and writes the interferograms), the `n2f` step of the 02 pipeline 18 instead of 21 s on Campi Flegrei and 9.3 instead of 9.6 s on the sample data, where the step is mostly its fixed costs. With the same input the phases differ from the TF32 convolutions used before by 4e-4 rad for the typical pixel, by 3e-3 rad (Campi Flegrei) to 8e-3 rad (sample data) for 99 % of the pixels and up to 0.3 rad (3 rad on the sample data) at single pixels of noise; the TF32 model was itself off strict float32 by 1.6e-3 and 3.4e-3 rad for 99 % of the pixels. The PS candidates of the 02 pipeline change by about 0.1 %: on Campi Flegrei 587 of 530 000 candidates of the temporal coherence are lost and 574 gained, the same order as between two runs of the same code (526 and 537), which give a random phase to the pixels without data. The CPU keeps float32, `n2fs3d` keeps float32 (TF32 on the GPU)

The worker processes of the commands (the GPU workers, and the CPU workers with `processes`) read the chunks of the next task while a task runs: the main process sends one task more than a worker runs at a time, and the worker reads its inputs before handing it to the task threads. A worker holds the inputs of one task more. The steps bound by reading gain the overlap: `shp-test` on Campi Flegrei 11.1 instead of 12.6 s, `temp-coh` on the GPU 11.1 instead of 12.0 s; steps bound by computing (the phase linking) or by reading alone (`amp-disp`) do not change; identical results. The GPU workers keep one task at a time: numba loads a kernel wrongly when two threads launch it for the first time at once

A number given to a command as text that is not one, e.g. an empty `--var NAME=` substituted into a pipeline step, is reported as such when the arguments are read (`mcf-pc: argument range_pixel_spacing = '' is not a number`) instead of failing inside the command with `could not convert string to float`

`temp-coh` (command and API) checks that the interferograms and the rslc have the same points or pixels and that the image pairs fit them, and names the likely cause (`intf` and `rslc` swapped); before, swapped inputs indexed outside the arrays and the GPU reported an illegal address

`load-gamma-flatten-rslc` failed with a numba typing error since the commands run their tasks through the executor: the numba decorator had moved from the flattening loop to the task that reads the GAMMA files and writes the zarr

The point cloud commands (`ras2pc`, `ras2pc-ras-chunk`, `pc2ras`, `pc-concat`, `pc-sort`, `pc-union`, `pc-intersect`, `pc-diff`, `pc-select-data`, `data-reduce`) run 4 threads by default instead of 1: their tasks are a few zarr chunk operations per channel, each a round trip to zarr's event loop of about 5 ms, and the threads overlap them. Campi Flegrei commands: `pc-union` 4 instead of 17 s, `pc-select-data` 3 instead of 11 s, `ras2pc` 4 instead of 15 s, `pc-concat` 5 instead of 13 s (wall time, the old numbers with a cold file cache); the same results. `threads_per_worker` still sets it

`shp-test` selects the SHPs in the same pass as the test: its outputs are now `is_shp` and `shp_num` (with `alpha`, default 0.05), what `select-shp` computes from the p values, and the p values are an optional output `pvalue` after the window sizes in the argument order (`shp_test(rslc, is_shp, shp_num, az_half_win, r_half_win, alpha=0.05, pvalue=None, ...)`). The 03 example no longer writes the p values (2 GB for the sample data) nor runs `select-shp`, which stays for p values kept with `pvalue`. `shp_test` + `select_shp` of the 03 pipeline: 13 instead of 31 s on the sample data, 20 instead of 29 s on Campi Flegrei (A100); the same SHPs

The window arrays of the SHP test (`shp-test` p values, `select-shp` SHP flags, both `(nlines, width, az_win, r_win)`, and the point windows of `ras2pc-ras-chunk`, `(n_points, az_win, r_win)`) are chunked with the whole window of a pixel in one chunk instead of one window cell per chunk (decision 0032): every block was written and read through 121 chunk operations for an 11 x 11 window. Campi Flegrei (981 x 4160 pixels, 92 images): `select-shp` 8.5 instead of 19 s, `ras2pc-ras-chunk` of the SHP flags 2.2 instead of 4.6 s; sample data (2500 x 1834, 17 images): 12 instead of 10 s and 3.4 instead of 9.3 s (a block of p values is now one 484 MB chunk, decompressed by one thread). The values are the same, arrays with the old chunks are still read, and `pc-concat` keeps the layout of its input

The example pipelines run `amp-disp` and `temp-coh` on the CPU (`cuda = false` in their steps): the commands read the stack once and reduce it, and starting a GPU worker costs more than the GPU saves (Campi Flegrei, 981 x 4160 x 92: the `amp-disp` command takes 7 instead of 15 s, `temp-coh` 9 instead of 16 s; the results differ by 1e-6). The GPU versions stay

rmm is no longer used (decision 0034): the GPU workers allocate through cupy's own memory pool and torch through its own caching allocator, nothing is reserved; the commands lose `rmm_pool_size` (pipeline files that set it fail with an unknown argument; the `rmm_pool_size` of `n2ft` described below is gone with it). On Campi Flegrei (A100) every step takes the same time with and without the rmm pool, and `n2ft --compile` tunes its kernels in every case. (`n2f` without a pool had torch allocating through rmm's plain cudaMalloc / cudaFree, 3 times slower: 72 instead of 23 s.) A worker process of a command that dies without reporting (killed, a crash in a kernel, a script without the `if __name__ == '__main__'` guard that `spawn` needs) is an error naming the worker and its exit code; before, the command waited for it without end

dask is no longer a dependency (decision 0034): the commands run their tasks with moraine's own workers, threads of the command's process (CPU) or processes started with `spawn`, one per GPU of `CUDA_VISIBLE_DEVICES` (`cuda`), allocating from cupy's memory pool. dask-cuda is no longer needed for the GPU; `psutil` is a dependency of its own. Breaking: the commands lose `**dask_cluster_arg`, so pipeline files with dask options in `[step.kw]` (e.g. `memory_limit`) fail with "unknown worker arguments"; `moraine.cli.dask_from_zarr`, `dask_from_zarr_overlap` and `dask_to_zarr` are removed and the zarr helpers (`parallel_read_zarr`, `parallel_write_zarr`, `ZarrDir`) live in `moraine.cli.zarr_`. A GPU command starts its workers in 2 to 3 s instead of 4 to 7 s and stops them at once; the pipelines 02 to 05 of Campi Flegrei on one A100 take 155 instead of 178 s (`amp-disp` 6 instead of 11 s, `shp-test` 16 instead of 20, the fused DS step 21 instead of 25, `pc-union` 4 instead of 7)

The commands run their work through `moraine.cli.executor` (decision 0033): a task is a plain function on chunks of zarr arrays (`Chunk`, `chunk_task`), run by an `Executor` in threads, processes or one process per GPU, instead of a dask array graph built by hand in every command. The results are the same (checked bit for bit on a crop of the sample data and on Campi Flegrei); the progress is logged at every tenth of the tasks instead of a progress bar, and the steps are faster without the graph (Campi Flegrei on one A100: 03_ds 109 -> 71 s, phase linking 51 -> 25 s). Output chunks are the chunk size given, no longer clipped to the array shape (e.g. (1000, 1000) instead of (981, 1000) for a raster of 981 lines). `math` writes the dtype of its expression (before: always float64, which dropped the imaginary part of complex results); `temp-coh` accepts an integer chunk size for point clouds (it failed before). `moraine.cli.dask_from_zarr`, `dask_from_zarr_overlap` and `dask_to_zarr` are no longer used by the commands

`pc-pyramid` and `ras-pyramid` render the channels in 8 forked processes (`n_workers`) instead of a dask cluster of one worker with 2 threads: the work per channel is Python code that threads cannot share, and the cluster took longer to start than the rendering. Campi Flegrei on an A100 node: the pyramid of 223 890 points x 92 images takes 6 instead of 25 s, of the 981 x 4160 x 92 rslc 8 instead of 28 s, of one 981 x 4160 raster 4 instead of 9 s; the pyramids are identical. The arguments `processes`, `threads_per_worker` and the dask cluster arguments of the two commands are gone: remove them from pipeline files that set them

The batch norm layers of the loaded `n2f` and `n2fs3d` models are folded into their convolutions (`fold_batch_norms` of the UNet): the model runs 17 % faster on an A100 (981 x 4160 pixels: 86 instead of 104 ms per interferogram); the results change by rounding (float32 up to 3e-5, typically 2e-4 rad with the TF32 convolutions of the GPU)

`n2f` (command) filters one output chunk of the raster with all image pairs per task instead of one image pair of the whole raster: the rslc of a processing chunk is read once for all pairs, the output chunk is written once, and the filtered interferograms stay on the GPU until the chunk is done. On 981 x 4160 Campi Flegrei pixels with 91 image pairs the computing takes 19 instead of 29 s on an A100 (command 31 instead of 39 s); the results are the same up to rounding and the random phase given to pixels without data. The processing chunks (`chunks`, with `depths` of overlap) are cut at the output chunks (`out_chunks`); a task holds the rslc of one processing chunk for all images and the filtered output chunk (8 bytes per pixel and image / image pair), on the GPU with `cuda`

`n2ft --compile` failed in the GPU workers when torch.compile tuned its kernels for the first time on a machine (`CUDAPluggableAllocator does not yet support getDeviceStats`): torch allocated from the rmm pool, which has no memory statistics. The command no longer makes an rmm pool by default (`rmm_pool_size=None`, the model allocates with torch); with a pool, the compiled model runs without the kernel tuning (about 2 times slower)

A model call of `n2ft` (API and command) filters as many interferograms of a processing chunk as a fifth of the GPU memory holds (about 8 kB per point and interferogram; 200 000 point-interferograms on the CPU as before) instead of 200 000 point-interferograms: more work per kernel. On an 80 GB A100 the eager model takes 0.24 instead of 0.35 µs per point and interferogram; the `n2ft` step of the 04 pipeline takes 29 instead of 37 s on Campi Flegrei (590 667 points, 91 pairs) and 17 instead of 19 s on the sample data; the phases change by rounding (at most 1.8e-4 rad)

The batch norm layers of the loaded `n2ft` model are replaced by the per channel affine function they compute in evaluation mode (`fold_batch_norms`): the eager model runs 8 % faster on an A100 (200 000 Campi Flegrei points x 91 pairs: 7.0 instead of 7.6 s), and the compiled model no longer compiles again (34 s each time) when the batch of interferograms crosses a size threshold of the cuDNN batch norm kernels; the phases change by rounding (at most 1.8e-4 rad)

The neighbour searches of the `n2ft` structure (16 and 3 nearest neighbours at the 4 sampling levels of a processing chunk) run in 8 threads instead of one per core: the structure of a 23 000-point chunk takes 70 instead of 190 ms on a 128-core machine (the queries 21 instead of 120 ms). It runs in parallel to the model, so the filtering only gets faster where the model is faster than the structure (`compile`, or a slow CPU)

`n2ft` with `compile` compiled the model a second time (14 s) when the interferograms of a processing chunk did not divide into its batches and the last batch was a single interferogram (e.g. 91 interferograms in batches of 9): the batches of a chunk are now as equal as possible and never a single interferogram, and the model sees the same tensor layout for every batch size, so the model is compiled once per process; on 590 667 Campi Flegrei points with 91 image pairs the compiled model takes 2.7 instead of 8.2 s per 200 000 points after its compilation

The halo search of `n2ft` (command and API: the k nearest neighbours of the points of every processing chunk) queries one KD-tree of all points in 32 threads instead of building a tree per chunk in joblib processes: 0.5 instead of 7.5 s for the 590 667 Campi Flegrei points (30 chunks) in the command, and 6 instead of 11 s for 10 million points; the main process needs 16 x `k` x `chunks` bytes per thread (41 MB by default). The `n2ft` step of the 04 pipeline takes 37 instead of 42 s on Campi Flegrei and 19 instead of 29 s on the sample data (A100). The halos are now exactly the nearest neighbours: before, the halo of a chunk was estimated from the k-th neighbour distances of its border points and could miss some (sample data: 15 of the 1.2 million halo points, in 6 of 50 chunks), and since the filter sees the whole chunk, the phases of those chunks change (p99 0.1 rad, refined points 352 433 instead of 352 434); on Campi Flegrei nothing changes. joblib is no longer a dependency

`n2ft` (API and command) filters several interferograms of the same points in one model call (up to 200 000 point-interferograms, about 1.6 GB of GPU memory), the command filters all image pairs of a processing chunk at once instead of one pair at a time, the sampling and neighbour structure and the phasors of the next two processing chunks are prepared in threads while the model filters the current one, and the phasors are normalized on the GPU: on 590 667 Campi Flegrei points with 91 image pairs the `n2ft` step takes 42 instead of 69 s on an A100 (sample data: 29 instead of 38 s); the phases differ by rounding (at most 1.4e-4 rad) and the refined points are the same. The command has a `compile` option like the API: the model is compiled with torch.compile in every worker (15-40 s once per worker, less when torch has cached it) and then runs about 3 times faster; by default (`compile` not given) it is compiled when points times image pairs is at least 1e8, which is where the compilation pays off (not for Campi Flegrei with 5.4e7, where the compiled command is slower); the command logs the choice

`emperical_co_emi_temp_coh_pc` (API and command) processes as many points at once as 1 GiB of coherence matrices hold instead of 1000: thousands of small GPU operations per batch made the command slow for few images. On the sample data (17 images, 4.1 M candidates) the command takes 41 instead of 75 s on an A100 and 28 instead of 42 s on the CPU; with 92 images the batch is about 32 000 points and a point takes 25 instead of 28 µs on the GPU. A task holds up to about 2 GiB (the coherence matrices and, on the GPU, the samples of their estimation; CPU peak 4.7 instead of 2.9 GB with the default 2 threads on the sample); `batch_size` still sets the number of points

The GPU kernels are cached on disk like the CPU functions (decision 0029): the first GPU `emi` or `emperical-co-emi-temp-coh-pc` of a process (every pipeline step, every dask worker) no longer compiles the EMI kernel for 15 s, it loads it in about 0.5 s; the other GPU kernels save 0.3-0.9 s each

The GPU `emi` chooses its block size (128, 256 or 512 threads) by the number of images: the size whose blocks fit the most times into a multiprocessor. With many images only one block fits and a wider block divides the work of a step among more warps: on an A100 per point 13 instead of 14 µs for 128 images (128 CPU cores 24 µs), 26 instead of 36 µs for 150 images (CPU 26 µs), 48 instead of 67 µs for 196 images (CPU 54 µs); up to about 100 images nothing changes

The GPU `emi` gave a wrong phase history (errors up to π: the second eigenvector) for rare points: when a pivot of the Sturm sequence of its eigenvalue search rounded to exactly 0, it was counted as non-negative but used as a negative pivot (2 of the 4.1 M DS candidates of the sample data). The pivot is now replaced before it is counted, as in LAPACK

The GPU kernel of `emi` is 2.1-2.5 times faster: on an A100 4.7 instead of 10.2 µs per point for 92 images (128 CPU cores: 11 µs), 2.1 instead of 5.1 µs for 60 images, 14 instead of 31 µs for 128 images (CPU 24 µs), 36 instead of 79 µs for 150 images (CPU 26 µs), 67 instead of 141 µs for 196 images (CPU 54 µs). The phases differ from before by float32 rounding (max phase difference to the float64 EMI on 2000 Campi Flegrei DS candidates, 92 images: p99 2.5e-4 instead of 3.3e-4 rad, max 1.1e-3 instead of 2.7e-3; CPU 2.7e-4 / 1.1e-3). The loops are controlled with 32-bit integers, the loops over the triangle of the matrices have a uniform trip count, the inverse is swept four pivots at a time, the matrix-vector products use 2 / 4 threads per row for small trailing blocks, and the kernel is compiled with at most 72 registers. The fused command `emperical-co-emi-temp-coh-pc` takes 30 instead of 36 µs per point on 20 000 Campi Flegrei candidates (92 images, batch 1000)

`select_shp` (command `select-shp`) selected the wrong pixels: it kept the pixels whose KS p value is below `p_max`, i.e. those whose amplitude distribution differs significantly from the centre pixel, so the DS candidates were mostly bright or dark pixels among different neighbours and their coherence was dominated by a few bright pixels (patchy phase in low coherence interferograms). The SHPs are now the pixels whose p value is at least the significance level, and the argument `p_max` is renamed `alpha` (default 0.05; a larger value keeps fewer SHPs). Pipeline files and scripts with `p_max` must change it to `alpha`. The KS p value of identical samples (the centre pixel itself) is now 1 instead of 0. Every result from `select-shp` on needs to be recomputed

`emi` (also in the commands `emi` and `emperical-co-emi-temp-coh-pc`) with the default `regularize=True` runs in a numba.cuda kernel on the GPU instead of cupy (decision 0031): on 20 000 Campi Flegrei DS candidates (92 images) on an A100 10 instead of 43 µs per point, and no GPU memory beyond the input and output instead of 445 kB per point (8.5 GiB for 20 000 points); the fused command takes 39 instead of 72 µs per point (batch 1000) and 35 instead of 68 µs with batches of 10 000 points, which need 2.0 instead of 6.0 GiB. The phases are closer to the CPU (max difference per point p99 3e-4 instead of 2e-3 rad); on the sample data the selected DS do not change. With 30 or fewer images the speed is about the same as before (10 images: 0.5 instead of 0.2 µs). With many images the GPU slows down faster than the CPU (128 images 31 µs, 196 images 132 µs per point; 128 CPU cores: 24, 54): see the 03 guide. `regularize=False` and more than 198 images (A100) still use cupy

With numba-cuda, GPU kernels failed to compile when the python of a conda environment was run without activating it (libnvvm not found: e.g. scripts, batch jobs or the pre-commit hook calling the python by its path); moraine now sets `CUDA_HOME` to that environment when neither `CUDA_HOME`, `CUDA_PATH` nor `CONDA_PREFIX` is set

`numba-cuda` (NVIDIA) is a GPU dependency, installed with conda like cupy (README): the GPU kernels are numba.cuda kernels (decision 0030), and it starts them about 4 times faster than the deprecated CUDA target built into numba (110 instead of 450 us per launch)

No cupy ElementwiseKernel is left; the GPU kernels are numba.cuda kernels or cupy matrix products. `temp_coh` (one warp per point) takes 1.8 instead of 7.9 ms for 981 x 1000 points and 91 image pairs on an A100, `amp_disp` (one warp per pixel) 1.7 instead of 2.2 ms for 981 x 1000 x 92, `emperical_co` 9 instead of 64 ms for 100 x 100 x 92; the small per pixel kernels of `n2f` / `n2fs3d` and `ad_intf_pc` take 0.1-1 ms more per call, the launch overhead of numba.cuda. The results are the same, or differ by float32 rounding; `emperical_co` ignores `block_size`

Compiled numba functions are cached on disk, in a directory named after a hash of the moraine sources (`~/.cache/moraine/numba/`, decision 0029): the first call in a process takes 0.4 instead of 11 s for the fused DS step and 0.2 instead of 5 s for the SHP test, once a version of the code has run. The functions that used numba's own cache (unwrapping, chunking, amplitude) could keep old machine code after a change of another file; they use the new cache too

`ks_test` (command `shp-test`) looks the p value up in a table of the n + 1 possible KS statistics and uses the symmetry of the test: 2.2 times faster on the CPU (300 x 1000 pixels, 92 images, 11 x 11 window: 158 instead of 344 ms), the same p values bit for bit; on the GPU a numba.cuda kernel replaces the cupy kernels (43 instead of 53 ms) and the results are now the same as on the CPU (they differed by up to 2e-7). With `return_dist` the CPU left `dist` uninitialized where the other pixel is nan; it is nan there now

`emperical_co_pc` (also in `emperical-co-pc` and `emperical-co-emi-temp-coh-pc`) computes the coherence of all image pairs of a point from one matrix product of its SHP samples: on 20 000 Campi Flegrei DS candidates (92 images, 4186 pairs) 2.0 instead of 37 µs per point on an A100 and 2.8 instead of 13 µs on 128 CPU cores; the fused command takes 73 instead of 258 µs per point on the GPU and 30 instead of 45 µs on the CPU. The GPU uses up to 1 GiB more memory. The coherence changes by float32 rounding (at most 2e-6), the DS selected on that sample do not change; `block_size` has no effect any more

`n2ft` (command) filters one output chunk of points with all image pairs per task instead of all points per image pair: the sampling and neighbour structure of every processing chunk is computed once instead of once per image pair, and a worker holds the rslc of its chunk instead of whole images. 531 k points (Campi Flegrei PS candidates) on one A100: 62 s instead of 438 s for 91 image pairs, worker peak memory 1.4 instead of 2.8 GB; on the CPU 289 s instead of 351 s for 10 image pairs. 10 million points, 2 image pairs: 125 s instead of 200 s, worker 1.4 instead of 2.3 GB, main process 1.1 instead of 1.2 GB. The results are identical; the processing chunks no longer cross the output chunks, and when `chunks` divides `out_chunks` (as in the examples) they are the same as before

Errors of the parallel zarr reads and writes of the commands (e.g. a corrupted chunk, a read-only output) are raised; before, a failed read returned uninitialized values and a failed write was ignored

`pc_union`, `pc_intersect` and `pc_diff` (commands `pc-union`, `pc-intersect`, `pc-diff`, `pc-select-data`) check that the indices are sorted and without duplicates, as `docs/contracts/data.md` requires: the check let unsorted indices through (`pc_union` then merged them wrongly) and rejected point clouds of zero or one point

`temp_coh` on the CPU no longer normalizes the arrays passed to it (`intf`, `rslc`) in place; the results are the same

`n2f` and `n2fs3d` no longer change the interferogram passed to them (pixels with an amplitude below 1e-30, no data in GAMMA files, were set to NaN in place); the results are the same

`examples/03_ds.toml` selects the DS by the connectivity of the coherent image pairs (`n_components == 1`) and the weighted temporal coherence (`t_coh_w_min`, 0.9) instead of the plain temporal coherence >= 0.8 (decision 0028, `docs/workflows/03_ds.md`): `ds_temp_coh(..., n_looks=...)` also returns `n_components`, the number of connected components of the graph of the images whose edges are the coherent image pairs (`alpha`, default 1e-3, is the probability that a point without any coherent pair has all images connected); the commands `ds-temp-coh` and `emperical-co-emi-temp-coh-pc` get the output `n_components` / `n_components_dir` and the input `alpha`. Breaking: the weighted outputs of `ds_temp_coh` and `emperical_co_emi_temp_coh_pc` have one more element. The weighted value is higher than the plain one, so 0.8 would keep too many points; on the sample data the selection is a little looser than the plain 0.8. The tutorials follow in a separate change.

`emi` on the CPU (also inside `emperical-co-emi-temp-coh-pc`) was about 2·10⁴ times slower than it should be on many-core machines: the multithreaded BLAS (MKL, OpenBLAS) started its own threads in every call from the numba threads (128 x 128 threads on 128 cores; 187 ms instead of 7.6 µs per point for 60 images). Its threads are now limited to one while the kernel runs (new dependency `threadpoolctl`). The kernel also computes only the smallest eigenpair (LAPACK `cheevr`) instead of all: 1.5-1.8 times faster, 3.4-5 µs per point for 60 images, 8-12 µs for 92 images on 128 cores; the phase histories change only by float32 rounding

Faster CPU kernels of decisions 0024 and 0025: `ds_temp_coh` 1.6-3 times (real arithmetic; 100 000 points x 4186 image pairs: 46 instead of 86 ms), the effective number of looks of the SHP sets 30-95 times (pair counts per lag by popcount; identical results), the speckle correlation of a 1000 x 1000 chunk 6 times (local power normalization in numba instead of scipy, 9 instead of 57 ms)

The DS tutorials (`nbs/Tutorials/CLI/*/03_ds.ipynb`, `nbs/Tutorials/DS_Processing.ipynb`) and `nbs/Introduction/software_architecture.ipynb` use `emi` without the EMI quality and select DS by temporal coherence, as `examples/03_ds.toml`; `Xinpu/03_ds.ipynb` also shows the effective number of looks of the SHPs (`emperical-co-pc --n_looks_dir`) and compares the temporal coherence with the weighted one and the effective number of image pairs (`ds-temp-coh --n_looks`, and the same from the fused command)

`emi` returns the phase history only and `emperical_co_emi_temp_coh_pc` returns `(ph, t_coh)` (with `weighted=True` also `t_coh_w`, `eff_n_pairs`): the EMI quality is no longer an output, and the commands lose `emi_quality` (`emi`) and `emi_quality_dir` (`emperical-co-emi-temp-coh-pc`). With the regularization it is not comparable between points; select DS by temporal coherence. `ph, quality = emi(coh)` becomes `ph = emi(coh)` (`emi(coh)[0]` still runs, but gives the phase history of the first point) (decision 0027)

Weighted DS temporal coherence: `ds_temp_coh(coh, ph, n_looks=...)` (command `ds-temp-coh` with the input `n_looks` and the outputs `t_coh_w`, `eff_n_pairs`) also returns the temporal coherence with the image pairs weighted by their squared coherence without the noise bias, so that incoherent pairs (long time spans in vegetation) do not lower it, and the effective number of image pairs it rests on. `n_looks` is the effective number of independent looks of the SHP set of each point, from the positions of the SHPs and the speckle correlation measured in the rslc (neighbouring pixels are correlated: with Sentinel-1 IW 121 SHPs are about 45 independent looks): `emperical_co_pc(..., return_n_looks=True)` (command `emperical-co-pc --n_looks_dir`). The fused `emperical_co_emi_temp_coh_pc(..., weighted=True)` (command: `t_coh_w_dir`, `eff_n_pairs_dir`) gives the same results. On Xinpu the points with temporal coherence < 0.6 but a weighted one >= 0.8 agree with their neighbours (spatial consistency 0.5 against 0.15 for a weighted one < 0.6). CPU numba, GPU numba.cuda: 5 ms for 100 000 points x 4186 image pairs on an A100 (decisions 0024, 0025, 0027)

`emi` and `emperical_co_emi_temp_coh_pc` (commands `emi`, `emperical-co-emi-temp-coh-pc`) regularize the coherence matrices that are not positive definite (`regularize`, default True). EMI needs a positive definite coherence magnitude matrix, which most DS candidates lack when the number of images approaches the number of independent looks of the SHPs: their phase history was noise (Campi Flegrei, 92 images, 11 x 11 window: 97 % of 1.79 million candidates). Such points get the coherence matrix (1 - beta) coh + beta I with the smallest beta their data need (median 0.27 there); points with a well conditioned positive definite matrix do not change. On Campi Flegrei 1 187 702 candidates reach a temporal coherence of 0.8 instead of 49 089, with continuous fringes of the caldera uplift. `examples/03_ds.toml` selects DS by temporal coherence alone; `regularize = false` gives the old results (decisions 0023, 0026)

`ds_temp_coh` on the GPU is 5 times faster (numba.cuda kernel with one warp per point; 100 000 points x 4186 image pairs: 3.9 ms instead of 22 ms on an A100); the results differ only by float32 rounding

`load-gamma-flatten-rslc` has the argument `gamma_threads` (default: the number of CPU cores, at most 64): the number of threads of each `phase_sim_orb` run (GAMMA uses 8 without it); on 128 cores one image takes 1.5 s instead of 10 s with the same result

The CLI tutorials (`nbs/Tutorials/CLI/`) use two sample data sets, one folder each with the same five notebooks (`01_load` ... `05_unwrap`): `Xinpu/` (landslide, Three Gorges, Sentinel-1 ascending) and `CampiFlegrei/` (caldera uplift, Sentinel-1 descending); the data are in `data/` and read from the GAMMA results

`moraine quicklook --extent` and `view(...).png(..., extent=...)` draw a part of the scene, from the finest pyramid level that fits, down to the data (degrees on web mercator maps, pixels on the radar grid); the title gives the extent and the level, `repr` of a view the extent and the finest cell of every layer, so that agents can zoom into what they look at

Data conventions (`docs/contracts/data.md`): stacks are chunked by blocks in space and one image (or image pair) per chunk; commands expect this layout (decision 0019)

The API modules moved from `moraine/` to `moraine/api/`, phase unwrapping to `moraine/api/unwrap/` (`mcf.py`, `emcf.py`, `gamma.py`, `delaunay_.py`; `moraine/pu.py` is split). `import moraine` still exports the same names (`moraine.mcf_pc`, ...); code importing modules directly changes, e.g. `from moraine.pu import mcf_pc` becomes `from moraine.api.unwrap import mcf_pc`. The downloaded deep learning models stay in `moraine/dl_model/`

Deep learning models run on PyTorch instead of ONNX Runtime; torch is an optional dependency (`pip install moraine[dl]`) and the models are downloaded as `.pth` files by `download_dl_model()`

`n2ft` results are reproducible (fixed farthest point sampling start)

Development moved from nbdev notebooks to plain python: packaging in `pyproject.toml`, numpy style docstrings, pytest tests in `tests/`; the nbdev documentation site is removed

`moraine ... --json` output has `version` and `ok` fields; pyramids record their format version in `0.zarr`; pipeline files may declare `[pipeline] version`. The formats are specified in `docs/contracts/`

New `emcf_pc`: extended minimum cost flow (EMCF) unwrapping of the interferograms of any network of image pairs (`image_pairs`, by default every image with the next three); the loops of the network are used against unwrapping errors; `temporal_cost` (constant or time span) and `spatial_cost` (constant, edge length, point `weight`; both constant by default) choose where corrections go; by default as many interferograms are unwrapped at the same time as the available cores and half of the available memory allow. On a synthetic benchmark with known truth (`tests/unwrap_benchmark.py`, every image with the next three): median 0.83 % wrong (point, interferogram) against 1.54 % for `mcf_pc`. Decisions 0012 (stage 3) and 0020. The `emcf-pc` command takes the image pairs (e.g. from `image-pairs --bandwidth 3`), the pixel spacings and optionally the dates instead of `meta.toml`, needs `ph` chunked one image per chunk and processes the points in blocks and the interferograms one by one (10 million points, 100 images: 8.2 GB peak memory with 4 interferograms at the same time); `misclosure`, the `pairs` output, `t_scale`, `bperp_scale`, `repair` and `exclude` are removed

New `unwrap_correct_closure_pc`: correction of unwrapping errors by phase closure per region, as in MintPy (Yunjun et al. 2019): where the unwrapped interferograms of a loop of image pairs do not add up to zero, the interferograms of every region (points connected without long edges, e.g. an island) are corrected by whole cycles; points of small regions one by one. Works after any unwrapper; returns the unwrapped phase of every image `ts` (n_points, nimages) relative to the reference image `ref` (default 0), whose differences close every loop, the per point `misclosure_fraction` (before the correction), `change_fraction` (interferograms changed by the correction, a point quality for masking; errors that close every loop are not seen) and the regions. After `emcf_pc` on the benchmark: median 0.81 % wrong. Decisions 0021 and 0022 (no separate time series inversion). Command `unwrap-correct-closure-pc` (per block of points, per interferogram, per block of points; optional outputs `misclosure_fraction`, `change_fraction` and `region`; 10 million points, 100 images, 294 interferograms: 145 s, 3.3 GB)

One viewer, `moraine.cli.view(data, ...)`, replaces the holoviews plots: interactive maps in Jupyter / VS Code notebooks (no server or port forwarding) of pyramids, rasters in memory and point data; `show=` a name ('phase', 'intf_seq', 'intf_all', 'coh', ...) or a function `lambda v, ref, sec: ...` whose arguments are sliders; `a * b` overlays views, `a + b` shows them side by side with linked zoom and pan; click a pixel / point for its time series (`series=`), double click for its reference; web mercator data over a satellite / street base map; polygons drawn on the map saved to `polygons='file.geojson'`; `.selected`, `.reference`, `.index` in python; `repr` describes a view and `.png(path)` saves an image. `moraine view PYRAMID ... -o view.ipynb [--show] [--dates]` writes a notebook of such maps and `moraine quicklook` draws with it (`--show`, the old `--post_proc` still works). Removed: `ras_plot`, `pc_plot`, `ts_plot`, `bg_alpha`, `view_pyramid` and the dependencies holoviews, bokeh, jupyter_bokeh; anywidget is a dependency

New command `polygon-mask`: bool mask of a raster or point cloud inside or outside the polygons of a GeoJSON file (longitude / latitude or radar grid coordinates); `moraine.read_polygons`, `write_polygons`, `polygons_contain`

`mcf_pc` / `mcf-pc`: new option `spatial_cost` ('constant', the default and the former behaviour, or 'length': phase jumps first on long edges). The `mcf-pc` command takes the pixel spacings (`range_pixel_spacing`, `azimuth_pixel_spacing`; the grid index is converted to meters), needs `ph` chunked one image per chunk and unwraps the interferograms in threads instead of dask processes: one copy of the network, `n_workers` by default bounded by the cores and half of the available memory; `threads_per_worker` and the dask cluster arguments are removed (10 million points, 16 interferograms, 8 at the same time: 28 s and 10 GB against 36 s and 17.5 GB with processes)

`mcf_pc` / `mcf-pc` use an own Delaunay triangulation and a successive shortest path min cost flow on its half-edges: exactly optimal, independent of the point order, about 7 times faster than before (10 million points in 16 s, GAMMA mcf_pt 35 s) with about 5 times less memory; new option `earth_cost` (default 1). OR-Tools is no longer a dependency

`gamma_mcf_pt`: float64 weights are converted to the FLOAT type mcf_pt reads (they were misread); the `gamma-mcf-pt` command passes `ref_point` on (it was ignored and the first point always used) and its default is 0, the first point as documented

`pc_pyramid` / `pc-pyramid`: the grid reaches the cells of the largest coordinates; the points of the last line and column were merged into the previous cells (one point per cell kept), and whether they were depended on rounding for coordinates that are not multiples of `ras_resolution`. Rebuild point cloud pyramids to see those points

Commands recognize outputs named without `/` or `.` (e.g. `--out_dir pyr`); `main()` returns 1 for a failed pipeline instead of raising `SystemExit`

`emi` command: `ref` was ignored, the first image was always the reference

Bugs squashed: CPU `ad_intf_pc` (undefined name), `isPD`/`nearestPD` on numpy arrays, `HilbertRtree.save`/`load` with zarr 3

## 0.9.0

Add mcf_pc API and CLI

Add interative point plot tool

## 0.8.5


## 0.8.4

Add n2ft model

Add API for plot

## 0.8.3

Move two deep learning models out of the package to prevent too big pypi package

update cli.co use image_pairs in memory rather than tnet file;

fix plot bugs in functions for general read zarr file, and change seq_intf from (i, i-1) to (i,i+1);

add multilook and intf to api.co;

finish gamma phase wrapping API and CLI;

move read/write gamma file to a seperate module from cli.load

## 0.8.2

bug fixed for cli.emperical_co_pc

### Bugs Squashed

- Data format scomplex is not supported. ([#20](https://github.com/kanglcn/moraine/issues/20))
  - ### Description of the problem

The package doesn't support scomplex data, so if the preprocessed rslc data is saved as scomplex, you are supposed to convert it to fcomplex and then process it with this package. 

### Minimal Complete Verifiable Example

_No response_

### Full error message

_No response_

### System information

```bash
numpy version 1.22.4
```


### Are you willing to help fix this bug?

No





## 0.8.1

implement parallel zarr io

modify `emperical_co_pc` and necessary utils
for independent ras chunkwise processing 

add `emperical_co_emi_temp_coh_pc` to 
- only processing a small batch in one chunk to prevent
holding the coherence matrix for all chunk which may exceed
memory limit;
- prevent writing coherence matrix which may exceed disk space limit.

## 0.8.0

- add cpu version of all functions and make use cpu as default
- add default args to dask cluster and allow users to configure that
- only calculate low tri of coherence matrix and copy its conj to up tri
- more flexiable plot functions
- carefully deal with nan values for all functions
- Update `pl.temp_coh` with elementwise kernel


## 0.7.0
Make cuda depedency optional

## 0.6.6
Test

## 0.6.5
Test

## 0.6.4
Test

## 0.6.3
Test

## 0.6.2
Test

## 0.6.1

Bug fix

## 0.6.0

Rename to Moraine

Add CLI plot

Package Reconstruction

## 0.5.1

Fix the doc generation

## 0.5.0

New features:

Add CLI for temporal coherence estimation for DS

Add dispersion index calculation for PS

Add logic operation for pc index

Add `transform` for coordinate reprojection

Add holoviews plot


Maintain:

Using `with ... as:` to prevent dask cluster unclosed

Remove all plot options in CLI

Set all thread per worker for local cuda cluster to 1

Use progressbar

Update doc theme

Reorginaze Tutorial

## 0.4.2

Finish point cloud manipulation functions and commands

Add utils for automatically determine chunk_size

## 0.4.1

Fix a small deploy issue

## 0.4.0

Modify gamma load function for more precise look vector

Add point cloud manipulation

Add Introduction section and add more detail on readme

Add utils, logger.zarr_info logger.darr_info

Rename emperical_co_sp to emperical_co_pc

## 0.3.2

Modify the gamma load funtion to make it faster

## 0.3.1

Add DS processing from CLI tutorial and fix bugs

## 0.3.0

Add interface to load gamma result

## 0.2.0

Add ks_test, select_ds_can, emperical_co_sp, emi CLI

Add log and sparse utils

Add test script

## 0.1.0

Refine phase linking EMI

Add function to estimate temporal coherence

Add tutorial for dask processing

Update the test in API notebooks

## 0.0.4

Add covariance/coherence matrix estimation for sparse data

Add phase linking method: EMI

Add tutorial for DS processing (still under construction)

## 0.0.3

Add ks test

Add covariance/coherence matrix estimation

Add tutorial for adaptive multilooking

## 0.0.2

Remove cupy requirement for automatically publish to pypi

## 0.0.1




