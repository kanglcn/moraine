# 0002 Deep learning inference uses PyTorch only

## Status

Accepted

## Date

2026-09-28

## Context

n2f, n2fs3d and n2ft ran with ONNX Runtime. Benchmarks on the sample data (634 249 points, A100):

- ONNX Runtime CUDA fails for n2ft with more than 65 535 points (cuDNN BatchNormalization limit).
- n2ft on CPU: 107 s with ONNX Runtime, 5.2 s with PyTorch; on GPU PyTorch works on all points (0.22 s).
- n2f / n2fs3d (UNet): ONNX and PyTorch are about as fast; outputs agree (mean phase difference ~5e-6 rad
  on CPU). On GPU both differ from CPU by ~6e-4 rad because of TF32 convolutions, only at pixels where
  the network output is nearly zero (noise).

Keeping two inference runtimes doubles the code paths and dependencies.

## Decision

All deep learning models run with PyTorch. Models are `state_dict` `.pth` files (n2f.pth, n2fs3d.pth from
github.com/kanglcn/n2f, n2ft.pth from github.com/kanglcn/n2ft) downloaded by
`moraine.download_dl_model()`, loaded once and cached by `moraine.dl._load_model`. PyTorch is an optional
dependency (`pip install moraine[dl]`) imported only when a model is used. cupy arrays are passed to
torch with DLPack; CLI GPU workers allocate torch memory from the rmm pool. n2ft uses a fixed farthest
point sampling start (`start_idx=0`) so results are reproducible.

## Consequences

- onnxruntime is no longer a dependency; `import moraine` does not import torch.
- TF32 stays enabled on GPU (5x faster than strict FP32 for the UNet); results differ slightly from CPU.

## Do not

- Do not add ONNX Runtime or another inference runtime back.
- Do not import torch at module level in code that `import moraine` loads.
- Do not change the model architectures in `moraine/unet_torch_.py` / `moraine/n2ft_torch_.py` without
  new weights: they must match the published `.pth` files.
