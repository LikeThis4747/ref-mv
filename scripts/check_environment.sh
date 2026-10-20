#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$REF_MV_ROOT"

"$REF_MV_PYTHON" - <<'PY'
from pathlib import Path
import os

import torch
import gaussian_renderer
import diff_surfel_rasterization
from diff_surfel_rasterization import _C as diff_surfel_cuda
import raytracing
import _raytracing
import cubemapencoder
import _cubemapencoder
from simple_knn import _C as simple_knn_cuda
import mvinverse
import ref_mv
from mvinverse.models.mvinverse import MVInverse

root = Path(os.environ["REF_MV_ROOT"]).resolve()
ref_root = (root / "third_party" / "ref_gaussian").resolve()
mv_root = (root / "third_party" / "mvinverse").resolve()

loaded = {
    "ref_mv": Path(ref_mv.__file__).resolve(),
    "gaussian_renderer": Path(gaussian_renderer.__file__).resolve(),
    "mvinverse": Path(mvinverse.__file__).resolve(),
    "diff_surfel_rasterization": Path(diff_surfel_rasterization.__file__).resolve(),
    "diff_surfel_cuda": Path(diff_surfel_cuda.__file__).resolve(),
    "raytracing": Path(raytracing.__file__).resolve(),
    "raytracing_cuda": Path(_raytracing.__file__).resolve(),
    "cubemapencoder": Path(cubemapencoder.__file__).resolve(),
    "cubemapencoder_cuda": Path(_cubemapencoder.__file__).resolve(),
    "simple_knn_cuda": Path(simple_knn_cuda.__file__).resolve(),
}

assert torch.cuda.is_available(), "CUDA is not available"
assert loaded["ref_mv"].is_relative_to(root), loaded
assert loaded["gaussian_renderer"].is_relative_to(ref_root), loaded
assert loaded["mvinverse"].is_relative_to(mv_root), loaded
assert loaded["diff_surfel_rasterization"].is_relative_to(ref_root), loaded
assert loaded["diff_surfel_cuda"].is_relative_to(ref_root), loaded
assert loaded["raytracing"].is_relative_to(ref_root), loaded
assert loaded["raytracing_cuda"].is_relative_to(ref_root), loaded
assert loaded["cubemapencoder"].is_relative_to(ref_root), loaded
assert loaded["cubemapencoder_cuda"].is_relative_to(ref_root), loaded
assert loaded["simple_knn_cuda"].is_relative_to(ref_root), loaded

print(f"python: {os.sys.executable}")
print(f"torch: {torch.__version__}")
print(f"cuda: {torch.version.cuda} / {torch.cuda.get_device_name(0)}")
for name, path in loaded.items():
    print(f"{name}: {path}")
print(f"MVInverse: {MVInverse.__module__}.{MVInverse.__name__}")
print("environment import check: PASS")
PY
