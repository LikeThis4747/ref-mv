#!/usr/bin/env bash

set -euo pipefail

REF_MV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REF_MV_ROOT
export REF_GAUSSIAN_ROOT="$REF_MV_ROOT/third_party/ref_gaussian"
export MVINVERSE_ROOT="$REF_MV_ROOT/third_party/mvinverse"
export REF_MV_PYTHON="/home/disk9/wsq/conda/envs/mv_refgaussian/bin/python"
export PYTHONPATH="$REF_GAUSSIAN_ROOT:$REF_GAUSSIAN_ROOT/submodules/diff-surfel-rasterization:$REF_GAUSSIAN_ROOT/submodules/cubemapencoder:$REF_GAUSSIAN_ROOT/submodules/raytracing:$REF_GAUSSIAN_ROOT/submodules/simple-knn:$MVINVERSE_ROOT:$REF_MV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
