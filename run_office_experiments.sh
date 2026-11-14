#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source scripts/env.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="data/office/office_0422_refgaussian_60views"
REF_LOG="$(mktemp)"
trap 'rm -f "$REF_LOG"' EXIT

echo "[GPU 0] Starting original Ref training to iteration 20000"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" third_party/ref_gaussian/train.py \
    -s "$DATASET" \
    --eval \
    -r 2 \
    --iterations 20000 \
    --indirect_from_iter 10000 \
    --volume_render_until_iter 0 \
    --initial 1 \
    --init_until_iter 3000 \
    --lambda_normal_smooth 0.45 \
    --save_iterations 20000 \
    > >(tee "$REF_LOG") 2>&1 &
REF_PID=$!

echo "[GPU 1] Starting joint training from iteration 1"
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" train.py \
    -s "$DATASET" \
    -r 2 \
    --volume_render_until_iter 0 \
    --iterations 30000 \
    --save_iterations 20000 30000 \
    --gs_feature_lr 0.001 \
    --joint_envmap_lr 0.001 \
    --gs_patch_lr 5e-5 \
    --neighbor_k 8 \
    --num_views 2 &
JOINT_START_PID=$!

echo "Waiting for original Ref training to finish before starting Ref-20000 -> joint..."
wait "$REF_PID"

REF_OUTPUT="$(sed -n 's/^Optimizing //p' "$REF_LOG" | head -n 1 | awk '{print $1}' || true)"
REF_OUTPUT="${REF_OUTPUT#./}"
if [[ -z "$REF_OUTPUT" ]]; then
    echo "Could not determine the automatically generated Ref output path." >&2
    exit 1
fi
echo "Detected Ref output: $REF_OUTPUT"

echo "[GPU 0] Starting joint training from Ref iteration 20000"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" train.py \
    -s "$DATASET" \
    -r 2 \
    --volume_render_until_iter 0 \
    --ref_model_path "$REF_OUTPUT" \
    --ref_iteration 20000 \
    --iterations 30000 \
    --save_iterations 30000 \
    --gs_feature_lr 0.001 \
    --joint_envmap_lr 0.001 \
    --gs_patch_lr 5e-5 \
    --neighbor_k 8 \
    --num_views 2 &
JOINT_AFTER_REF_PID=$!

wait "$JOINT_START_PID"
wait "$JOINT_AFTER_REF_PID"

echo "All office experiments completed."
