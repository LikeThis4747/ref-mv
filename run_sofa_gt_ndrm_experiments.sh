#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source scripts/env.sh

DATASET="${REF_MV_ROOT}/data/sofa_0511_refgaussian_sample70_blendercoords"
OUTPUT_ROOT="${REF_MV_ROOT}/output/sofa_0511_refgaussian_sample70_blendercoords"

# 1) Skipped: an existing ref_GT_ND_r2_20k checkpoint is reused below.
# CUDA_VISIBLE_DEVICES=0 "${REF_MV_PYTHON}" third_party/ref_gaussian/train.py \
#   -s "${DATASET}" \
#   --eval -r 2 \
#   --iterations 20000 --save_iterations 20000 \
#   --initial 1 --init_until_iter 3000 \
#   --volume_render_until_iter 0 --indirect_from_iter 10000 \
#   --use_gt_normal_loss --use_gt_depth_loss \
#   --lambda_gt_normal 0.05 --lambda_gt_depth 0.01 \
#   --experiment_name ref_GT_ND_r2_20k

# 2) Joint from scratch, joining after the 3000-iteration initial stage.
CUDA_VISIBLE_DEVICES=0 "${REF_MV_PYTHON}" train.py \
  -s "${DATASET}" \
  --eval -r 2 \
  --iterations 30000 --save_iterations 30000 \
  --initial 1 --init_until_iter 3000 \
  --volume_render_until_iter 0 --indirect_from_iter 20000 \
  --gs_feature_lr 0.001 --joint_envmap_lr 0.001 --gs_patch_lr 5e-5 \
  --neighbor_k 8 --num_views 2 \
  --use_gt_normal_loss --use_gt_depth_loss \
  --lambda_gt_normal 0.05 --lambda_gt_depth 0.01 \
  --experiment_name joint_start3k_GT_ND_nv2_r2_30k

# 3) Joint from the GT-normal/depth original Ref checkpoint saved above.
REF_DIR_NAME="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
  -name '*-ref_GT_ND_r2_20k' -printf '%f\n' | sort | tail -n 1)"
if [[ -z "${REF_DIR_NAME}" ]]; then
  echo "Could not find the Ref GT-ND 20k output under ${OUTPUT_ROOT}" >&2
  exit 1
fi
REF_MODEL_PATH="${OUTPUT_ROOT}/${REF_DIR_NAME}"
if [[ ! -f "${REF_MODEL_PATH}/point_cloud/iteration_20000/point_cloud.ply" ]]; then
  echo "Ref iteration 20000 is missing from ${REF_MODEL_PATH}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "${REF_MV_PYTHON}" train.py \
  -s "${DATASET}" \
  --eval -r 2 \
  --initial 0 --init_until_iter 0 \
  --volume_render_until_iter 0 --indirect_from_iter 20000 \
  --ref_model_path "${REF_MODEL_PATH}" --ref_iteration 20000 \
  --iterations 30000 --save_iterations 30000 \
  --gs_feature_lr 0.001 --joint_envmap_lr 0.001 --gs_patch_lr 5e-5 \
  --neighbor_k 8 --num_views 2 \
  --use_gt_normal_loss --use_gt_depth_loss \
  --lambda_gt_normal 0.05 --lambda_gt_depth 0.01 \
  --experiment_name joint_start20k_GT_ND_nv2_r30k
