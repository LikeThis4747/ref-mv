#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# The training script uses both GPUs, so both must have enough free memory.
GPU_IDS=(0 1)
MIN_FREE_GIB="${MIN_FREE_GIB:-25}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-1800}"
MIN_FREE_MIB=$((MIN_FREE_GIB * 1024))

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; cannot monitor GPU memory." >&2
    exit 1
fi

now_epoch=$(TZ=Asia/Shanghai date +%s)
today_six=$(TZ=Asia/Shanghai date -d 'today 06:00:00' +%s)
if (( now_epoch < today_six )); then
    start_epoch=$today_six
else
    start_epoch=$(TZ=Asia/Shanghai date -d 'tomorrow 06:00:00' +%s)
fi

start_text=$(TZ=Asia/Shanghai date -d "@$start_epoch" '+%Y-%m-%d %H:%M:%S %Z')
echo "Monitoring starts at $start_text."
echo "Requirement: GPU ${GPU_IDS[*]} each have > ${MIN_FREE_GIB} GiB free memory."
echo "Check interval: ${CHECK_INTERVAL_SECONDS}s."

if (( now_epoch < start_epoch )); then
    sleep "$((start_epoch - now_epoch))"
fi

while true; do
    timestamp=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')
    gpu_status=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    ready=1

    for gpu_id in "${GPU_IDS[@]}"; do
        free_mib=$(awk -F',' -v gpu="$gpu_id" '$1 + 0 == gpu {gsub(/[[:space:]]/, "", $2); print $2}' <<< "$gpu_status")
        if [[ -z "$free_mib" ]]; then
            echo "[$timestamp] Could not read free memory for GPU $gpu_id." >&2
            ready=0
            continue
        fi

        if (( free_mib <= MIN_FREE_MIB )); then
            echo "[$timestamp] GPU $gpu_id: ${free_mib} MiB free (waiting)."
            ready=0
        else
            echo "[$timestamp] GPU $gpu_id: ${free_mib} MiB free (ready)."
        fi
    done

    if (( ready )); then
        echo "[$timestamp] Memory condition satisfied; starting experiments."
        exec bash run_office_experiments.sh
    fi

    echo "[$timestamp] Next check in ${CHECK_INTERVAL_SECONDS}s."
    sleep "$CHECK_INTERVAL_SECONDS"
done
