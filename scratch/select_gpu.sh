#!/usr/bin/env bash
# Pick the GPU with the most free memory and export CUDA_VISIBLE_DEVICES.
# Usage:
#   source scratch/select_gpu.sh          # sets CUDA_VISIBLE_DEVICES in current shell
#   export CUDA_VISIBLE_DEVICES=$(bash scratch/select_gpu.sh)  # subprocess form

_pick_gpu() {
    # Returns the GPU index with the highest free memory (MiB).
    nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits \
    | awk -F', ' '{ if ($2 > max) { max=$2; idx=$1 } } END { print idx }'
}

_GPU_ID=$(_pick_gpu)

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Script is being sourced — set the variable in the caller's shell.
    export CUDA_VISIBLE_DEVICES="$_GPU_ID"
    echo "[select_gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "$_GPU_ID" | awk -F', ' '{printf "%d/%d MiB free", $1, $2}'))"
else
    # Script is run directly — just print the index for $(...) capture.
    echo "$_GPU_ID"
fi

unset -f _pick_gpu
unset _GPU_ID
