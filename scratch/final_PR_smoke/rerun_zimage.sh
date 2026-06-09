#!/usr/bin/env bash
# Re-run Z-Image smoke tests at the correct 1024×1024 resolution.
# The initial smoke_all_models.sh run used Z-Image's default 360×640.
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

MODELS_DIR="/miele/brian/modelscope"
ZIMAGE_MODEL="$MODELS_DIR/Tongyi-MAI/Z-Image"
RESULTS_DIR="$(dirname "$0")/results"
TIMINGS_LOG="$(dirname "$0")/timings.log"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
STEPS=50

echo "Config: steps=$STEPS seed=$SEED GPU=$CUDA_VISIBLE_DEVICES"
echo "Prompt: $PROMPT"

run_gen() {
    local label="$1"
    local ext="${2:-png}"
    shift 2
    local outfile="$RESULTS_DIR/${label}.${ext}"
    echo ""
    echo "=== $label ==="
    echo "  output: $outfile"
    local t0; t0=$(date +%s%N)
    sglang generate \
        --output-file-path "$outfile" \
        --seed "$SEED" \
        "$@"
    local t1; t1=$(date +%s%N)
    local elapsed; elapsed=$(python3 -c "print(f'{($t1-$t0)/1e9:.2f}')")
    echo "  -> saved: $outfile  (wall time: ${elapsed}s)"
    echo "${label},${elapsed}" >> "$TIMINGS_LOG"
}

run_gen "zimage_fullres_1024" "png" \
    --model-path "$ZIMAGE_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$STEPS" \
    --height 1024 --width 1024 \
    --dit-cpu-offload false

run_gen "zimage_dct_rewind_d0.05_1024" "png" \
    --model-path "$ZIMAGE_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$STEPS" \
    --height 1024 --width 1024 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

echo ""
echo "Z-Image re-run complete. Results in: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"/zimage_*1024*.png
