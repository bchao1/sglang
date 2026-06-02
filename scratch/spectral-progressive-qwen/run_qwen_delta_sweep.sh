#!/usr/bin/env bash
# =============================================================================
# Qwen-Image progressive resolution — delta sweep for speedup vs delta curve
# =============================================================================
# Runs fullres + 5 delta values, saves delta_timing.json for the speedup plot.
#
# Usage:
#   bash scratch/spectral-progressive-qwen/run_qwen_delta_sweep.sh
# =============================================================================
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

QWEN_MODEL="/miele/brian/modelscope/Qwen/Qwen-Image"
STEPS=30
SEED=42
HEIGHT=1024
WIDTH=1024
PROMPT="Golden hour over misty mountain peaks, dramatic cinematic light, photorealistic"
SWEEP_ID="delta_sweep_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRATCH_DIR/spectral-progressive-qwen/results/$SWEEP_ID"
mkdir -p "$RESULTS_DIR"

echo "Sweep ID: $SWEEP_ID"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Steps:    $STEPS  Seed: $SEED  Res: ${HEIGHT}x${WIDTH}"
echo "Results:  $RESULTS_DIR"
echo ""

TIMING_JSON="$RESULTS_DIR/delta_timing.json"
echo "[" > "$TIMING_JSON"
FIRST=1

run_delta() {
    local delta="$1"
    local label="delta_$(echo "$delta" | tr '.' '_')"
    local outfile="$RESULTS_DIR/${label}.png"
    echo ""
    echo "--- delta=$delta ---"
    local t0 t1
    t0=$(date +%s%3N)
    local log
    log=$(sglang generate \
        --model-path "$QWEN_MODEL" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta "$delta" \
        --save-output 2>&1)
    t1=$(date +%s%3N)
    local wall_s
    wall_s=$(echo "scale=2; $(( t1 - t0 )) / 1000" | bc)

    local denoise_s=""
    if echo "$log" | grep -q "Progressive denoising done in"; then
        denoise_s=$(echo "$log" | grep -oP "Progressive denoising done in \K[0-9.]+" | head -1)
    fi

    echo "  Wall: ${wall_s}s  Denoise: ${denoise_s:-N/A}s"

    if [[ "$FIRST" -eq 0 ]]; then echo "," >> "$TIMING_JSON"; fi
    FIRST=0
    printf '  {"delta": %s, "wall_s": %s, "denoise_s": %s}' \
        "$delta" "$wall_s" "${denoise_s:-null}" >> "$TIMING_JSON"
}

run_delta 0.05
run_delta 0.10
run_delta 0.20
run_delta 0.50

echo "" >> "$TIMING_JSON"
echo "]" >> "$TIMING_JSON"

echo ""
echo "Delta sweep complete: $TIMING_JSON"
cat "$TIMING_JSON"
