#!/usr/bin/env bash
# Quick smoke test: progressive resolution growing for Qwen-Image
# Runs fullres + several dct_rewind configs for a single prompt.
# Usage:
#   bash scratch/spectral-progressive-qwen/test_progressive_qwen_gen.sh
#   bash scratch/spectral-progressive-qwen/test_progressive_qwen_gen.sh --steps 20
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

QWEN_MODEL="/miele/brian/modelscope/Qwen/Qwen-Image"
RESULTS_DIR="$SCRATCH_DIR/spectral-progressive-qwen/results"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
STEPS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)  STEPS="$2";  shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Config: steps=$STEPS  seed=$SEED  GPU=$CUDA_VISIBLE_DEVICES"
echo "Prompt: $PROMPT"
echo "Model:  $QWEN_MODEL"
echo "Results: $RESULTS_DIR"

run_gen() {
    local label="$1"
    shift 1
    local outfile="$RESULTS_DIR/${label}.png"
    echo ""
    echo "=== $label ==="
    echo "  output: $outfile"
    time sglang generate \
        --model-path "$QWEN_MODEL" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        "$@"
    echo "  -> saved: $outfile"
}

# ── Baseline: full resolution ─────────────────────────────────────────────────
run_gen "qwen_fullres"

# ── Progressive: 1 level ──────────────────────────────────────────────────────
run_gen "qwen_dct_rewind_L1_d0.05" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

run_gen "qwen_dct_rewind_L1_d0.10" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.10

run_gen "qwen_dct_rewind_L1_d0.20" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.20

echo ""
echo "Done. Results:"
ls -lh "$RESULTS_DIR"
