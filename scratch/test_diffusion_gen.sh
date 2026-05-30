#!/usr/bin/env bash
# Scratch test: sglang diffusion image generation
# Tests FLUX.1-dev and Z-Image / Z-Image-Turbo using pre-downloaded models.
# Images saved to scratch/results/ (git-excluded).
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"
export SGLANG_DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA

MODELS_DIR="/miele/brian/modelscope"
RESULTS_DIR="$SCRATCH_DIR/results"
mkdir -p "$RESULTS_DIR"

PROMPT="A beautiful mountain lake at golden hour, photorealistic"

run_gen() {
    local label="$1"
    local model_path="$2"
    shift 2
    local outfile="$RESULTS_DIR/${label}.png"
    echo ""
    echo "=== $label ==="
    echo "  model: $model_path"
    echo "  output: $outfile"
    sglang generate \
        --model-path "$model_path" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --seed 42 \
        "$@"
    echo "  -> saved: $outfile"
}

# FLUX.1-dev
run_gen "flux1_dev" \
    "$MODELS_DIR/black-forest-labs/FLUX.1-dev" \
    --num-inference-steps 50

# Z-Image
run_gen "z_image" \
    "$MODELS_DIR/Tongyi-MAI/Z-Image" \
    --num-inference-steps 50

# Z-Image-Turbo (distilled — 4 steps is its native range, but match 50 for comparison)
run_gen "z_image_turbo" \
    "$MODELS_DIR/Tongyi-MAI/Z-Image-Turbo" \
    --num-inference-steps 50

echo ""
echo "Done. Results:"
ls -lh "$RESULTS_DIR"
