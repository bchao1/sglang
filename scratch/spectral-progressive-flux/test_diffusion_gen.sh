#!/usr/bin/env bash
# Scratch test: sglang diffusion image generation
# Tests FLUX.1-dev and Z-Image / Z-Image-Turbo using pre-downloaded models.
# Images saved to scratch/results/ (git-excluded).
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

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
        --attention-backend torch_sdpa \
        --seed 42 \
        "$@"
    echo "  -> saved: $outfile"
}

# FLUX.1-dev
run_gen "flux1_dev" \
    "$MODELS_DIR/black-forest-labs/FLUX.1-dev" \
    --num-inference-steps 50

# Z-Image requires CUTLASS DSL which needs cuda-python 13.x → CUDA 13 runtime.
# Driver 560.35.05 only supports CUDA 12.6, so Z-Image cannot run on this machine.
# Uncomment when driver is updated to support CUDA 13.
# run_gen "z_image" \
#     "$MODELS_DIR/Tongyi-MAI/Z-Image" \
#     --num-inference-steps 50 \
#     --guidance-scale 1

echo ""
echo "Done. Results:"
ls -lh "$RESULTS_DIR"
