#!/usr/bin/env bash
# Scratch test: progressive resolution growing for Z-Image
# Runs fullres baseline and several dct_rewind configurations,
# saves side-by-side PNGs to scratch/results/progressive_zimage/.
# Usage:
#   bash scratch/spectral-progressive-z-image/test_progressive_zimage_gen.sh
#   bash scratch/spectral-progressive-z-image/test_progressive_zimage_gen.sh --steps 20
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

MODELS_DIR="/miele/brian/modelscope"
ZIMAGE_MODEL="$MODELS_DIR/Tongyi-MAI/Z-Image"
RESULTS_DIR="$SCRATCH_DIR/spectral-progressive-z-image/results"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
STEPS=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)  STEPS="$2";  shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Config: steps=$STEPS  seed=$SEED"
echo "Prompt: $PROMPT"
echo "Model:  $ZIMAGE_MODEL"
echo "Results: $RESULTS_DIR"

run_gen() {
    local label="$1"
    local model_path="$2"
    shift 2
    local outfile="$RESULTS_DIR/${label}.png"
    echo ""
    echo "=== $label ==="
    echo "  output: $outfile"
    time sglang generate \
        --model-path "$model_path" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        "$@"
    echo "  -> saved: $outfile"
}

# ── Baseline: full resolution ─────────────────────────────────────────────────
run_gen "zimage_fullres" "$ZIMAGE_MODEL"

# ── Progressive: 1 level (64×64 → 128×128 latent) ────────────────────────────
run_gen "zimage_dct_rewind_L1_d0.01" "$ZIMAGE_MODEL" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.01

run_gen "zimage_dct_rewind_L1_d0.05" "$ZIMAGE_MODEL" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# ── Progressive: 2 levels (32×32 → 64×64 → 128×128, 3-stage) ────────────────
run_gen "zimage_dct_rewind_L2_d0.01" "$ZIMAGE_MODEL" \
    --progressive-mode dct_rewind \
    --progressive-levels 2 \
    --progressive-delta 0.01

# ── Progressive: plain DCT (no rewind, for comparison) ───────────────────────
run_gen "zimage_dct_plain_L1" "$ZIMAGE_MODEL" \
    --progressive-mode dct \
    --progressive-levels 1 \
    --progressive-delta 0.01

echo ""
echo "Done. Results:"
ls -lh "$RESULTS_DIR"
echo ""
echo "Compare with ImageMagick montage:"
echo "  montage $RESULTS_DIR/*.png -geometry 512x512+4+4 $RESULTS_DIR/montage.png"
