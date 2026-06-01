#!/usr/bin/env bash
# Scratch test: progressive resolution growing for FLUX.1-dev
# Runs fullres baseline and several dct_rewind configurations,
# saves side-by-side PNGs to scratch/results/progressive/.
# Usage:
#   bash scratch/test_progressive_gen.sh
#   bash scratch/test_progressive_gen.sh --steps 20          # faster debug run
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

MODELS_DIR="/miele/brian/modelscope"
FLUX_MODEL="$MODELS_DIR/black-forest-labs/FLUX.1-dev"
RESULTS_DIR="$SCRATCH_DIR/results/progressive"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
STEPS=50

# Override defaults from CLI flags (e.g. --steps 20 for quick runs)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --seed)   SEED="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Config: steps=$STEPS  seed=$SEED"
echo "Prompt: $PROMPT"
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

# ── Baseline: full resolution, all 50 steps at 128×128 latent ─────────────────
run_gen "flux_fullres" "$FLUX_MODEL"

# ── Progressive: 1 level (64×64 → 128×128 latent, ~2-stage) ──────────────────
run_gen "flux_dct_rewind_L1_d0.01" "$FLUX_MODEL" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.01

# ── Progressive: 1 level, tighter delta (transitions earlier) ─────────────────
run_gen "flux_dct_rewind_L1_d0.05" "$FLUX_MODEL" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# ── Progressive: 2 levels (32×32 → 64×64 → 128×128, 3-stage) ────────────────
run_gen "flux_dct_rewind_L2_d0.01" "$FLUX_MODEL" \
    --progressive-mode dct_rewind \
    --progressive-levels 2 \
    --progressive-delta 0.01

# ── Progressive: plain DCT (no rewind, for comparison) ───────────────────────
run_gen "flux_dct_plain_L1" "$FLUX_MODEL" \
    --progressive-mode dct \
    --progressive-levels 1 \
    --progressive-delta 0.01

echo ""
echo "Done. Results:"
ls -lh "$RESULTS_DIR"
echo ""
echo "Compare with ImageMagick montage:"
echo "  montage $RESULTS_DIR/*.png -geometry 512x512+4+4 $RESULTS_DIR/montage.png && eog $RESULTS_DIR/montage.png"
