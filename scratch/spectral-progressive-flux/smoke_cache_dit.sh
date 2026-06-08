#!/usr/bin/env bash
# Smoke test: Cache-DiT × Progressive resolution on FLUX.1-dev.
#
# Runs four configurations:
#   1. fullres        + no cache-dit  (baseline)
#   2. fullres        + cache-dit
#   3. dct_rewind L1  + no cache-dit
#   4. dct_rewind L1  + cache-dit     ← the newly fixed combo
#
# Usage:
#   bash scratch/spectral-progressive-flux/smoke_cache_dit.sh
#   bash scratch/spectral-progressive-flux/smoke_cache_dit.sh --steps 10
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

MODELS_DIR="/miele/brian/modelscope"
FLUX1_MODEL="$MODELS_DIR/black-forest-labs/FLUX.1-dev"
RESULTS_DIR="$(dirname "$0")/results/cache_dit_smoke"
TIMINGS_LOG="$RESULTS_DIR/timings.log"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
STEPS=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Config: steps=$STEPS  GPU=$CUDA_VISIBLE_DEVICES"
echo "Model: $FLUX1_MODEL"
echo "Results: $RESULTS_DIR"
echo ""
echo "# Cache-DiT smoke test — $(date)" > "$TIMINGS_LOG"
echo "label,wall_secs" >> "$TIMINGS_LOG"

run_gen() {
    local label="$1"
    shift
    local outfile="$RESULTS_DIR/${label}.png"
    echo ""
    echo "=== $label ==="
    echo "  output: $outfile"
    local t0 t1 elapsed
    t0=$(date +%s%N)
    sglang generate \
        --model-path "$FLUX1_MODEL" \
        --prompt "$PROMPT" \
        --attention-backend torch_sdpa \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        --seed "$SEED" \
        --output-file-path "$outfile" \
        "$@"
    t1=$(date +%s%N)
    elapsed=$(python3 -c "print(f'{($t1-$t0)/1e9:.2f}')")
    echo "  -> saved: $outfile  (${elapsed}s)"
    echo "${label},${elapsed}" >> "$TIMINGS_LOG"
}

# 1. fullres, no cache-dit
run_gen "fullres_no_cache"

# 2. fullres + cache-dit
SGLANG_CACHE_DIT_ENABLED=1 run_gen "fullres_cache_dit"

# 3. progressive, no cache-dit
run_gen "progressive_no_cache" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# 4. progressive + cache-dit  (the newly fixed combination)
SGLANG_CACHE_DIT_ENABLED=1 run_gen "progressive_cache_dit" \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

echo ""
echo "===================================================================="
echo "All four configs passed."
echo ""
cat "$TIMINGS_LOG"
echo ""
echo "Output files:"
ls -lh "$RESULTS_DIR"/*.png
