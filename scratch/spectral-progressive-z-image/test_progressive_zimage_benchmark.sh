#!/usr/bin/env bash
# Benchmark: progressive resolution growing for Z-Image vs fullres baseline.
# Measures wall-clock time for each config, outputs a timing table.
# Usage:
#   bash scratch/spectral-progressive-z-image/test_progressive_zimage_benchmark.sh
#   bash scratch/spectral-progressive-z-image/test_progressive_zimage_benchmark.sh --steps 20
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

MODELS_DIR="/miele/brian/modelscope"
ZIMAGE_MODEL="$MODELS_DIR/Tongyi-MAI/Z-Image"
BENCH_ID="bench_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRATCH_DIR/spectral-progressive-z-image/results/$BENCH_ID"
mkdir -p "$RESULTS_DIR"

PROMPT="A golden hour mountain landscape with soft clouds"
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

echo "Bench ID: $BENCH_ID"
echo "Config:   steps=$STEPS  seed=$SEED"
echo "Model:    $ZIMAGE_MODEL"
echo "Results:  $RESULTS_DIR"

declare -A TIMES

bench_run() {
    local label="$1"
    local model_path="$2"
    shift 2
    local outfile="$RESULTS_DIR/${label}.png"
    echo ""
    echo "--- $label ---"
    local t0 t1
    t0=$(date +%s%N)
    sglang generate \
        --model-path "$model_path" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        "$@" 2>&1 | grep -E "seconds|step|stage|rewind|Progressive|Pixel data" || true
    t1=$(date +%s%N)
    local elapsed_ms=$(( (t1 - t0) / 1000000 ))
    local elapsed_s
    elapsed_s=$(echo "scale=2; $elapsed_ms / 1000" | bc)
    TIMES["$label"]="$elapsed_s"
    echo "  Wall-clock: ${elapsed_s}s"
}

# ── Group A: core configs ──────────────────────────────────────────────────────
bench_run "A1_fullres"          "$ZIMAGE_MODEL"
bench_run "A2_dct_rw_L1_d0.01" "$ZIMAGE_MODEL" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.01
bench_run "A3_dct_rw_L1_d0.05" "$ZIMAGE_MODEL" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05
bench_run "A4_dct_rw_L2_d0.01" "$ZIMAGE_MODEL" \
    --progressive-mode dct_rewind --progressive-levels 2 --progressive-delta 0.01

echo ""
echo "============================================================"
echo "Timing summary ($BENCH_ID, steps=$STEPS)"
echo "------------------------------------------------------------"
printf "%-30s %8s %8s\n" "Config" "Time(s)" "Speedup"
echo "------------------------------------------------------------"
t_base="${TIMES[A1_fullres]}"
for key in A1_fullres A2_dct_rw_L1_d0.01 A3_dct_rw_L1_d0.05 A4_dct_rw_L2_d0.01; do
    t="${TIMES[$key]}"
    speedup=$(echo "scale=2; $t_base / $t" | bc)
    printf "%-30s %8s %8s\n" "$key" "${t}s" "${speedup}x"
done
echo "============================================================"
echo "Images saved to: $RESULTS_DIR"
