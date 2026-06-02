#!/usr/bin/env bash
# =============================================================================
# Qwen-Image progressive resolution — Group A timing benchmark
# =============================================================================
# Runs fullres + progressive configs, reports wall-clock and denoising timing.
# Saves timing_group_a.json for gen_speedup_plot_qwen.py.
#
# Usage:
#   bash scratch/spectral-progressive-qwen/run_qwen_benchmark.sh
#   bash scratch/spectral-progressive-qwen/run_qwen_benchmark.sh --steps 30
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
BENCH_ID="bench_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRATCH_DIR/spectral-progressive-qwen/results/$BENCH_ID"
mkdir -p "$RESULTS_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)  STEPS="$2";   shift 2 ;;
        --height) HEIGHT="$2";  shift 2 ;;
        --width)  WIDTH="$2";   shift 2 ;;
        --prompt) PROMPT="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Bench ID: $BENCH_ID"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Steps:    $STEPS  Seed: $SEED  Res: ${HEIGHT}x${WIDTH}"
echo "Model:    $QWEN_MODEL"
echo "Results:  $RESULTS_DIR"
echo ""

TIMING_JSON="$RESULTS_DIR/timing_group_a.json"
echo "[" > "$TIMING_JSON"
FIRST=1

run_config() {
    local label="$1"
    local outfile="$RESULTS_DIR/${label}.png"
    shift 1
    echo ""
    echo "--- $label ---"
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
        --save-output \
        "$@" 2>&1)
    t1=$(date +%s%3N)
    local wall_s
    wall_s=$(echo "scale=2; $(( t1 - t0 )) / 1000" | bc)

    # Extract denoising time from log
    local denoise_s=""
    if echo "$log" | grep -q "Progressive denoising done in"; then
        denoise_s=$(echo "$log" | grep -oP "Progressive denoising done in \K[0-9.]+" | head -1)
    fi
    if [[ -z "$denoise_s" ]]; then
        local avg_step
        avg_step=$(echo "$log" | grep -oP "average time per step: \K[0-9.]+" | head -1)
        if [[ -n "$avg_step" ]]; then
            denoise_s=$(echo "scale=2; $avg_step * $STEPS" | bc)
        fi
    fi
    if [[ -z "$denoise_s" ]]; then
        denoise_s=$(echo "$log" | grep -oP "generated successfully in \K[0-9.]+" | head -1 || echo "")
    fi

    echo "  Wall:    ${wall_s}s"
    echo "  Denoise: ${denoise_s:-N/A}s"
    echo "$log" | grep -E "Stage [0-9]|rewind:|Progressive|step.*sigma" | head -5 || true

    if [[ "$FIRST" -eq 0 ]]; then echo "," >> "$TIMING_JSON"; fi
    FIRST=0
    printf '  {"label": "%s", "wall_s": %s, "denoise_s": %s}' \
        "$label" "$wall_s" "${denoise_s:-null}" >> "$TIMING_JSON"
}

# Group A: fullres + progressive configs
run_config "A1_fullres"
run_config "A2_dct_rw_L1_d0.05" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05
run_config "A3_dct_rw_L1_d0.10" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.10
run_config "A4_dct_rw_L1_d0.20" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.20

echo "" >> "$TIMING_JSON"
echo "]" >> "$TIMING_JSON"

# Summary table
echo ""
echo "============================================================"
echo "Group A timing summary ($BENCH_ID, steps=$STEPS)"
echo "------------------------------------------------------------"
python3 - "$TIMING_JSON" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
base = next((r["denoise_s"] or r["wall_s"] for r in data if "A1" in r["label"]), None)
print(f"{'Config':<28} {'Wall':>8} {'Denoise':>9} {'Speedup':>8}")
print("-" * 58)
for r in data:
    t = r["denoise_s"] or r["wall_s"]
    spd = f"{base/t:.2f}x" if base and t else "?"
    print(f"{r['label']:<28} {r['wall_s']:>7.2f}s {(r['denoise_s'] or r['wall_s']):>8.2f}s {spd:>8}")
EOF

echo "============================================================"
echo "Results: $RESULTS_DIR"
