#!/usr/bin/env bash
# =============================================================================
# Progressive Resolution Benchmark Suite — FLUX.2-klein-4B
# =============================================================================
#
# PURPOSE
#   Measure wall-clock denoising time for fullres vs progressive dct_rewind
#   at levels=1 / delta=0.05 (first test — A and beta are Flux.1 placeholders).
#
# GROUPS
#   A  Pure baseline: fullres vs progressive, GPU-resident, no opts
#
# USAGE
#   bash scratch/spectral-progressive-flux2/bench_progressive_flux2.sh [--steps N] [--group A|all]
#
# REQUIREMENTS
#   GPU with >=20 GB VRAM (FLUX.2-klein-4B is ~8 GB; plenty of headroom on A6000)
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FLUX2_MODEL="/miele/brian/modelscope/black-forest-labs/FLUX.2-klein-4B"
STEPS=30
SEED=42
GROUP="A"
PROMPT="A serene mountain lake at golden hour, photorealistic"

# Use GPU 1 (idle, 48 GB free)
export CUDA_VISIBLE_DEVICES=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)  STEPS="$2"; shift 2 ;;
        --seed)   SEED="$2";  shift 2 ;;
        --group)  GROUP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/bench_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "run_id\ttotal_s\tdenoise_s\tavg_step_s" > "$TIMING_LOG"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $FLUX2_MODEL"
echo "Results: $RESULTS_DIR"
echo "Config: steps=$STEPS seed=$SEED"
echo ""

# ---------------------------------------------------------------------------
# run_gen <label> <extra_flags...>
# ---------------------------------------------------------------------------
run_gen() {
    local label="$1"; shift
    local outfile="$RESULTS_DIR/${label}.png"
    local logfile="$RESULTS_DIR/${label}.log"
    echo ""
    echo "─── $label ──────────────────────────────────────────────────"

    time sglang generate \
        --model-path "$FLUX2_MODEL" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        "$@" \
        2>&1 | tee "$logfile"

    # Parse timing from logs (strip ANSI colour codes first)
    local clean_log
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")

    local total_s denoise_s avg_s
    total_s=$(echo "$clean_log" | grep -oP "generated successfully in \K[\d.]+" 2>/dev/null || echo "NA")
    avg_s=$(echo "$clean_log"   | grep -oP "average time per step: \K[\d.]+"     2>/dev/null || echo "NA")

    if denoise_s=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" 2>/dev/null | head -1); then
        : # got it from the progressive log line
    elif [[ "$avg_s" != "NA" ]]; then
        denoise_s=$(echo "scale=2; $avg_s * $STEPS" | bc -l)
    else
        denoise_s="NA"
    fi

    echo -e "${label}\t${total_s}\t${denoise_s}\t${avg_s}" >> "$TIMING_LOG"
    echo "  ✓  total=${total_s}s  denoise_loop=${denoise_s}s  avg=${avg_s}s/step"
}

# =============================================================================
# GROUP A — Pure baseline: fullres vs progressive, GPU-resident, no opts
# =============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "A" ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  GROUP A  Pure baseline — GPU-resident, no optimizations    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # A1: fullres (reference)
    run_gen "A1_fullres"

    # A2: progressive dct_rewind L1 δ=0.05 (transition ~step 28/50 → ~step 17/30)
    run_gen "A2_prog_L1_d0.05" \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta 0.05
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BENCHMARK COMPLETE                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Results: $RESULTS_DIR"
echo ""
echo "Timing (raw):"
column -t -s $'\t' "$TIMING_LOG"
echo ""

A1_denoise=$(grep "^A1_fullres" "$TIMING_LOG" | awk -F'\t' '{print $3}' 2>/dev/null || echo "")
if [[ -n "$A1_denoise" && "$A1_denoise" != "NA" ]]; then
    echo "Speedup vs A1_fullres DiT loop (${A1_denoise}s):"
    printf "  %-42s  %8s  %8s  %8s\n" "run_id" "total_s" "denoise_s" "speedup"
    printf "  %-42s  %8s  %8s  %8s\n" "------" "-------" "---------" "-------"
    while IFS=$'\t' read -r run_id total denoise avg; do
        [[ "$run_id" == "run_id" || "$denoise" == "NA" || "$denoise" == "denoise_s" ]] && continue
        speedup=$(echo "scale=2; $A1_denoise / $denoise" | bc 2>/dev/null || echo "?")
        printf "  %-42s  %8.1f  %9.2f  %7sx\n" "$run_id" "${total:-0}" "${denoise:-0}" "$speedup"
    done < "$TIMING_LOG"
fi

echo ""
echo "Logs:    $RESULTS_DIR/*.log"
echo "Images:  $RESULTS_DIR/*.png"
