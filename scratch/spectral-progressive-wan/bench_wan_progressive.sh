#!/usr/bin/env bash
# =============================================================================
# Progressive Resolution Benchmark — Wan T2V 1.3B
# =============================================================================
#
# PURPOSE
#   Compare fullres vs dct_rewind progressive for Wan T2V.
#   Parameters exactly match wavelet-diffusion/inference_progressive.py:
#     height=480, width=832, num_frames=81, steps=50, guidance_scale=5.0
#     flow_shift=5.0 (via --flow-shift, overriding sglang default of 3.0)
#
# USAGE
#   bash scratch/spectral-progressive-wan/bench_wan_progressive.sh [--delta D] [--levels L]
#
#   Defaults: delta=0.05, levels=1
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scratch/select_gpu.sh"

WAN_MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
STEPS=50
SEED=42
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
GUIDANCE_SCALE=5.0
FLOW_SHIFT=5.0   # matches wavelet-diffusion/inference_progressive.py WAN_SHIFT=5.0
DELTA=0.05
LEVELS=1
PROMPT="A curious raccoon explores a lush forest with dappled sunlight, photorealistic"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delta)   DELTA="$2";   shift 2 ;;
        --levels)  LEVELS="$2";  shift 2 ;;
        --prompt)  PROMPT="$2";  shift 2 ;;
        --seed)    SEED="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/bench_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "run_id\ttotal_s\tdenoise_s\tavg_step_s\ttransition_step" > "$TIMING_LOG"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $WAN_MODEL"
echo "Results: $RESULTS_DIR"
echo "Config: steps=$STEPS seed=$SEED ${HEIGHT}x${WIDTH} frames=$NUM_FRAMES"
echo "        guidance=$GUIDANCE_SCALE flow_shift=$FLOW_SHIFT delta=$DELTA levels=$LEVELS"
echo ""

# ---------------------------------------------------------------------------
# run_gen <label> <extra_flags...>
# ---------------------------------------------------------------------------
run_gen() {
    local label="$1"; shift
    local outfile="$RESULTS_DIR/${label}.mp4"
    local logfile="$RESULTS_DIR/${label}.log"
    echo ""
    echo "─── $label ──────────────────────────────────────────────────"

    time sglang generate \
        --model-path "$WAN_MODEL" \
        --prompt "$PROMPT" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --num-frames "$NUM_FRAMES" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --flow-shift "$FLOW_SHIFT" \
        --dit-cpu-offload false \
        "$@" \
        2>&1 | tee "$logfile"

    local total_s denoise_s avg_s trans_step
    local clean_log
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")
    total_s=$(echo "$clean_log" | grep -oP "generated successfully in \K[\d.]+" || echo "NA")
    avg_s=$(echo "$clean_log" | grep -oP "average time per step: \K[\d.]+" || echo "NA")
    if denoise_done=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" 2>/dev/null | head -1); then
        denoise_s="$denoise_done"
    elif [[ "$avg_s" != "NA" ]]; then
        denoise_s=$(echo "scale=2; $avg_s * $STEPS" | bc -l)
    else
        denoise_s="NA"
    fi
    # Capture transition step from progressive log
    trans_step=$(echo "$clean_log" | grep -oP "Stage \d/\d: \d+x\d+ latent, steps \[0, \K\d+" 2>/dev/null | head -1 || echo "NA")
    echo -e "${label}\t${total_s}\t${denoise_s}\t${avg_s}\t${trans_step}" >> "$TIMING_LOG"
    echo "  ✓  total=${total_s}s  denoise=${denoise_s}s  avg=${avg_s}s/step  transition_step=${trans_step}"
}

# =============================================================================
# R1: fullres baseline
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  R1  Fullres baseline — GPU-resident, no opts               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
run_gen "R1_fullres"

# =============================================================================
# R2: progressive dct_rewind L1 δ=0.05
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  R2  Progressive dct_rewind L${LEVELS} δ=${DELTA}                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
run_gen "R2_prog_L${LEVELS}_d${DELTA}" \
    --progressive-mode dct_rewind \
    --progressive-levels "$LEVELS" \
    --progressive-delta "$DELTA"

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

R1_denoise=$(grep "^R1_fullres" "$TIMING_LOG" | awk -F'\t' '{print $3}')
if [[ -n "$R1_denoise" && "$R1_denoise" != "NA" ]]; then
    echo "Speedup vs R1_fullres DiT loop (${R1_denoise}s):"
    printf "  %-40s  %8s  %9s  %10s  %8s\n" "run_id" "total_s" "denoise_s" "trans_step" "speedup"
    printf "  %-40s  %8s  %9s  %10s  %8s\n" "------" "-------" "---------" "----------" "-------"
    while IFS=$'\t' read -r run_id total denoise avg trans; do
        [[ "$run_id" == "run_id" || "$denoise" == "NA" || "$denoise" == "denoise_s" ]] && continue
        speedup=$(echo "scale=2; $R1_denoise / $denoise" | bc 2>/dev/null || echo "?")
        printf "  %-40s  %8.1f  %9.2f  %10s  %7sx\n" "$run_id" "${total:-0}" "${denoise:-0}" "$trans" "$speedup"
    done < "$TIMING_LOG"
fi

echo ""
echo "Videos: $RESULTS_DIR/*.mp4"
echo ""
echo "To compare visually:"
echo "  ffplay $RESULTS_DIR/R1_fullres.mp4"
echo "  ffplay $RESULTS_DIR/R2_prog_L${LEVELS}_d${DELTA}.mp4"
