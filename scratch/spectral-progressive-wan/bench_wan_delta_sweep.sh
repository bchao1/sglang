#!/usr/bin/env bash
# =============================================================================
# Wan T2V Progressive — delta sweep benchmark
# Runs ONE fullres baseline + dct_rewind L1 for each δ in DELTAS.
# Parameters match wavelet-diffusion/inference_progressive.py exactly.
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
FLOW_SHIFT=5.0
LEVELS=1
DELTAS=(0.01 0.02 0.05 0.1)
PROMPT="A Formula 1 race car speeding through a circuit at sunset, motion blur, photorealistic"

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/delta_sweep_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "run_id\ttotal_s\tdenoise_s\tavg_step_s\ttransition_step\tspeedup" > "$TIMING_LOG"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $WAN_MODEL"
echo "Results: $RESULTS_DIR"
echo "Config: steps=$STEPS seed=$SEED ${HEIGHT}x${WIDTH} frames=$NUM_FRAMES"
echo "        guidance=$GUIDANCE_SCALE flow_shift=$FLOW_SHIFT levels=$LEVELS"
echo "Prompt: $PROMPT"
echo ""

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
    trans_step=$(echo "$clean_log" | grep -oP "Stage \d/\d: \d+x\d+ latent, steps \[0, \K\d+" 2>/dev/null | head -1 || echo "NA")

    # Compute speedup vs fullres (set by caller via FULLRES_DENOISE env)
    local speedup="baseline"
    if [[ -n "${FULLRES_DENOISE:-}" && "$denoise_s" != "NA" ]]; then
        speedup=$(echo "scale=2; $FULLRES_DENOISE / $denoise_s" | bc -l)
        speedup="${speedup}x"
    fi

    echo -e "${label}\t${total_s}\t${denoise_s}\t${avg_s}\t${trans_step}\t${speedup}" >> "$TIMING_LOG"
    echo "  ✓  total=${total_s}s  denoise=${denoise_s}s  transition_step=${trans_step}  speedup=${speedup}"
}

# =============================================================================
# Fullres baseline (run once)
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  R1  Fullres baseline                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
run_gen "R1_fullres"

# Extract fullres denoise time for speedup calculations
FULLRES_DENOISE=$(grep "^R1_fullres" "$TIMING_LOG" | awk -F'\t' '{print $3}')
export FULLRES_DENOISE
echo "Fullres denoise: ${FULLRES_DENOISE}s"

# =============================================================================
# Progressive sweep over delta values
# =============================================================================
for DELTA in "${DELTAS[@]}"; do
    DLABEL=$(echo "$DELTA" | tr '.' '_')
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  dct_rewind L${LEVELS} δ=${DELTA}                                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    run_gen "R_prog_L${LEVELS}_d${DLABEL}" \
        --progressive-mode dct_rewind \
        --progressive-levels "$LEVELS" \
        --progressive-delta "$DELTA"
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  DELTA SWEEP COMPLETE                                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Results: $RESULTS_DIR"
echo ""
echo "Timing:"
column -t -s $'\t' "$TIMING_LOG"
echo ""
echo "Videos: $RESULTS_DIR/*.mp4"
