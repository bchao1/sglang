#!/usr/bin/env bash
# =============================================================================
# Full Benchmark Suite — Qwen-Image
# 10 prompts × {fullres, dct_rewind L1 δ=0.05, δ=0.10} = 30 generations
# Generates per-prompt side-by-side strips + timing table
# Modelled after scratch/spectral-progressive-flux2/bench_full_suite.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN_MODEL="/miele/brian/modelscope/Qwen/Qwen-Image"
STEPS=30
SEED=42

[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && source /home/brianchc/sglang/scratch/select_gpu.sh
export FLASHINFER_DISABLE_VERSION_CHECK=1

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/full_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "prompt_id\tmode\ttotal_s\tdenoise_s\tavg_step_s" > "$TIMING_LOG"

echo "GPU: $CUDA_VISIBLE_DEVICES  Steps=$STEPS  Seed=$SEED"
echo "Results: $RESULTS_DIR"
echo ""

PROMPTS=(
    "Golden hour over misty mountain peaks, dramatic cinematic light, photorealistic"
    "Gothic cathedral interior with stained glass windows, volumetric light shafts, 8K"
    "Close-up portrait of an elderly woman with weathered skin, natural outdoor light, film grain"
    "Hong Kong neon-lit street at night, rain reflections, teal and magenta, cinematic"
    "Macro photograph of a vintage pocket watch on velvet, bokeh, studio lighting"
    "Snow leopard mid-leap in Himalayan blizzard, motion blur, National Geographic style"
    "Minimalist Japanese tea room, morning light, cherry blossoms through paper screen door"
    "Aerial view of turquoise sea over coral reef, Maldives, drone shot, midday sun"
    "Arizona slot canyon, swirling sandstone walls, single beam of orange sunlight from above"
    "Ancient floating islands with waterfalls, lush forests, misty atmosphere, concept art"
)

LABELS=(
    "01_landscape"
    "02_architecture"
    "03_portrait"
    "04_cityscape"
    "05_object"
    "06_wildlife"
    "07_interior"
    "08_seascape"
    "09_desert"
    "10_fantasy"
)

for i in "${!PROMPTS[@]}"; do
    printf "%02d %s: %s\n" "$i" "${LABELS[$i]}" "${PROMPTS[$i]}"
done > "$RESULTS_DIR/prompts.txt"

run_gen() {
    local prompt_id="$1" mode_label="$2" outfile="$3" prompt="$4"
    shift 4
    local logfile="${outfile%.png}.log"
    echo ""
    echo "── ${LABELS[$prompt_id]} ${mode_label} ──────────────────────────"

    sglang generate \
        --model-path "$QWEN_MODEL" \
        --prompt "$prompt" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --height 1024 --width 1024 \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        "$@" \
        2>&1 | tee "$logfile"

    local clean_log total_s avg_s denoise_s
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")
    total_s=$(echo "$clean_log" | grep -oP "generated successfully in \K[\d.]+"                   || echo "NA")
    avg_s=$(echo "$clean_log"   | grep -oP "average time per step: \K[\d.]+"                       || echo "NA")
    denoise_s=$(echo "$clean_log" | grep -oP "finished in \K[\d.]+" | head -1                     || echo "NA")
    if [[ "$denoise_s" == "NA" ]] && [[ "$avg_s" != "NA" ]]; then
        denoise_s=$(echo "scale=2; $avg_s * $STEPS" | bc -l)
    fi

    echo -e "${prompt_id}\t${mode_label}\t${total_s}\t${denoise_s}\t${avg_s}" >> "$TIMING_LOG"
    echo "  ✓ total=${total_s}s  denoise=${denoise_s}s  avg_step=${avg_s}s"
}

for i in "${!PROMPTS[@]}"; do
    pid=$i
    echo ""
    echo "╔══ ${LABELS[$i]} ($((i+1))/${#PROMPTS[@]}) ════════════════════════════"
    echo "║ ${PROMPTS[$i]}"
    echo "╚══════════════════════════════════════════════════════════"

    run_gen "$pid" "fullres" \
        "$RESULTS_DIR/${LABELS[$i]}_fullres.png" "${PROMPTS[$i]}"

    run_gen "$pid" "prog_d05" \
        "$RESULTS_DIR/${LABELS[$i]}_prog_d05.png" "${PROMPTS[$i]}" \
        --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05

    run_gen "$pid" "prog_d10" \
        "$RESULTS_DIR/${LABELS[$i]}_prog_d10.png" "${PROMPTS[$i]}" \
        --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.10
done

# ── Timing summary ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BENCHMARK COMPLETE                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Results: $RESULTS_DIR"
echo ""
column -t -s $'\t' "$TIMING_LOG"

# Per-prompt speedup table
echo ""
printf "%-16s  %8s  %8s  %8s  %8s  %8s\n" \
    "label" "fr_den" "d05_den" "d10_den" "spd_d05" "spd_d10"
printf "%-16s  %8s  %8s  %8s  %8s  %8s\n" \
    "----------------" "------" "-------" "-------" "-------" "-------"

fr_total=0; d05_total=0; d10_total=0; count=0
for i in "${!PROMPTS[@]}"; do
    pid=$i
    fr=$(grep  "^${pid}	fullres"   "$TIMING_LOG" | awk -F'\t' '{print $4}')
    d05=$(grep "^${pid}	prog_d05"  "$TIMING_LOG" | awk -F'\t' '{print $4}')
    d10=$(grep "^${pid}	prog_d10"  "$TIMING_LOG" | awk -F'\t' '{print $4}')
    [[ "$fr" == "NA" || "$d05" == "NA" || "$d10" == "NA" ]] && continue
    spd05=$(echo "scale=2; $fr/$d05" | bc 2>/dev/null || echo "?")
    spd10=$(echo "scale=2; $fr/$d10" | bc 2>/dev/null || echo "?")
    printf "%-16s  %8.2f  %8.2f  %8.2f  %7sx  %7sx\n" \
        "${LABELS[$i]}" "$fr" "$d05" "$d10" "$spd05" "$spd10"
    fr_total=$(echo "$fr_total + $fr"   | bc)
    d05_total=$(echo "$d05_total + $d05" | bc)
    d10_total=$(echo "$d10_total + $d10" | bc)
    count=$((count+1))
done

if [[ $count -gt 0 ]]; then
    fr_avg=$(echo "scale=2; $fr_total/$count" | bc)
    d05_avg=$(echo "scale=2; $d05_total/$count" | bc)
    d10_avg=$(echo "scale=2; $d10_total/$count" | bc)
    spd05_avg=$(echo "scale=2; $fr_avg/$d05_avg" | bc)
    spd10_avg=$(echo "scale=2; $fr_avg/$d10_avg" | bc)
    printf "%-16s  %8.2f  %8.2f  %8.2f  %7sx  %7sx\n" \
        "AVG" "$fr_avg" "$d05_avg" "$d10_avg" "$spd05_avg" "$spd10_avg"
fi

echo ""
echo "Logs: $RESULTS_DIR/*.log"
echo "RESULTS_DIR=$RESULTS_DIR"
