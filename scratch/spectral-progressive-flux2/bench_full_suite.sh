#!/usr/bin/env bash
# =============================================================================
# Full Benchmark Suite — FLUX.2-klein-4B
# 10 prompts × {fullres, dct_rewind L1 δ=0.05, δ=0.10} = 30 generations
# Generates per-prompt side-by-side strips + timing table
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLUX2_MODEL="/miele/brian/modelscope/black-forest-labs/FLUX.2-klein-4B"
STEPS=30
SEED=42

source /home/brianchc/sglang/scratch/select_gpu.sh
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
    "A misty forest at golden hour, warm amber and burnt sienna tones, cinematic color grading, photorealistic"
    "Portrait of a woman in soft rose-gold twilight, warm blush and peach skin tones, Kodak film look, shallow depth of field"
    "Neon-lit Tokyo street at midnight, deep teal shadows, magenta and violet reflections on wet pavement, cyberpunk mood"
    "Tuscany vineyard at sunset, rich terracotta and ochre, long golden light raking across rolling hills"
    "Arctic tundra at blue hour, cool cerulean and indigo, a lone wolf silhouetted against pale lavender sky"
    "Smoky jazz club interior, amber tungsten light, deep mahogany browns, hazy atmosphere, analogue grain"
    "Cherry blossoms at dusk, desaturated periwinkle sky, delicate dusty pink petals, soft diffused light"
    "Desert mesa at dawn, vivid orange sandstone glowing in first light, long purple shadows, cloudless cyan sky"
    "Underwater coral reef scene, luminescent teal water, warm sunbeams filtering down, rich saturated marine colors"
    "Autumn maple forest floor, deep crimson and cadmium orange leaves, misty morning light, moody chiaroscuro"
)

for i in "${!PROMPTS[@]}"; do
    printf "%02d: %s\n" "$i" "${PROMPTS[$i]}"
done > "$RESULTS_DIR/prompts.txt"

run_gen() {
    local prompt_id="$1" mode_label="$2" outfile="$3" prompt="$4"
    shift 4
    local logfile="${outfile%.png}.log"
    echo ""
    echo "── prompt_${prompt_id} ${mode_label} ──────────────────────────"

    sglang generate \
        --model-path "$FLUX2_MODEL" \
        --prompt "$prompt" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        "$@" \
        2>&1 | tee "$logfile"

    local clean_log total_s avg_s denoise_s
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")
    total_s=$(echo "$clean_log" | grep -oP "generated successfully in \K[\d.]+" || echo "NA")
    avg_s=$(echo "$clean_log"   | grep -oP "average time per step: \K[\d.]+"     || echo "NA")
    if ! denoise_s=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" 2>/dev/null | head -1); then
        [[ "$avg_s" != "NA" ]] && denoise_s=$(echo "scale=2; $avg_s * $STEPS" | bc -l) || denoise_s="NA"
    fi

    echo -e "${prompt_id}\t${mode_label}\t${total_s}\t${denoise_s}\t${avg_s}" >> "$TIMING_LOG"
    echo "  ✓ total=${total_s}s  denoise=${denoise_s}s"
}

for i in "${!PROMPTS[@]}"; do
    pid=$(printf "%02d" "$i")
    echo ""
    echo "╔══ Prompt $pid / ${#PROMPTS[@]} ════════════════════════════════"
    echo "║ ${PROMPTS[$i]}"
    echo "╚══════════════════════════════════════════════════════════"

    run_gen "$pid" "fullres" \
        "$RESULTS_DIR/prompt_${pid}_fullres.png" "${PROMPTS[$i]}"

    run_gen "$pid" "prog_d05" \
        "$RESULTS_DIR/prompt_${pid}_prog_d05.png" "${PROMPTS[$i]}" \
        --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05

    run_gen "$pid" "prog_d10" \
        "$RESULTS_DIR/prompt_${pid}_prog_d10.png" "${PROMPTS[$i]}" \
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
printf "%-6s  %8s  %8s  %8s  %8s  %8s\n" \
    "prompt" "fr_den" "d05_den" "d10_den" "spd_d05" "spd_d10"
printf "%-6s  %8s  %8s  %8s  %8s  %8s\n" \
    "------" "------" "-------" "-------" "-------" "-------"

fr_total=0; d05_total=0; d10_total=0; count=0
for i in "${!PROMPTS[@]}"; do
    pid=$(printf "%02d" "$i")
    fr=$(grep  "^${pid}	fullres"   "$TIMING_LOG" | awk -F'\t' '{print $4}')
    d05=$(grep "^${pid}	prog_d05"  "$TIMING_LOG" | awk -F'\t' '{print $4}')
    d10=$(grep "^${pid}	prog_d10"  "$TIMING_LOG" | awk -F'\t' '{print $4}')
    [[ "$fr" == "NA" || "$d05" == "NA" || "$d10" == "NA" ]] && continue
    spd05=$(echo "scale=2; $fr/$d05" | bc 2>/dev/null || echo "?")
    spd10=$(echo "scale=2; $fr/$d10" | bc 2>/dev/null || echo "?")
    printf "%-6s  %8.2f  %8.2f  %8.2f  %7sx  %7sx\n" \
        "$pid" "$fr" "$d05" "$d10" "$spd05" "$spd10"
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
    printf "%-6s  %8.2f  %8.2f  %8.2f  %7sx  %7sx\n" \
        "AVG" "$fr_avg" "$d05_avg" "$d10_avg" "$spd05_avg" "$spd10_avg"
fi

# ── Side-by-side comparison strips (3-way: fullres | d=0.05 | d=0.10) ────────
if command -v montage &>/dev/null; then
    echo ""
    echo "Generating per-prompt 3-way comparison strips..."
    for i in "${!PROMPTS[@]}"; do
        pid=$(printf "%02d" "$i")
        fr="$RESULTS_DIR/prompt_${pid}_fullres.png"
        d05="$RESULTS_DIR/prompt_${pid}_prog_d05.png"
        d10="$RESULTS_DIR/prompt_${pid}_prog_d10.png"
        [[ -f "$fr" && -f "$d05" && -f "$d10" ]] || continue
        montage "$fr" "$d05" "$d10" \
            -geometry 512x512+2+2 -tile 3x1 \
            -label 'fullres' -label 'δ=0.05' -label 'δ=0.10' \
            "$RESULTS_DIR/prompt_${pid}_3way.png"
    done

    # Full montage grid (prompts as rows, modes as columns)
    fullres_imgs=("$RESULTS_DIR"/prompt_*_fullres.png)
    d05_imgs=("$RESULTS_DIR"/prompt_*_prog_d05.png)
    d10_imgs=("$RESULTS_DIR"/prompt_*_prog_d10.png)
    montage "${fullres_imgs[@]}" "${d05_imgs[@]}" "${d10_imgs[@]}" \
        -geometry 256x256+2+2 -tile 3x10 \
        "$RESULTS_DIR/montage_3way.png"
    echo "Montage: $RESULTS_DIR/montage_3way.png"
fi

echo ""
echo "Logs: $RESULTS_DIR/*.log"
echo "RESULTS_DIR=$RESULTS_DIR"
