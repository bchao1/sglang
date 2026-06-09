#!/usr/bin/env bash
# =============================================================================
# 10-Prompt Quality Benchmark — FLUX.2-klein-4B
# dct_rewind L1 δ=0.05 vs fullres, same 10 cinematic prompts as Flux.1
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLUX2_MODEL="/miele/brian/modelscope/black-forest-labs/FLUX.2-klein-4B"
STEPS=30
SEED=42

export CUDA_VISIBLE_DEVICES=6
export FLASHINFER_DISABLE_VERSION_CHECK=1

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/quality_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "prompt_id\tmode\ttotal_s\tdenoise_s" > "$TIMING_LOG"

echo "GPU: $CUDA_VISIBLE_DEVICES  Model: $FLUX2_MODEL"
echo "Steps=$STEPS  Seed=$SEED  Results: $RESULTS_DIR"
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

# Save prompts index
for i in "${!PROMPTS[@]}"; do
    printf "%02d: %s\n" "$i" "${PROMPTS[$i]}"
done > "$RESULTS_DIR/prompts.txt"

run_gen() {
    local prompt_id="$1" mode="$2" outfile="$3" prompt="$4"
    local logfile="${outfile%.png}.log"
    local extra_flags=()
    if [[ "$mode" == "prog" ]]; then
        extra_flags=(--progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05)
    fi

    sglang generate \
        --model-path "$FLUX2_MODEL" \
        --prompt "$prompt" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        "${extra_flags[@]}" \
        2>&1 | tee "$logfile"

    local clean_log
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")
    local total_s denoise_s
    total_s=$(echo "$clean_log" | grep -oP "generated successfully in \K[\d.]+" || echo "NA")
    if ! denoise_s=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" 2>/dev/null | head -1); then
        local avg_s
        avg_s=$(echo "$clean_log" | grep -oP "average time per step: \K[\d.]+" || echo "NA")
        [[ "$avg_s" != "NA" ]] && denoise_s=$(echo "scale=2; $avg_s * $STEPS" | bc -l) || denoise_s="NA"
    fi
    echo -e "${prompt_id}\t${mode}\t${total_s}\t${denoise_s}" >> "$TIMING_LOG"
    echo "  ✓ prompt_${prompt_id} ${mode}: total=${total_s}s denoise=${denoise_s}s"
}

for i in "${!PROMPTS[@]}"; do
    pid=$(printf "%02d" "$i")
    echo ""
    echo "══════ Prompt $pid / ${#PROMPTS[@]} ══════"
    echo "${PROMPTS[$i]}"

    run_gen "$pid" "fullres" "$RESULTS_DIR/prompt_${pid}_fullres.png" "${PROMPTS[$i]}"
    run_gen "$pid" "prog"    "$RESULTS_DIR/prompt_${pid}_prog.png"    "${PROMPTS[$i]}"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  QUALITY BENCHMARK COMPLETE                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Results: $RESULTS_DIR"
echo ""
column -t -s $'\t' "$TIMING_LOG"

# Speedup summary per prompt
echo ""
echo "Per-prompt speedup (fullres→prog, denoising loop):"
for i in "${!PROMPTS[@]}"; do
    pid=$(printf "%02d" "$i")
    fr=$(grep "^${pid}	fullres" "$TIMING_LOG" | awk -F'\t' '{print $4}')
    pr=$(grep "^${pid}	prog"    "$TIMING_LOG" | awk -F'\t' '{print $4}')
    [[ "$fr" == "NA" || "$pr" == "NA" ]] && continue
    spd=$(echo "scale=2; $fr / $pr" | bc 2>/dev/null || echo "?")
    printf "  prompt_%s: fullres=%.1fs  prog=%.1fs  speedup=%sx\n" "$pid" "$fr" "$pr" "$spd"
done

echo ""
echo "Montage: montage $RESULTS_DIR/prompt_*_fullres.png $RESULTS_DIR/prompt_*_prog.png \\"
echo "         -geometry 512x512+2+2 -tile 2x $RESULTS_DIR/montage.png"
