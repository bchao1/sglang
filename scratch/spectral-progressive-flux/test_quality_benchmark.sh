#!/usr/bin/env bash
# =============================================================================
# Progressive vs Fullres — Color/Cinematic Quality Benchmark
# =============================================================================
#
# PURPOSE
#   Test the hypothesis that progressive resolution growing produces better
#   color grading, tints, and cinematic color descriptions than fullres,
#   because the low-resolution stages lock in the global color palette and
#   overall vibe (e.g., golden hour) before fine detail is added.
#
# DESIGN
#   10 prompts, all centered on cinematic color / lighting descriptions.
#   Each prompt is run twice (same seed):
#     - fullres (baseline)
#     - progressive dct_rewind L1 δ=0.05 (best-quality progressive config)
#   Side-by-side montage per prompt for manual visual inspection.
#
# USAGE
#   bash scratch/test_quality_benchmark.sh [--steps N] [--seed N]
#
# OUTPUT
#   scratch/results/quality_YYYYMMDD_HHMMSS/
#     prompt_{00..09}_fullres.png
#     prompt_{00..09}_prog.png
#     montage.png          (2-column: fullres | prog, one row per prompt)
#     prompts.txt          (prompt index → text)
# =============================================================================
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

FLUX_MODEL="/miele/brian/modelscope/black-forest-labs/FLUX.1-dev"
STEPS=50
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --seed)  SEED="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRATCH_DIR/results/quality_${TS}"
mkdir -p "$RESULTS_DIR"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Results: $RESULTS_DIR"
echo "Config: steps=$STEPS seed=$SEED"
echo ""

# ---------------------------------------------------------------------------
# 10 prompts — all emphasize cinematic color, tint, or lighting mood.
# These test whether progressive generation (which locks in global color
# palette at low resolution before adding detail) produces richer color
# grading than standard fullres generation.
# ---------------------------------------------------------------------------
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

# Write prompt list for reference
{
    for i in "${!PROMPTS[@]}"; do
        printf "%02d  %s\n" "$i" "${PROMPTS[$i]}"
    done
} > "$RESULTS_DIR/prompts.txt"

# ---------------------------------------------------------------------------
# Helper: generate one image
# ---------------------------------------------------------------------------
gen_image() {
    local label="$1"
    local prompt="$2"
    shift 2
    local outfile="$RESULTS_DIR/${label}.png"
    local logfile="$RESULTS_DIR/${label}.log"
    echo "  → $label"
    sglang generate \
        --model-path "$FLUX_MODEL" \
        --prompt "$prompt" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        "$@" \
        2>&1 | tee "$logfile" | grep -E "generated successfully|ERROR|error" || true
}

# ---------------------------------------------------------------------------
# Main loop: each prompt → fullres + progressive
# ---------------------------------------------------------------------------
for i in "${!PROMPTS[@]}"; do
    idx=$(printf "%02d" "$i")
    prompt="${PROMPTS[$i]}"
    echo ""
    echo "── Prompt $idx ──────────────────────────────────────────────────"
    echo "   $prompt"

    gen_image "prompt_${idx}_fullres" "$prompt"
    gen_image "prompt_${idx}_prog" "$prompt" \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta 0.05
done

# ---------------------------------------------------------------------------
# Montage: 2-column grid (fullres | prog), one row per prompt
# ---------------------------------------------------------------------------
echo ""
echo "Building montage..."
montage_inputs=()
for i in "${!PROMPTS[@]}"; do
    idx=$(printf "%02d" "$i")
    montage_inputs+=("$RESULTS_DIR/prompt_${idx}_fullres.png" "$RESULTS_DIR/prompt_${idx}_prog.png")
done

if command -v montage &>/dev/null; then
    montage "${montage_inputs[@]}" \
        -geometry 512x512+4+4 \
        -tile 2x \
        -label '%f' \
        "$RESULTS_DIR/montage.png" 2>/dev/null || true
    echo "Montage: $RESULTS_DIR/montage.png"
else
    echo "(imagemagick not available — skip montage)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  QUALITY BENCHMARK COMPLETE                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Results: $RESULTS_DIR"
echo "Prompts: $RESULTS_DIR/prompts.txt"
echo ""
echo "Visual inspection guide:"
echo "  - Compare fullres vs prog side by side per prompt"
echo "  - Progressive hypothesis: prog locks in color palette at low-res stage,"
echo "    producing more saturated / consistent color grading on descriptions"
echo "    like 'golden hour', 'neon', 'amber', 'cinematic color grading'."
echo "  - If prog images show stronger color coherence: hypothesis confirmed."
echo "  - Check for artifacts or detail loss in prog (DCT ringing, aliasing)."
