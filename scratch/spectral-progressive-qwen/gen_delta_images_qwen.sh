#!/usr/bin/env bash
# Generate progressive images at a specified delta for all 10 PR prompts.
# Also captures fullres images if --fullres flag is set.
#
# Usage:
#   bash gen_delta_images_qwen.sh <delta> <out_dir>
#   bash gen_delta_images_qwen.sh fullres <out_dir>   # generate fullres baseline
set -euo pipefail

DELTA="${1:?Usage: gen_delta_images_qwen.sh <delta|fullres> <out_dir>}"
OUT_DIR="${2:?Usage: gen_delta_images_qwen.sh <delta|fullres> <out_dir>}"

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"
mkdir -p "$OUT_DIR"

QWEN_MODEL="/miele/brian/modelscope/Qwen/Qwen-Image"
STEPS=30
SEED=42
HEIGHT=1024
WIDTH=1024

PROMPTS=(
    "01_landscape|Golden hour over misty mountain peaks, dramatic cinematic light, photorealistic"
    "02_architecture|Gothic cathedral interior with stained glass windows, volumetric light shafts, 8K"
    "03_portrait|Close-up portrait of an elderly woman with weathered skin, natural outdoor light, film grain"
    "04_cityscape|Hong Kong neon-lit street at night, rain reflections, teal and magenta, cinematic"
    "05_object|Macro photograph of a vintage pocket watch on velvet, bokeh, studio lighting"
    "06_wildlife|Snow leopard mid-leap in Himalayan blizzard, motion blur, National Geographic style"
    "07_interior|Minimalist Japanese tea room, morning light, cherry blossoms through paper screen door"
    "08_seascape|Aerial view of turquoise sea over coral reef, Maldives, drone shot, midday sun"
    "09_desert|Arizona slot canyon, swirling sandstone walls, single beam of orange sunlight from above"
    "10_fantasy|Ancient floating islands with waterfalls, lush forests, misty atmosphere, concept art"
)

TIMING_JSON="$OUT_DIR/timing.json"
echo "[" > "$TIMING_JSON"
FIRST=1

for entry in "${PROMPTS[@]}"; do
    label="${entry%%|*}"
    prompt="${entry#*|}"

    if [[ "$DELTA" == "fullres" ]]; then
        outfile="$OUT_DIR/${label}_fullres.png"
        suffix_label="fullres"
        extra_args=()
    else
        outfile="$OUT_DIR/${label}_progressive.png"
        suffix_label="delta=$DELTA"
        extra_args=(
            --progressive-mode dct_rewind
            --progressive-levels 1
            --progressive-delta "$DELTA"
        )
    fi

    echo ""
    echo "--- $label ($suffix_label) ---"
    local_t0=$(date +%s%3N)
    log=$(sglang generate \
        --model-path "$QWEN_MODEL" \
        --prompt "$prompt" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --height "$HEIGHT" --width "$WIDTH" \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        --save-output \
        "${extra_args[@]}" 2>&1)
    local_t1=$(date +%s%3N)
    wall_s=$(echo "scale=2; $(( local_t1 - local_t0 )) / 1000" | bc)

    denoise_s=""
    if echo "$log" | grep -q "Progressive denoising done in"; then
        denoise_s=$(echo "$log" | grep -oP "Progressive denoising done in \K[0-9.]+" | head -1)
    fi
    if [[ -z "$denoise_s" ]]; then
        denoise_s=$(echo "$log" | grep -oP "generated successfully in \K[0-9.]+" | head -1 || echo "")
    fi

    echo "  Wall: ${wall_s}s  Denoise: ${denoise_s:-N/A}s  -> $outfile"

    if [[ "$FIRST" -eq 0 ]]; then echo "," >> "$TIMING_JSON"; fi
    FIRST=0
    printf '  {"label": "%s", "delta": "%s", "wall_s": %s, "denoise_s": %s}' \
        "$label" "$DELTA" "$wall_s" "${denoise_s:-null}" >> "$TIMING_JSON"
done

echo "" >> "$TIMING_JSON"
echo "]" >> "$TIMING_JSON"
echo ""
echo "Done: $OUT_DIR"
ls -lh "$OUT_DIR"/*.png | awk '{print $5, $9}'
