#!/usr/bin/env bash
# Generate progressive images at a specified delta for all 10 PR prompts.
# Usage: bash gen_delta_images.sh <delta> <out_dir>
set -euo pipefail

DELTA="${1:?Usage: gen_delta_images.sh <delta> <out_dir>}"
OUT_DIR="${2:?Usage: gen_delta_images.sh <delta> <out_dir>}"
mkdir -p "$OUT_DIR"

ZIMAGE_MODEL="/miele/brian/modelscope/Tongyi-MAI/Z-Image"
STEPS=50
SEED=42

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
    outfile="$OUT_DIR/${label}_progressive.png"
    echo ""
    echo "--- $label (delta=$DELTA) ---"
    log=$(CUDA_VISIBLE_DEVICES=4 sglang generate \
        --model-path "$ZIMAGE_MODEL" \
        --prompt "$prompt" \
        --output-file-path "$outfile" \
        --attention-backend torch_sdpa \
        --seed "$SEED" \
        --height 1024 --width 1024 \
        --num-inference-steps "$STEPS" \
        --dit-cpu-offload false \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta "$DELTA" \
        --save-output 2>&1)
    denoise=$(echo "$log" | grep -oP "Progressive denoising done in \K[0-9.]+" | head -1)
    echo "  Denoise: ${denoise:-N/A}s"
    if [[ "$FIRST" -eq 0 ]]; then echo "," >> "$TIMING_JSON"; fi
    FIRST=0
    printf '  {"label": "%s", "denoise_prog": %s}' "$label" "${denoise:-null}" >> "$TIMING_JSON"
done

echo "" >> "$TIMING_JSON"
echo "]" >> "$TIMING_JSON"
echo ""
echo "Done: $OUT_DIR"
ls -lh "$OUT_DIR"/*.png | awk '{print $5, $9}'
