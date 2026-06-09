#!/usr/bin/env bash
# =============================================================================
# Ideogram 4 delta sweep benchmark — 48-step V4_QUALITY_48 preset
#
# Runs fullres baseline + dct_rewind L1 for deltas 0.01, 0.02, 0.05, 0.1
# using the V4_QUALITY_48 preset (48 denoising steps, std=1.5).
#
# Extracts ONLY the DIT denoising loop time (not VAE decode, not full wall time).
#
# Output: results/delta_sweep_48step_<TS>/results.json  (read by gen_speedup_plot.py)
#
# Usage (from repo root):
#   source scratch/select_gpu.sh
#   bash scratch/spectral-progressive-ideogram/bench_ideogram_48step.sh
# =============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=$(bash scratch/select_gpu.sh 2>/dev/null || echo "0")

MODEL_PATH="/miele/brian/modelscope/ideogram-4-fp8"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/delta_sweep_48step_${TS}"
mkdir -p "$RESULTS_DIR"

PROMPTS_JSON="/home/brianchc/ideogram4/experiments/prompts_snow_leopard.json"
PROMPT=$(python3 -c "
import json, sys
with open('$PROMPTS_JSON') as f:
    data = json.load(f)
caption = data[0]['caption']
print(json.dumps(caption))
")
SEED=42
DELTAS=(0.01 0.02 0.05 0.1)
STEPS=48
PRESET="V4_QUALITY_48"

# Write a temporary config file so sglang picks up the preset
CFG_FILE="$RESULTS_DIR/preset_cfg.json"
echo "{\"preset\": \"$PRESET\"}" > "$CFG_FILE"

JSON_OUT="$RESULTS_DIR/results.json"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo "Prompt: $PROMPT"
echo "Preset: $PRESET ($STEPS steps)"
echo "Deltas: ${DELTAS[*]}"
echo ""

LAST_DENOISE_S=""

# ---------------------------------------------------------------------------
# run_gen <label> <logfile> <extra_flags...>
# Sets LAST_DENOISE_S from the sglang output log.
# ---------------------------------------------------------------------------
run_gen() {
    local label="$1"
    local logfile="$2"
    shift 2

    echo ""
    echo "─── $label ──────────────────────────────────────────────────────"
    sglang generate \
        --model-path "$MODEL_PATH" \
        --config "$CFG_FILE" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --attention-backend torch_sdpa \
        --dit-cpu-offload false \
        "$@" \
        2>&1 | tee "$logfile"

    local clean_log
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")

    # Progressive mode → "Progressive denoising done in X.XXs"
    local denoise_s=""
    if denoise_s=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" | head -1 2>/dev/null) && [[ -n "$denoise_s" ]]; then
        :
    else
        # Fullres → "average time per step: X.XXXX seconds"
        local avg_s
        avg_s=$(echo "$clean_log" | grep -oP "average time per step: \K[\d.]+" | head -1 || echo "")
        if [[ -n "$avg_s" ]]; then
            denoise_s=$(python3 -c "print(f'{$avg_s * $STEPS:.2f}')")
        fi
    fi

    LAST_DENOISE_S="${denoise_s:-}"
    echo "  → denoise_loop=${LAST_DENOISE_S:-N/A}s"
}

# ---------------------------------------------------------------------------
# Fullres baseline
# ---------------------------------------------------------------------------
run_gen "Ideogram4_fullres_48" \
    "$RESULTS_DIR/Ideogram4_fullres.log" \
    --output-file-path "$RESULTS_DIR/Ideogram4_fullres.png"
FULLRES_DENOISE="$LAST_DENOISE_S"

# Open JSON
printf '{\n  "Ideogram 4 (48-step)": {\n    "steps": %d,\n    "fullres_denoise_s": %s,\n    "points": [\n' \
    "$STEPS" "${FULLRES_DENOISE:-null}" > "$JSON_OUT"

# ---------------------------------------------------------------------------
# dct_rewind sweeps
# ---------------------------------------------------------------------------
FIRST_POINT=1
for delta in "${DELTAS[@]}"; do
    dlabel="${delta//./_}"
    label="Ideogram4_dct_rewind_d${dlabel}"
    logfile="$RESULTS_DIR/${label}.log"

    run_gen "$label" "$logfile" \
        --output-file-path "$RESULTS_DIR/${label}.png" \
        --progressive-mode dct_rewind \
        --progressive-levels 1 \
        --progressive-delta "$delta"

    speedup="null"
    if [[ -n "$FULLRES_DENOISE" && -n "$LAST_DENOISE_S" ]]; then
        speedup=$(python3 -c "print(f'{$FULLRES_DENOISE / $LAST_DENOISE_S:.4f}')")
    fi

    if [[ "$FIRST_POINT" -eq 0 ]]; then printf ",\n" >> "$JSON_OUT"; fi
    FIRST_POINT=0
    printf '      {"delta": %s, "denoise_s": %s, "speedup": %s}' \
        "$delta" "${LAST_DENOISE_S:-null}" "$speedup" >> "$JSON_OUT"
done

printf "\n    ]\n  }\n}\n" >> "$JSON_OUT"

echo ""
echo "======================================================================"
echo "BENCHMARK COMPLETE"
echo "Results: $RESULTS_DIR"
echo "JSON:    $JSON_OUT"
echo ""
cat "$JSON_OUT"
