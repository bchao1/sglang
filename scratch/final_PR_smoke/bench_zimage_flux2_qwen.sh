#!/usr/bin/env bash
# Re-run delta sweep for Z-Image, FLUX.2-klein-4B, and Qwen-Image only.
# FLUX.1-dev and Wan data are taken from run 1 (already correct).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

MODELS_DIR="/miele/brian/modelscope"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/partial_sweep_${TS}"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
DELTAS=(0.01 0.02 0.05 0.1 0.2)

JSON_OUT="$RESULTS_DIR/results.json"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Results: $RESULTS_DIR"
echo ""

printf "{\n" > "$JSON_OUT"
FIRST_MODEL=1
LAST_DENOISE_S=""

run_gen() {
    local label="$1"
    local logfile="$2"
    local steps="$3"
    shift 3

    echo ""
    echo "─── $label ───────────────────────────────────────────────────────"
    sglang generate \
        --num-inference-steps "$steps" \
        --seed "$SEED" \
        "$@" \
        2>&1 | tee "$logfile"

    local clean_log
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")

    local denoise_s=""
    if denoise_s=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" | head -1 2>/dev/null) && [[ -n "$denoise_s" ]]; then
        :
    else
        local avg_s
        avg_s=$(echo "$clean_log" | grep -oP "average time per step: \K[\d.]+" | head -1 || echo "")
        if [[ -n "$avg_s" ]]; then
            denoise_s=$(python3 -c "print(f'{$avg_s * $steps:.2f}')")
        fi
    fi

    LAST_DENOISE_S="${denoise_s:-}"
    echo "  → denoise_loop=${LAST_DENOISE_S:-N/A}s"
}

bench_model() {
    local json_key="$1"
    local steps="$2"
    local ext="$3"
    shift 3
    local common_flags=("$@")

    echo ""
    echo "======================================================================"
    echo "  MODEL: $json_key  (steps=$steps)"
    echo "======================================================================"

    local label_base="${json_key//[^a-zA-Z0-9]/_}_fullres"
    run_gen "$label_base" "$RESULTS_DIR/${label_base}.log" "$steps" \
        --output-file-path "$RESULTS_DIR/${label_base}.${ext}" \
        "${common_flags[@]}"
    local fullres_denoise="$LAST_DENOISE_S"

    if [[ "$FIRST_MODEL" -eq 0 ]]; then printf ",\n" >> "$JSON_OUT"; fi
    FIRST_MODEL=0
    printf '  "%s": {\n    "steps": %d,\n    "fullres_denoise_s": %s,\n    "points": [\n' \
        "$json_key" "$steps" "${fullres_denoise:-null}" >> "$JSON_OUT"

    local first_point=1
    for delta in "${DELTAS[@]}"; do
        local dlabel="${delta//./_}"
        local label_d="${json_key//[^a-zA-Z0-9]/_}_d${dlabel}"
        run_gen "$label_d" "$RESULTS_DIR/${label_d}.log" "$steps" \
            --output-file-path "$RESULTS_DIR/${label_d}.${ext}" \
            --progressive-mode dct_rewind \
            --progressive-levels 1 \
            --progressive-delta "$delta" \
            "${common_flags[@]}"
        local prog_denoise="$LAST_DENOISE_S"

        local speedup="null"
        if [[ -n "$fullres_denoise" && -n "$prog_denoise" ]]; then
            speedup=$(python3 -c "print(f'{$fullres_denoise / $prog_denoise:.4f}')")
        fi

        if [[ "$first_point" -eq 0 ]]; then printf ",\n" >> "$JSON_OUT"; fi
        first_point=0
        printf '      {"delta": %s, "denoise_s": %s, "speedup": %s}' \
            "$delta" "${prog_denoise:-null}" "$speedup" >> "$JSON_OUT"
    done

    printf "\n    ]\n  }" >> "$JSON_OUT"
}

# ============================================================================
# FLUX.2-klein-4B  (50 steps, 1024×1024)
# ============================================================================
bench_model "FLUX.2-klein-4B" 50 "png" \
    --model-path "$MODELS_DIR/black-forest-labs/FLUX.2-klein-4B" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --dit-cpu-offload false

# ============================================================================
# Z-Image  (50 steps, 1024×1024)
# ============================================================================
bench_model "Z-Image" 50 "png" \
    --model-path "$MODELS_DIR/Tongyi-MAI/Z-Image" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --height 1024 --width 1024 \
    --dit-cpu-offload false

# ============================================================================
# Qwen-Image  (50 steps, 1024×1024)
# --vae-cpu-offload offloads the VAE decoder to CPU to avoid OOM.
# The DiT stays GPU-resident (--dit-cpu-offload false) for accurate timing.
# ============================================================================
bench_model "Qwen-Image" 50 "png" \
    --model-path "$MODELS_DIR/Qwen/Qwen-Image" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --height 1024 --width 1024 \
    --dit-cpu-offload false \
    --vae-cpu-offload

printf "\n}\n" >> "$JSON_OUT"

echo ""
echo "======================================================================"
echo "PARTIAL BENCHMARK COMPLETE"
echo "JSON: $JSON_OUT"
cat "$JSON_OUT"
