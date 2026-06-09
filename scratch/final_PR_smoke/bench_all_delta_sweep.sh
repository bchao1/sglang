#!/usr/bin/env bash
# =============================================================================
# All-models delta sweep benchmark — DIT denoising speedup only
#
# Runs fullres baseline + dct_rewind L1 for deltas 0.01, 0.02, 0.05, 0.1, 0.2
# for every model on a SINGLE prompt. Extracts ONLY the DIT denoising loop time
# (not VAE decode, not total wall time).
#
# Qwen-Image uses --vae-cpu-offload to keep VAE off the GPU while keeping
# --dit-cpu-offload false so the DiT stays GPU-resident for accurate timing.
#
# Output: results/delta_sweep_<TS>/results.json  (read by gen_delta_sweep_plot.py)
#
# Usage:
#   bash scratch/final_PR_smoke/bench_all_delta_sweep.sh
# =============================================================================
set -euo pipefail

# Force GPU 1 — user requirement: only run on purely empty GPU
export CUDA_VISIBLE_DEVICES=1

MODELS_DIR="/miele/brian/modelscope"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/delta_sweep_${TS}"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
DELTAS=(0.01 0.02 0.05 0.1 0.2)

JSON_OUT="$RESULTS_DIR/results.json"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Results: $RESULTS_DIR"
echo "Prompt: $PROMPT"
echo "Deltas: ${DELTAS[*]}"
echo ""

# JSON accumulator
printf "{\n" > "$JSON_OUT"
FIRST_MODEL=1

# ---------------------------------------------------------------------------
# run_gen <label> <logfile> <steps> <extra_flags...>
# Returns denoise_s via LAST_DENOISE_S env var.
# For fullres: uses "average time per step" * steps.
# For progressive: uses "Progressive denoising done in X".
# ---------------------------------------------------------------------------
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

    # Progressive mode: "Progressive denoising done in X.XXs"
    local denoise_s=""
    if denoise_s=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" | head -1 2>/dev/null) && [[ -n "$denoise_s" ]]; then
        :
    else
        # Fullres: "average time per step: X.XXXX seconds"
        local avg_s
        avg_s=$(echo "$clean_log" | grep -oP "average time per step: \K[\d.]+" | head -1 || echo "")
        if [[ -n "$avg_s" ]]; then
            denoise_s=$(python3 -c "print(f'{$avg_s * $steps:.2f}')")
        fi
    fi

    LAST_DENOISE_S="${denoise_s:-}"
    echo "  → denoise_loop=${LAST_DENOISE_S:-N/A}s"
}

# ---------------------------------------------------------------------------
# bench_model <json_key> <steps> <ext> <common_flags...>
# Runs fullres + all delta values, appends JSON block.
# For Qwen, caller passes --vae-cpu-offload in common_flags.
# ---------------------------------------------------------------------------
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

    # Fullres baseline
    local label_base
    label_base="${json_key//[^a-zA-Z0-9]/_}_fullres"
    run_gen "$label_base" \
        "$RESULTS_DIR/${label_base}.log" \
        "$steps" \
        --output-file-path "$RESULTS_DIR/${label_base}.${ext}" \
        "${common_flags[@]}"
    local fullres_denoise="$LAST_DENOISE_S"

    # JSON: open model block
    if [[ "$FIRST_MODEL" -eq 0 ]]; then printf ",\n" >> "$JSON_OUT"; fi
    FIRST_MODEL=0
    printf '  "%s": {\n    "steps": %d,\n    "fullres_denoise_s": %s,\n    "points": [\n' \
        "$json_key" "$steps" "${fullres_denoise:-null}" >> "$JSON_OUT"

    local first_point=1
    for delta in "${DELTAS[@]}"; do
        local dlabel="${delta//./_}"
        local label_d="${json_key//[^a-zA-Z0-9]/_}_d${dlabel}"
        run_gen "$label_d" \
            "$RESULTS_DIR/${label_d}.log" \
            "$steps" \
            --output-file-path "$RESULTS_DIR/${label_d}.${ext}" \
            --progressive-mode dct_rewind \
            --progressive-levels 1 \
            --progressive-delta "$delta" \
            "${common_flags[@]}"
        local prog_denoise="$LAST_DENOISE_S"

        # Compute speedup
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
# FLUX.1-dev  (50 steps, 1024×1024, no offload)
# ============================================================================
FLUX1_MODEL="$MODELS_DIR/black-forest-labs/FLUX.1-dev"
bench_model "FLUX.1-dev" 50 "png" \
    --model-path "$FLUX1_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --dit-cpu-offload false

# ============================================================================
# FLUX.2-klein-4B  (50 steps, 1024×1024, no offload)
# ============================================================================
FLUX2_MODEL="$MODELS_DIR/black-forest-labs/FLUX.2-klein-4B"
bench_model "FLUX.2-klein-4B" 50 "png" \
    --model-path "$FLUX2_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --dit-cpu-offload false

# ============================================================================
# Z-Image  (50 steps, 1024×1024, no offload)
# ============================================================================
ZIMAGE_MODEL="$MODELS_DIR/Tongyi-MAI/Z-Image"
bench_model "Z-Image" 50 "png" \
    --model-path "$ZIMAGE_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --height 1024 --width 1024 \
    --dit-cpu-offload false

# ============================================================================
# Wan 2.1 T2V 1.3B  (50 steps, 480×832, 81 frames, no offload)
# ============================================================================
WAN_MODEL="$MODELS_DIR/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_PROMPT="$PROMPT"
bench_model "Wan 2.1 T2V 1.3B" 50 "mp4" \
    --model-path "$WAN_MODEL" \
    --prompt "$WAN_PROMPT" \
    --attention-backend torch_sdpa \
    --num-frames 81 \
    --height 480 \
    --width 832 \
    --guidance-scale 5.0 \
    --flow-shift 5.0 \
    --dit-cpu-offload false

# ============================================================================
# Qwen-Image  (50 steps, 1024×1024)
# --vae-cpu-offload: VAE decodes on CPU to save GPU memory
# --dit-cpu-offload false: DiT stays GPU-resident for accurate denoising timing
# ============================================================================
QWEN_MODEL="$MODELS_DIR/Qwen/Qwen-Image"
bench_model "Qwen-Image" 50 "png" \
    --model-path "$QWEN_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --height 1024 --width 1024 \
    --dit-cpu-offload false \
    --vae-cpu-offload true

# Close JSON
printf "\n}\n" >> "$JSON_OUT"

echo ""
echo "======================================================================"
echo "BENCHMARK COMPLETE"
echo "Results: $RESULTS_DIR"
echo "JSON:    $JSON_OUT"
echo ""
cat "$JSON_OUT"
