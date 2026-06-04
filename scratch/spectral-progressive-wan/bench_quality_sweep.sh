#!/usr/bin/env bash
# =============================================================================
# Wan T2V Quality Sweep — 10 cinematic prompts × 4 configs (fullres + 3 deltas)
# 720P · 81 frames · 50 steps · guidance=5.0 · flow_shift=5.0
# Dispatches across 3 free GPUs (0, 8, 9) in parallel.
# Each GPU handles its prompt slice sequentially; all 3 GPUs run concurrently.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WAN_MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
STEPS=50
SEED=42
HEIGHT=720
WIDTH=1280
NUM_FRAMES=81
GUIDANCE_SCALE=5.0
FLOW_SHIFT=5.0
LEVELS=1
DELTAS=(fullres 0.01 0.02 0.05)
PROMPTS_FILE="$SCRIPT_DIR/quality_prompts.txt"
FREE_GPUS=(0 8 9)

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/quality_sweep_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "prompt_id\tconfig\ttotal_s\tdenoise_s\ttransition_step\tgpu" > "$TIMING_LOG"

echo "Results: $RESULTS_DIR"
echo "GPUs: ${FREE_GPUS[*]}"
echo "Resolution: ${HEIGHT}x${WIDTH}, ${NUM_FRAMES} frames, ${STEPS} steps"
echo ""

# Read prompts into array
mapfile -t PROMPTS < "$PROMPTS_FILE"
N_PROMPTS=${#PROMPTS[@]}
N_GPUS=${#FREE_GPUS[@]}

# ──────────────────────────────────────────────────────────────────────────────
# Per-GPU worker function: runs all configs for each assigned prompt
# ──────────────────────────────────────────────────────────────────────────────
run_gpu_worker() {
    local gpu_id="$1"
    local -a prompt_indices=("${@:2}")
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    # Each GPU gets a unique port range to avoid EADDRINUSE collisions
    local base_port=$((30100 + gpu_id * 10))

    echo "[GPU $gpu_id] Starting: prompts ${prompt_indices[*]} (base_port=$base_port)"

    for pidx in "${prompt_indices[@]}"; do
        local prompt="${PROMPTS[$pidx]}"
        local pid_label=$(printf "p%02d" "$((pidx + 1))")
        echo "[GPU $gpu_id] Prompt $pid_label: ${prompt:0:60}..."

        for config in "${DELTAS[@]}"; do
            local label="${pid_label}_${config}"
            local outfile="$RESULTS_DIR/${label}.mp4"
            local logfile="$RESULTS_DIR/${label}.log"

            if [[ "$config" == "fullres" ]]; then
                local extra_flags=()
            else
                local extra_flags=(
                    --progressive-mode dct_rewind
                    --progressive-levels "$LEVELS"
                    --progressive-delta "$config"
                )
            fi

            echo "[GPU $gpu_id] Running $label..."
            sglang generate \
                --model-path "$WAN_MODEL" \
                --prompt "$prompt" \
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
                --master-port "$base_port" \
                "${extra_flags[@]}" \
                > "$logfile" 2>&1

            local clean_log total_s denoise_s trans_step
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

            # Atomic append to shared timing log
            flock "$TIMING_LOG" \
                echo -e "${pid_label}\t${config}\t${total_s}\t${denoise_s}\t${trans_step}\tGPU${gpu_id}" \
                >> "$TIMING_LOG"

            echo "[GPU $gpu_id] ✓ $label: total=${total_s}s denoise=${denoise_s}s trans=${trans_step}"
        done
    done
    echo "[GPU $gpu_id] All prompts done."
}

# ──────────────────────────────────────────────────────────────────────────────
# Distribute prompts round-robin across GPUs
# ──────────────────────────────────────────────────────────────────────────────
declare -A GPU_PROMPTS
for gpu in "${FREE_GPUS[@]}"; do
    GPU_PROMPTS[$gpu]=""
done

for i in "${!PROMPTS[@]}"; do
    gpu_slot="${FREE_GPUS[$((i % N_GPUS))]}"
    GPU_PROMPTS[$gpu_slot]+=" $i"
done

echo "Prompt assignment:"
for gpu in "${FREE_GPUS[@]}"; do
    echo "  GPU $gpu: prompts${GPU_PROMPTS[$gpu]}"
done
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Launch all GPU workers in parallel
# ──────────────────────────────────────────────────────────────────────────────
PIDS=()
for gpu in "${FREE_GPUS[@]}"; do
    read -ra indices <<< "${GPU_PROMPTS[$gpu]}"
    run_gpu_worker "$gpu" "${indices[@]}" &
    PIDS+=($!)
    echo "Launched GPU $gpu worker (pid ${PIDS[-1]})"
done

echo ""
echo "All workers running. Waiting for completion..."

# Wait for all workers and collect exit codes
ALL_OK=true
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Worker pid $pid failed!"
        ALL_OK=false
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  QUALITY SWEEP COMPLETE                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Results: $RESULTS_DIR"
echo ""
echo "Timing:"
column -t -s $'\t' "$TIMING_LOG"
echo ""
echo "Videos: $RESULTS_DIR/*.mp4 ($(ls "$RESULTS_DIR"/*.mp4 2>/dev/null | wc -l) files)"

if $ALL_OK; then
    echo "All runs completed successfully."
else
    echo "Some runs failed — check *.log files in $RESULTS_DIR"
    exit 1
fi
