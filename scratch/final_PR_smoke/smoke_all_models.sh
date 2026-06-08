#!/usr/bin/env bash
# Unified smoke test: progressive resolution growing — all models.
#
# Runs fullres baseline + dct_rewind L1 delta=0.05 for each model:
#   FLUX.1-dev, FLUX.2-klein-4B, Z-Image, Wan 2.1 T2V 1.3B, Qwen-Image
#
# Results saved to scratch/final_PR_smoke/results/
# Timings logged to scratch/final_PR_smoke/timings.log
#
# Usage:
#   bash scratch/final_PR_smoke/smoke_all_models.sh
#   bash scratch/final_PR_smoke/smoke_all_models.sh --steps 20  # quick debug run
set -euo pipefail

SCRATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

MODELS_DIR="/miele/brian/modelscope"
RESULTS_DIR="$(dirname "$0")/results"
TIMINGS_LOG="$(dirname "$0")/timings.log"
mkdir -p "$RESULTS_DIR"

PROMPT="A serene mountain lake at golden hour, photorealistic"
SEED=42
IMAGE_STEPS=50
VIDEO_STEPS=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)  IMAGE_STEPS="$2"; VIDEO_STEPS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --seed)   SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Config: image_steps=$IMAGE_STEPS  video_steps=$VIDEO_STEPS  seed=$SEED  GPU=$CUDA_VISIBLE_DEVICES"
echo "Prompt: $PROMPT"
echo "Results: $RESULTS_DIR"
echo ""

# Clear/init timings log
echo "# Smoke test timings — $(date)" > "$TIMINGS_LOG"
echo "# GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}")" >> "$TIMINGS_LOG"
echo "label,wall_secs" >> "$TIMINGS_LOG"

run_gen() {
    local label="$1"
    local ext="${2:-png}"
    shift 2
    local outfile="$RESULTS_DIR/${label}.${ext}"
    echo ""
    echo "=== $label ==="
    echo "  output: $outfile"
    local t0
    t0=$(date +%s%N)
    sglang generate \
        --output-file-path "$outfile" \
        --seed "$SEED" \
        "$@"
    local t1
    t1=$(date +%s%N)
    local elapsed
    elapsed=$(python3 -c "print(f'{($t1-$t0)/1e9:.2f}')")
    echo "  -> saved: $outfile  (wall time: ${elapsed}s)"
    echo "${label},${elapsed}" >> "$TIMINGS_LOG"
}

# ============================================================================
# FLUX.1-dev  (image, 50 steps, 1024×1024)
# ============================================================================
FLUX1_MODEL="$MODELS_DIR/black-forest-labs/FLUX.1-dev"

run_gen "flux1_fullres" "png" \
    --model-path "$FLUX1_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$IMAGE_STEPS" \
    --dit-cpu-offload false

run_gen "flux1_dct_rewind_d0.05" "png" \
    --model-path "$FLUX1_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$IMAGE_STEPS" \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# ============================================================================
# FLUX.2-klein-4B  (image, 30 steps, 1024×1024)
# ============================================================================
FLUX2_MODEL="$MODELS_DIR/black-forest-labs/FLUX.2-klein-4B"
FLUX2_STEPS=30

run_gen "flux2_fullres" "png" \
    --model-path "$FLUX2_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$FLUX2_STEPS" \
    --dit-cpu-offload false

run_gen "flux2_dct_rewind_d0.05" "png" \
    --model-path "$FLUX2_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$FLUX2_STEPS" \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# ============================================================================
# Z-Image  (image, 50 steps, 1024×1024)
# ============================================================================
ZIMAGE_MODEL="$MODELS_DIR/Tongyi-MAI/Z-Image"

run_gen "zimage_fullres" "png" \
    --model-path "$ZIMAGE_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$IMAGE_STEPS" \
    --dit-cpu-offload false

run_gen "zimage_dct_rewind_d0.05" "png" \
    --model-path "$ZIMAGE_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$IMAGE_STEPS" \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# ============================================================================
# Wan 2.1 T2V 1.3B  (video, 50 steps, 480×832, 81 frames)
# ============================================================================
WAN_MODEL="$MODELS_DIR/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_PROMPT="A cheetah sprinting across the Serengeti at sunset, slow motion, photorealistic"

run_gen "wan_fullres" "mp4" \
    --model-path "$WAN_MODEL" \
    --prompt "$WAN_PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$VIDEO_STEPS" \
    --num-frames 81 \
    --height 480 \
    --width 832 \
    --guidance-scale 5.0 \
    --flow-shift 5.0 \
    --dit-cpu-offload false

run_gen "wan_dct_rewind_d0.05" "mp4" \
    --model-path "$WAN_MODEL" \
    --prompt "$WAN_PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$VIDEO_STEPS" \
    --num-frames 81 \
    --height 480 \
    --width 832 \
    --guidance-scale 5.0 \
    --flow-shift 5.0 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# ============================================================================
# Qwen-Image  (image, 30 steps, 1024×1024)
# NOTE: may OOM at final decoding stage; --dit-cpu-offload false is required
# for accurate denoising timing but can be removed if VRAM is insufficient.
# ============================================================================
QWEN_MODEL="$MODELS_DIR/Qwen/Qwen-Image"
QWEN_STEPS=30

run_gen "qwen_fullres" "png" \
    --model-path "$QWEN_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$QWEN_STEPS" \
    --dit-cpu-offload false

run_gen "qwen_dct_rewind_d0.05" "png" \
    --model-path "$QWEN_MODEL" \
    --prompt "$PROMPT" \
    --attention-backend torch_sdpa \
    --num-inference-steps "$QWEN_STEPS" \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "===================================================================="
echo "Smoke test complete. Results in: $RESULTS_DIR"
echo "Timings log: $TIMINGS_LOG"
echo ""
cat "$TIMINGS_LOG"
echo ""
echo "All output files:"
ls -lh "$RESULTS_DIR"
