#!/usr/bin/env bash
# =============================================================================
# Qwen-Image progressive resolution — full experiment suite
# =============================================================================
# Step 1: Group A benchmark (fullres + δ=0.05, 0.10, 0.20) timing
# Step 2: Delta sweep (δ=0.05, 0.10, 0.20, 0.50) for speedup plot
# Step 3: 10-prompt images at δ=0.05 (likely best quality)
# Step 4: 10-prompt images at δ=0.10 (second candidate)
# Step 5: Fullres baseline 10-prompt images
# Step 6: Generate speedup plot + 3-way comparisons
#
# Usage:
#   bash scratch/spectral-progressive-qwen/run_all_experiments.sh
#   bash scratch/spectral-progressive-qwen/run_all_experiments.sh --steps 30
#
# Individual steps can be run separately after interruption.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRATCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRATCH_DIR/select_gpu.sh"

STEPS=30
SKIP_BENCHMARK=0
SKIP_SWEEP=0
SKIP_IMAGES=0
SKIP_PLOTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)           STEPS="$2";         shift 2 ;;
        --skip-benchmark)  SKIP_BENCHMARK=1;   shift ;;
        --skip-sweep)      SKIP_SWEEP=1;       shift ;;
        --skip-images)     SKIP_IMAGES=1;      shift ;;
        --skip-plots)      SKIP_PLOTS=1;       shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# ── Step 1: Group A benchmark ─────────────────────────────────────────────────
if [[ "$SKIP_BENCHMARK" -eq 0 ]]; then
    echo ""
    echo "================================================================"
    echo "STEP 1: Group A benchmark (fullres + δ=0.05, 0.10, 0.20)"
    echo "================================================================"
    bash "$SCRIPT_DIR/run_qwen_benchmark.sh" --steps "$STEPS"
fi

# ── Step 2: Delta sweep ───────────────────────────────────────────────────────
if [[ "$SKIP_SWEEP" -eq 0 ]]; then
    echo ""
    echo "================================================================"
    echo "STEP 2: Delta sweep (δ=0.05, 0.10, 0.20, 0.50)"
    echo "================================================================"
    bash "$SCRIPT_DIR/run_qwen_delta_sweep.sh"
fi

# ── Step 3-5: 10-prompt images ────────────────────────────────────────────────
if [[ "$SKIP_IMAGES" -eq 0 ]]; then
    D05_DIR="$RESULTS_DIR/pr_images"
    D10_DIR="$RESULTS_DIR/pr_images_d10"
    FULLRES_DIR="$RESULTS_DIR/pr_images"   # fullres goes into same dir as d0.05

    echo ""
    echo "================================================================"
    echo "STEP 3: 10-prompt images — δ=0.05"
    echo "================================================================"
    mkdir -p "$D05_DIR"
    bash "$SCRIPT_DIR/gen_delta_images_qwen.sh" 0.05 "$D05_DIR"

    echo ""
    echo "================================================================"
    echo "STEP 4: 10-prompt images — δ=0.10"
    echo "================================================================"
    mkdir -p "$D10_DIR"
    bash "$SCRIPT_DIR/gen_delta_images_qwen.sh" 0.10 "$D10_DIR"

    echo ""
    echo "================================================================"
    echo "STEP 5: Fullres baseline 10 prompts"
    echo "================================================================"
    # Fullres goes into pr_images alongside d0.05 images (for 3-way assembly)
    bash "$SCRIPT_DIR/gen_delta_images_qwen.sh" fullres "$D05_DIR"
fi

# ── Step 6: Plots and comparisons ─────────────────────────────────────────────
if [[ "$SKIP_PLOTS" -eq 0 ]]; then
    echo ""
    echo "================================================================"
    echo "STEP 6: Speedup plot + 3-way comparison strips"
    echo "================================================================"
    conda run -n genAI python3 "$SCRIPT_DIR/gen_speedup_plot_qwen.py"
    conda run -n genAI python3 "$SCRIPT_DIR/gen_3way_comparison_qwen.py"
fi

echo ""
echo "================================================================"
echo "All experiments complete."
echo "Results: $RESULTS_DIR"
echo "Visuals: $SCRIPT_DIR/pr_visuals/"
echo "================================================================"
