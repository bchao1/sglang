#!/usr/bin/env bash
# =============================================================================
# 720p Smoke Test — fullres vs δ=0.01 vs δ=0.05, single Chinese prompt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scratch/select_gpu.sh"

WAN_MODEL="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
STEPS=50
SEED=42
HEIGHT=720
WIDTH=1280
NUM_FRAMES=81
GUIDANCE_SCALE=5.0
FLOW_SHIFT=5.0
LEVELS=1

PROMPT="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。"
NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results/smoke_720p_${TS}"
mkdir -p "$RESULTS_DIR"
TIMING_LOG="$RESULTS_DIR/timing.tsv"
echo -e "run_id\ttotal_s\tdenoise_s\tavg_step_s\ttransition_step" > "$TIMING_LOG"

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $WAN_MODEL"
echo "Results: $RESULTS_DIR"
echo "Resolution: ${HEIGHT}x${WIDTH} (progressive auto-aligns to 704x1280)"
echo "Config: steps=$STEPS seed=$SEED frames=$NUM_FRAMES guidance=$GUIDANCE_SCALE flow_shift=$FLOW_SHIFT"
echo ""

# ---------------------------------------------------------------------------
# run_gen <label> <extra_flags...>
# ---------------------------------------------------------------------------
run_gen() {
    local label="$1"; shift
    local outfile="$RESULTS_DIR/${label}.mp4"
    local logfile="$RESULTS_DIR/${label}.log"
    echo ""
    echo "─── $label ──────────────────────────────────────────────────"

    conda run -n genAI --no-capture-output sglang generate \
        --model-path "$WAN_MODEL" \
        --prompt "$PROMPT" \
        --negative-prompt "$NEGATIVE_PROMPT" \
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
        "$@" \
        2>&1 | tee "$logfile"

    local clean_log total_s denoise_s avg_s trans_step
    clean_log=$(sed 's/\x1b\[[0-9;]*m//g' "$logfile")
    total_s=$(echo "$clean_log" | grep -oP "generated successfully in \K[\d.]+" || echo "NA")
    avg_s=$(echo "$clean_log"   | grep -oP "average time per step: \K[\d.]+"    || echo "NA")
    if denoise_done=$(echo "$clean_log" | grep -oP "Progressive denoising done in \K[\d.]+" 2>/dev/null | head -1); then
        denoise_s="$denoise_done"
    elif [[ "$avg_s" != "NA" ]]; then
        denoise_s=$(echo "scale=2; $avg_s * $STEPS" | bc -l)
    else
        denoise_s="NA"
    fi
    trans_step=$(echo "$clean_log" | grep -oP "Stage \d/\d: \d+x\d+ latent, steps \[0, \K\d+" 2>/dev/null | head -1 || echo "NA")
    echo -e "${label}\t${total_s}\t${denoise_s}\t${avg_s}\t${trans_step}" >> "$TIMING_LOG"
    echo "  ✓  total=${total_s}s  denoise=${denoise_s}s  avg=${avg_s}s/step  transition_step=${trans_step}"
}

# =============================================================================
# R1: fullres baseline
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  R1  Fullres baseline 720p                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
run_gen "R1_fullres"

# =============================================================================
# R2: progressive dct_rewind δ=0.01
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  R2  Progressive dct_rewind L${LEVELS} δ=0.01                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
run_gen "R2_prog_L${LEVELS}_d0.01" \
    --progressive-mode dct_rewind \
    --progressive-levels "$LEVELS" \
    --progressive-delta 0.01

# =============================================================================
# R3: progressive dct_rewind δ=0.05
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  R3  Progressive dct_rewind L${LEVELS} δ=0.05                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
run_gen "R3_prog_L${LEVELS}_d0.05" \
    --progressive-mode dct_rewind \
    --progressive-levels "$LEVELS" \
    --progressive-delta 0.05

# =============================================================================
# Smoke Tests
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SMOKE TESTS                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
PASS=0; FAIL=0
smoke_check() {
    local name="$1"; local cond="$2"
    if eval "$cond"; then
        echo "  PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"
        FAIL=$((FAIL + 1))
    fi
}

# All three videos exist and are non-empty
for label in "R1_fullres" "R2_prog_L${LEVELS}_d0.01" "R3_prog_L${LEVELS}_d0.05"; do
    f="$RESULTS_DIR/${label}.mp4"
    smoke_check "video exists: $label"        "[[ -f '$f' ]]"
    smoke_check "video non-empty: $label"     "[[ -s '$f' ]]"
done

# Progressive runs should be faster than fullres (denoise_s)
R1_denoise=$(grep "^R1_fullres" "$TIMING_LOG" | awk -F'\t' '{print $3}')
R2_denoise=$(grep "^R2_prog"   "$TIMING_LOG" | grep "d0.01" | awk -F'\t' '{print $3}')
R3_denoise=$(grep "^R3_prog"   "$TIMING_LOG" | grep "d0.05" | awk -F'\t' '{print $3}')

if [[ "$R1_denoise" != "NA" && -n "$R1_denoise" ]]; then
    [[ "$R2_denoise" != "NA" && -n "$R2_denoise" ]] && \
        smoke_check "δ=0.01 faster than fullres" \
            "[[ \$(echo \"$R2_denoise < $R1_denoise\" | bc -l) -eq 1 ]]"
    [[ "$R3_denoise" != "NA" && -n "$R3_denoise" ]] && \
        smoke_check "δ=0.05 faster than fullres" \
            "[[ \$(echo \"$R3_denoise < $R1_denoise\" | bc -l) -eq 1 ]]"
    [[ "$R2_denoise" != "NA" && "$R3_denoise" != "NA" && -n "$R2_denoise" && -n "$R3_denoise" ]] && \
        smoke_check "δ=0.05 faster than δ=0.01 (coarser start → more steps at low-res)" \
            "[[ \$(echo \"$R3_denoise < $R2_denoise\" | bc -l) -eq 1 ]]"
fi

# Transition step sanity: δ=0.01 should transition earlier than δ=0.05
R2_trans=$(grep "^R2_prog" "$TIMING_LOG" | grep "d0.01" | awk -F'\t' '{print $5}')
R3_trans=$(grep "^R3_prog" "$TIMING_LOG" | grep "d0.05" | awk -F'\t' '{print $5}')
if [[ "$R2_trans" != "NA" && "$R3_trans" != "NA" && -n "$R2_trans" && -n "$R3_trans" ]]; then
    smoke_check "δ=0.01 transitions earlier than δ=0.05" \
        "[[ $R2_trans -lt $R3_trans ]]"
fi

echo ""
echo "Smoke result: ${PASS} passed, ${FAIL} failed"

# =============================================================================
# Timing Summary
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TIMING SUMMARY                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
column -t -s $'\t' "$TIMING_LOG"

if [[ -n "$R1_denoise" && "$R1_denoise" != "NA" ]]; then
    echo ""
    echo "Speedup vs fullres DiT loop (${R1_denoise}s):"
    printf "  %-38s  %8s  %9s  %10s  %8s\n" "run_id" "total_s" "denoise_s" "trans_step" "speedup"
    printf "  %-38s  %8s  %9s  %10s  %8s\n" "------" "-------" "---------" "----------" "-------"
    while IFS=$'\t' read -r run_id total denoise avg trans; do
        [[ "$run_id" == "run_id" || "$denoise" == "NA" || "$denoise" == "denoise_s" ]] && continue
        speedup=$(echo "scale=2; $R1_denoise / $denoise" | bc 2>/dev/null || echo "?")
        printf "  %-38s  %8.1f  %9.2f  %10s  %7sx\n" "$run_id" "${total:-0}" "${denoise:-0}" "$trans" "$speedup"
    done < "$TIMING_LOG"
fi

echo ""
echo "Videos:"
for f in "$RESULTS_DIR"/*.mp4; do
    size=$(du -h "$f" | awk '{print $1}')
    echo "  $f  ($size)"
done
echo ""

[[ $FAIL -eq 0 ]] || { echo "SMOKE TESTS FAILED ($FAIL failures)"; exit 1; }
echo "All smoke tests passed."
