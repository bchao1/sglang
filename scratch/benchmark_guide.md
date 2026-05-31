# Progressive Resolution Benchmarking Guide

## Purpose
This document explains the benchmark design, how to reproduce results, and how
to interpret them.  Keep it updated as new experiments are added.

---

## Quick Start
```bash
# Full speed benchmark (Groups A + D, 50 steps, seed 42)
bash scratch/test_progressive_benchmark.sh

# Single group
bash scratch/test_progressive_benchmark.sh --group A
bash scratch/test_progressive_benchmark.sh --group D

# Custom steps / seed
bash scratch/test_progressive_benchmark.sh --steps 20 --seed 0

# Color/cinematic quality experiment (10 prompts, fullres vs progressive)
bash scratch/test_quality_benchmark.sh

# Legacy sweep (various progressive configs)
bash scratch/test_progressive_gen.sh --steps 50
```

---

## Experimental Design

### Controlled Variables (same for ALL runs)
| Variable | Value |
|---|---|
| Model | FLUX.1-dev (`/miele/brian/modelscope/black-forest-labs/FLUX.1-dev`) |
| Resolution | 1024×1024 |
| Attention backend | `torch_sdpa` |
| Steps | 50 (default) |
| Seed | 42 (default) |
| DIT offload | `false` — transformer GPU-resident for all runs |
| Text/VAE offload | layerwise (auto, small overhead, not measured) |

**Why disable DIT CPU offload?**  
With offload enabled, every step pays ~0.41 s to PCIe-transfer the 22 GB
transformer regardless of sequence length.  This constant overhead dilutes the
quadratic attention savings that make progressive generation fast, depressing
measured speedup from ~1.6× to ~1.35×.  The paper measures pure compute; we
match that by disabling offload.

### Benchmark Groups

| Group | Config | Purpose |
|---|---|---|
| **A** | Fullres + Progressive, GPU-resident, no opts | Core comparison; mirrors paper setup |
| **D** | Fullres + torch.compile vs Progressive (no compile) | Fairness test: best-effort fullres vs progressive without incompatible opts |

> **Note on omitted groups:**
> - **TeaCache (Groups B/C)** — `FluxTransformer2DModel.forward()` does not call
>   `TeaCacheMixin` hooks.  `--enable-teacache` is a silent no-op for FLUX.1.
>   TeaCache tests omitted until FLUX TeaCache is implemented.
> - **STA (Sliding Tile Attention)** — not implemented for FLUX.1 in the current
>   codebase.  No `--enable-sta` flag exists.

### Configs Within Group A

| ID | Config | Stage split (50 steps) | Expected speedup |
|---|---|---|---|
| A1 | fullres | 50 @ 128² | 1.00× (baseline) |
| A2 | dct_rewind L1 δ=0.01 | 18@64² + 32@128² | ~1.37× (token-step) |
| A3 | dct_rewind L1 δ=0.05 | 28@64² + 22@128² | ~1.72× (token-step) |
| A4 | dct_rewind L2 δ=0.01 | 10@32² + 8@64² + 32@128² | ~1.45× (token-step) |

**Stage split** = how many denoising steps run at each latent resolution.
Transition points are computed from the Bayes-optimal spectrum criterion
(paper Eq. 142-146) applied to FLUX's actual sigma schedule (dynamic shift
μ=1.15 for 1024×1024).

**Token-step speedup** = fullres_token_steps / progressive_token_steps (linear
model of compute).  Actual speedup may differ due to fixed-overhead per step
and quadratic attention for long sequences.

---

## How to Read the Results

### Speedup formula
```
speedup = A1_total_s / run_total_s
```

### Key comparisons
- **A1 vs A2/A3/A4**: Does progressive outperform fullres without any optimization?
- **D1 vs A3**: Does fullres + torch.compile beat progressive (no compile)?
  - If D1 < A3: progressive is competitive even without compile.
  - If D1 > A3: progressive still achieves similar quality at lower cost.
- **D2 == A3**: D2 is a control (same config as A3) run during the D session to
  confirm reproducibility.

---

## Adding New Experiments

1. Add a new `run_gen` call in the appropriate Group in `test_progressive_benchmark.sh`.
2. Or add a new Group section (copy Group C pattern).
3. Results auto-appear in the timing TSV and speedup summary.

Common additions:
```bash
# Flash Attention backend (if available)
run_gen "A_fa_fullres" --attention-backend fa

# Different progressive delta
run_gen "A_prog_L1_d0.02" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.02

# dct (no rewind) vs dct_rewind comparison
run_gen "A_prog_L1_dct_plain" \
    --progressive-mode dct --progressive-levels 1 --progressive-delta 0.01
```

---

## Known Optimization Compatibility

| Optimization | Fullres | Progressive | Notes |
|---|---|---|---|
| **torch.compile** | ✅ | ❌ Broken | Fixed sequence length in compiled kernel; incompatible with stage transition |
| TeaCache | ❌ Not impl. | ❌ Not impl. | `FluxTransformer2DModel.forward()` does not call `TeaCacheMixin` hooks — no-op for FLUX |
| STA (Sliding Tile Attn) | ❌ Not impl. | ❌ Not impl. | No `--enable-sta` flag exists for FLUX in current codebase |
| Cache-Dit | ✅ | ❌ Broken | Step cache indexed by step count; stage-1 short-seq cache incompatible with stage-2 |
| Layerwise offload | ✅ | ✅ | Component-level, unaffected |
| LoRA | ✅ | ✅ | Weight-level, unaffected |

**Implication for benchmarking:** The only optimization that applies to fullres but not
progressive is `torch.compile`.  Group D tests this directly.

---

## Quality Benchmark — Color Grading Hypothesis

**Hypothesis:** Progressive resolution growing may produce better cinematic color
fidelity than fullres because the low-resolution stage(s) lock in the global color
palette and mood before fine detail is added.  A prompt like "golden hour" or
"neon-lit Tokyo" will imprint its color style at 64×64 latent resolution, and
subsequent upsampled stages refine texture within that established palette.

### Prompts (10 total, color/cinematic focus)
See `scratch/test_quality_benchmark.sh` for the full list.  Themes covered:
- Golden hour / warm amber tones
- Twilight rose-gold portraits
- Neon cyberpunk blues / magentas
- Tuscan ochre sunsets
- Arctic blue-hour
- Jazz-club amber tungsten
- Pastel cherry blossom
- Desert dawn orange
- Underwater teal
- Autumn crimson / cadmium orange

### Running
```bash
bash scratch/test_quality_benchmark.sh             # 50 steps, seed 42
bash scratch/test_quality_benchmark.sh --steps 20  # faster draft pass
```

### Evaluation
- Visual comparison of `prompt_NN_fullres.png` vs `prompt_NN_prog.png`
- Progressive wins if: more saturated / coherent color, stronger mood atmosphere
- Fullres wins if: sharper fine detail, less chromatic fringing

---

## Result History

| Date | Steps | A1 (s) | A2 speedup | A3 speedup | A4 speedup | D1 speedup | Notes |
|---|---|---|---|---|---|---|---|
| 2026-05-30 | 20 | 24.73 | 1.28× | 1.37× | 1.26× | — | with DIT offload (biased) |
| 2026-05-31 | 50 | 57.82 | 1.35× | 1.56× | 1.41× | — | with DIT offload (biased) |
| 2026-05-31 | 50 | 36.44 | 1.32× | 1.62× | 1.38× | 0.22×† | GPU-resident A6000, no offload |

†D1 torch.compile: 161.5s first-run (127s Triton autotune), ~35.5s steady-state (3% gain vs A1). Progressive beats compiled fullres by 1.59× even in steady state. Token-step speedup (formula): A2=1.37×, A3=1.72×, A4=1.44×. Wall-clock is 94–96% of theoretical.

---

## DCT Correctness Reference

GPU implementation (torch.fft Makhoul) vs scipy:
- Relative error: **1.7e-7** (when using same noise)
- All intermediate computation in **float32**; only output cast to bfloat16
- RNG: PyTorch vs numpy generates different numbers for same seed (statistically
  equivalent; bit-identical reproduction requires same RNG)

---

## Files
```
scratch/
  test_progressive_benchmark.sh   # Main benchmark script (Groups A + D)
  test_quality_benchmark.sh       # Color/cinematic quality experiment (10 prompts)
  test_progressive_gen.sh         # Legacy sweep script (varies δ/levels)
  benchmark_guide.md              # This file
  results/bench_YYYYMMDD_HHMMSS/  # Per-run outputs (speed benchmark)
  results/quality_YYYYMMDD_HHMMSS/ # Per-run outputs (quality benchmark)
    *.png                         # Generated images
    *.log                         # Full sglang output per run
    timing.tsv                    # run_id / total_s / denoise_s / avg_step_s

python/sglang/multimodal_gen/
  runtime/pipelines_core/stages/progressive_resolution/
    spectral_ops.py     # GPU DCT-II / IDCT-II (torch.fft)
    scheduler_utils.py  # Stage transition math
    upsample.py         # dct_upsample_2d, apply_upsample
    denoising.py        # ProgressiveDenoisingStage base class
  runtime/pipelines/
    flux_progressive.py # FluxProgressiveDenoisingStage + freqs_cis update
    flux.py             # Modified to use FluxProgressiveDenoisingStage
  configs/sample/sampling_params.py  # +progressive_mode/levels/delta
```
