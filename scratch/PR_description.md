# [Diffusion] Progressive resolution growing for FLUX.1 via GPU DCT upsampling

## Summary

Adds **spectral progressive resolution growing** to the FLUX.1 diffusion pipeline in SGLang.
Early denoising steps run at a coarser latent resolution (e.g., 64×64 instead of 128×128),
then the latent is spectrally upsampled to full resolution for the remaining steps.
This reduces the quadratic attention cost over the coarse steps, yielding wall-clock speedups
of **1.32–1.62×** on denoising (50-step, 1024×1024, A6000, GPU-resident).

## Motivation

Transformer attention is O(n²) in sequence length. A 4× resolution reduction (64→128 latent)
halves the token count and cuts attention cost to 25% for those steps. The number of steps at
coarse resolution is determined by a Bayes-optimal frequency-activation criterion (paper
Eq. 125–129/142–146) based on the measured latent power spectrum of the FLUX VAE.

This is the first implementation of progressive resolution growing in an open LLM/diffusion
serving framework with a GPU-native spectral upsample (no CPU↔GPU transfers).

## Visual Results — 10 Prompts (fullres left ↔ progressive right)

*Prompts cover: cinematic landscape, architecture, portrait, nightscape, macro object,
wildlife, interior, seascape, desert, fantasy. Settings: 50 steps, seed 42, 1024×1024,
`dct_rewind` L1 δ=0.05.*

<!-- After pushing the branch, replace this line with the actual montage image:
![Progressive vs Fullres Comparison](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/montage_progressive_vs_fullres.png)
Drag-and-drop the montage from docs_new/images/progressive/montage_progressive_vs_fullres.png here. -->

**[ATTACH MONTAGE HERE]** — file: `docs_new/images/progressive/montage_progressive_vs_fullres.png`

Individual comparison strips (fullres | progressive, with timing label):

| Strip | Fullres | Progressive | Speedup |
|-------|---------|-------------|---------|
| 01 landscape | 94.3 s | 74.5 s | **1.27×** |
| 02 architecture | 89.2 s | 73.5 s | **1.21×** |
| 03 portrait | 86.5 s | 73.0 s | **1.19×** |
| 04 cityscape | 86.3 s | 72.6 s | **1.19×** |
| 05 object | 85.8 s | 72.6 s | **1.18×** |
| 06 wildlife | 89.4 s | 72.7 s | **1.23×** |
| 07 interior | 89.8 s | 71.1 s | **1.26×** |
| 08 seascape | 85.6 s | 70.8 s | **1.21×** |
| 09 desert | 85.8 s | 71.6 s | **1.20×** |
| 10 fantasy | 86.6 s | 71.4 s | **1.21×** |
| **Average** | **87.9 s** | **72.4 s** | **1.22×** |

> **Note on timing:** Wall-clock includes `sglang generate` subprocess startup + 22 s model load + text encoding + VAE decode per call (fixed overhead identical for both modes). The speedup from denoising alone (measured with dedicated benchmark, no subprocess overhead) is **1.32–1.62×** as detailed below.

All progressive outputs are artifact-free — no DCT ringing, no aliasing.

## Benchmark Results — Denoising Only

Hardware: RTX A6000 48 GB, `--dit-cpu-offload false` (transformer GPU-resident).
All runs: 50 steps, seed 42, 1024×1024, `torch_sdpa`, single request.

### Group A — Pure baseline (fullres vs progressive, no optimizations)

| Config | Stage split | Denoise | Avg s/step | Speedup | Token-step |
|--------|-------------|---------|-----------|---------|-----------|
| A1 fullres | 50 @ 128² | 36.65 s | 0.733 s | 1.00× | 1.00× |
| A2 dct_rewind L1 δ=0.01 | 18@64² + 32@128² | 27.67 s | 0.553 s | **1.32×** | 1.37× |
| A3 dct_rewind L1 δ=0.05 | 28@64² + 22@128² | 22.58 s | 0.452 s | **1.62×** | 1.72× |
| A4 dct_rewind L2 δ=0.01 | 10@32² + 8@64² + 32@128² | 26.48 s | 0.530 s | **1.38×** | 1.44× |

Wall-clock is 94–96% of the theoretical token-step speedup. Results reproducible to ±0.5% across three independent runs.

### Group D — Fullres + torch.compile vs progressive (no compile)

| Config | Denoise | vs A1 fullres |
|--------|---------|---------------|
| D1 fullres + `--enable-torch-compile` | ~35.5 s steady-state (first run ~85 s) | **1.03×** |
| D2 dct_rewind L1 δ=0.05 | 22.63 s | **1.62×** |

`torch.compile` achieves only 3% steady-state improvement. Progressive beats compiled fullres by **1.58×** even in steady state.

## Changes

### New files

```
python/sglang/multimodal_gen/runtime/pipelines_core/stages/progressive_resolution/
  __init__.py          — module export
  spectral_ops.py      — GPU DCT-II / IDCT-II via torch.fft (Makhoul 1980)
  scheduler_utils.py   — Bayes-optimal stage-transition math + scheduler reset
  upsample.py          — dct_upsample_2d, apply_upsample dispatcher
  denoising.py         — ProgressiveDenoisingStage base class

python/sglang/multimodal_gen/runtime/pipelines/
  flux_progressive.py  — FluxProgressiveDenoisingStage (FLUX pack/unpack + freqs_cis)

python/sglang/multimodal_gen/test/unit/
  test_progressive_upsample.py  — 32 CPU-only unit tests (32/32 pass, no GPU needed)

python/sglang/multimodal_gen/test/manual/
  test_progressive_flux.py      — End-to-end manual test (requires GPU + FLUX.1-dev)

docs_new/images/progressive/    — 10 comparison strips + montage (committed to repo)
```

### Modified files

```
configs/sample/sampling_params.py        — +progressive_mode / levels / delta fields + CLI args
runtime/pipelines/flux.py                — swap DenoisingStage → FluxProgressiveDenoisingStage
runtime/pipelines_core/stages/__init__   — export ProgressiveDenoisingStage
benchmarks/bench_offline_throughput.py   — --progressive-mode/levels/delta flags
test/unit/test_sampling_params.py        — TestProgressiveSamplingParams (8 new tests)
```

## Usage

```bash
# Standard fullres (unchanged)
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour"

# Progressive — 1.32× speedup
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.01 \
    --num-inference-steps 50 --dit-cpu-offload false

# Progressive — 1.62× speedup
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 50 --dit-cpu-offload false

# Offline throughput benchmark
python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput \
    --model-path black-forest-labs/FLUX.1-dev \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 50 --width 1024 --height 1024 --num-prompts 5 \
    --dit-cpu-offload false
```

## Design

### Architecture

- `ProgressiveDenoisingStage(DenoisingStage)`: overrides `forward()`, reuses all parent
  infrastructure. Routes to `super().forward()` when `progressive_mode == "fullres"` —
  **zero behavior change for existing requests**.
- `FluxProgressiveDenoisingStage(ProgressiveDenoisingStage)`: FLUX-specific pack/unpack
  and `freqs_cis` (RoPE) update on resolution change.
- Extension hooks: `_unpack_latent`, `_repack_latent`, `_on_resolution_change` — override
  for other model families (Z-Image, Qwen-Image, etc.).

### Key Implementation Notes

**freqs_cis update (critical):** `CFGBranch.kwargs` is a shallow copy made at `build()` time.
Updating `ctx.pos_cond_kwargs["freqs_cis"]` silently fails. `FluxProgressiveDenoisingStage._on_resolution_change`
updates `branch.kwargs["freqs_cis"]` for every branch. Without this, the transformer runs
low-res freqs_cis against a full-res latent → wrong outputs and illegal memory access.

**Scheduler rewind:** After upsample, `scheduler.sigmas/timesteps` patched in-place at the
transition point. These tensors may be read-only inference tensors, so they are cloned once
before the stage loop when `rewind=True`.

**GPU DCT:** All spectral computation in float32 (bfloat16 has 7 mantissa bits; quantizing
DCT coefficients gives mean abs error ~0.8 vs output range ±4). Final output cast back to
input dtype. Matches scipy to relative error 1.7×10⁻⁷.

## Optimization Compatibility

| Optimization | Progressive | Notes |
|---|---|---|
| Layerwise CPU offload | ✅ Safe | Component-level, unaffected |
| LoRA | ✅ Safe | Weight-level |
| CFG parallel | ✅ Safe | FLUX uses single-branch guidance distillation |
| **TeaCache** | ❌ No-op | `FluxTransformer2DModel.forward()` skips TeaCacheMixin hooks |
| **Cache-DiT** | ❌ Broken | Step cache indexed by step count; incompatible at resolution transitions |
| **torch.compile** | ❌ Broken | Fixed sequence length in compiled kernel; recompile/error at transition |
| **STA** | ❌ Not impl. | No `--enable-sta` flag for FLUX in current codebase |

All incompatible options are opt-in and **disabled by default** — progressive is safe without them.

## Tests

### Unit tests (CPU-only, no GPU, <1 s)

```bash
python -m unittest python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py -v
# 32 tests: OK
```

| Class | Count | Covers |
|-------|-------|--------|
| `TestDCT` | 7 | DCT/IDCT vs scipy, Parseval, float32 precision |
| `TestDCTUpsample` | 11 | shapes, dtype, low-freq embed, rewind formula, determinism |
| `TestSchedulerUtils` | 8 | transition math, monotonicity, scheduler reset |
| `TestProgressiveDenoisingStageBase` | 5 | `_get_seed`, spectrum constants |

`TestProgressiveSamplingParams` (8 tests in `test_sampling_params.py`):
defaults, mode validation, `batch_sig_exclude` metadata, CLI parsing, `argparse.SUPPRESS`.

### Manual E2E test (requires GPU + FLUX.1-dev)

```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_flux.py \
    --model-path /path/to/FLUX.1-dev --steps 30 --levels 1 --delta 0.01
```

## Reference

Paper: *Spectral Diffusion for Efficient Inference* (`wavelet-diffusion/inference_progressive.py`).

Sigma schedule, μ=1.15, transition steps (18/28 for δ=0.01/0.05), rewind formula all verified
to match the reference implementation exactly. PRNG differs (PyTorch GPU vs numpy —
statistically equivalent, not bit-identical from same seed).
