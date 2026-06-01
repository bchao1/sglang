# [Diffusion] Progressive resolution growing for FLUX.1 via GPU DCT upsampling

## Summary

Adds **spectral progressive resolution growing** to the FLUX.1 diffusion pipeline in SGLang.
Early denoising steps run at a coarser latent resolution (e.g., 64×64 instead of 128×128),
then the latent is spectrally upsampled to full resolution for the remaining steps.
This reduces the quadratic attention cost over the coarse steps, yielding wall-clock speedups
of **1.32–1.62×** on FLUX.1-dev 50-step generation at 1024×1024 with no perceptible quality
loss.

## Motivation

Transformer attention is O(n²) in sequence length. A 4× resolution reduction (64→128 latent)
halves the token count and cuts attention cost to 25% for those steps. The number of steps at
coarse resolution is determined by a Bayes-optimal frequency-activation criterion (paper
Eq. 125–129/142–146) based on the latent power spectrum of the FLUX VAE.

This is the first implementation of progressive resolution growing in an open LLM/diffusion
serving framework with a GPU-native spectral upsample (no CPU↔GPU transfers).

## Changes

### New files

```
python/sglang/multimodal_gen/runtime/pipelines_core/stages/progressive_resolution/
  __init__.py          — module export (SPDX header)
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
```

### Modified files

```
configs/sample/sampling_params.py   — +progressive_mode / progressive_levels / progressive_delta
                                      fields (batch_sig_exclude=True) + CLI args

runtime/pipelines/flux.py           — replace add_standard_denoising_stage() with
                                      _add_flux_denoising_stage() (backward-compatible)

runtime/pipelines_core/stages/__init__.py — export ProgressiveDenoisingStage

benchmarks/bench_offline_throughput.py   — --progressive-mode/levels/delta flags

test/unit/test_sampling_params.py   — TestProgressiveSamplingParams (8 new tests)
```

## Usage

### Single-image generation (CLI)

```bash
# Fullres baseline (unchanged behavior)
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour"

# Progressive: 18 low-res steps + 32 full-res steps (1.32× speedup)
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.01 \
    --num-inference-steps 50 --dit-cpu-offload false

# More aggressive: 28 low-res steps + 22 full-res steps (1.62× speedup)
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 50 --dit-cpu-offload false
```

### Python API

```python
from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

gen = DiffGenerator.from_pretrained("black-forest-labs/FLUX.1-dev")
result = gen.generate(sampling_params=SamplingParams(
    prompt="A serene mountain lake at golden hour",
    num_inference_steps=50,
    progressive_mode="dct_rewind",
    progressive_levels=1,
    progressive_delta=0.05,
))
```

### Benchmark

```bash
# Compare modes using the offline throughput benchmark
python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput \
    --model-path black-forest-labs/FLUX.1-dev \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 50 --width 1024 --height 1024 \
    --num-prompts 5 --dit-cpu-offload false
```

## Benchmark Results

Hardware: RTX A6000 48 GB, `--dit-cpu-offload false` (transformer GPU-resident).
All runs: 50 steps, seed 42, 1024×1024, `torch_sdpa`, single request.

### Group A — Pure baseline (fullres vs progressive, no optimizations)

| Config | Stage split | Denoise | Avg s/step | Speedup | Token-step |
|--------|-------------|---------|-----------|---------|-----------|
| A1 fullres | 50 @ 128² | 36.65 s | 0.733 s | 1.00× | 1.00× |
| A2 dct_rewind L1 δ=0.01 | 18@64² + 32@128² | 27.67 s | 0.553 s | **1.32×** | 1.37× |
| A3 dct_rewind L1 δ=0.05 | 28@64² + 22@128² | 22.58 s | 0.452 s | **1.62×** | 1.72× |
| A4 dct_rewind L2 δ=0.01 | 10@32² + 8@64² + 32@128² | 26.48 s | 0.530 s | **1.38×** | 1.44× |

Wall-clock is 94–96% of the theoretical token-step speedup. The ~5% gap is fixed per-step
overhead (scheduler `.step()`, memory alloc) that does not scale with token count.
Results are reproducible to ±0.5% across three independent runs.

### Group D — Fullres + torch.compile vs progressive (no compile)

| Config | Denoise | vs A1 fullres |
|--------|---------|---------------|
| D1 fullres + `--enable-torch-compile` | ~35.5 s steady-state (first run ~85 s) | **1.03×** |
| D2 dct_rewind L1 δ=0.05 | 22.63 s | **1.62×** |

`torch.compile` achieves only 3% steady-state improvement on A6000 (requires ~50 s
first-step Triton graph tracing). Progressive generation beats compiled fullres by
**1.58×** even in steady state, without requiring any compilation.

### Earlier results (H100 80 GB, 20 steps, `--dit-cpu-offload true` — biased by PCIe overhead)

| Config | Denoise | Total | Speedup |
|--------|---------|-------|---------|
| fullres | 22.72 s | 24.73 s | 1.00× |
| dct_rewind L1 δ=0.01 | 17.20 s | 19.27 s | **1.28×** |
| dct_rewind L1 δ=0.05 | 15.91 s | 18.01 s | **1.37×** |
| dct_rewind L2 δ=0.01 | 17.50 s | 19.61 s | **1.26×** |

The lower speedup with CPU offload is expected: each step pays ~0.41 s for PCIe transfer
of the 22 GB transformer, diluting the attention savings.

## Image Quality

<!-- Attach side-by-side montage here: fullres vs progressive for same seed/prompt -->
<!-- Example: scratch/results/quality_20260531_162803/montage.png -->

All progressive outputs are artifact-free (no DCT ringing, no aliasing). Output file sizes
are comparable to fullres (0.99–1.97 MB vs 1.08–1.98 MB, 10-prompt quality benchmark).

Quality benchmark (50 steps, seed 42, 8 complete pairs): fullres avg 39.1 s, progressive
avg 24.7 s → **1.58× speedup**. Progressive images have subtly different compositions
(global scene layout committed early at low resolution) but equivalent or better color
fidelity for strongly spatial-global prompts (e.g., desert mesa: deeper red-orange in
progressive).

## Design

### Architecture

- `ProgressiveDenoisingStage(DenoisingStage)`: overrides `forward()`, reuses all parent
  infrastructure. Routes to `super().forward()` when `progressive_mode == "fullres"`,
  preserving identical behavior for standard requests.
- `FluxProgressiveDenoisingStage(ProgressiveDenoisingStage)`: FLUX-specific pack/unpack
  and `freqs_cis` (RoPE) update on resolution change.
- Extension hooks: `_unpack_latent`, `_repack_latent`, `_on_resolution_change` — can be
  overridden for Z-Image, Qwen-Image, etc.

### GPU DCT

DCT-II via torch.fft using the Makhoul (1980) algorithm. All intermediate computation
stays in float32 to avoid bfloat16 quantization error (~0.8 mean abs error vs ±4 range).
Output is cast back to the input dtype. Matches scipy to relative error 1.7e-7.
No `.cpu()` or `.numpy()` calls anywhere in the path.

### Key Design Notes

**freqs_cis update (critical)**: `CFGBranch.kwargs` is a shallow copy made at `build()` time.
Updating `ctx.pos_cond_kwargs["freqs_cis"]` alone silently fails to reach the transformer.
`FluxProgressiveDenoisingStage._on_resolution_change` updates `branch.kwargs["freqs_cis"]`
directly for every branch. Without this fix, the transformer runs low-res freqs_cis
against a full-res latent → wrong outputs and illegal memory access in the attention kernel.

**Scheduler rewind**: After upsampling, `scheduler.sigmas/timesteps` are patched in-place
at the transition point. These tensors may be created inside `torch.inference_mode` (making
them read-only), so they are cloned once before the stage loop when `rewind=True`.

## Optimization Compatibility

| Optimization | Progressive | Notes |
|---|---|---|
| Layerwise CPU offload | ✅ Safe | Component-level, unaffected |
| LoRA | ✅ Safe | Weight-level |
| CFG parallel | ✅ Safe | FLUX uses single-branch guidance distillation |
| **TeaCache** | ❌ No-op | `FluxTransformer2DModel.forward()` skips TeaCacheMixin hooks — produces no errors but zero speedup |
| **Cache-DiT** | ❌ Broken | Step cache indexed by step count; incompatible at stage transitions |
| **torch.compile** | ❌ Broken | Fixed sequence length in compiled kernel; recompile or error at transition |
| **STA** | ❌ Not impl. | No `--enable-sta` flag exists for FLUX in current codebase |

All incompatible options are opt-in and disabled by default — progressive generation is safe without them.

## Tests

### Unit tests (CPU-only, no GPU required)

```bash
python -m unittest python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py -v
# 32 tests, <1 s
```

**Coverage:**
- `TestDCT` (7 tests): DCT-II/IDCT-II matches scipy (all sizes), Parseval identity, float32 precision
- `TestDCTUpsample` (11 tests): shapes, dtype preservation, low-freq embed, rewind formula, t_eff > σ_t, determinism, error handling
- `TestSchedulerUtils` (8 tests): stage transition math, multi-level ordering, δ monotonicity, scheduler reset
- `TestProgressiveDenoisingStageBase` (5 tests): `_get_seed` fallback logic, FLUX spectrum constants

Additional tests in `test_sampling_params.py` (`TestProgressiveSamplingParams`, 8 tests):
field defaults, valid modes, `batch_sig_exclude` metadata, CLI parsing, `argparse.SUPPRESS` behavior.

### Manual / E2E test (requires GPU + FLUX.1-dev checkpoint)

```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_flux.py \
    --model-path /path/to/FLUX.1-dev \
    --steps 30 --levels 1 --delta 0.01 \
    --output-dir /tmp/progressive_flux_test
```

Verifies: fullres and progressive both produce valid images, progressive output differs
from fullres (actually changes computation), no black images or artifacts.

## Reference

- Paper: "Spectral Diffusion for Efficient Inference" (see `wavelet-diffusion/inference_progressive.py`)
- Sigma schedule, μ=1.15, transition steps (18/28), rewind formula all verified to match
  the reference implementation exactly.
- PRNG differs (PyTorch GPU vs numpy — statistically equivalent, not bit-identical from same seed).
