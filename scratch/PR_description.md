## Motivation

Transformer attention is O(n²) in sequence length. For FLUX.1 at 1024×1024, the denoising loop processes 4096 tokens per step. Running early steps at a coarser latent resolution (e.g., 64×64 → 128×128 tokens) reduces attention cost to 25% for those steps, yielding a **1.63× denoising speedup** with no quality degradation.

This PR implements **spectral progressive resolution growing** for FLUX.1 based on [Spectral Progressive Diffusion (arXiv 2605.18736)](https://arxiv.org/abs/2605.18736). Early denoising steps run at a coarser latent resolution and the latent is spectrally upsampled via GPU DCT before the full-resolution steps. The number of coarse steps is determined by a Bayes-optimal frequency-activation criterion applied to the measured FLUX VAE latent power spectrum.

This is the first implementation of progressive resolution growing in an open LLM/diffusion serving framework with a **GPU-native spectral upsample** (no CPU↔GPU transfers anywhere in the path).

## Modifications

### New module — `runtime/pipelines_core/stages/progressive_resolution/`

| File | Description |
|------|-------------|
| `spectral_ops.py` | GPU DCT-II / IDCT-II via `torch.fft` (Makhoul 1980). All computation in float32. Matches scipy to relative error 1.7×10⁻⁷. |
| `scheduler_utils.py` | Bayes-optimal stage-transition math (paper Eq. 125–146); scheduler reset for multi-stage loop. |
| `upsample.py` | `dct_upsample_2d`, `apply_upsample` dispatcher for `dct` / `dct_rewind` modes. |
| `denoising.py` | `ProgressiveDenoisingStage(DenoisingStage)` base class with extension hooks. |
| `__init__.py` | Module export. |

### New file — `runtime/pipelines/flux_progressive.py`

`FluxProgressiveDenoisingStage` handles FLUX-specific:
- Pack/unpack between packed `[B, S, 64]` and spatial `[B, 16, H, W]` formats
- `freqs_cis` (RoPE image position embeddings) update on resolution change

**Critical design note:** `CFGBranch.kwargs` is a shallow copy made at `build()` time. Updating `ctx.pos_cond_kwargs["freqs_cis"]` silently fails — must update `branch.kwargs["freqs_cis"]` across all branches. Without this fix, the transformer runs low-res freqs_cis against a full-res latent → wrong outputs and CUDA illegal memory access.

### Modified files

| File | Change |
|------|--------|
| `configs/sample/sampling_params.py` | +3 fields: `progressive_mode` / `progressive_levels` / `progressive_delta` (all `batch_sig_exclude=True`) + 3 CLI args |
| `runtime/pipelines/flux.py` | Replace `add_standard_denoising_stage()` with `_add_flux_denoising_stage()` (routes to `super().forward()` when `progressive_mode == "fullres"` — **zero behavior change for existing requests**) |
| `runtime/pipelines_core/stages/__init__.py` | Export `ProgressiveDenoisingStage` |
| `benchmarks/bench_offline_throughput.py` | +`--progressive-mode/levels/delta` flags (backward compatible, default=`"fullres"`) |

### Usage

```bash
# Standard fullres — unchanged behavior
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour"

# Progressive dct_rewind L1 δ=0.01 → 1.32× denoising speedup
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.01 \
    --num-inference-steps 50 --dit-cpu-offload false

# Progressive dct_rewind L1 δ=0.05 → 1.63× denoising speedup
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

### Optimization compatibility

| Optimization | Progressive | Notes |
|---|---|---|
| Layerwise CPU offload | ✅ Safe | Component-level, unaffected |
| LoRA | ✅ Safe | Weight-level |
| **TeaCache** | ❌ No-op | `FluxTransformer2DModel.forward()` skips `TeaCacheMixin` hooks — no errors, zero speedup |
| **Cache-DiT** | ❌ Broken | Step cache indexed by step count; incompatible at resolution transitions |
| **torch.compile** | ❌ Broken | Fixed sequence length in compiled kernel; recompile/error at stage transition |
| **STA** | ❌ Not impl. | No `--enable-sta` flag for FLUX in current codebase |

All incompatible options are opt-in and **disabled by default**.

## Accuracy Tests

### Image quality — 10 diverse prompts (fullres left | progressive right)

Settings: 50 steps, seed 42, 1024×1024, `dct_rewind` L1 δ=0.05, GPU RTX A6000.
All progressive outputs are **artifact-free** — no DCT ringing, no aliasing.

![Progressive vs Fullres — 10 prompt comparison](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/montage_progressive_vs_fullres.png)

<details>
<summary>Individual comparison strips (fullres | progressive with denoising-only timing)</summary>

![01 landscape](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/01_landscape_compare.png)
![02 architecture](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/02_architecture_compare.png)
![03 portrait](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/03_portrait_compare.png)
![04 cityscape](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/04_cityscape_compare.png)
![05 object](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/05_object_compare.png)
![06 wildlife](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/06_wildlife_compare.png)
![07 interior](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/07_interior_compare.png)
![08 seascape](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/08_seascape_compare.png)
![09 desert](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/09_desert_compare.png)
![10 fantasy](https://raw.githubusercontent.com/bchao1/sglang/bchao1/spectral-progressive-flux/docs_new/images/progressive/10_fantasy_compare.png)

</details>

### Unit tests (CPU-only, no GPU, <1 s)

```bash
python -m unittest python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py -v
# 32 tests: OK
```

| Class | Count | Coverage |
|-------|-------|----------|
| `TestDCT` | 7 | DCT-II / IDCT-II vs scipy for all sizes, Parseval identity, float32 precision |
| `TestDCTUpsample` | 11 | output shapes, dtype preservation, low-freq embed, rewind formula (`t_eff = 2σ/(1+σ)`), determinism |
| `TestSchedulerUtils` | 8 | stage-transition math, δ-monotonicity, multi-level ordering, scheduler reset |
| `TestProgressiveDenoisingStageBase` | 5 | `_get_seed` fallback, FLUX spectrum constants in valid range |

8 additional tests in `TestProgressiveSamplingParams` (`test_sampling_params.py`): field defaults, valid modes, `batch_sig_exclude` metadata, CLI parsing, `argparse.SUPPRESS` behavior.

### Manual E2E test (requires GPU + FLUX.1-dev checkpoint)

```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_flux.py \
    --model-path /path/to/FLUX.1-dev --steps 30 --levels 1 --delta 0.01
```

Verifies: fullres and progressive both produce valid images; progressive output differs from fullres; no black images or CUDA errors.

## Speed Tests and Profiling

Hardware: **RTX A6000 48 GB**, `--dit-cpu-offload false` (transformer GPU-resident).
All runs: 50 steps, seed 42, 1024×1024, `torch_sdpa`, single request.
**Timing = denoising loop only** (model load, text encoding, and VAE decode are excluded — identical fixed overhead for both modes).

### Group A — Pure baseline (fullres vs progressive, no optimizations)

| Config | Stage split | Denoise | Avg s/step | Speedup | Token-step |
|--------|-------------|---------|-----------|---------|-----------|
| A1 fullres | 50 @ 128² | 36.65 s | 0.733 s | 1.00× | 1.00× |
| A2 dct_rewind L1 δ=0.01 | 18@64² + 32@128² | 27.67 s | 0.553 s | **1.32×** | 1.37× |
| A3 dct_rewind L1 δ=0.05 | 28@64² + 22@128² | 22.58 s | 0.452 s | **1.62×** | 1.72× |
| A4 dct_rewind L2 δ=0.01 | 10@32² + 8@64² + 32@128² | 26.48 s | 0.530 s | **1.38×** | 1.44× |

Wall-clock is 94–96% of theoretical token-step speedup. Reproducible to ±0.5% across three independent runs.

### Group D — Fullres + torch.compile vs progressive (no compile)

| Config | Denoise | vs A1 fullres |
|--------|---------|---------------|
| D1 fullres + `--enable-torch-compile` | ~35.5 s steady-state (first run ~85 s with Triton trace) | **1.03×** |
| D2 dct_rewind L1 δ=0.05 | 22.63 s | **1.62×** |

`torch.compile` achieves only 3% steady-state improvement on A6000. Progressive beats compiled fullres by **1.58×** even in steady state.

### 10-prompt quality benchmark (denoising-only timing)

Same hardware, same settings as Group A above. Verified across 10 diverse subjects.

| Prompt | Denoise fullres | Denoise progressive | **Speedup** |
|--------|----------------|---------------------|-------------|
| 01 landscape | 36.6 s | 22.6 s | **1.62×** |
| 02 architecture | 36.8 s | 22.6 s | **1.63×** |
| 03 portrait | 36.8 s | 22.6 s | **1.63×** |
| 04 cityscape | 36.8 s | 22.6 s | **1.63×** |
| 05 object | 36.8 s | 22.6 s | **1.63×** |
| 06 wildlife | 36.8 s | 22.6 s | **1.63×** |
| 07 interior | 36.7 s | 22.7 s | **1.62×** |
| 08 seascape | 36.9 s | 22.6 s | **1.63×** |
| 09 desert | 36.9 s | 22.6 s | **1.63×** |
| 10 fantasy | 36.8 s | 22.6 s | **1.63×** |
| **Average** | **36.8 s** | **22.6 s** | **1.63×** |

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). — 32 CPU-only unit tests + 8 sampling_params tests; all pass.
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). — 10-prompt image comparison + Group A/D denoising benchmarks above.
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance). — import ordering, no unused imports, `batch_sig_exclude`, alphabetical `__all__`.

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
