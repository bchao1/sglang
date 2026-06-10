<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.io to discuss further. -->

## Motivation

Extends Spectral Progressive Diffusion (see [merged PR — FLUX.1/2/Z-Image/Wan/Qwen](https://github.com/sgl-project/sglang/pull/27524)) to **Ideogram 4**, adding coarse-to-fine denoising with GPU-based DCT spectral upsample.

Ideogram 4 has a dual-transformer architecture: a conditional transformer (text + image tokens) and an unconditional transformer (image tokens only, zero LLM features). Both must be refreshed at the stage transition, which required a new `_refresh_cache_dit_context` hook in the progressive base class.

## Modifications

| File | Change |
|------|--------|
| `progressive_resolution/ideogram.py` | New: `Ideogram4ProgressiveDenoisingStage` — all 5 extension hooks + dual-transformer Cache-DiT refresh |
| `progressive_resolution/denoising.py` | Added `_refresh_cache_dit_context` hook to support models with multiple transformers |
| `runtime/pipelines/ideogram.py` | `_create_denoising_stage` → `_Ideogram4DenoisingStageRouter` (zero overhead at `progressive_mode="fullres"`) |

New fixes:
| File | Change |
|------|--------|
| `progressive_resolution/flux.py` | Module-level `_flux_pack`/`_flux_unpack` aliases (import fix for existing tests) |
| `progressive_resolution/qwen_image.py` | Module-level `_qwen_image_pack`/`_qwen_image_unpack` aliases (import fix) |

## Accuracy Tests

Visual quality comparison (1024×1024, 48-step, RTX A6000):

<details>
<summary>fullres vs dct_rewind δ=0.01 and δ=0.05 — smoke prompt</summary>

Prompt: *"A serene mountain lake at golden hour, photorealistic"*

| fullres | δ=0.01 | δ=0.05 |
|---------|--------|--------|
| ![fullres](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-ideogram/results/delta_sweep_48step_20260609_091051/Ideogram4_fullres.png) | ![d0.01](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-ideogram/results/delta_sweep_48step_20260609_091051/Ideogram4_dct_rewind_d0_01.png) | ![d0.05](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-ideogram/results/delta_sweep_48step_20260609_091051/Ideogram4_dct_rewind_d0_05.png) |

</details>

## Speed Tests and Profiling

Hardware: RTX A6000 48 GB, `torch_sdpa`, `--dit-cpu-offload false`. Timing = denoising loop only.
Latent: 64×64 = 4096 tokens at 1024×1024 (`_latent_scale_factor = patch_size × ae_scale_factor = 16`).

### Ideogram 4 — 20-step

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 20 @ 64² | 53.99 s | 1.00× |
| dct_rewind L1 δ=0.01 | 6@32² + 14@64² | 43.47 s | **1.24×** |
| dct_rewind L1 δ=0.02 | 7@32² + 13@64² | 41.69 s | **1.30×** |
| dct_rewind L1 δ=0.05 | 9@32² + 11@64² | 38.14 s | **1.42×** |
| dct_rewind L1 δ=0.10 | 11@32² + 9@64² | 34.60 s | **1.56×** |

### Ideogram 4 — 48-step

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 48 @ 64² | 130.92 s | 1.00× |
| dct_rewind L1 δ=0.01 | 12@32² + 36@64² | 109.79 s | **1.19×** |
| dct_rewind L1 δ=0.02 | 16@32² + 32@64² | 102.67 s | **1.28×** |
| dct_rewind L1 δ=0.05 | 21@32² + 27@64² | 93.83 s | **1.40×** |
| dct_rewind L1 δ=0.10 | 26@32² + 22@64² | 84.94 s | **1.54×** |

### Speedup vs δ

![Ideogram 4 speedup vs delta](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-ideogram/speedup_ideogram4_combined.png)

### Usage

```bash
sglang generate \
    --model-path ideogram-ai/ideogram-4 \
    --prompt "A serene mountain lake at golden hour, photorealistic" \
    --height 1024 --width 1024 \
    --num-inference-steps 20 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05
```

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). — `pre-commit run --all-files` passes.
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). — Added `TestIdeogram4LatentAdapters` (pack/unpack roundtrip, row-major, FLUX.2 consistency) and `TestIdeogram4OnResolutionChange` (token doubling, text invariance, grid coordinates) to `test_progressive.py`.
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). — Ideogram 4 section to be added to `docs_new/docs/sglang-diffusion/progressive_resolution.mdx`.
- [x] Provide accuracy and speed benchmark results — speed tables for 20-step and 48-step, visual quality comparison above.
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
