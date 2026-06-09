# Spectral Progressive Diffusion — Ideogram 4

Extends the spectral progressive resolution framework to Ideogram 4.

## Status

**Branch:** `bchao1/spectral-progressive-ideogram`
**Base:** `bchao1/spectral-progressive-flux` (extends the 5-model progressive PR)

## Model-specific challenges

**Dual-transformer architecture:**
- `transformer` — conditional (processes text + image tokens jointly)
- `unconditional_transformer` — unconditional (image tokens only, zero LLM features)
- Both must be cache-refreshed at stage transitions — requires a new
  `_refresh_cache_dit_context` hook in `ProgressiveDenoisingStage` base class.

**Scheduler (`Ideogram4Scheduler`):**
- Logit-normal noise schedule with resolution-aware mean shift (`get_schedule_for_resolution`).
- Does NOT use `scheduler.sigmas` natively; the progressive base reads `scheduler.sigmas`
  for stage transitions, so `Ideogram4Scheduler` must synthesize a `sigmas` tensor from
  its internal `schedule_values` array. Done in `_on_resolution_change` setup.

**Position IDs / attention masks:**
- Text tokens (first `max_text_tokens` positions) are invariant across resolution changes.
- Image tail grows: new grid positions are `[h*W + w → (t=0, h, w)] + IMAGE_POSITION_OFFSET`.
- `_on_resolution_change` rebuilds `position_ids`, `segment_ids`, `indicator`, attention
  masks, and `neg_llm_features` for the new grid size.

## Files changed

| File | Change |
|------|--------|
| `progressive_resolution/ideogram.py` | New: `Ideogram4ProgressiveDenoisingStage` |
| `progressive_resolution/denoising.py` | Added `_refresh_cache_dit_context` hook (multi-transformer support) |
| `runtime/pipelines/ideogram.py` | Added `_create_denoising_stage` → `ProgressiveDenoisingStageRouter` |
| `progressive_resolution/flux.py` | Added module-level `_flux_pack`/`_flux_unpack` aliases (import fix) |
| `progressive_resolution/qwen_image.py` | Added `_qwen_image_pack`/`_qwen_image_unpack` aliases (import fix) |

## Spectrum constants

```python
IDEOGRAM_SPECTRUM_A    = 203.615097
IDEOGRAM_SPECTRUM_BETA = 1.915461
```

Using FLUX.1-dev VAE values as placeholder (Ideogram-specific fit pending).

## Benchmark Results

Hardware: RTX A6000 48 GB, `torch_sdpa`, no CPU offload.
Latent: 64×64 = 4096 tokens at 1024×1024.

### 20-step (default schedule)

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 20 @ 64² | 53.99 s | 1.00× |
| dct_rewind δ=0.01 | 6@32² + 14@64² | 43.47 s | **1.24×** |
| dct_rewind δ=0.02 | 7@32² + 13@64² | 41.69 s | **1.30×** |
| dct_rewind δ=0.05 | 9@32² + 11@64² | 38.14 s | **1.42×** |
| dct_rewind δ=0.10 | 11@32² + 9@64² | 34.60 s | **1.56×** |

### 48-step

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 48 @ 64² | 130.92 s | 1.00× |
| dct_rewind δ=0.01 | 12@32² + 36@64² | 109.79 s | **1.19×** |
| dct_rewind δ=0.02 | 16@32² + 32@64² | 102.67 s | **1.28×** |
| dct_rewind δ=0.05 | 21@32² + 27@64² | 93.83 s | **1.40×** |
| dct_rewind δ=0.10 | 26@32² + 22@64² | 84.94 s | **1.54×** |

Results from: `results/delta_sweep_20260609_090215/` (20-step) and
`results/delta_sweep_48step_20260609_091051/` (48-step).
