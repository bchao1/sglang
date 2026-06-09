# Spectral Progressive Diffusion — Ideogram 4

Extends the spectral progressive resolution framework to Ideogram 4, enabling
coarse-to-fine denoising with DCT-II spectral upsample between stages.

## Model Overview

**Ideogram 4** uses a dual-transformer architecture:
- `transformer` — conditional (processes text + image tokens jointly)
- `unconditional_transformer` — unconditional (image tokens only, zero LLM features)

Latent format: `[B, grid_h * grid_w, in_channels=128]` (row-major, same as FLUX.2)
where `grid_h = height // 16`, `grid_w = width // 16` (patch_size=2 × ae_scale_factor=8)

Scheduler: `Ideogram4Scheduler` — logit-normal noise schedule with resolution-aware
mean shift (`get_schedule_for_resolution`). Does NOT use `scheduler.sigmas` for stage
transitions (uses its own schedule_values array). The base progressive stage reads
`scheduler.sigmas` for stage transition thresholds — so `Ideogram4Scheduler` must
provide a sigmas tensor. See implementation notes.

## Implementation

### New file: `progressive_resolution/ideogram.py`

`Ideogram4ProgressiveDenoisingStage(ProgressiveDenoisingStage, Ideogram4DenoisingStage)`

Multiple inheritance with explicit `DenoisingStage.__init__` call to bypass the
incompatible parent signatures.

**Hooks implemented:**
| Hook | What it does |
|------|-------------|
| `_latent_scale_factor` | Returns `patch_size * ae_scale_factor = 16` |
| `_unpack_latent` | `[B, S, C] → [B, C, H, W]` (row-major, identical to FLUX.2) |
| `_repack_latent` | `[B, C, H, W] → [B, S, C]` (row-major) |
| `_generate_initial_noise` | `in_channels=128` directly, float32 (no //4), row-major pack |
| `_on_resolution_change` | Rebuilds `position_ids`, `segment_ids`, `indicator`, attn masks, `neg_llm_features` for new grid |
| `_refresh_cache_dit_context` | Refreshes both `transformer` AND `unconditional_transformer` |

**_on_resolution_change design:**
- Text portion (first `max_text_tokens` positions) is invariant across resolution changes
- Only the image tail grows: new grid position IDs are `[h*W+c → (t=0, h, w)] + IMAGE_POSITION_OFFSET`
- Patches `batch.extra["ideogram4"]` and `ctx.extra` in-place

### Modified files

`denoising.py` — Added `_refresh_cache_dit_context` hook to `ProgressiveDenoisingStage`
so that subclasses with multiple transformers (like Ideogram) can refresh all of them.

`flux.py`, `qwen_image.py` — Added `_flux_pack`/`_flux_unpack` and
`_qwen_image_pack`/`_qwen_image_unpack` module-level aliases to fix broken imports
in the pre-existing test file.

`ideogram.py` (pipeline) — `create_pipeline_stages` now wraps denoising in a
`ProgressiveDenoisingStageRouter` via `_create_denoising_stage`. Non-progressive
requests (progressive_mode="fullres") zero-overhead to `Ideogram4DenoisingStage`;
progressive requests dispatch to `Ideogram4ProgressiveDenoisingStage`.

### Tests

`test_progressive.py` — Added:
- `TestIdeogram4LatentAdapters`: pack/unpack roundtrip + row-major order + matches FLUX.2
- `TestIdeogram4OnResolutionChange`: token count doubling, text invariance, grid coordinates

## Spectrum Constants (placeholder)

```
IDEOGRAM_SPECTRUM_A = 203.615097
IDEOGRAM_SPECTRUM_BETA = 1.915461
```

Using FLUX.1-dev VAE values as placeholder until Ideogram-specific coefficients are
fitted from Aesthetics-Train-V2 or a comparable dataset of Ideogram-decoded latents.

## Benchmark Results

TODO: run bench once GPU smoke tests confirm correctness.

Expected: ~1.5–2× denoise speedup at progressive_delta=0.05 (based on FLUX.1 results).
