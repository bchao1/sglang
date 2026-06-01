# Z-Image Spectral Progressive Diffusion — SGLang Integration Log

## Status: 🚧 In Progress
**Branch:** `bchao1/spectral-progressive-z-image`
**Base:** `bchao1/spectral-progressive-flux` (extends spectral progressive infrastructure)

---

## Goal
Extend the spectral progressive diffusion framework (DCT rewind) to support Z-Image models
in addition to FLUX.1. Reuse the `ProgressiveDenoisingStage` base class; implement
Z-Image-specific `_unpack_latent` / `_repack_latent` / `_on_resolution_change` hooks.

Reference: `bchao1/spectral-progressive-flux` feature for FLUX.1 implementation.

---

## Design Decisions

### Scope (this PR)
- Z-Image pipeline support via `ZImageProgressiveDenoisingStage`
- Reuse all spectral ops from `progressive_resolution/` (no duplication)
- Identify Z-Image VAE scale factor, patch size, and latent packing format
- Verify transition step formula and rewind logic carry over unchanged

### Architecture
- `ZImageProgressiveDenoisingStage(ProgressiveDenoisingStage)`: implements model-specific hooks
  - `_unpack_latent` / `_repack_latent`: Z-Image latent format (TBD — audit pipeline)
  - `_on_resolution_change`: update pos embeddings / freqs_cis equivalent for Z-Image transformer
- Z-Image pipeline: swap its `DenoisingStage` → `ZImageProgressiveDenoisingStage`
- `SamplingParams`: no new fields needed (reuses `progressive_mode/levels/delta`)

### Files to Create
```
runtime/pipelines/
  z_image_progressive.py  # ZImageProgressiveDenoisingStage (Z-Image-specific hooks)
```

### Files to Modify
```
runtime/pipelines/z_image.py (or equivalent)   # swap DenoisingStage → ZImageProgressiveDenoisingStage
```

---

## Z-Image Architecture Notes

| Parameter | Value | Notes |
|-----------|-------|-------|
| VAE | FluxVAEConfig | **Same VAE as FLUX.1-dev** |
| VAE scale factor | 8 | pixel → latent: H_lat = H_px / 8 |
| Latent channels (C) | 16 | `in_channels = 16 = num_channels_latents` |
| Spatial patch size | 2 | PATCH_SIZE = 2 (H and W) |
| Temporal patch size | 1 | F_PATCH_SIZE = 1 (image = single frame) |
| Latent shape (denoising loop) | [B, C, 1, H_lat, W_lat] | 5-D with frame dim (F=1) |
| Latent dtype | float32 | Z-Image uses fp32 latents (unlike FLUX bfloat16) |
| Pos embedding | RoPE, (cap_freqs_cis, x_freqs_cis) | Tuple: caption + image RoPE positions |
| Scheduler shift | Dynamic mu (same formula as FLUX) | prepare_mu in pipeline |
| CFG | Traditional dual-pass | Should use guidance = True |
| Token count (1024²) | 1 × 64 × 64 = 4096 (same as FLUX) | |

**Key implementation notes:**
1. `_unpack_latent`: squeeze(2) removes F=1 dim → [B, C, H, W] for DCT ops
2. `_repack_latent`: unsqueeze(2) adds F=1 dim → [B, C, 1, H, W]
3. `_generate_initial_noise`: uses `in_channels` directly (=16), NOT `//4`
4. `_on_resolution_change`: recomputes full freqs_cis tuple (no caching — caption-length dependency)
5. freqs_cis is a tuple `(cap_freqs_cis, x_freqs_cis)` — both updated at transition

---

## Spectrum Constants (Z-Image VAE)
Same as FLUX.1-dev — Z-Image uses the same FluxVAEConfig.
Confirmed by reference implementation (inference_progressive.py):
```
A, beta = FLUX_SPECTRUM_A, FLUX_SPECTRUM_BETA  # same VAE as FLUX
```
- A = 203.615097
- beta = 1.915461

---

## Files Created
```
runtime/pipelines/zimage_progressive.py   # ZImageProgressiveDenoisingStage
test/manual/test_progressive_zimage.py    # E2E manual test (needs GPU + model)
```

## Files Modified
```
runtime/pipelines/zimage_pipeline.py     # swap DenoisingStage → ZImageProgressiveDenoisingStage
test/unit/test_progressive_upsample.py  # +TestZImagePackUnpack (10 new tests, CPU-only)
```

## Scratch Scripts
```
scratch/spectral-progressive-z-image/
  test_progressive_zimage_gen.sh         # fullres + 4 progressive configs
  test_progressive_zimage_benchmark.sh   # timing table (Group A equiv)
```

---

## Benchmark Results (TBD — pending GPU run)

Expected speedup formula (token-step, cfg_passes=2, patch_size=1):
- Z-Image has cfg_passes=2 (dual CFG) vs FLUX's 1
- Token-step speedup formula still applies (2 passes cancel out in ratio)
- Baseline expected: same as FLUX (same VAE, same latent size)

---

## Change Log
- 2026-06-01: Branch created from bchao1/spectral-progressive-flux; setup complete
- 2026-06-01: Implementation complete — ZImageProgressiveDenoisingStage, pipeline wiring,
              unit tests (41/41 pass), manual test, scratch scripts
