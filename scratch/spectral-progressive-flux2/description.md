# Progressive Resolution Growing — FLUX.2 Integration Log

## Status: 🚧 Implementation complete — benchmarking pending
**Branch:** `bchao1/spectral-progressive-flux2`
**Base:** `bchao1/spectral-progressive-flux` (extends Flux.1 progressive work)

---

## Goal
Extend spectral progressive denoising (DCT rewind) to the FLUX.2 pipeline in SGLang.
FLUX.2 uses a different latent packing format and 4-D RoPE positional embeddings compared to FLUX.1.

---

## Key Differences from FLUX.1

| | FLUX.1 | FLUX.2 |
|--|--------|--------|
| VAE scale factor | 8 | 8 |
| Effective latent scale | 8 (spatial at 1/8 pixel) | 16 (spatial at 1/16 pixel, extra 2× patchify) |
| Spatial latent shape | (B, 16, H/8, W/8) | (B, 64, H/16, W/16) |
| Packed format | (B, (H/16)*(W/16), 64) — 2×2 patchify | (B, (H/16)*(W/16), 64) — row-major reshape |
| Positional IDs | img_ids: (H/8//2, W/8//2, 3) — 3D | latent_ids: (H_lat*W_lat, 4) — 4D, read from batch.latent_ids |
| Text encoders | CLIP + T5 | Mistral (single encoder) |
| Pipeline type | T2I | TI2I (supports conditioning image) |

---

## Design Decisions

### Latent Scale Hook
- Added `_latent_scale_factor(server_args) -> int` to `ProgressiveDenoisingStage` base class
- Default: `vae_scale_factor` (preserves FLUX.1 behavior)
- FLUX.2 override: `vae_scale_factor * 2 = 16`
- Used in `forward()` for H_lat/W_lat computation, batch.height/width updates, and new_h/w_pixel

### Spectrum Constants (placeholder)
- Using FLUX.1-dev VAE values: A=203.615097, beta=1.915461
- TODO: fit FLUX.2-specific constants from FLUX.2 VAE latent statistics

### latent_ids Management
- In `_generate_initial_noise`: compute `_prepare_latent_ids(noise_spatial)` and assign to `batch.latent_ids`
  so that `_prepare_denoising_loop` sees the correct grid for the initial (low-res) resolution
- In `_on_resolution_change`: recompute `latent_ids` for the new spatial size; update `batch.latent_ids`
  before calling `prepare_pos_cond_kwargs` (which reads `batch.latent_ids` for the 4D RoPE)

### Files Changed
```
python/sglang/multimodal_gen/runtime/pipelines_core/stages/progressive_resolution/denoising.py
  + _latent_scale_factor() hook; forward() uses latent_scale instead of vae_scale_factor

runtime/pipelines/flux_2_progressive.py  (NEW)
  Flux2ProgressiveDenoisingStage: _latent_scale_factor, _unpack_latent, _repack_latent,
  _generate_initial_noise (sets batch.latent_ids), _on_resolution_change (updates
  batch.latent_ids + freqs_cis cache + CFG branch update)

runtime/pipelines/flux_2.py  (MODIFIED)
  Flux2Pipeline.create_pipeline_stages: manually assemble TI2I stages, use
  _add_flux2_denoising_stage() instead of add_standard_ti2i_stages()
```

---

## Benchmark Results

### FLUX.2-klein-4B, 30 steps, 1024×1024, seed=42, torch_sdpa, GPU-resident (A6000 48GB)

Warm-GPU runs (GPU fully warmed from prior runs, no JIT cold-start bias):

| Config | Stage split | Denoise | Total | Speedup |
|--------|------------|---------|-------|---------|
| A1 fullres | 30 @ 64² | 9.72s | 12.33s | 1.00× |
| A2 dct_rewind L1 δ=0.05 | 18@32² + 12@64² | 5.48s | 7.91s | **1.77×** |

Token-step analysis (quadratic attention model):
- Fullres: 30 × 4096 = 122,880 token-steps
- Progressive: 18×1024 + 12×4096 = 67,584 token-steps
- Expected speedup: **1.81×**
- Wall-clock efficiency: **98%** (vs 94-96% for Flux.1; FLUX.2-klein-4B fixed overhead is smaller)

Stage transition log (confirmed correct):
- `Progressive denoising: mode=dct_rewind levels=1 delta=0.050 initial=32x32`
- `Stage 1/2: 32x32 latent, steps [0, 18)` — 18 steps at 1024 tokens
- `rewind: sigma=0.8500 → t_eff=0.9189 at step 18`
- `Updated latent_ids and freqs_cis for 64x64 latent (pixel 1024x1024) across 1 branch(es)`
- `Stage 2/2: 64x64 latent, steps [18, 30)` — 12 steps at 4096 tokens

### Notes
- Spectrum constants (A=203.615097, beta=1.915461) are Flux.1-dev placeholders
- Transition at step 18/30 (delta=0.05): matches expected ~60% low-res split
- FLUX.2-klein is distilled (no guidance), uses Qwen3 text encoder, Flux2Transformer2DModel
- No image condition used (text-only generation, ImageVAEEncodingStage skips gracefully)

---

## Change Log
- 2026-06-01: Initial implementation (Flux.2 progressive denoising)
- 2026-06-01: Benchmark on FLUX.2-klein-4B: 1.77× denoising speedup vs 1.81× predicted
              (98% wall-clock efficiency). All stage transitions correct.
