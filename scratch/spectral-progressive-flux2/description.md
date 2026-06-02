# Progressive Resolution Growing — FLUX.2 Integration Log

## Status: benchmark complete — PR ready
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

test/unit/test_progressive_upsample.py  (MODIFIED)
  +20 Flux.2-specific unit tests (52 total, all pass CPU-only)
```

---

## Benchmark Results

### FLUX.2-klein-4B, 30 steps, 1024×1024, seed=42, torch_sdpa, GPU-resident (A6000 48GB)

**10-prompt benchmark (warm GPU, all prompts consistent to ±0.03s):**

| Config | Stage split | Denoise (avg) | Avg s/step | Speedup | Token-step |
|--------|------------|--------------|-----------|---------|-----------|
| fullres | 30 @ 64² | 9.72 s | 0.324 s | 1.00× | 1.00× |
| dct_rewind L1 δ=0.05 | 18@32² + 12@64² | 5.50 s | 0.183 s | **1.77×** | 1.82× |
| dct_rewind L1 δ=0.10 | 20@32² + 10@64² | 5.03 s | 0.168 s | **1.93×** | 2.00× |

Wall-clock efficiency: **97%** across both configs (gap is fixed per-step overhead).

**Per-prompt table (denoising loop, seconds):**

| Prompt | fullres | δ=0.05 | δ=0.10 | spd δ=0.05 | spd δ=0.10 |
|--------|---------|--------|--------|-----------|-----------|
| 00 misty forest | 9.70 | 5.49 | 5.00 | 1.77× | 1.94× |
| 01 rose-gold portrait | 9.70 | 5.50 | 5.06 | 1.76× | 1.92× |
| 02 neon Tokyo | 9.72 | 5.52 | 5.05 | 1.76× | 1.92× |
| 03 Tuscany vineyard | 9.71 | 5.53 | 5.03 | 1.76× | 1.93× |
| 04 Arctic tundra | 9.72 | 5.48 | 5.00 | 1.77× | 1.94× |
| 05 jazz club | 9.75 | 5.48 | 5.03 | 1.78× | 1.94× |
| 06 cherry blossoms | 9.74 | 5.49 | 5.05 | 1.77× | 1.93× |
| 07 desert mesa | 9.74 | 5.50 | 5.05 | 1.77× | 1.93× |
| 08 coral reef | 9.73 | 5.49 | 5.04 | 1.77× | 1.93× |
| 09 autumn maples | 9.71 | 5.47 | 5.02 | 1.78× | 1.93× |
| **AVG** | **9.72** | **5.50** | **5.03** | **1.77×** | **1.93×** |

### Stage transition log (confirmed correct)
```
Progressive denoising: mode=dct_rewind levels=1 delta=0.050 initial=32x32
Stage 1/2: 32x32 latent, steps [0, 18)
  rewind: sigma=0.8500 → t_eff=0.9189 at step 18
Updated latent_ids and freqs_cis for 64x64 latent (pixel 1024x1024) across 1 branch(es)
Stage 2/2: 64x64 latent, steps [18, 30)
Progressive denoising done in 5.49s (avg 0.1830s/step)

Progressive denoising: mode=dct_rewind levels=1 delta=0.100 initial=32x32
Stage 1/2: 32x32 latent, steps [0, 20)
  rewind: sigma=0.8095 → t_eff=0.8947 at step 20
Updated latent_ids and freqs_cis for 64x64 latent (pixel 1024x1024) across 1 branch(es)
Stage 2/2: 64x64 latent, steps [20, 30)
Progressive denoising done in 5.00s (avg 0.1667s/step)
```

### Notes
- Spectrum constants (A=203.615097, beta=1.915461) are Flux.1-dev placeholders
- FLUX.2-klein is distilled (no guidance), uses Qwen3 text encoder, Flux2Transformer2DModel
- Speedups are 97% of token-step prediction — same efficiency as Flux.1
- All 10 prompts produce artifact-free images across all 3 modes

### Visuals
- `results/full_20260601_195648/prompt_NN_3way.png` — per-prompt 3-way strip (fullres | δ=0.05 | δ=0.10)
- `results/full_20260601_195648/montage_3way.png` — full 3×10 grid

---

## Change Log
- 2026-06-01: Initial implementation (Flux.2 progressive denoising)
- 2026-06-01: Benchmark on FLUX.2-klein-4B: 1.77× (δ=0.05) and 1.93× (δ=0.10) denoising speedup
              across 10 prompts. 97% wall-clock efficiency. All transitions correct.
- 2026-06-01: Added 20 Flux.2-specific unit tests (52 total, all pass CPU-only, 14s)
