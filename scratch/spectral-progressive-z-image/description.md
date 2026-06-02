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

## Bugs Fixed During Development (pre-existing Z-Image issues)

1. **negative_prompt_embeds batch dimension mismatch** — `_append_negative_text_outputs` used
   `pe.shape[0]` as the target batch size. For Z-Image's 2-D embeddings `[seq_len, dim]`, `shape[0]`
   is the sequence length, not the batch size (1). Fix: detect `ndim == 2` → treat as batch=1.

2. **CuTeDSL fused kernel import failure** — Z-Image model and layernorm forward_cuda unconditionally
   import `cutlass.cute` (CUTLASS 3.x Python bindings). `cutlass.cute` is not installed in the genAI
   env. Fix: `_has_cutlass_cute` pre-check at module import; guards all three CUTLASS paths in
   `zimage.py` and `layernorm.py`. Fallback to native RMSNorm path.

---

## Benchmark Results (GPU: RTX A6000 48GB, `--dit-cpu-offload false`, 50 steps, 1024×1024)

### Group A — core configs

| Config | Stage split | Denoise | Avg s/step | Speedup |
|--------|-------------|---------|-----------|---------|
| fullres | 50 @ 128² | 52.72 s | 1.054 s | 1.00× |
| dct_rewind L1 δ=0.01 | 26@64² + 24@128² | 32.46 s | 0.649 s | **1.62×** |
| dct_rewind L1 δ=0.05 | 35@64² + 15@128² | 25.41 s | 0.508 s | **2.07×** |
| dct_rewind L2 δ=0.01 | 15@32² + 11@64² + 24@128² | 30.02 s | 0.600 s | **1.76×** |

### Delta sweep (L1, each run on dedicated GPU)

| δ | Denoise | Speedup |
|---|---------|---------|
| 0.01 | 34.38 s | 1.53× |
| 0.05 | 26.03 s | 2.03× |
| 0.10 | 22.65 s | **2.33×** ⭐ recommended |
| 0.20 | 19.77 s | **2.66×** |
| 0.50 | 17.74 s | **2.97×** |

Z-Image consistently achieves higher progressive speedups than FLUX.1 at the same δ because
dual CFG doubles the absolute attention savings.

### FLUX.1 vs Z-Image comparison

| δ | FLUX.1 | Z-Image | Ratio |
|---|--------|---------|-------|
| 0.01 | 1.32× | 1.53× | +16% |
| 0.05 | 1.63× | 2.03× | +25% |
| 0.10 | 1.83× | 2.33× | +27% |

---

## Change Log
- 2026-06-01: Branch created from bchao1/spectral-progressive-flux; setup complete
- 2026-06-01: Implementation complete — ZImageProgressiveDenoisingStage, pipeline wiring,
              unit tests (41/41 pass), manual test, scratch scripts
- 2026-06-01: Fixed two pre-existing Z-Image bugs (negative embedding batch dim, CuTeDSL import)
- 2026-06-01: All experiments complete. Group A: 1.62×/2.07×/1.76× at δ=0.01/0.05/L2.
              Delta sweep: up to 2.97× at δ=0.50, best tradeoff 2.33× at δ=0.10.
              10-prompt quality comparison (fullres | δ=0.05 | δ=0.10): all outputs artifact-free.
              PR_description.md written.
