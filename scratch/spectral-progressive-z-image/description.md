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

## Z-Image Architecture Notes (TBD — fill in after audit)
- VAE scale factor: TBD
- Patch size: TBD
- Latent packing format: TBD
- Positional embedding mechanism: TBD (rope/freqs_cis equivalent?)
- Transformer attention mechanism: TBD

---

## Spectrum Constants (Z-Image VAE — TBD)
- A = TBD
- beta = TBD

---

## Benchmark Results (TBD)

---

## Change Log
- 2026-06-01: Branch created from bchao1/spectral-progressive-flux; setup complete
