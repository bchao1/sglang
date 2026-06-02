# spectral-progressive-qwen

## Goal
Extend progressive resolution growing (spectral DCT upsample) to the
Qwen-Image T2I pipeline family. Built on top of `bchao1/spectral-progressive-flux`.

## Architecture

Follows the same class hierarchy as FLUX.1 and Z-Image progressive stages:
```
ProgressiveDenoisingStage          (base, pipelines_core/.../denoising.py)
├── FluxProgressiveDenoisingStage  (FLUX.1, pipelines/flux_progressive.py)
├── ZImageProgressiveDenoisingStage (Z-Image, pipelines/zimage_progressive.py)
└── QwenImageProgressiveDenoisingStage  ← this PR
```

## Key implementation details

### Latent format
Qwen-Image uses the same 2×2 patchify convention as FLUX.1-dev:
- `in_channels = 64`, spatial channels C = 16
- Pack: [B, 16, H, W] → [B, S, 64] where S = (H//2) * (W//2)
- Unpack: [B, S, 64] → [B, 16, H, W]
- `_qwen_image_pack` / `_qwen_image_unpack` are numerically identical to
  `_flux_pack` / `_flux_unpack` — same patchification, same channel count.

### Resolution change hook
Qwen DiT `forward()` consumes both:
1. `freqs_cis` — (img_cache, txt_cache) RoPE tensors (size depends on sequence length)
2. `img_shapes` — list of [(T, H//2, W//2)] tuples, used by `build_modulate_index`

Both are updated in `_on_resolution_change` across all CFG branches and
in `ctx.pos_cond_kwargs`.

### mu parameter
`mu` (flow-matching scheduler shift) is computed once for the final resolution
at timestep-preparation time and is not re-computed between progressive stages.
This is an approximation — sigma rewind already corrects the effective noise
level at each stage transition, making mu re-computation negligible in practice.

### Spectrum constants
```
A    = 203.615097
beta = 1.915461
```
Using FLUX.1-dev fitted values as placeholder (both VAEs produce 16-channel
spatial latents with similar frequency roll-off).
TODO: fit from Qwen-Image VAE latents on Aesthetics-Train-V2.

## Files

### New
- `python/sglang/multimodal_gen/runtime/pipelines/qwen_image_progressive.py`
  - `_qwen_image_unpack` / `_qwen_image_pack`
  - `QwenImageProgressiveDenoisingStage(ProgressiveDenoisingStage)`
  - `QwenImageProgressivePipeline(QwenImagePipeline)`

- `python/sglang/multimodal_gen/test/unit/test_progressive_qwen_image.py`
  - 29 CPU-only unit tests (see table below)

## Unit tests (CPU-only, no GPU, no model)

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestQwenImagePack` | 9 | shapes (pack+unpack), roundtrip (both directions), dtype, spatial ordering, matches `_pack_latents` |
| `TestQwenImageProgressiveStage` | 14 | instantiation, inheritance, spectrum constants, _unpack/_repack shape + roundtrip, _on_resolution_change: no-cfg, freqs update, img_shapes update, null-freqs skip, missing-key skip, null-shapes skip, joint update |
| `TestQwenImageProgressivePipeline` | 3 | pipeline_name, inheritance, required modules |
| `TestQwenPackIntegrationWithProgressiveBase` | 3 | spatial channel count, token dim = 64, 2× upsample → 4× sequence length |

Run:
```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_qwen_image.py -v
```

## GPU E2E benchmark (TODO — fill after GPU run)

Hardware:
Model:
Steps: 30
Seed: 42
Resolution:

| Config | Stage split | Denoise | Speedup |
|--------|------------|---------|---------|
| fullres | | | 1.00× |
| dct_rewind L1 δ=0.05 | | | |
| dct_rewind L1 δ=0.10 | | | |
