# Progressive Resolution Growing — SGLang Integration Log

## Goal
Integrate spectral progressive diffusion (DCT rewind) into the FLUX.1 pipeline in SGLang.
Reference: `wavelet-diffusion/inference_progressive.py`.

## Design Decisions

### Scope (this PR)
- **DCT only** (`dct_rewind` as default, `dct` as non-rewind variant)
- FLUX.1 first; skeleton extensible to Z-Image, Qwen-Image (via `_unpack_latent` / `_repack_latent` hooks)
- No sequence parallelism support (guarded with an error check)
- No CPU↔GPU transfers for spectral ops — pure torch.fft GPU implementation

### Architecture
- `ProgressiveDenoisingStage(DenoisingStage)`: overrides `forward()`, reuses all parent infrastructure
  - Routes to `super().forward()` when `progressive_mode == "fullres"` (existing path unchanged)
  - `_unpack_latent` / `_repack_latent` / `_on_resolution_change` hooks for model-specific logic
- `FluxPipeline` replaces its `DenoisingStage` with `FluxProgressiveDenoisingStage`
  - Backward-compatible: when `progressive_mode == "fullres"` (default), behavior is identical
- `SamplingParams`: +`progressive_mode`, +`progressive_levels`, +`progressive_delta`

### Files Created
```
python/sglang/multimodal_gen/runtime/pipelines_core/stages/progressive_resolution/
  __init__.py
  spectral_ops.py      # GPU DCT-II / IDCT-II via torch.fft (no CPU transfer)
  scheduler_utils.py   # Stage transitions, find_transition_steps, reset_scheduler_state
  upsample.py          # dct_upsample_2d_gpu, apply_upsample dispatcher
  denoising.py         # ProgressiveDenoisingStage base class

runtime/pipelines/
  flux_progressive.py  # FluxProgressiveDenoisingStage (FLUX-specific hooks)
```

### Files Modified
```
configs/sample/sampling_params.py        # +progressive_mode/levels/delta
runtime/pipelines/flux.py               # swap DenoisingStage → FluxProgressiveDenoisingStage
runtime/pipelines_core/stages/__init__  # export ProgressiveDenoisingStage
```

### Tests
```
test/manual/test_progressive_flux.py    # Manual tier (needs GPU + model)
```

## Spectrum Constants (FLUX VAE, Aesthetics-Train-V2, 105k images)
- A = 203.615097
- beta = 1.915461

## Key Implementation Notes

### GPU DCT
- Implement DCT-II via torch.fft using the Makhoul (1980) algorithm
- 2D DCT = two separable 1D DCTs (along H, then W)
- All operations on GPU; no `.cpu()` or `.numpy()` calls in the hot path

### Latent Dimensions (FLUX.1)
- VAE scale factor: 8
- Patch size: 2
- Full pixel 1024×1024 → latent 128×128 → seq_len = 64×64 = 4096 → packed [B, 4096, 64]
- With levels=1: initial latent 64×64 → seq_len 32×32 = 1024 → packed [B, 1024, 64]

### Resolution Change (FLUX-specific)
- After upsample: update `batch.height/width` to new pixel dims
- Recompute `freqs_cis` (rotary embeddings for img_ids) via `pipeline_config.prepare_pos_cond_kwargs`
- Cache per `(h_lat, w_lat)` to avoid redundant computation

### Scheduler Rewind (dct_rewind)
- After upsampling, patch `ctx.scheduler.sigmas[transition_step]` to `t_eff = 2*sigma_t/(1+sigma_t)`
- Patch `ctx.scheduler.timesteps[transition_step]` to `t_eff * 1000`
- Call `_reset_scheduler_loop_state(ctx.scheduler)` and set `_step_index = transition_step`

## Change Log
- 2026-05-29: Initial implementation
