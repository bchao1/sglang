# Progressive Resolution Growing — SGLang Integration Log

## Status: ✅ Working — feature branch ready
**Branch:** `bchao1/spectral-progressive-flux`
**Base:** `main` (upstream clean, no feature code on main)
**Commit:** `fafb6993e` feat(diffusion): progressive resolution growing for FLUX.1 via GPU DCT upsampling

---

## Goal
Integrate spectral progressive diffusion (DCT rewind) into the FLUX.1 pipeline in SGLang.
Reference: `wavelet-diffusion/inference_progressive.py`.

---

## Design Decisions

### Scope (this PR)
- **DCT only** (`dct_rewind` as default, `dct` as non-rewind variant)
- FLUX.1 first; skeleton extensible to Z-Image, Qwen-Image (via `_unpack_latent` / `_repack_latent` hooks)
- No sequence parallelism support (guarded with an explicit error check)
- No CPU↔GPU transfers for spectral ops — pure torch.fft GPU implementation

### Architecture
- `ProgressiveDenoisingStage(DenoisingStage)`: overrides `forward()`, reuses all parent infrastructure
  - Routes to `super().forward()` when `progressive_mode == "fullres"` (existing path unchanged)
  - `_unpack_latent` / `_repack_latent` / `_on_resolution_change` hooks for model-specific logic
- `FluxPipeline` replaces its `DenoisingStage` with `FluxProgressiveDenoisingStage`
  - Backward-compatible: when `progressive_mode == "fullres"` (default), behaviour is identical
- `SamplingParams`: +`progressive_mode`, +`progressive_levels`, +`progressive_delta` with CLI flags

### Files Created
```
python/sglang/multimodal_gen/runtime/pipelines_core/stages/progressive_resolution/
  __init__.py
  spectral_ops.py      # GPU DCT-II / IDCT-II via torch.fft (no CPU transfer)
  scheduler_utils.py   # Stage transitions, find_transition_steps, reset_scheduler_state
  upsample.py          # dct_upsample_2d, apply_upsample dispatcher
  denoising.py         # ProgressiveDenoisingStage base class

runtime/pipelines/
  flux_progressive.py  # FluxProgressiveDenoisingStage (FLUX-specific hooks)

test/unit/test_progressive_upsample.py   # 17 CPU-only unit tests (17/17 pass)
test/manual/test_progressive_flux.py     # End-to-end manual test (needs GPU + model)
```

### Files Modified
```
configs/sample/sampling_params.py        # +progressive_mode/levels/delta fields + CLI args
runtime/pipelines/flux.py               # swap DenoisingStage → FluxProgressiveDenoisingStage
runtime/pipelines_core/stages/__init__  # export ProgressiveDenoisingStage
```

---

## Spectrum Constants (FLUX VAE, Aesthetics-Train-V2, 105k images)
- A = 203.615097
- beta = 1.915461

---

## Key Implementation Notes

### GPU DCT
- DCT-II via torch.fft using the Makhoul (1980) algorithm; verified against scipy
- 2D DCT = two separable 1D DCTs (along H, then W)
- All operations on GPU; no `.cpu()` or `.numpy()` calls anywhere in the path

### Latent Dimensions (FLUX.1)
- VAE scale factor: 8, patch size: 2
- Full pixel 1024×1024 → latent 128×128 → packed [B, 4096, 64]
- levels=1 initial stage: 64×64 latent → packed [B, 1024, 64]
- levels=2 initial stage: 32×32 latent → packed [B, 256, 64]

### Resolution Change (FLUX-specific) — critical design note
- `CFGBranch.kwargs` is a shallow copy made at `build()` time.
  Updating `ctx.pos_cond_kwargs` alone does NOT reach the transformer.
  Must update `branch.kwargs["freqs_cis"]` directly across all branches.
- Cache `freqs_cis` per `(h_lat, w_lat)` to avoid redundant recomputation.

### Scheduler Rewind (dct_rewind)
- After upsample: `t_eff = 2*sigma_t/(1+sigma_t)` (always > sigma_t for sigma_t < 1)
- Patch `scheduler.sigmas/timesteps` and `ctx.timesteps` at transition point
- Clone scheduler tensors first (they may be inference tensors → in-place update forbidden)
- Reset `scheduler._step_index = transition_step`

---

## Bugs Fixed During Development

1. **InferenceMode tensor in-place update** — `scheduler.sigmas/timesteps` created inside
   `inference_mode`; cloned once before stage loop when `rewind=True`.

2. **Stale `raw_latent_shape`** — overwrote `batch.latents` with low-res initial noise and
   set `raw_latent_shape=[1,1024,64]`; `maybe_unpad_latents` then truncated the final
   full-res `[1,4096,64]` latent to 1024 tokens → black images.
   Fix: `batch.raw_latent_shape = ctx.latents.shape` before `_finalize_denoising_loop`.

3. **`SamplingParams.add_cli_args` missing entries** — progressive fields not auto-discovered;
   added explicit `add_argument` calls for `--progressive-mode/levels/delta`.

4. **`CFGBranch.kwargs` stale freqs_cis** — root cause of all black images and L2 CUDA OOB
   crash. `cfg_policy.build()` does `{**image_kwargs, **pos_cond_kwargs}` (shallow copy);
   updating `ctx.pos_cond_kwargs["freqs_cis"]` was silently ignored. Transformer ran with
   low-res freqs_cis (1024 tokens) against a full-res latent (4096 tokens), causing wrong
   outputs and eventually illegal memory access in the attention kernel.
   Fix: update `branch.kwargs["freqs_cis"]` for every branch in `ctx.cfg_policy.branches`.

---

## Benchmark Results (FLUX.1-dev, 1024×1024, torch_sdpa, GPU: H100 80GB)

### 20 steps
| Config | Stage split | Denoise | Total | Speedup |
|--------|------------|---------|-------|---------|
| fullres | 20 @ 128² | 22.72s | 24.73s | 1.00× |
| dct_rewind L1 δ=0.01 | 8@64²+12@128² | 17.20s | 19.27s | **1.28×** |
| dct_rewind L1 δ=0.05 | 12@64²+8@128² | 15.91s | 18.01s | **1.37×** |
| dct_rewind L2 δ=0.01 | 4@32²+4@64²+12@128² | 17.50s | 19.61s | **1.26×** |
| dct_plain  L1 δ=0.01 | 8@64²+12@128² | 16.25s | 18.22s | **1.36×** |

### 50 steps (production-quality)
| Config | Stage split | Avg s/step | Denoise | Total | Speedup |
|--------|------------|-----------|---------|-------|---------|
| fullres | 50 @ 128² | 1.095s | 54.75s | 57.82s | 1.00× |
| dct_rewind L1 δ=0.01 | 18@64²+32@128² | 0.811s | 40.53s | 42.74s | **1.35×** |
| dct_rewind L1 δ=0.05 | 28@64²+22@128² | 0.698s | 34.87s | 37.15s | **1.56×** |
| dct_rewind L2 δ=0.01 | 10@32²+8@64²+32@128² | 0.776s | 38.78s | 41.01s | **1.41×** |
| dct_plain  L1 δ=0.01 | 18@64²+32@128² | 0.770s | 38.48s | 40.58s | **1.43×** |

All outputs are valid images (1.3–1.6 MB PNG, mean pixel ≈ 110–120, std ≈ 52–70).

---

## Git State
```
main                        ← upstream clean, no feature code
bchao1/spectral-progressive-flux  ← all feature work (1 commit, fafb6993e)
```

To switch to feature branch:
```bash
git checkout bchao1/spectral-progressive-flux
```

To run benchmark:
```bash
bash scratch/test_progressive_gen.sh --steps 50
```

To run unit tests (no GPU needed):
```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py -v
```

---

## Change Log
- 2026-05-29: Initial implementation
- 2026-05-30: Bugs 1–3 fixed; first successful end-to-end run (1.26–1.38× at 20 steps)
- 2026-05-31: Bug 4 fixed (CFGBranch.kwargs freqs_cis); full 50-step benchmark (1.35–1.56×);
              feature branch `bchao1/spectral-progressive-flux` created, main restored to clean upstream
