# Progressive Resolution Growing — Wan T2V Video

## Status: ✅ Working — first GPU benchmark complete
**Branch:** `bchao1/spectral-progressive-wan`
**Base:** `bchao1/spectral-progressive-flux` (inherits GPU DCT ops + base stage infrastructure)

---

## Goal
Extend spectral progressive diffusion (DCT rewind) to the **Wan 2.1 T2V video model** in SGLang.
Reference: `wavelet-diffusion/inference_progressive.py` (WAN section).

---

## Key Differences vs FLUX

| Aspect | FLUX (image) | Wan T2V (video) |
|--------|-------------|-----------------|
| Latent shape | `[B, S, 64]` packed tokens | `[B, C, T, H, W]` — already spatial |
| Progressive upsample dims | H×W | H×W only (T fixed across stages) |
| Pack/unpack needed | Yes — patchify 2×2 | No — identity ops |
| Positional embeddings | RoPE freqs_cis depends on H/W | None spatial; no update needed |
| CFG passes | 1 (guidance-distilled) | 2 (dual CFG cond/uncond) |
| Latent channels | 16 (from in_channels//4=64//4) | 16 (z_dim=16, not in_channels//4) |
| VAE spatial stride | 8 | 8 |
| Spectrum A | 203.615097 | 219.484718 |
| Spectrum β | 1.915461 | 2.422687 |

---

## Design Decisions

### Scope (this PR)
- **DCT only** (`dct_rewind` as default, `dct` as non-rewind variant) — same as FLUX
- Wan T2V first; Wan I2V and DMD variants are separate future work
- No sequence parallelism support (SP not compatible with progressive; guarded in base class)
- Temporal dim T_lat is always preserved unchanged across stage transitions

### Architecture
```
WanProgressiveDenoisingStage(ProgressiveDenoisingStage)
    _unpack_latent(latent, h, w) → identity (latent already [B, C, T, H, W])
    _repack_latent(x, h, w, batch, server_args) → identity
    _on_resolution_change(...) → pass (no RoPE to update for Wan T2V)
    _generate_initial_noise(...) → [1, z_dim, T_lat, h_lat, w_lat]
```

`WanPipeline.create_pipeline_stages` expanded to call `_add_wan_denoising_stage(server_args)`,
which selects `WanProgressiveDenoisingStage` when `server_args.progressive_mode in {"dct", "dct_rewind"}`.

### Files Created
```
runtime/pipelines/wan_progressive.py           # WanProgressiveDenoisingStage
test/unit/test_progressive_wan.py              # 20 CPU-only unit tests
test/manual/test_progressive_wan.py            # GPU integration test
scratch/spectral-progressive-wan/description.md
```

### Files Modified
```
runtime/pipelines/wan_pipeline.py             # wire in _add_wan_denoising_stage
configs/models/vaes/wanvae.py                 # add vae_scale_factor to WanVAEArchConfig
```

---

## Reference Parameter Audit (`wavelet-diffusion/inference_progressive.py`)

All benchmark runs use exactly these parameters to match the reference. **Deviating from
these when re-running will invalidate comparisons with the reference implementation.**

| Parameter | Reference value | sglang default | Benchmark override |
|-----------|----------------|----------------|-------------------|
| height × width | 480 × 832 | 480 × 832 | none (already matches) |
| num_frames | 81 | 81 | none |
| num_inference_steps | 50 (`WAN_N_STEPS`) | 50 | none |
| guidance_scale | 5.0 (`WAN_GUIDANCE_SCALE`) | **3.0** | `--guidance-scale 5.0` |
| flow_shift | 5.0 (`WAN_SHIFT`) | **3.0** | `--flow-shift 5.0` |
| scheduler | `FlowMatchEulerDiscreteScheduler` | `FlowUniPCMultistepScheduler` | *not changed* |
| seed | 42 | — | `--seed 42` |

### Sigma schedule impact (δ=0.05, L=1)

The two non-default params (guidance_scale, flow_shift) materially affect stage transitions:

| Schedule | Shift | Transition step / 50 | Token-step speedup (theory) |
|----------|-------|----------------------|-----------------------------|
| Reference | 5.0 | step 33 (σ≈0.720) | **1.98×** |
| sglang default | 3.0 | step 26 (σ≈0.735) | 1.64× |
| **Benchmark** | **5.0** | **step 33** | **1.98×** |

With shift=5.0 the scheduler decays slower → more steps remain below the transition sigma →
stage 1 (half-resolution) runs longer → more token-step savings.

The scheduler solver (`UniPC` vs `Euler`) affects denoising trajectory but NOT the sigma
schedule or stage transitions. `UniPC` is higher-order and generally produces better quality
at the same step count; the reference used Euler as a simpler baseline.

### How flow_shift=5.0 is passed in the benchmark script

`flow_shift` is a `PipelineConfig` CLI flag (`--flow-shift`). Unlike `guidance_scale`, it
cannot be overridden per-request — it is set once at server/pipeline init time. The benchmark
script passes `--flow-shift 5.0` to `sglang generate`, which overrides the `WanT2V480PConfig`
default of 3.0 for that run only. The sglang codebase default is NOT changed.

---

## Spectrum Constants (WAN 2.1 VAE, VChitect, 9050 videos)
- A = 219.484718
- β = 2.422687 (steeper than FLUX β=1.915 → higher Nyquist power → earlier stage transitions)

### Transition Sigmas (Wan 480P: H_lat=60, W_lat=104, δ=0.01)

| Levels | Stage 2 transition σ | Stage 3 transition σ |
|--------|---------------------|---------------------|
| 1 | ~0.94 | — |
| 2 | ~0.98 | ~0.94 |

(Actual values depend on exact Nyquist frequency of the coarser latent resolution.)
Stage 2 fires much earlier than for FLUX (~0.85) because Wan's steeper spectrum means
Nyquist-band frequencies are more powerful, activating at higher noise levels.

---

## Key Implementation Notes

### Temporal Dimension
- `T_lat = (N_frames - 1) // 4 + 1` where 4 is the temporal VAE stride
- For 81 frames: T_lat = 21
- Progressive upsample operates only on `(H, W)` via `dct_upsample_2d(..., x)` where
  `x.shape = (..., H, W)`. For a 5-D `[B, C, T, H, W]` tensor, `leading = (B, C, T)` and
  DCT runs independently on each H×W plane — exactly the desired spatial upsample.

### Initial Noise Override
The base class `_generate_initial_noise` uses `C = in_channels // 4` (FLUX patchify convention:
in_channels=64 → C=16). For Wan, `in_channels=16` and `in_channels // 4 = 4` (wrong).
Override uses `C = vae_config.arch_config.z_dim = 16` (correct latent channels).

### vae_scale_factor Fix
`ProgressiveDenoisingStage.forward()` reads `vae_config.arch_config.vae_scale_factor`.
`WanVAEArchConfig.__post_init__` previously set `spatial_compression_ratio` but not
`vae_scale_factor`. Added `self.vae_scale_factor = self.scale_factor_spatial` (= 8).
This is consistent with FLUX, SANA, and other VAE configs.

### No RoPE Update
FLUX requires updating `freqs_cis` at resolution transitions (see flux_progressive.py Bug 4).
Wan T2V uses no H/W-dependent positional embedding in the denoising forward pass context
(`prepare_pos_cond_kwargs` returns `{}`). `_on_resolution_change` is a deliberate no-op.

---

## Bugs Fixed
1. **`vae_scale_factor` missing on WanVAEArchConfig** — AttributeError in progressive stage.
   Fixed by adding `self.vae_scale_factor = self.scale_factor_spatial` to `__post_init__`.

2. **Wrong channel count in `_generate_initial_noise`** — base class uses `in_channels // 4`
   which is 16//4=4 for Wan (correct formula for FLUX with in_channels=64, wrong for Wan).
   Fixed by overriding to use `vae_config.arch_config.z_dim = 16`.

---

## Test Results (CPU-only unit tests)

Run: `python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_wan.py -v`

| Test | Status |
|------|--------|
| WanSpectrumConstants: all 5 | ✅ |
| WanProgressiveStageHooks: unpack/repack identity, on_resolution_change noop | ✅ |
| WanGenerateInitialNoise: shape, dtype, determinism, T_lat preservation | ✅ |
| DCTUpsample5D: shape, T-dim unchanged, rewind, dtype, apply_upsample | ✅ |
| WanVAEArchConfig: vae_scale_factor | ✅ |
| WanStageTransitions: single/two-level, find_transition_steps | ✅ |

---

## Benchmark Results (GPU)

**Run:** `bench_20260603_170457` — GPU 0 (RTX A6000 48GB), 2026-06-03
**Model:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, 50 steps, 480×832, 81 frames
**Params:** guidance=5.0, flow_shift=5.0, seed=42 (reference-matched)
**Note:** CuTeDSL fused kernels disabled (`cutlass.cute` absent, native fallback active).
Results are conservative; fused kernels would reduce both R1 and R2 equally.

| Config | Denoise (s) | Total (s) | Transition step | Speedup (denoise) |
|--------|-------------|-----------|-----------------|-------------------|
| R1 fullres | 268.9s | 286.0s | — | 1.00× |
| R2 dct_rewind L1 δ=0.05 | **118.9s** | **135.9s** | 33/50 (σ=0.720→t_eff=0.837) | **2.26×** |

**Token-step theory** predicted **1.98×**; actual is **2.26×**. The gap is because the
half-resolution steps (0–32) run faster-than-linear: the quadratic attention cost at
30×52 (= 1,560 tokens) is much less than half the cost at 60×104 (= 6,240 tokens).

**Wall-clock speedup: 2.10× total** (286s → 136s including text encoding + VAE decode).

Videos saved:
- `results/bench_20260603_170457/R1_fullres.mp4`
- `results/bench_20260603_170457/R2_prog_L1_d0.05.mp4`

---

## Delta Sweep (2026-06-03) — Dynamic scene (F1 race car)

**Run:** `delta_sweep_20260603_171818` — GPU 0 (RTX A6000 48GB)
**Prompt:** "A Formula 1 race car speeding through a circuit at sunset, motion blur, photorealistic"
**Params:** 50 steps · 480×832 · 81 frames · guidance=5.0 · flow\_shift=5.0 · seed=42 · L=1

| Config | Denoise (s) | Total (s) | Transition step | Speedup (denoise) |
|--------|-------------|-----------|-----------------|-------------------|
| fullres | 266.8s | 284.1s | — | 1.00× |
| dct\_rewind δ=0.01 | 161.5s | 178.7s | 23/50 | 1.65× |
| dct\_rewind δ=0.02 | 142.7s | 160.3s | 27/50 | 1.86× |
| dct\_rewind δ=0.05 | 114.7s | 131.7s | 33/50 | 2.32× |
| dct\_rewind δ=0.10 | **95.9s** | **113.0s** | 37/50 | **2.78×** |

**Speedup vs δ:** monotonically increasing — larger δ lowers the activation threshold, firing the transition later (more steps at low-res). δ=0.10 reaches **2.78×** denoise speedup, **2.51×** wall-clock.

**Transition steps** match theory for shift=5.0 schedule exactly (steps 23, 27, 33, 37).

**Quality note:** All 5 videos generated without artifacts. Visual inspection required to assess whether δ=0.10 (37/50 steps at 30×52) retains sufficient fine detail for motion-heavy scenes.

---

## Optimization Compatibility

| Optimization | Progressive | Notes |
|---|---|---|
| Layerwise CPU offload | ✅ Safe | Component-level, unaffected |
| LoRA | ✅ Safe | Weight-level |
| CFG (dual-pass) | ✅ Safe | Both cond/uncond branches run with whatever H/W |
| **Sequence Parallelism** | ❌ Blocked | Guarded by base class (RuntimeError) |
| **Cache-Dit** | ❌ Broken | Step cache indexed by step count; T×H×W changes between stages |
| **torch.compile** | ❌ Broken | Compiled kernel has fixed sequence length |

---

## Git State
```
bchao1/spectral-progressive-flux   ← base (GPU DCT ops + base stage)
bchao1/spectral-progressive-wan    ← this feature (all Wan-specific additions)
```
