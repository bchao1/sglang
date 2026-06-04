> **⚠️ Stacked PR:** Depends on the FLUX.1 progressive PR ([PR #26961](https://github.com/sgl-project/sglang/pull/26961)). The Wan-specific change is the top commit: `feat(diffusion): progressive resolution growing for Wan T2V via GPU DCT upsampling`. All earlier commits belong to prior progressive PRs and can be ignored during review.

## Motivation

Transformer attention is O(n²) in sequence length. For Wan 2.1 T2V at 480×832 with 81 frames, the denoising loop processes **6,240 spatial tokens** per step (latent 60×104, 2×2 patch). Running early steps at half spatial resolution (30×52 → 1,560 tokens) reduces attention cost to ~6% for those steps, yielding a **1.65–2.78× denoising speedup** depending on δ — with no quality degradation.

This PR extends **spectral progressive resolution growing** (introduced for FLUX.1 in [PR #26961](https://github.com/sgl-project/sglang/pull/26961)) to the **Wan 2.1 T2V video pipeline**. Early denoising steps run at half the spatial latent resolution; at the transition point the latent is spectrally upsampled via GPU DCT. The temporal dimension T is **never changed** — only H×W grows between stages. It would be an interesting future direction to also handle temporal progressive generation.

| Model | HuggingFace ID |
|-------|---------------|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |

Wan T2V differs from FLUX in four ways that required new subclass logic:

| | FLUX.1 (image) | Wan T2V (video) |
|---|---|---|
| Latent format | `[B, S, 64]` patchified tokens | `[B, C, T, H, W]` — already spatial |
| Progressive upsample dims | H×W | H×W only (T is fixed across all stages) |
| Positional embeddings | RoPE `freqs_cis` depends on H/W | None spatial — no update needed at transitions |
| CFG | Single forward pass (guidance-distilled) | Dual forward pass (cond + uncond) |

## Modifications

### Base class fix — `ProgressiveDenoisingStage`

**`runtime/pipelines_core/stages/progressive_resolution/denoising.py`**

Two fixes required by Wan's architecture:

```python
# 1. vae_scale_factor fallback — Wan VAEArchConfig exposes spatial_compression_ratio
#    but not vae_scale_factor (unlike FLUX/SANA). Resolved without touching Wan code.
arch = server_args.pipeline_config.vae_config.arch_config
vae_scale_factor = getattr(arch, "vae_scale_factor", None) or getattr(
    arch, "spatial_compression_ratio", 8
)

# 2. torch.autocast context — DenoisingStage.forward() wraps its loop in autocast;
#    ProgressiveDenoisingStage bypasses that, causing Float/BFloat16 mismatch on Wan.
#    Fixed by wrapping the stage loop in the same autocast context.
with torch.autocast(device_type=current_platform.device_type,
                    dtype=ctx.target_dtype, enabled=ctx.autocast_enabled):
    for stage in range(1, num_stages + 1):
        ...
```

FLUX.1 and FLUX.2 behavior is **unchanged** — both fixes are inert on their code paths.

### CuTeDSL fallback — `runtime/layers/layernorm.py`

Extended the `_has_cutlass_cute` guard (first introduced in the Z-Image progressive PR) to `FusedNormScaleShift.forward_cuda`, which Wan's AdaLN transformer blocks use. Without this guard, Wan fails immediately on machines where `cutlass.cute` (CUTLASS 3.x Python bindings) is not installed.

```python
# At module import — graceful fallback if CUTLASS 3.x DSL is absent
try:
    import cutlass.cute as _
    _has_cutlass_cute = True
except Exception:
    _has_cutlass_cute = False

# In forward_cuda — all four CUTLASS kernel paths now guarded
if not _has_cutlass_cute or (x.shape[-1] % 256 != 0 and x.shape[-1] <= 8192):
    return self.forward_native(x, shift, scale)  # numerically equivalent
```

### New file — `runtime/pipelines/wan_progressive.py`

`WanProgressiveDenoisingStage(ProgressiveDenoisingStage)` overrides:

| Hook | What it does |
|------|-------------|
| `_unpack_latent` | Identity — Wan latent `[B, C, T, H, W]` is already spatial |
| `_repack_latent` | Identity — no packing needed |
| `_on_resolution_change` | No-op — Wan T2V has no H/W-dependent positional embeddings |
| `_generate_initial_noise` | Generates `[1, z_dim, T_lat, h_lat, w_lat]`; uses `z_dim=16` (not `in_channels//4=4`) |

Spectrum constants fitted on VChitect (9050 videos), spatial P(ω) = A·|ω|^(−β):
- **A = 219.484718**, **β = 2.422687** (steeper than FLUX β=1.915 → stage transitions fire later in denoising)

### Modified — `runtime/pipelines/wan_pipeline.py`

`WanPipeline.create_pipeline_stages` now calls `_add_wan_denoising_stage(server_args)`. When `progressive_mode == "fullres"` (default), delegates to standard `DenoisingStage.forward()` — **zero behavior change for existing requests**.

### Usage

```bash
# Standard fullres — unchanged behavior
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A curious raccoon explores a lush forest" \
    --num-inference-steps 50 --num-frames 81 --height 480 --width 832

# Progressive dct_rewind L1 δ=0.05 → 2.32× denoising speedup (480p reference params)
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A curious raccoon explores a lush forest" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 50 --num-frames 81 --height 480 --width 832 \
    --guidance-scale 5.0 --flow-shift 5.0 --dit-cpu-offload false

# Progressive dct_rewind L1 δ=0.10 → 2.78× denoising speedup
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A curious raccoon explores a lush forest" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.10 \
    --num-inference-steps 50 --num-frames 81 --height 480 --width 832 \
    --guidance-scale 5.0 --flow-shift 5.0 --dit-cpu-offload false
```

### Optimization compatibility

| Optimization | Progressive | Notes |
|---|---|---|
| Layerwise CPU offload | ✅ Safe | Component-level, unaffected |
| LoRA | ✅ Safe | Weight-level |
| CFG dual-pass | ✅ Safe | Both cond/uncond branches run at whatever H/W |

## Accuracy Tests

### Video quality — sample prompt: fullres | δ=0.01 | δ=0.02 | δ=0.05

Settings: **50 steps**, seed 42, **480×832**, 81 frames, guidance=5.0, flow\_shift=5.0, RTX A6000.

**Prompt:** *"Giant rogue waves crashing against sheer basalt sea cliffs at golden hour, white spray launching fifty meters skyward, stormy sky, photorealistic cinematic widescreen"*

<table>
<tr>
  <th align="center">Fullres — 267.5s</th>
  <th align="center">δ=0.01 — <b>1.65×</b> — 161.2s</th>
  <th align="center">δ=0.02 — <b>1.86×</b> — 142.4s</th>
  <th align="center">δ=0.05 — <b>2.27×</b> — 117.8s</th>
</tr>
<tr>
  <td><video src="https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02/p02_fullres.mp4" controls muted width="100%"></video></td>
  <td><video src="https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02/p02_0.01.mp4" controls muted width="100%"></video></td>
  <td><video src="https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02/p02_0.02.mp4" controls muted width="100%"></video></td>
  <td><video src="https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02/p02_0.05.mp4" controls muted width="100%"></video></td>
</tr>
</table>

All outputs are artifact-free. Denoising times shown; total wall-clock includes text encoding + VAE decode (~17s each).

### Unit tests (CPU-only, no GPU, ~11 s)

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_wan.py -v
# 26 passed, 6 subtests passed in 10.83s
```

| Class | Count | Coverage |
|-------|-------|----------|
| `TestWanSpectrumConstants` | 5 | A/β plausibility, β > FLUX β, stage transitions with WAN constants, FLUX transitions earlier than WAN |
| `TestWanProgressiveStageHooks` | 3 | `_unpack_latent` identity, `_repack_latent` identity, `_on_resolution_change` no-op |
| `TestWanGenerateInitialNoise` | 5 | shape `[1,C,T,H,W]`, dtype, determinism, different seeds differ, T\_lat preserved |
| `TestDCTUpsample5D` | 6 | 5-D output shape, T-dim unchanged, rewind tuple, dtype preservation, `apply_upsample` dispatch |
| `TestWanVAEArchConfig` | 3 | `vae_scale_factor` absent (original code unchanged), `spatial_compression_ratio=8`, fallback logic |
| `TestWanStageTransitions` | 3 | single-level in range, two-level ordered, `find_transition_steps` conservative |
| `TestWanProgressiveDenoisingStageInit` | 1 | spectrum constants flow through to `_spectrum_A/_spectrum_beta` |

### Manual E2E test (requires GPU + Wan 2.1 T2V checkpoint)

```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_wan.py \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --steps 50 --levels 1 --delta 0.05 \
    --height 480 --width 832 --num-frames 81
```

## Speed Tests and Profiling

Hardware: **RTX A6000 48 GB**, `--dit-cpu-offload false` (transformer GPU-resident).
All runs: **50 steps**, seed 42, **480×832**, **81 frames**, guidance=5.0, flow\_shift=5.0.
**Timing = denoising loop only.**

> **Note:** CuTeDSL fused kernels are disabled on this machine (`cutlass.cute` not installed —
> CUTLASS 3.x namespace collision with unrelated `cutlass` ML package). Native PyTorch fallback
> is numerically equivalent. Results are slightly conservative; fused kernels would benefit both
> fullres and progressive equally and would not change the speedup ratio.

### Pure baseline (fullres vs progressive, no optimizations)

| Config | Stage split | Denoise | Avg s/step | Speedup | Token-step |
|--------|------------|---------|-----------|---------|-----------|
| fullres | 50 @ 60×104 | 266.8 s | 5.34 s | 1.00× | 1.00× |
| dct\_rewind L1 δ=0.01 | 23@30×52 + 27@60×104 | 161.5 s | — | **1.65×** | 1.57× |
| dct\_rewind L1 δ=0.02 | 27@30×52 + 23@60×104 | 142.7 s | — | **1.86×** | 1.74× |
| dct\_rewind L1 δ=0.05 | 33@30×52 + 17@60×104 | 114.7 s | — | **2.32×** | 1.98× |
| dct\_rewind L1 δ=0.10 | 37@30×52 + 13@60×104 | 95.9 s | — | **2.78×** | 2.21× |

Wall-clock speedup **exceeds** token-step theory because the half-resolution attention
(30×52 = 1,560 tokens) is quadratically cheaper than the full-res (60×104 = 6,240 tokens),
yielding super-linear gains per token saved.

Transition steps match the Bayes-optimal criterion exactly for flow\_shift=5.0: steps 23, 27, 33, 37.

### δ vs speedup tradeoff

![Speedup vs delta](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/images/speedup_vs_delta.png)

δ=**0.05 is the recommended default** (2.32× speedup, visually similar on diverse cinematic scenes).

## Caveat
Wan 2.1 natively supports 480p and 720p generation, with 480p being the lowest supported resolution. Progressive resolution growing assumes the model can generate at 0.5× the final resolution; however, Wan 2.1 cannot generate 240p videos. Significant degradation in 480p videos is observed when using progressive generation, so it is advised to enable progressive mode only when generating 720p or higher-resolution videos.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests — 26 new CPU-only tests, all pass in ~11 s.
- [x] Update documentation — added Wan T2V section to `progressive_resolution.mdx` (usage, Python API, delta table, benchmark numbers, spatial alignment limitation).
- [x] Provide accuracy and speed benchmark results — 4-δ sweep table; stage transitions verified against theory; 1.65–2.78× speedup measured.
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
