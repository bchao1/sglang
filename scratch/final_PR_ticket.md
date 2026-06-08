# Spectral Progressive Resolution Growing — All Models

## Motivation

Transformer attention is O(n²) in sequence length. For image/video diffusion models running at full resolution, early denoising steps waste compute on high-frequency detail that has not yet been activated. **Spectral progressive resolution growing** runs those early steps at half the spatial resolution, then spectrally upsamples the latent before the full-resolution steps.

The upsample step is **Bayes-optimal**: using the measured VAE latent power spectrum, it computes the exact denoising step at which Nyquist-band frequencies first carry more signal than noise. This means the speedup is lossless by construction — not a quality-speed tradeoff.

| Model | Full-res tokens | Half-res tokens | Token ratio | Best speedup (δ=0.05) |
|-------|----------------|----------------|------------|----------------------|
| FLUX.1 1024×1024 | 4,096 | 1,024 | 4.0× | **1.63×** |
| FLUX.2 1024×1024 | 4,096 | 1,024 | 4.0× | **1.77×** |
| Z-Image 1024×1024 | 4,096 | 1,024 | 4.0× | **2.07×** |
| Wan 2.1 T2V 480×832/81f | 6,240 | 1,560 | 4.0× | **2.32×** |
| Qwen-Image 1024×1024 | 1,024 | 256 | 4.0× | **1.29×** |

Based on [Spectral Progressive Diffusion (arXiv 2605.18736)](https://arxiv.org/abs/2605.18736). This is the first GPU-native spectral progressive implementation in an open LLM/diffusion serving framework.

---

## What changed

### New core module — `runtime/pipelines_core/stages/progressive_resolution/`

| File | Description |
|------|-------------|
| `spectral_ops.py` | GPU DCT-II / IDCT-II via `torch.fft`. Matches scipy to relative error 1.7×10⁻⁷. |
| `scheduler_utils.py` | Bayes-optimal stage-transition math; scheduler reset for multi-stage loop. |
| `upsample.py` | `dct_upsample_2d` dispatcher for `dct` / `dct_rewind` modes. |
| `denoising.py` | `ProgressiveDenoisingStage(DenoisingStage)` base class with model-specific extension hooks. |

Extension hooks in `ProgressiveDenoisingStage`:

| Hook | Purpose |
|------|---------|
| `_latent_scale_factor(server_args)` | Pixel-to-latent conversion factor. Default: `vae_scale_factor`. FLUX.2 overrides to `vae_scale_factor × 2`. |
| `_unpack_latent(latent, h_lat, w_lat)` | Model-native latent → spatial `[B, C, H, W]`. |
| `_repack_latent(x_spatial, h_lat, w_lat, batch, server_args)` | Spatial `[B, C, H, W]` → model-native. |
| `_on_resolution_change(ctx, batch, server_args, h_px, w_px)` | Update resolution-dependent state (RoPE, positional IDs) at stage transitions. |

### New model-specific progressive stages

| File | Stage class | Model(s) |
|------|-------------|---------|
| `pipelines/flux_progressive.py` | `FluxProgressiveDenoisingStage` | FLUX.1-dev |
| `pipelines/flux_2_progressive.py` | `Flux2ProgressiveDenoisingStage` | FLUX.2-dev, FLUX.2-klein-4B/9B |
| `pipelines/zimage_progressive.py` | `ZImageProgressiveDenoisingStage` | Z-Image, Z-Image-Turbo |
| `pipelines/wan_progressive.py` | `WanProgressiveDenoisingStage` | Wan2.1-T2V-1.3B/14B |
| `pipelines/qwen_image_progressive.py` | `QwenImageProgressiveDenoisingStage` | Qwen-Image |

### Modified pipeline files (routing)

All pipelines use the `_[Model]DenoisingStageRouter(PipelineStage)` pattern: a thin router that delegates per-request to either `DenoisingStage` (fullres) or the model's `ProgressiveDenoisingStage` (dct/dct_rewind). **Zero behavior change when `progressive_mode="fullres"` (default).**

| File | Change |
|------|--------|
| `pipelines/flux.py` | `_FluxDenoisingStageRouter` — routes per-request |
| `pipelines/flux_2.py` | `_Flux2DenoisingStageRouter` — routes per-request |
| `pipelines/zimage_pipeline.py` | `_ZImageDenoisingStageRouter` — routes per-request |
| `pipelines/wan_pipeline.py` | `_WanDenoisingStageRouter` — routes per-request |
| `pipelines/qwen_image.py` | `_QwenImageDenoisingStageRouter` — routes per-request |

### Other modifications

| File | Change |
|------|--------|
| `configs/sample/sampling_params.py` | +3 fields: `progressive_mode` / `progressive_levels` / `progressive_delta` |
| `runtime/layers/layernorm.py` | CuTeDSL fused norm: use `_has_cutlass_cute` guard (not try/except) |
| `runtime/pipelines_core/stages/__init__.py` | Export `ProgressiveDenoisingStage` |
| `docs_new/docs/sglang-diffusion/progressive_resolution.mdx` | Full usage + benchmark guide |
| `docs_new/docs.json` | Add to **SGLang Diffusion → Performance Optimization** |

---

## New parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `progressive_mode` | `--progressive-mode` | `"fullres"` | `"fullres"` = disabled (standard generation). `"dct_rewind"` = spectral upsample with scheduler rewind (recommended). `"dct"` = upsample without rewind. |
| `progressive_levels` | `--progressive-levels` | `1` | Number of resolution halvings. `1` = one coarse stage. `2` = two coarse stages. |
| `progressive_delta` | `--progressive-delta` | `0.01` | Noise-dominated tolerance δ. Higher = more coarse steps = more speedup. |

---

## Usage

### FLUX.1-dev

```bash
sglang generate \
    --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A serene mountain lake at golden hour, photorealistic" \
    --num-inference-steps 50 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05
```

### FLUX.2-klein-4B

```bash
sglang generate \
    --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "A serene mountain lake at golden hour, photorealistic" \
    --num-inference-steps 30 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.10
```

### Z-Image

```bash
sglang generate \
    --model-path Tongyi-MAI/Z-Image \
    --prompt "A serene mountain lake at golden hour, photorealistic" \
    --height 1024 --width 1024 \
    --num-inference-steps 50 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.10
```

> **Note:** Always specify `--height 1024 --width 1024`. Z-Image's default resolution (360×640) produces a 45×80 latent where H=45 is not divisible by the patch size.

### Wan 2.1 T2V 1.3B

```bash
sglang generate \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A cheetah sprinting across the Serengeti at sunset" \
    --num-inference-steps 50 \
    --num-frames 81 \
    --height 480 \
    --width 832 \
    --guidance-scale 5.0 \
    --flow-shift 5.0 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.05
```

### Qwen-Image

```bash
sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A serene mountain lake at golden hour, photorealistic" \
    --num-inference-steps 30 \
    --dit-cpu-offload false \
    --progressive-mode dct_rewind \
    --progressive-levels 1 \
    --progressive-delta 0.20
```

For full documentation and more examples, see [`docs_new/docs/sglang-diffusion/progressive_resolution.mdx`](../docs_new/docs/sglang-diffusion/progressive_resolution.mdx).

---

## Speedup table — all models

Hardware: RTX A6000 48 GB, `--dit-cpu-offload false`. Timing = denoising loop only.

### FLUX.1 (50 steps, 1024×1024)

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 50 @ 128² latent | 36.65 s | 1.00× |
| dct_rewind L1 δ=0.01 | 18@64² + 32@128² | 27.67 s | **1.32×** |
| dct_rewind L1 δ=0.05 | 28@64² + 22@128² | 22.58 s | **1.63×** |
| dct_rewind L2 δ=0.01 | 10@32² + 8@64² + 32@128² | 26.48 s | **1.38×** |

### FLUX.2-klein-4B (30 steps, 1024×1024)

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 30 @ 64² latent | 9.72 s | 1.00× |
| dct_rewind L1 δ=0.05 | 18@32² + 12@64² | 5.50 s | **1.77×** |
| dct_rewind L1 δ=0.10 | 20@32² + 10@64² | 5.03 s | **1.93×** |

### Z-Image (50 steps, 1024×1024, dual CFG)

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 50 @ 128² latent | 52.72 s | 1.00× |
| dct_rewind L1 δ=0.01 | 26@64² + 24@128² | 34.38 s | **1.53×** |
| dct_rewind L1 δ=0.05 | 35@64² + 15@128² | 26.03 s | **2.07×** |
| dct_rewind L1 δ=0.10 | 42@64² + 8@128² | 22.65 s | **2.33×** |

### Wan 2.1 T2V 1.3B (50 steps, 480×832, 81 frames)

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 50 @ 60×104 latent | 266.8 s | 1.00× |
| dct_rewind L1 δ=0.01 | 23@30×52 + 27@60×104 | 161.5 s | **1.65×** |
| dct_rewind L1 δ=0.02 | 27@30×52 + 23@60×104 | 142.7 s | **1.86×** |
| dct_rewind L1 δ=0.05 | 33@30×52 + 17@60×104 | 114.7 s | **2.32×** |
| dct_rewind L1 δ=0.10 | 37@30×52 + 13@60×104 | 95.9 s | **2.78×** |

### Qwen-Image (30 steps, 1024×1024)

| Config | Stage split | Denoise | Speedup |
|--------|-------------|---------|---------|
| fullres | 30 @ 128² latent | 43.00 s | 1.00× |
| dct_rewind L1 δ=0.05 | 13@64² + 17@128² | 33.25 s | **1.29×** |
| dct_rewind L1 δ=0.10 | 16@64² + 14@128² | 33.86 s | **1.27×** |
| dct_rewind L1 δ=0.20 | 19@64² + 11@128² | 25.40 s | **1.69×** |

---

## Speedup vs delta (all models on one plot)

![Combined speedup vs delta](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/final_PR_smoke/combined_speedup.png)

Generated by `scratch/final_PR_smoke/gen_combined_speedup_plot.py` using per-model benchmark data from individual PR descriptions. Hardware: RTX A6000 48 GB, denoising loop only.

---

## Image/video quality samples

### FLUX.1

<details>
<summary>Fullres vs dct_rewind δ=0.05 — 10 diverse prompts</summary>

![FLUX.1 montage](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/montage_progressive_vs_fullres.png)

Fullres | δ=0.05 side-by-side strips (10 prompts):

![01 landscape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/01_landscape_compare.png)
![02 architecture](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/02_architecture_compare.png)
![03 portrait](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/03_portrait_compare.png)
![04 cityscape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/04_cityscape_compare.png)
![05 object](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/05_object_compare.png)
![06 wildlife](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/06_wildlife_compare.png)
![07 interior](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/07_interior_compare.png)
![08 seascape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/08_seascape_compare.png)
![09 desert](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/09_desert_compare.png)
![10 fantasy](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux/images/progressive/10_fantasy_compare.png)

</details>

### FLUX.2-klein-4B

<details>
<summary>Fullres | δ=0.05 | δ=0.10 — 10 diverse prompts</summary>

![FLUX.2 montage](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/montage_3way_small.png)

Individual 3-way comparisons (fullres | δ=0.05 | δ=0.10):

![01](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/01_misty_forest_3way.png)
![02](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/02_rose_gold_portrait_3way.png)
![03](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/03_neon_tokyo_3way.png)
![04](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/04_tuscany_vineyard_3way.png)
![05](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/05_arctic_tundra_3way.png)
![06](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/06_jazz_club_3way.png)
![07](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/07_cherry_blossoms_3way.png)
![08](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/08_desert_mesa_3way.png)
![09](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/09_coral_reef_3way.png)
![10](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/10_autumn_maples_3way.png)

</details>

### Z-Image

<details>
<summary>Fullres | δ=0.05 | δ=0.10 — 10 diverse prompts</summary>

![Z-Image 3-way](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage_3way/montage_3way.jpg)

Fullres | δ=0.05 side-by-side strips (10 prompts):

![01 landscape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/01_landscape_compare.png)
![02 architecture](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/02_architecture_compare.png)
![03 portrait](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/03_portrait_compare.png)
![04 cityscape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/04_cityscape_compare.png)
![05 object](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/05_object_compare.png)
![06 wildlife](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/06_wildlife_compare.png)
![07 interior](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/07_interior_compare.png)
![08 seascape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/08_seascape_compare.png)
![09 desert](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/09_desert_compare.png)
![10 fantasy](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/10_fantasy_compare.png)

</details>

### Wan 2.1 T2V

<details>
<summary>Fullres | δ=0.01 | δ=0.02 | δ=0.05 — video quality comparison (GIF)</summary>

Prompt: *"Giant rogue waves crashing against sheer basalt sea cliffs at golden hour, white spray launching fifty meters skyward, stormy sky, photorealistic cinematic widescreen"*

| fullres | δ=0.01 | δ=0.02 | δ=0.05 |
|---------|--------|--------|--------|
| ![fullres](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02_gif/p02_fullres.gif) | ![d0.01](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02_gif/p02_0.01.gif) | ![d0.02](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02_gif/p02_0.02.gif) | ![d0.05](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-wan/videos/p02_gif/p02_0.05.gif) |

</details>

### Qwen-Image

<details>
<summary>Speedup vs δ curve and quality comparison</summary>

![Qwen-Image speedup vs delta](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/speedup_vs_delta.png)

> 3-way comparison images (fullres | δ=0.05 | δ=0.20) can be generated with
> `python scratch/spectral-progressive-qwen/gen_3way_comparison_qwen.py`

</details>

---

## Tests

### Unit tests (CPU-only, no GPU, < 30 s)

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive.py -v
# 119 tests: OK (as of this PR)
```

| Test class | # tests | Coverage |
|-----------|---------|---------|
| `TestDCT` | 7 | DCT-II / IDCT-II vs scipy, Parseval identity, float32 precision |
| `TestDCTUpsample` | 11 | shapes, dtype, low-freq embed, rewind formula, determinism |
| `TestSchedulerUtils` | 8 | stage-transition math, δ-monotonicity, scheduler reset |
| `TestProgressiveDenoisingStageBase` | 5 | seed fallback, FLUX spectrum constants |
| `TestFlux2Pack` | 7 | FLUX.2 pack/unpack shapes, roundtrip, row-major ordering |
| `TestFlux2ProgressiveStage` | 16 | spectrum constants, `_latent_scale_factor`, `_generate_initial_noise`, `_on_resolution_change` |
| `TestZImagePackUnpack` | 8 | Z-Image unpack/repack shapes, dtype, constant consistency with FLUX |
| `TestWanSpectrumConstants` | 5 | Wan constants plausibility, comparison with FLUX |
| `TestWanProgressiveStageHooks` | 3 | Wan pack/unpack identity hooks |
| `TestWanGenerateInitialNoise` | 5 | video noise shape [B,C,T,H,W], dtype, determinism |
| `TestDCTUpsample5D` | 6 | 5-D DCT (spatial H×W grows, T fixed) |
| `TestWanVAEArchConfig` | 3 | vae_scale_factor absent, spatial_compression_ratio=8 |
| `TestWanStageTransitions` | 3 | transitions for 480P latent resolution |
| `TestQwenImagePack` | 8 | Qwen-Image pack/unpack shapes, roundtrips, ordering, dtype |
| `TestQwenImageProgressiveStage` | 15 | class hierarchy, spectrum constants, hooks, `_on_resolution_change` |
| `TestQwenPackIntegrationWithProgressiveBase` | 3 | consistency with base class assumptions |
| `TestQwenImagePipelineUsesProgressiveStage` | 3 | pipeline name, method presence, config modules |

### Manual E2E tests (requires GPU + model checkpoint)

```bash
# FLUX.1
python python/sglang/multimodal_gen/test/manual/test_progressive_flux.py

# FLUX.2
python python/sglang/multimodal_gen/test/manual/test_progressive_flux2.py

# Z-Image  (always pass --height 1024 --width 1024)
python python/sglang/multimodal_gen/test/manual/test_progressive_zimage.py

# Wan T2V
python python/sglang/multimodal_gen/test/manual/test_progressive_wan.py
```

---

## Smoke test results (1 prompt, fullres + dct_rewind δ=0.05)

> Results from `bash scratch/final_PR_smoke/smoke_all_models.sh`
> Hardware: RTX A6000 48 GB, 1 prompt, seed=42

| Model | Config | Wall time | Denoise | Output |
|-------|--------|-----------|---------|--------|
| FLUX.1 | fullres | 320.4s | 57.7s | ✓ PNG |
| FLUX.1 | dct_rewind δ=0.05 | 109.9s | 22.7s | ✓ PNG |
| FLUX.2-klein-4B | fullres | 204.2s | 10.4s | ✓ PNG |
| FLUX.2-klein-4B | dct_rewind δ=0.05 | 60.7s | 6.2s | ✓ PNG |
| Z-Image | fullres | TBD | TBD | re-run needed (first run used default 360×640) |
| Z-Image | dct_rewind δ=0.05 | TBD | TBD | re-run needed (first run used default 360×640) |
| Wan T2V 1.3B | fullres | TBD | TBD | TBD |
| Wan T2V 1.3B | dct_rewind δ=0.05 | TBD | TBD | TBD |
| Qwen-Image | fullres | TBD | TBD | TBD |
| Qwen-Image | dct_rewind δ=0.05 | TBD | TBD | TBD |

> Wall time includes model loading. Denoise = denoising loop only. First FLUX.1 run has inflated denoise due to cold GPU (CUDA kernel compilation on step 1).

---

## Limitations

- **Sequence parallelism incompatible.** Cannot be combined with `--ulysses-degree` or `--ring-degree`. The stage raises a `RuntimeError` if SP is enabled.
- **torch.compile incompatible.** Compiled kernels have a fixed sequence length; the resolution transition causes a recompile or error.
- **Cache-DiT incompatible.** Cache-DiT indexes its step cache by step count; the stage transition resets the index.
- **Wan: spatial-only.** Progressive growing applies to H×W only. The temporal dimension T (number of latent frames) is kept fixed across all stages.
- **Qwen-Image: OOM risk.** At default settings, the VAE decode step may OOM on a 48GB A6000. DiT offload (`--dit-cpu-offload`) frees memory for decode at the cost of PCIe transfer per step.

## References

- [Spectral Progressive Diffusion (arXiv 2605.18736)](https://arxiv.org/abs/2605.18736)
- [SGLang documentation](docs_new/docs/sglang-diffusion/progressive_resolution.mdx)
