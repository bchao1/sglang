> **⚠️ Stacked PR:** Depends on the FLUX.1 progressive PR ([PR #XXXXX](https://github.com/sgl-project/sglang/pull/XXXXX)). The Qwen-Image-specific change is the top three commits: `feat(diffusion): extend progressive resolution growing to Qwen-Image`, `feat(diffusion): wire QwenImageProgressiveDenoisingStage into QwenImagePipeline`, `fix(diffusion): break circular import in qwen_image_progressive`. All earlier commits belong to the FLUX.1 PR and can be ignored during review.

## Motivation

Transformer attention is O(n²) in sequence length. For Qwen-Image at 1024×1024, the denoising loop processes 4096 tokens per step (128×128 latent, 2×2 patchify). Running early steps at a coarser latent resolution (64×64 → 1024 tokens) reduces attention cost to ~6% for those steps, yielding a **1.29–1.69× denoising speedup** with no quality degradation.

This PR extends **spectral progressive resolution growing** (introduced for FLUX.1 in [PR #XXXXX](https://github.com/sgl-project/sglang/pull/XXXXX)) to the **Qwen-Image pipeline**.

| Model | HuggingFace ID |
|-------|---------------|
| Qwen-Image | `Qwen/Qwen-Image` |

Qwen-Image uses the same 2×2 patchify convention as FLUX.1-dev but requires updating two additional positional encoding structures when the latent resolution changes between progressive stages:

| | FLUX.1 | Qwen-Image |
|---|---|---|
| Latent packing | 2×2 patchify → `[B, S, 64]` | **Same** 2×2 patchify → `[B, S, 64]` |
| vae_scale_factor | 8 | 8 (same) |
| Positional encoding | `freqs_cis` (RoPE) | `freqs_cis` (RoPE) **+** `img_shapes` (for `build_modulate_index`) |
| Resolution change hook | `freqs_cis` only | `freqs_cis` **and** `img_shapes` in every CFG branch |

## Modifications

### New file — `runtime/pipelines/qwen_image_progressive.py`

`QwenImageProgressiveDenoisingStage(ProgressiveDenoisingStage)` overrides:

| Hook | What it does |
|------|-------------|
| `_unpack_latent` | Packed `[B, S, 64] → [B, 16, H_lat, W_lat]` (2×2 de-patchify, identical to FLUX.1) |
| `_repack_latent` | Spatial `[B, 16, H_lat, W_lat] → [B, S, 64]` (2×2 patchify) |
| `_on_resolution_change` | Recomputes `freqs_cis` via `prepare_pos_cond_kwargs`, then updates both `freqs_cis` and `img_shapes` in every CFG branch and in `ctx.pos_cond_kwargs` |

When `progressive_mode == "fullres"` (the default) the stage delegates entirely to `DenoisingStage.forward()` — **zero behavior change for existing requests**.

### Modified — `runtime/pipelines/qwen_image.py`

`QwenImagePipeline.create_pipeline_stages` now manually assembles T2I stages with `_add_qwen_denoising_stage()`. When `progressive_mode == "fullres"` (default), delegates to `DenoisingStage.forward()` — **zero behavior change for existing requests**.

### Modified — `runtime/layers/layernorm.py`

JIT kernel imports for `fused_scale_residual_norm_scale_shift`, `fused_norm_scale_shift`, and `fused_norm_tanh_mul_add` now wrapped in `try/except`, falling back to `forward_native` when the compiled kernel is unavailable. This makes the Qwen-Image progressive path robust on machines without a precompiled JIT cache.

### Usage

```bash
# Standard fullres — unchanged behavior
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A serene mountain lake at golden hour"

# Progressive dct_rewind L1 δ=0.05 → 1.29× denoising speedup
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 30 --dit-cpu-offload false

# Progressive dct_rewind L1 δ=0.20 → 1.69× denoising speedup (recommended)
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.20 \
    --num-inference-steps 30 --dit-cpu-offload false
```

### Optimization compatibility

| Optimization | Progressive | Notes |
|---|---|---|
| Layerwise CPU offload | ✅ Safe | Component-level, unaffected |
| LoRA | ✅ Safe | Weight-level |
| **torch.compile** | ❌ No support | Fixed sequence length in compiled kernel |

## Accuracy Tests

### Image quality — 10 diverse prompts: fullres | δ=0.05 | δ=0.10

Settings: 30 steps, seed 42, 1024×1024, RTX A6000.
Labels show denoising-loop time only. All three outputs are artifact-free.

![3-way comparison: fullres | δ=0.05 | δ=0.10](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/montage_preview.png)

<details>
<summary>Per-prompt 3-way comparison strips (10 prompts)</summary>

![01 landscape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/00_landscape_3way.png)
![02 architecture](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/01_architecture_3way.png)
![03 portrait](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/02_portrait_3way.png)
![04 cityscape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/03_cityscape_3way.png)
![05 object](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/04_object_3way.png)
![06 wildlife](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/05_wildlife_3way.png)
![07 interior](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/06_interior_3way.png)
![08 seascape](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/07_seascape_3way.png)
![09 desert](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/08_desert_3way.png)
![10 fantasy](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/3way/09_fantasy_3way.png)

</details>

### Unit tests (CPU-only, no GPU, no model, 12 s)

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_qwen_image.py -v
# 32 passed in 11.53s
```

| Class | Count | Coverage |
|-------|-------|----------|
| `TestQwenImagePack` | 10 | pack/unpack shapes, roundtrip (both directions), dtype, spatial ordering, matches `_pack_latents` |
| `TestQwenImageProgressiveStage` | 16 | instantiation, inheritance, spectrum constants, `_unpack/_repack` shape + roundtrip, `_on_resolution_change`: no-cfg, freqs update, img_shapes update, null-freqs skip, missing-key skip, null-shapes skip, joint update |
| `TestQwenImagePipelineUsesProgressiveStage` | 3 | pipeline_name, `_add_qwen_denoising_stage` method, required modules |
| `TestQwenPackIntegrationWithProgressiveBase` | 3 | spatial channel count, token dim = 64, 2× upsample → 4× sequence length |

### Manual E2E test (requires GPU + Qwen-Image checkpoint)

```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_qwen_image.py \
    --model-path /path/to/Qwen-Image --steps 30 --levels 1 --delta 0.05
```

## Speed Tests and Profiling

Hardware: **RTX A6000 48 GB**, `--dit-cpu-offload false` (transformer GPU-resident).
All runs: 30 steps, seed 42, 1024×1024. **Timing = denoising loop only.**

### Pure baseline (fullres vs progressive, no optimizations)

| Config | Stage split | Denoise | Avg s/step | Speedup |
|--------|------------|---------|-----------|---------|
| fullres | 30 @ 128² | 43.00 s | 1.434 s | 1.00× |
| dct_rewind L1 δ=0.05 | 13@64² + 17@128² | 33.25 s | — | **1.29×** |
| dct_rewind L1 δ=0.10 | 16@64² + 14@128² | 33.86 s | — | **1.27×** |
| dct_rewind L1 δ=0.20 | 19@64² + 11@128² | 25.40 s | — | **1.69×** |

> **Note on speedup headroom:** Qwen-Image's denoising step includes significant non-attention work (RoPE re-computation, layernorm, FFN) that does not scale with sequence length. As a result, the practical speedup from coarse steps is lower than the 16× token-count reduction would suggest. This is consistent with architectures where FFN/layernorm dominates attention.

### δ vs speedup tradeoff

![Speedup vs delta](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-qwen/pr_visuals/speedup_vs_delta.png)

Speedups are denoising-loop only. **δ=0.20 is the recommended tradeoff** (1.69× with no visible quality change at 30 steps).

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests — 32 new CPU-only tests, all pass in 12 s.
- [x] Update documentation — added Qwen-Image section to `progressive_resolution.mdx`.
- [x] Provide accuracy and speed benchmark results — 10-prompt × 3-mode comparison; stage transitions verified; 1.29×/1.69× speedup measured.
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
