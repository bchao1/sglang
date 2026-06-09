> **⚠️ Stacked PR:** Depends on the FLUX.1 progressive PR ([PR #XXXX](https://github.com/sgl-project/sglang/pull/XXXX)). The Z-Image-specific changes are the top 2 commits: `feat: progressive resolution growing for Z-Image` and `fix: Z-Image negative embedding batch dim, CuTeDSL fallback`. All earlier commits belong to the FLUX PR and can be ignored during review.

## Motivation

Transformer attention is O(n²) in sequence length. For Z-Image at 1024×1024, each denoising step runs **dual CFG** (two forward passes) on 4096 tokens. Running early steps at a coarser latent resolution (e.g., 64×64 → 128×128 tokens) reduces per-step attention cost to 6.25% for those steps, yielding a **2.07× denoising speedup** at δ=0.05.

This PR extends the spectral progressive resolution framework from FLUX.1 ([PR #26961](https://github.com/sgl-project/sglang/pull/26961)) to **Z-Image**, implementing the model-specific hooks required to handle Z-Image's 5-D latent format and caption+image RoPE positional embeddings.

## Modifications

### New file — `runtime/pipelines/zimage_progressive.py`

`ZImageProgressiveDenoisingStage` handles Z-Image-specific:

| Hook | What it does |
|------|-------------|
| `_generate_initial_noise` | Generates `[B, C, 1, H, W]` noise (5-D, float32); overrides base class which uses FLUX-specific `in_channels // 4` formula |
| `_unpack_latent` | `squeeze(2)` removes the frame dim: `[B, C, 1, H, W] → [B, C, H, W]` for DCT spectral ops |
| `_repack_latent` | `unsqueeze(2)` restores frame dim: `[B, C, H, W] → [B, C, 1, H, W]` after upsample |
| `_on_resolution_change` | Recomputes the full `(cap_freqs_cis, x_freqs_cis)` tuple for the new resolution and updates all CFG branches (2 branches = cond + uncond) |

**Key design note:** Z-Image's freqs_cis encodes both caption and image positions in a single tuple; the image RoPE offsets depend on caption length. We recompute the full tuple on every stage transition (no cache) to avoid cross-request contamination.

### Modified files

| File | Change |
|------|--------|
| `runtime/pipelines/zimage_pipeline.py` | Replace `add_standard_t2i_stages()` (which creates a plain `DenoisingStage`) with explicit stage setup calling `_add_zimage_denoising_stage()` — routes to `DenoisingStage.forward()` when `progressive_mode == "fullres"` (**zero behavior change for existing requests**) |
| `runtime/pipelines_core/stages/text_encoding.py` | Fix `_append_negative_text_outputs`: `target_batch_sizes` was using `pe.shape[0]` (= seq_len for Z-Image's 2-D embeddings) instead of batch size, causing `negative_prompt_embeds batch dimension mismatch`. Fix: detect 2-D tensors and treat as batch=1. |
| `runtime/models/dits/zimage.py` | Guard fused CuTeDSL norm kernel with `_has_cutlass_cute` flag (graceful fallback to native RMSNorm when CUTLASS Python bindings are unavailable). |
| `runtime/layers/layernorm.py` | Same `_has_cutlass_cute` guard in three `forward_cuda` paths. |
| `test/unit/test_progressive_upsample.py` | +10 `TestZImagePackUnpack` CPU-only tests |
| `test/manual/test_progressive_zimage.py` | E2E manual test (requires GPU + Z-Image checkpoint) |

### Spectrum constants

Z-Image uses the same VAE as FLUX.1 (`FluxVAEConfig`), so the power-law spectrum constants are identical:
```
ZIMAGE_SPECTRUM_A = 203.615097    # same as FLUX
ZIMAGE_SPECTRUM_BETA = 1.915461   # same as FLUX
```
Confirmed by the reference implementation: `A, beta = FLUX_SPECTRUM_A, FLUX_SPECTRUM_BETA  # same VAE as FLUX`.

### Usage

```bash
# Standard fullres — unchanged behavior
sglang generate --model-path Tongyi-MAI/Z-Image \
    --prompt "A serene mountain lake at golden hour" \
    --height 1024 --width 1024

# Progressive dct_rewind L1 δ=0.05 → 2.07× denoising speedup
sglang generate --model-path Tongyi-MAI/Z-Image \
    --prompt "A serene mountain lake at golden hour" \
    --height 1024 --width 1024 \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 50 --dit-cpu-offload false

# Progressive dct_rewind L1 δ=0.10 → 2.33× denoising speedup
sglang generate --model-path Tongyi-MAI/Z-Image \
    --prompt "A serene mountain lake at golden hour" \
    --height 1024 --width 1024 \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.10 \
    --num-inference-steps 50 --dit-cpu-offload false
```

**Note:** Always specify `--height 1024 --width 1024` (or another resolution where H_lat and W_lat are divisible by 2). Z-Image's default resolution (360×640) produces a 45×80 latent where H=45 is not divisible by the patch size.

## Accuracy Tests

### Image quality — 10 diverse prompts: fullres | δ=0.05 | δ=0.10

Settings: 50 steps, seed 42, 1024×1024, GPU RTX A6000, `--dit-cpu-offload false`.
Labels show denoising-loop time only. All outputs are artifact-free.

![3-way comparison: fullres | δ=0.05 | δ=0.10](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage_3way/montage_3way.jpg)

<details>
<summary>Fullres vs δ=0.05 side-by-side strips (10 prompts)</summary>

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

### Unit tests (CPU-only, no GPU, <1 s)

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py -v
# 41 tests: OK  (includes 10 new TestZImagePackUnpack tests)
```

| Class | Count | Coverage |
|-------|-------|----------|
| `TestDCT` | 7 | DCT-II / IDCT-II vs scipy, Parseval identity, float32 precision |
| `TestDCTUpsample` | 11 | output shapes, dtype, low-freq embed, rewind formula, determinism |
| `TestSchedulerUtils` | 8 | stage transitions, δ-monotonicity, multi-level ordering, scheduler reset |
| `TestProgressiveDenoisingStageBase` | 5 | `_get_seed` fallback, FLUX spectrum constants |
| `TestZImagePackUnpack` | **10** | **squeeze/unsqueeze round-trip, dtype preservation, spectrum constants match FLUX, stage transitions** |

### Manual E2E test (requires GPU + Z-Image checkpoint)

```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_zimage.py \
    --model-path /path/to/Tongyi-MAI/Z-Image \
    --height 1024 --width 1024 --steps 30 --levels 1 --delta 0.05
```

## Speed Tests and Profiling

Hardware: **RTX A6000 48 GB**, `--dit-cpu-offload false` (transformer GPU-resident), 50 steps, seed 42, 1024×1024.
**Timing = denoising loop only** (model load, text encoding, VAE decode excluded).

> **Why Z-Image speedups are larger than FLUX.1:** Z-Image uses dual CFG (2 forward passes per step), so attention computation represents a larger fraction of each step's cost. Progressive growing — which reduces attention from O(4096²) to O(1024²) per low-res step — saves 16× the attention compute, and since dual CFG doubles the absolute attention cost at full-res, the overall ratio to fixed overhead is more favorable.

### Group A — pure baseline (fullres vs progressive, no optimizations)

| Config | Stage split | Denoise | Avg s/step | Speedup | Token-step |
|--------|-------------|---------|-----------|---------|-----------|
| fullres | 50 @ 128² | 52.72 s | 1.054 s | 1.00× | 1.00× |
| dct_rewind L1 δ=0.01 | 26@64² + 24@128² | 32.46 s | 0.649 s | **1.62×** | 2.00× |
| dct_rewind L1 δ=0.05 | 35@64² + 15@128² | 25.41 s | 0.508 s | **2.07×** | 2.75× |
| dct_rewind L2 δ=0.01 | 15@32² + 11@64² + 24@128² | 30.02 s | 0.600 s | **1.76×** | 2.28× |

> Wall-clock speedup is lower than token-step speedup due to fixed per-step overhead (scheduler `.step()`, memory allocs, freqs_cis recomputation at stage transition). The gap is larger for Z-Image than FLUX because Z-Image has more fixed overhead per step (dual CFG text encoding, RoPE computation for both branches).

### δ vs speedup tradeoff

![Speedup vs delta](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-z-image/pr_visuals/progressive_zimage/speedup_vs_delta.png)

Speedups measured on RTX A6000, 50 steps, 1024×1024, denoising loop only. **δ=0.10 is the recommended tradeoff** (2.33× speedup with visually equivalent quality).

| δ | Stage split | Denoise | Speedup | Notes |
|---|-------------|---------|---------|-------|
| 0.01 | 26@64² + 24@128² | 34.38 s | **1.53×** | Conservative |
| 0.05 | 35@64² + 15@128² | 26.03 s | **2.03×** | Default recommendation |
| **0.10** | 42@64² + 8@128² | 22.65 s | **2.33×** | ⭐ Best quality/speed tradeoff |
| 0.20 | 42@64² + 8@128² | 19.77 s | **2.66×** | More aggressive |
| 0.50 | 48@64² + 2@128² | 17.74 s | **2.97×** | Maximum tested |

### FLUX.1 vs Z-Image progressive speedup comparison

Z-Image consistently achieves higher progressive speedups than FLUX.1 at the same δ because dual CFG doubles the absolute attention savings:

| δ | FLUX.1 speedup | Z-Image speedup | Ratio |
|---|---------------|-----------------|-------|
| 0.01 | 1.32× | **1.53×** | +16% |
| 0.05 | 1.63× | **2.03×** | +25% |
| 0.10 | 1.83× | **2.33×** | +27% |

## Checklist

- [x] Format your code according to [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). — 10 new CPU-only unit tests in `TestZImagePackUnpack`; all 41 tests pass.
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). — 10-prompt image comparison + Group A / delta-sweep benchmarks above.
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).



















<!-- pr-states:start -->
---
### CI States

Latest PR Test (Base): <!-- slot:pr-test:start -->:x: [Run #26791815970](https://github.com/bchao1/sglang/actions/runs/26791815970)<!-- slot:pr-test:end -->
Latest PR Test (Extra): <!-- slot:pr-test-extra:start -->:x: [Run #26791815898](https://github.com/bchao1/sglang/actions/runs/26791815898)<!-- slot:pr-test-extra:end -->
<!-- pr-states:end -->
