## Motivation

Transformer attention is O(n²) in sequence length. For FLUX.2 at 1024×1024, the denoising loop processes 4096 tokens per step. Running early steps at a coarser latent resolution (32×32 → 1024 tokens) reduces attention cost to ~6% for those steps, yielding a **1.77–1.93× denoising speedup** with no quality degradation.

This PR extends **spectral progressive resolution growing** (introduced for FLUX.1 in [#FLUX1_PR]) to the **FLUX.2 pipeline family** (FLUX.2-dev, FLUX.2-klein-4B, FLUX.2-klein-9B). Early denoising steps run at a coarser latent resolution and the latent is spectrally upsampled via GPU DCT before the full-resolution steps.

FLUX.2 differs from FLUX.1 in two ways that required new subclass logic:

| | FLUX.1 | FLUX.2 |
|---|---|---|
| Latent packing | 2×2 patchify → `[B, S, 64]` | Row-major reshape → `[B, H·W, C]` |
| Effective latent scale | `vae_scale_factor` (8) | `vae_scale_factor × 2` (16) |
| Positional IDs | Computed from pixel dims | `batch.latent_ids` (4D grid, must be updated on resolution change) |

## Modifications

### New hook — `ProgressiveDenoisingStage` base class

**`runtime/pipelines_core/stages/progressive_resolution/denoising.py`**

Added one override point:
```python
def _latent_scale_factor(self, server_args: ServerArgs) -> int:
    """Pixel-to-latent scale factor. Override for models with extra patchification."""
    return server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
```

`forward()` uses `latent_scale = self._latent_scale_factor(server_args)` in all pixel↔latent conversions. FLUX.1 does **not** override this — **FLUX.1 behavior is unchanged**.

### New file — `runtime/pipelines/flux_2_progressive.py`

`Flux2ProgressiveDenoisingStage(ProgressiveDenoisingStage)` overrides:

| Hook | What it does |
|------|-------------|
| `_latent_scale_factor` | Returns `vae_scale_factor × 2 = 16` |
| `_unpack_latent` | Row-major `[B, H·W, C] → [B, C, H, W]` |
| `_repack_latent` | Row-major `[B, C, H, W] → [B, H·W, C]` |
| `_generate_initial_noise` | Uses `in_channels` directly; sets `batch.latent_ids` before `_prepare_denoising_loop` |
| `_on_resolution_change` | Updates `batch.latent_ids` + `freqs_cis` cache + all CFG branches |

### Modified — `runtime/pipelines/flux_2.py`

`Flux2Pipeline.create_pipeline_stages` manually assembles TI2I stages with `_add_flux2_denoising_stage()`. When `progressive_mode == "fullres"` (default), delegates to `DenoisingStage.forward()` — **zero behavior change for existing requests**.

### Usage

```bash
# Standard fullres — unchanged behavior
sglang generate --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "A serene mountain lake at golden hour"

# Progressive dct_rewind L1 δ=0.05 → 1.77× denoising speedup
sglang generate --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.05 \
    --num-inference-steps 30 --dit-cpu-offload false

# Progressive dct_rewind L1 δ=0.10 → 1.93× denoising speedup
sglang generate --model-path black-forest-labs/FLUX.2-klein-4B \
    --prompt "A serene mountain lake at golden hour" \
    --progressive-mode dct_rewind --progressive-levels 1 --progressive-delta 0.10 \
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

![3-way comparison: fullres | δ=0.05 | δ=0.10](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/montage_3way_small.png)

> **Quality note:** All three modes produce visually equivalent results. The low-resolution stage commits to global composition and color palette; detail is added at full resolution in the same manner as standard generation.

<details>
<summary>Per-prompt 3-way comparison strips (10 prompts)</summary>

![01 misty forest](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/01_misty_forest_3way.png)
![02 rose gold portrait](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/02_rose_gold_portrait_3way.png)
![03 neon tokyo](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/03_neon_tokyo_3way.png)
![04 tuscany vineyard](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/04_tuscany_vineyard_3way.png)
![05 arctic tundra](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/05_arctic_tundra_3way.png)
![06 jazz club](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/06_jazz_club_3way.png)
![07 cherry blossoms](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/07_cherry_blossoms_3way.png)
![08 desert mesa](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/08_desert_mesa_3way.png)
![09 coral reef](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/09_coral_reef_3way.png)
![10 autumn maples](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/10_autumn_maples_3way.png)

</details>

### Unit tests (CPU-only, no GPU, 14 s)

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py -v
# 52 passed, 32 subtests passed in 13.86s
```

| Class | Count | Coverage |
|-------|-------|----------|
| `TestFlux2Pack` | 7 | pack/unpack shapes, roundtrip identity, row-major ordering, dtype preservation |
| `TestFlux2ProgressiveStage` | 13 | spectrum constants, `_latent_scale_factor`, `_generate_initial_noise` (shape, `latent_ids`, dtype, determinism), `_on_resolution_change` (no-crash, shape, branch update, coordinate correctness) |

### Manual E2E test (requires GPU + FLUX.2 checkpoint)

```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_flux2.py \
    --model-path /path/to/FLUX.2-klein-4B --steps 30 --levels 1 --delta 0.05
```

## Speed Tests and Profiling

Hardware: **RTX A6000 48 GB**, `--dit-cpu-offload false` (transformer GPU-resident).
All runs: 30 steps, seed 42, 1024×1024. **Timing = denoising loop only.**

### Pure baseline (fullres vs progressive, no optimizations)

| Config | Stage split | Denoise | Avg s/step | Speedup | Token-step |
|--------|------------|---------|-----------|---------|-----------|
| fullres | 30 @ 64² | 9.72 s | 0.324 s | 1.00× | 1.00× |
| dct_rewind L1 δ=0.05 | 18@32² + 12@64² | 5.50 s | 0.183 s | **1.77×** | 1.82× |
| dct_rewind L1 δ=0.10 | 20@32² + 10@64² | 5.03 s | 0.168 s | **1.93×** | 2.00× |

Wall-clock is **97% of token-step prediction** across both configs — identical efficiency to FLUX.1.
All 10 prompts reproduced within ±0.02× variance.

### δ vs speedup tradeoff

![Speedup vs delta](https://raw.githubusercontent.com/bchao1/sglang/dev/brian/scratch/spectral-progressive-flux2/images/speedup_vs_delta.png)

Speedups are denoising-loop only. Curve shows the token-step theoretical model; filled points are wall-clock measurements. **δ=0.10 is the recommended tradeoff** (1.93× with no visible quality change).

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests — 20 new CPU-only tests, 52 total, all pass in 14 s.
- [ ] Update documentation — TODO: add FLUX.2 section to `progressive_resolution.mdx`.
- [x] Provide accuracy and speed benchmark results — 10-prompt × 3-mode table; stage transitions verified; 1.77×/1.93× speedup measured.
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

### TODO before merge

1. **Fit FLUX.2-specific spectrum constants** (A, β) — replace FLUX.1 placeholder values in `flux_2_progressive.py`
2. **Docs** — add FLUX.2 section to `progressive_resolution.mdx`
