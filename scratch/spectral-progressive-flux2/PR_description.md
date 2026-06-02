## Motivation

This PR extends **spectral progressive resolution growing** (introduced for FLUX.1 in [#FLUX1_PR]) to the **FLUX.2 pipeline family** (FLUX.2-dev, FLUX.2-klein-4B, FLUX.2-klein-9B).

FLUX.2's attention cost is also O(n²) in sequence length. At 1024×1024 it processes 4096 tokens per step. Running early denoising steps at 32×32 latent (1024 tokens) reduces per-step attention cost to ~6% for those steps.

**Measured speedup on FLUX.2-klein-4B (30 steps, 1024×1024, A6000 — averaged across 10 diverse prompts):**

| Config | Stage split | Denoise | Speedup |
|--------|------------|---------|---------|
| fullres | 30 @ 64² latent | 9.72 s | 1.00× |
| dct_rewind L1 δ=0.05 | 18@32² + 12@64² | 5.50 s | **1.77×** |
| dct_rewind L1 δ=0.10 | 20@32² + 10@64² | 5.03 s | **1.93×** |

Wall-clock efficiency: **97% of token-step prediction** across both configs and all 10 prompts (±0.02× variance).

FLUX.2 differs from FLUX.1 in two ways that required new subclass logic:

| Difference | FLUX.1 | FLUX.2 |
|---|---|---|
| Latent packing | 2×2 patchify → `[B, S, 64]` | Row-major reshape → `[B, H·W, C]` |
| Effective latent scale | `vae_scale_factor` (8) | `vae_scale_factor × 2` (16) |
| Positional IDs | Computed from pixel dims | `batch.latent_ids` (4D grid, must be updated on resolution change) |

---

## Modifications

All changes are **additive and backward-compatible**. When `progressive_mode == "fullres"` (the default), `Flux2ProgressiveDenoisingStage` delegates to `DenoisingStage.forward()` — identical to previous behavior.

### `ProgressiveDenoisingStage` base class — one new hook

**`runtime/pipelines_core/stages/progressive_resolution/denoising.py`**

Added one override point (6 lines):
```python
def _latent_scale_factor(self, server_args: ServerArgs) -> int:
    """Pixel-to-latent scale factor. Override for models with extra patchification."""
    return server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
```

`forward()` now uses `latent_scale = self._latent_scale_factor(server_args)` in all six pixel↔latent conversions. FLUX.1 does **not** override this — **FLUX.1 behavior is unchanged**.

### New file — `runtime/pipelines/flux_2_progressive.py`

`Flux2ProgressiveDenoisingStage(ProgressiveDenoisingStage)` overrides five hooks:

| Hook | What it does |
|------|-------------|
| `_latent_scale_factor` | Returns `vae_scale_factor × 2 = 16` |
| `_unpack_latent` | Row-major reshape `[B, H·W, C] → [B, C, H, W]` |
| `_repack_latent` | Row-major reshape `[B, C, H, W] → [B, H·W, C]` |
| `_generate_initial_noise` | Uses `in_channels` directly (not `//4`); sets `batch.latent_ids` for the initial low-res grid before `_prepare_denoising_loop` builds freqs_cis |
| `_on_resolution_change` | Recomputes `batch.latent_ids` for the new spatial size; updates `freqs_cis` cache and all CFG branches |

### Modified — `runtime/pipelines/flux_2.py`

`Flux2Pipeline.create_pipeline_stages` now manually assembles TI2I stages with `_add_flux2_denoising_stage()` instead of `add_standard_ti2i_stages()` (which doesn't support a custom denoising stage factory). The assembled stages are identical — only the denoising stage class changes.

### No changes to

- `SamplingParams` — `progressive_mode` / `progressive_levels` / `progressive_delta` already registered for all diffusion models (from FLUX.1 PR)

### Usage

Works with all FLUX.2 model variants (`FLUX.2-dev`, `FLUX.2-klein-4B`, `FLUX.2-klein-9B`):

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

---

## Accuracy Tests

### Stage transition logs (confirmed correct)

**δ=0.05 — 18 low-res steps + 12 full-res steps:**
```
Progressive denoising: mode=dct_rewind levels=1 delta=0.050 initial=32x32
Stage 1/2: 32x32 latent, steps [0, 18)
  rewind: sigma=0.8500 → t_eff=0.9189 at step 18
Updated latent_ids and freqs_cis for 64x64 latent (pixel 1024x1024) across 1 branch(es)
Stage 2/2: 64x64 latent, steps [18, 30)
Progressive denoising done in 5.49s (avg 0.1830s/step)
```

**δ=0.10 — 20 low-res steps + 10 full-res steps:**
```
Progressive denoising: mode=dct_rewind levels=1 delta=0.100 initial=32x32
Stage 1/2: 32x32 latent, steps [0, 20)
  rewind: sigma=0.8095 → t_eff=0.8947 at step 20
Updated latent_ids and freqs_cis for 64x64 latent (pixel 1024x1024) across 1 branch(es)
Stage 2/2: 64x64 latent, steps [20, 30)
Progressive denoising done in 5.00s (avg 0.1667s/step)
```

- Initial latent 32×32 = 1024px ÷ (8×2) ÷ 2 — effective scale factor 16 applied correctly ✓
- `batch.latent_ids` and `freqs_cis` updated at every transition ✓
- Rewind formula verified ✓
- All 30 images (10 prompts × 3 modes) artifact-free ✓

### Unit tests

**52 total, all CPU-only, 14 seconds:**

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py -v
# 52 passed, 16 warnings, 32 subtests passed in 13.86s
```

New tests added in this PR (`TestFlux2Pack` + `TestFlux2ProgressiveStage`, 20 tests):

| Class | Count | Coverage |
|-------|-------|----------|
| `TestFlux2Pack` | 7 | pack/unpack shapes, roundtrip identity, row-major ordering, dtype preservation |
| `TestFlux2ProgressiveStage` | 13 | spectrum constants, `_latent_scale_factor`, `_generate_initial_noise` (shape, latent_ids, dtype, determinism), `_on_resolution_change` (no-crash, shape, branch update, coord correctness) |

Manual E2E test (requires GPU + FLUX.2 checkpoint):
```bash
python python/sglang/multimodal_gen/test/manual/test_progressive_flux2.py \
    --model-path /path/to/FLUX.2-klein-4B --steps 30 --levels 1 --delta 0.05
```

---

## Speed Tests

Hardware: **RTX A6000 48 GB**, `--dit-cpu-offload false`, `torch_sdpa`.
Model: **FLUX.2-klein-4B**, 30 steps, seed 42, 1024×1024.
Timing = warm-GPU denoising loop only (model load, text encoding, VAE decode excluded).

### 10-prompt benchmark — all prompts, all modes

| Prompt | fullres | δ=0.05 | δ=0.10 | spd δ=0.05 | spd δ=0.10 |
|--------|---------|--------|--------|-----------|-----------|
| 00 misty forest | 9.70 s | 5.49 s | 5.00 s | 1.77× | 1.94× |
| 01 rose-gold portrait | 9.70 s | 5.50 s | 5.06 s | 1.76× | 1.92× |
| 02 neon Tokyo | 9.72 s | 5.52 s | 5.05 s | 1.76× | 1.92× |
| 03 Tuscany vineyard | 9.71 s | 5.53 s | 5.03 s | 1.76× | 1.93× |
| 04 Arctic tundra | 9.72 s | 5.48 s | 5.00 s | 1.77× | 1.94× |
| 05 jazz club | 9.75 s | 5.48 s | 5.03 s | 1.78× | 1.94× |
| 06 cherry blossoms | 9.74 s | 5.49 s | 5.05 s | 1.77× | 1.93× |
| 07 desert mesa | 9.74 s | 5.50 s | 5.05 s | 1.77× | 1.93× |
| 08 coral reef | 9.73 s | 5.49 s | 5.04 s | 1.77× | 1.93× |
| 09 autumn maples | 9.71 s | 5.47 s | 5.02 s | 1.78× | 1.93× |
| **AVG** | **9.72 s** | **5.50 s** | **5.03 s** | **1.77×** | **1.93×** |

### Token-step analysis

| Config | Token-steps | Expected | Actual | Efficiency |
|--------|------------|----------|--------|-----------|
| fullres | 30×4096 = 122,880 | 1.00× | 1.00× | — |
| δ=0.05 | 18×1024 + 12×4096 = 67,584 | 1.82× | 1.77× | **97%** |
| δ=0.10 | 20×1024 + 10×4096 = 61,440 | 2.00× | 1.93× | **97%** |

The 3% gap is fixed per-step overhead (scheduler `.step()`, memory alloc) that doesn't scale with token count — identical to FLUX.1's efficiency.

### Quality comparisons

3-way strips (fullres | δ=0.05 | δ=0.10) generated for all 10 prompts. All outputs are artifact-free with visually equivalent quality across modes.

> **Note on spectrum constants:** Stage-transition thresholds use power-law coefficients `A=203.615097, β=1.915461` fitted on the FLUX.1-dev VAE. FLUX.2-specific coefficients will be fitted before the final merge (see Checklist item 1).

---

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). — `pre-commit run --all-files` passes clean.
- [x] Add unit tests — 20 new CPU-only tests, 52 total, all pass in 14s.
- [ ] Update documentation — TODO: add FLUX.2 section to `progressive_resolution.mdx` with benchmark table.
- [x] Provide accuracy and speed benchmark results — 10-prompt × 3-mode table; stage transitions verified; 1.77×/1.93× speedup measured.
- [x] Follow the SGLang code style guidance.

### TODO before merge

1. **Fit FLUX.2-specific spectrum constants** (A, β) — replace FLUX.1 placeholder in `flux_2_progressive.py`
2. **Docs** — add FLUX.2 section to `progressive_resolution.mdx`
