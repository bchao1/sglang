# PR Submission Steps

Branch: `bchao1/spectral-progressive-flux`
Target:  `sgl-project/sglang` → `main`

---

## Step 1 — Run pre-commit checks

```bash
cd /home/brianchc/sglang

pip install pre-commit        # if not already installed
pre-commit install
pre-commit run --all-files    # fix any lint / formatting issues
```

If pre-commit modifies files, re-stage and amend:
```bash
git add -u
git commit --amend --no-edit
```

---

## Step 2 — Run unit tests (CPU-only, no GPU needed)

```bash
python -m unittest discover \
  -s python/sglang/multimodal_gen/test/unit \
  -p "test_progressive_upsample.py" -v
# Expected: 32 tests OK

python -m unittest discover \
  -s python/sglang/multimodal_gen/test/unit \
  -p "test_sampling_params.py" -v
# Expected: 31 tests OK (includes 8 TestProgressiveSamplingParams tests)
```

---

## Step 3 — Wait for v2 image generation to finish

The current generation run (`gen_pr_comparisons.py`) is writing updated comparison
strips with **denoising-only timing** labels to `docs_new/images/progressive/`.
Watch for the `bej8h6e90` background task to complete, then verify:

```bash
ls -lt docs_new/images/progressive/*_compare.png  # all should have today's timestamp
cat scratch/results/pr_images/timing.json         # should have denoise_fullres / denoise_prog fields
```

---

## Step 4 — Stage all changes and commit

```bash
# Stage updated images (re-stage; some were modified by v2 run):
git add docs_new/images/progressive/

# Stage all code changes (tests, cleanup, benchmark, .gitignore):
git add \
  .gitignore \
  python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py \
  python/sglang/multimodal_gen/configs/sample/sampling_params.py \
  python/sglang/multimodal_gen/runtime/pipelines/flux.py \
  python/sglang/multimodal_gen/runtime/pipelines/flux_progressive.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/__init__.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/progressive_resolution/ \
  python/sglang/multimodal_gen/test/manual/test_progressive_flux.py \
  python/sglang/multimodal_gen/test/unit/test_progressive_upsample.py \
  python/sglang/multimodal_gen/test/unit/test_sampling_params.py

# Verify: scratch/ must NOT appear in staged files
git status --short | grep scratch   # expect: no output (scratch/ is gitignored)
git status --short | grep "\.claude" # expect: no output (.claude/ is gitignored)

# Commit everything
git commit -m "feat(diffusion): progressive resolution growing for FLUX.1 via GPU DCT upsampling"
```

---

## Step 5 — Push the branch

```bash
git push origin bchao1/spectral-progressive-flux
```

---

## Step 6 — Open the Pull Request on GitHub

1. Go to: https://github.com/sgl-project/sglang/compare/main...bchao1:sglang:bchao1/spectral-progressive-flux

2. Click **"Create pull request"**

3. **Title** (copy exactly):
   ```
   [Diffusion] Progressive resolution growing for FLUX.1 via GPU DCT upsampling
   ```

4. **Body**: paste the contents of `scratch/PR_description.md`
   - Images use raw GitHub URLs that become active once the branch is pushed.
   - No drag-and-drop needed — images are committed to the repo.

5. **Labels** (if you have permission): `diffusion`, `enhancement`, `performance`

---

## Step 7 — Trigger CI

Once the PR is open, comment:
```
/tag-run-ci-label
```

If tests fail:
```
/rerun-failed-ci
```

---

## Step 8 — Respond to review

Key points:

- **Backward compat**: `progressive_mode="fullres"` (default) — identical to original behavior for all existing requests. Zero changes to non-FLUX pipelines.
- **New files only**: All progressive code is in new files under `progressive_resolution/`. Only three existing files are modified: `flux.py` (1 method swap), `stages/__init__.py` (+1 export), `sampling_params.py` (+3 fields + CLI args).
- **torch.compile incompatibility**: Known, documented. Only affects opt-in flag that is off by default.
- **Sequence parallelism guard**: `if get_sp_world_size() > 1: raise RuntimeError(...)` prevents silent wrong behavior with SP.

---

## Files changed (summary)

| File | Change |
|------|--------|
| `.gitignore` | Add `scratch/`, `.claude/`, `!docs_new/images/progressive/*.png` |
| `sampling_params.py` | +3 fields (`progressive_mode/levels/delta`, all `batch_sig_exclude=True`) + CLI args |
| `flux.py` | Replace `add_standard_denoising_stage()` → `_add_flux_denoising_stage()` |
| `flux_progressive.py` | **New**: FLUX-specific progressive stage (pack/unpack + freqs_cis update) |
| `stages/__init__.py` | Export `ProgressiveDenoisingStage` |
| `progressive_resolution/` | **New module** (5 files): spectral_ops, scheduler_utils, upsample, denoising, `__init__` |
| `bench_offline_throughput.py` | +3 `--progressive-*` flags (backward compatible, default=fullres) |
| `test_progressive_upsample.py` | **New**: 32 CPU-only unit tests (unittest.TestCase) |
| `test_sampling_params.py` | +8 progressive field tests |
| `test_progressive_flux.py` | **New**: manual E2E test (GPU required) |
| `docs_new/images/progressive/` | 10 comparison strips + montage (raw GitHub URLs in PR body) |

---

## Compatibility notes

- Branch diverges from `main` at `a5e6a8887` — no upstream changes to any of our files since then.
- `sampling_params.py` fields have `batch_sig_exclude=True` — different requests can mix progressive/fullres in the same server without batching conflicts.
- `stages/__init__.py` addition is alphabetically ordered (`progressive_resolution` after `ltx_2_denoising`).
- `flux.py` import order follows the pattern in `qwen_image.py` (`disaggregation` < `pipelines` < `pipelines_core`).
