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

If pre-commit modifies files, `git add` and amend the commit:
```bash
git add -u
git commit --amend --no-edit
```

---

## Step 2 — Verify unit tests pass

```bash
# CPU-only, no GPU needed, <1 s
python -m unittest discover \
  -s python/sglang/multimodal_gen/test/unit \
  -p "test_progressive_upsample.py" -v

# 32 tests expected: OK

# Sampling-params progressive fields
python -m unittest discover \
  -s python/sglang/multimodal_gen/test/unit \
  -p "test_sampling_params.py" -v

# TestProgressiveSamplingParams: 8 tests expected OK (among 31 total)
```

---

## Step 3 — Stage all PR files and commit

```bash
# Comparison images are already staged (git add docs_new/images/progressive/ already done).
# Stage remaining code and test changes:
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
  python/sglang/multimodal_gen/test/unit/test_sampling_params.py \
  docs_new/images/progressive/

git status   # verify only the files above show as staged; scratch/ must NOT appear

git commit -m "feat(diffusion): progressive resolution growing for FLUX.1 via GPU DCT upsampling"
```

---

## Step 4 — Push the branch

```bash
git push origin bchao1/spectral-progressive-flux
```

---

## Step 5 — Open the Pull Request on GitHub

1. Go to: https://github.com/sgl-project/sglang/compare/main...bchao1:sglang:bchao1/spectral-progressive-flux

2. Click **"Create pull request"**

3. **Title** (copy exactly):
   ```
   [Diffusion] Progressive resolution growing for FLUX.1 via GPU DCT upsampling
   ```

4. **Body**: paste the contents of `scratch/PR_description.md`

5. **Upload comparison images** (drag-and-drop into the PR body editor):
   - `docs_new/images/progressive/montage_progressive_vs_fullres.png`
   - Replace `[MONTAGE_PLACEHOLDER]` in the PR body with the uploaded URL
   - Optionally upload individual comparison strips from `docs_new/images/progressive/`

6. **Labels** (if you have permission): `diffusion`, `enhancement`, `performance`

---

## Step 6 — Trigger CI

Once the PR is open, comment:
```
/tag-run-ci-label
```

If any tests fail, rerun with:
```
/rerun-failed-ci
```

---

## Step 7 — Respond to review

Key points to explain if asked:

- **Backward compat**: `progressive_mode="fullres"` (default) is identical to the old behavior. No existing tests or workflows are affected.
- **FLUX-only**: The stage is guarded by `if get_sp_world_size() > 1: raise RuntimeError(...)` for sequence parallelism. Other model families are unaffected.
- **torch.compile incompatibility**: Known limitation — documented in PR and in optimization compatibility table.
- **freqs_cis update**: The shallow-copy issue in `CFGBranch.kwargs` is a FLUX architectural constraint; the fix is in `FluxProgressiveDenoisingStage._on_resolution_change`.

---

## Files changed (summary)

| File | Change |
|------|--------|
| `.gitignore` | Add `scratch/`, `!docs_new/images/progressive/*.png` |
| `sampling_params.py` | +3 fields (`progressive_mode/levels/delta`) + CLI args |
| `flux.py` | Replace `add_standard_denoising_stage()` with `_add_flux_denoising_stage()` |
| `flux_progressive.py` | **New**: FLUX-specific progressive stage |
| `stages/__init__.py` | Export `ProgressiveDenoisingStage` |
| `progressive_resolution/` | **New module**: spectral_ops, scheduler_utils, upsample, denoising |
| `bench_offline_throughput.py` | +progressive flags |
| `test_progressive_upsample.py` | **New**: 32 CPU-only unit tests |
| `test_sampling_params.py` | +8 progressive field tests |
| `test_progressive_flux.py` | **New**: manual E2E test |
| `docs_new/images/progressive/` | Comparison images + montage |
