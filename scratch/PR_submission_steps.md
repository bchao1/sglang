# PR Submission Steps

Branch: `bchao1/spectral-progressive-flux`  
Target:  `sgl-project/sglang` → `main`

---

## Step 1 — Wait for image generation to finish

The background run (`gen_pr_comparisons.py`, GPU 0) is regenerating all 10 comparison
strips with **denoising-only** timing labels (~1.62× speedup shown on each image).

Wait until **all 10 strips** are updated and the montage is regenerated:

```bash
# Check: all should show today's date at 12:xx
ls -la docs_new/images/progressive/*_compare.png | awk '{print $6,$7,$NF}' | sed 's|.*/||'

# Check: timing.json should have denoise_fullres / denoise_prog fields
python3 -c "import json; d=json.load(open('scratch/results/pr_images/timing.json')); print('OK' if 'denoise_fullres' in d[0] else 'NOT READY')"
```

---

## Step 2 — Update the PR description timing table

Once `timing.json` has denoising data, update the table in `scratch/PR_description.md`:

```python
python3 -c "
import json
d = json.load(open('scratch/results/pr_images/timing.json'))
for r in d:
    print(f\"| {r['id']:02d} {r['label']:<12} | {r['denoise_fullres']:.1f} s | {r['denoise_prog']:.1f} s | **{r['speedup_denoise']:.2f}x** |\")
"
```

Replace the `| 03–10 | ~37 s | ~23 s | **~1.6×** |` placeholder row in `PR_description.md`
with the actual per-prompt values printed above.

---

## Step 3 — Run pre-commit checks

```bash
cd /home/brianchc/sglang
pip install pre-commit        # if not already installed
pre-commit install
pre-commit run --all-files
```

Pre-commit runs BEFORE the commit. If it modifies any files, just re-add them:
```bash
git add -u
# Then proceed to step 4 (no amend needed yet — you haven't committed yet)
```

---

## Step 4 — Run unit tests

```bash
python -m unittest discover \
  -s python/sglang/multimodal_gen/test/unit \
  -p "test_progressive_upsample.py" -v
# Expected: 32 tests OK

python -m unittest discover \
  -s python/sglang/multimodal_gen/test/unit \
  -p "test_sampling_params.py" -v
# Expected: 31 tests OK
```

---

## Step 5 — Stage all changes and commit

```bash
# Stage updated images (re-stage everything; v2 run modified strips 01-10 + montage):
git add docs_new/images/progressive/

# Stage all code changes:
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

# Sanity check: scratch/ and .claude/ must NOT appear
git status --short | grep -E "scratch|\.claude"  # expect: no output

# Commit
git commit -m "test(diffusion): add unit tests, benchmark flags, and comparison images for progressive generation"
```

> **Note:** The branch already has 3 earlier commits from the initial implementation. This becomes the 4th commit. GitHub squashes on merge, so the PR history is fine as-is.

If pre-commit hook fails on commit, fix the issue and run `git commit` again (do NOT use `--no-verify` or `--amend`).

---

## Step 6 — Push to your fork

```bash
git push origin bchao1/spectral-progressive-flux
```

`origin` = `https://github.com/bchao1/sglang.git` (your fork) ✓

---

## Step 7 — Open the Pull Request

1. Go to your fork: **https://github.com/bchao1/sglang**
   GitHub will show a yellow banner: **"bchao1/spectral-progressive-flux had recent pushes — Compare & pull request"** — click that button.

   Or go directly to:
   ```
   https://github.com/sgl-project/sglang/compare/main...bchao1:bchao1/spectral-progressive-flux
   ```

2. **Base**: `sgl-project/sglang:main`  
   **Compare**: `bchao1:bchao1/spectral-progressive-flux`

3. **Title** (copy exactly):
   ```
   [Diffusion] Progressive resolution growing for FLUX.1 via GPU DCT upsampling
   ```

4. **Body**: paste `scratch/PR_description.md`
   - All 10 comparison images and the montage use raw GitHub URLs — they render automatically (no drag-and-drop needed).

5. **Labels** (if you have permission): `diffusion`, `enhancement`, `performance`

---

## Step 8 — Trigger CI

Comment on the PR:
```
/tag-run-ci-label
```

If tests fail:
```
/rerun-failed-ci
```

---

## Step 9 — Respond to review

Key talking points:

| Question | Answer |
|----------|--------|
| Backward compat? | `progressive_mode="fullres"` default = identical to existing behavior. Zero change for non-FLUX pipelines. |
| Why only FLUX? | `_unpack_latent`/`_repack_latent`/`_on_resolution_change` hooks make it easy to add other models later. |
| torch.compile incompatible? | Yes, documented in PR. Only affects an opt-in flag that's off by default. |
| Sequence parallelism? | Guarded with `RuntimeError` — fails loud, not silently. |
| Image quality? | No artifacts. All 10 prompts produce artifact-free output; images in PR show this. |

---

## Files changed

| File | Change |
|------|--------|
| `.gitignore` | Add `scratch/`, `.claude/`, `!docs_new/images/progressive/*.png` |
| `sampling_params.py` | +3 fields (`progressive_mode/levels/delta`, `batch_sig_exclude=True`) + CLI args |
| `flux.py` | `add_standard_denoising_stage()` → `_add_flux_denoising_stage()` (1 line swap) |
| `flux_progressive.py` | **New**: FLUX-specific progressive stage |
| `stages/__init__.py` | +1 export: `ProgressiveDenoisingStage` |
| `progressive_resolution/` | **New module** (5 files) |
| `bench_offline_throughput.py` | +3 `--progressive-*` flags |
| `test_progressive_upsample.py` | **New**: 32 CPU-only unit tests |
| `test_sampling_params.py` | +8 tests for progressive fields |
| `test_progressive_flux.py` | **New**: manual E2E test |
| `docs_new/images/progressive/` | 10 comparison strips + montage |

## Compatibility verified

- No upstream changes to our files since branch point (`a5e6a8887`)
- `sampling_params.py`: `batch_sig_exclude=True` — requests with different modes batch correctly
- `stages/__init__.py`: alphabetically ordered (`progressive_resolution` after `ltx_2_denoising`)
- `flux.py`: import order matches `qwen_image.py` pattern
- 32 unit tests + 8 sampling_params tests all pass
