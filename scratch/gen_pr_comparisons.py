#!/usr/bin/env python3
"""
Generate fullres vs progressive side-by-side comparisons for the PR.

10 prompts: cinematic landscape, architecture, human portrait,
objects, fantasy, wildlife, interior, seascape, desert, nightscape.

Outputs:
  scratch/results/pr_images/      individual PNGs + timing log
  docs_new/images/progressive/    side-by-side strips + montage (committed to repo)

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scratch/gen_pr_comparisons.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLUX_MODEL = "/miele/brian/modelscope/black-forest-labs/FLUX.1-dev"
STEPS = 50
SEED = 42
HEIGHT = 1024
WIDTH = 1024
DELTA = 0.05
LEVELS = 1

OUT_DIR = Path(__file__).parent / "results" / "pr_images"
DOC_DIR = Path(__file__).parent.parent / "docs_new" / "images" / "progressive"

PROMPTS = [
    ("landscape",    "Golden hour over misty mountain peaks, dramatic cinematic light, photorealistic"),
    ("architecture", "Gothic cathedral interior with stained glass windows, volumetric light shafts, 8K"),
    ("portrait",     "Close-up portrait of an elderly woman with weathered skin, natural outdoor light, film grain"),
    ("cityscape",    "Hong Kong neon-lit street at night, rain reflections, teal and magenta, cinematic"),
    ("object",       "Macro photograph of a vintage pocket watch on velvet, bokeh, studio lighting"),
    ("wildlife",     "Snow leopard mid-leap in Himalayan blizzard, motion blur, National Geographic style"),
    ("interior",     "Minimalist Japanese tea room, morning light, cherry blossoms through paper screen door"),
    ("seascape",     "Aerial view of turquoise sea over coral reef, Maldives, drone shot, midday sun"),
    ("desert",       "Arizona slot canyon, swirling sandstone walls, single beam of orange sunlight from above"),
    ("fantasy",      "Ancient floating islands with waterfalls, lush forests, misty atmosphere, concept art"),
]

BASE_FLAGS = [
    "--model-path", FLUX_MODEL,
    "--attention-backend", "torch_sdpa",
    "--num-inference-steps", str(STEPS),
    "--seed", str(SEED),
    "--height", str(HEIGHT),
    "--width", str(WIDTH),
    "--dit-cpu-offload", "false",
    "--save-output",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_generate(prompt: str, outfile: Path, extra_flags: list = None) -> float:
    """Run `sglang generate` and return wall-clock seconds."""
    cmd = (
        [sys.executable, "-m", "sglang.multimodal_gen.runtime.entrypoints.generate"]
        + BASE_FLAGS
        + ["--prompt", prompt, "--output-file-path", str(outfile)]
        + (extra_flags or [])
    )
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        # Try the sglang generate entrypoint directly
        cmd2 = (
            ["python3", "-m", "sglang", "generate"]
            + BASE_FLAGS
            + ["--prompt", prompt, "--output-file-path", str(outfile)]
            + (extra_flags or [])
        )
        t0 = time.perf_counter()
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if result2.returncode != 0:
            print(f"  ERROR stdout: {result2.stdout[-500:]}")
            print(f"  ERROR stderr: {result2.stderr[-500:]}")
    return elapsed


def try_entrypoints(prompt: str, outfile: Path, extra_flags: list = None) -> tuple[float, bool]:
    """Try multiple entrypoints; return (elapsed, success)."""
    attempts = [
        ["sglang", "generate"],
        [sys.executable, "-m", "sglang.multimodal_gen.runtime.entrypoints.cli"],
    ]
    flags = BASE_FLAGS + ["--prompt", prompt, "--output-file-path", str(outfile)] + (extra_flags or [])

    # Primary: use what the benchmark script uses
    cmd = ["sglang", "generate"] + flags
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if r.returncode == 0 and outfile.exists():
        return elapsed, True

    print(f"  sglang generate failed (rc={r.returncode}), stderr: {r.stderr[-300:]}")
    return elapsed, False


def make_label_image(text: str, width: int, height: int = 36,
                     bg=(20, 20, 20), fg=(255, 255, 255)) -> Image.Image:
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=fg, font=font)
    return img


def make_side_by_side(img_fr: Image.Image, img_pr: Image.Image,
                      label: str, t_fr: float, t_pr: float,
                      out_path: Path, thumb_w: int = 512):
    img_fr = img_fr.resize((thumb_w, thumb_w), Image.LANCZOS)
    img_pr = img_pr.resize((thumb_w, thumb_w), Image.LANCZOS)
    speedup = t_fr / t_pr

    lbl_fr = make_label_image(f"Fullres  {t_fr:.1f}s", thumb_w)
    lbl_pr = make_label_image(
        f"Progressive δ=0.05  {t_pr:.1f}s  ({speedup:.2f}×)", thumb_w,
        bg=(0, 60, 30),
    )

    strip_w = thumb_w * 2 + 6
    strip_h = thumb_w + lbl_fr.height + 28
    strip = Image.new("RGB", (strip_w, strip_h), (40, 40, 40))

    title_bar = make_label_image(label.upper().replace("_", " "), strip_w, height=28, bg=(60, 60, 60))
    strip.paste(title_bar, (0, 0))

    y_img = 28
    strip.paste(img_fr, (0, y_img))
    strip.paste(img_pr, (thumb_w + 6, y_img))
    strip.paste(lbl_fr, (0, y_img + thumb_w))
    strip.paste(lbl_pr, (thumb_w + 6, y_img + thumb_w))

    strip.save(out_path, quality=95)
    print(f"  Saved comparison: {out_path.name}")


def make_montage(strip_paths: list, out_path: Path, cols: int = 2):
    strips = [Image.open(p) for p in strip_paths if Path(p).exists()]
    if not strips:
        print("No strips to montage.")
        return
    sw, sh = strips[0].size
    rows = (len(strips) + cols - 1) // cols
    mw = sw * cols + (cols - 1) * 4
    mh = sh * rows + (rows - 1) * 4
    montage = Image.new("RGB", (mw, mh), (20, 20, 20))
    for i, strip in enumerate(strips):
        r, c = divmod(i, cols)
        montage.paste(strip, (c * (sw + 4), r * (sh + 4)))
    montage.save(out_path, quality=92)
    print(f"\nMontage saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {FLUX_MODEL}")
    print(f"Steps:  {STEPS}  Seed: {SEED}  Resolution: {WIDTH}x{HEIGHT}")
    print(f"Mode:   dct_rewind  levels={LEVELS}  delta={DELTA}")
    print(f"Output: {OUT_DIR}")
    print()

    # Check that sglang generate is available
    check = subprocess.run(["sglang", "generate", "--help"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        print("ERROR: 'sglang generate' not found. Is sglang installed?")
        print(check.stderr[:500])
        sys.exit(1)

    prog_flags = [
        "--progressive-mode", "dct_rewind",
        "--progressive-levels", str(LEVELS),
        "--progressive-delta", str(DELTA),
    ]

    results = []
    strip_paths = []

    for i, (label, prompt) in enumerate(PROMPTS, 1):
        print(f"\n=== [{i:02d}/10] {label} ===")
        print(f"  {prompt[:80]}")

        out_fr = OUT_DIR / f"{i:02d}_{label}_fullres.png"
        out_pr = OUT_DIR / f"{i:02d}_{label}_progressive.png"

        # Fullres
        t_fr, ok_fr = try_entrypoints(prompt, out_fr)
        if not ok_fr:
            print(f"  SKIP {label} fullres — generation failed")
            continue
        print(f"  fullres:     {t_fr:.2f}s ✓")

        # Progressive
        t_pr, ok_pr = try_entrypoints(prompt, out_pr, prog_flags)
        if not ok_pr:
            print(f"  SKIP {label} progressive — generation failed")
            continue
        speedup = t_fr / t_pr
        print(f"  progressive: {t_pr:.2f}s  ({speedup:.2f}×) ✓")

        results.append({
            "id": i, "label": label,
            "t_fullres": round(t_fr, 2), "t_prog": round(t_pr, 2),
            "speedup": round(speedup, 3),
        })

        # Side-by-side
        strip_path = DOC_DIR / f"{i:02d}_{label}_compare.png"
        make_side_by_side(Image.open(out_fr), Image.open(out_pr),
                          label, t_fr, t_pr, strip_path)
        strip_paths.append(strip_path)

    if not results:
        print("No successful generations. Exiting.")
        sys.exit(1)

    # Montage
    make_montage(strip_paths, DOC_DIR / "montage_progressive_vs_fullres.png")

    # Timing JSON
    with open(OUT_DIR / "timing.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print()
    print("=" * 68)
    print(f"{'Label':<14}  {'Fullres':>8}  {'Progressive':>11}  {'Speedup':>8}")
    print("-" * 68)
    total_fr = total_pr = 0.0
    for r in results:
        total_fr += r["t_fullres"]
        total_pr += r["t_prog"]
        print(f"{r['label']:<14}  {r['t_fullres']:>7.2f}s  {r['t_prog']:>10.2f}s  {r['speedup']:>7.2f}×")
    if len(results) > 1:
        print("-" * 68)
        avg_speedup = total_fr / total_pr
        print(f"{'AVERAGE':<14}  {total_fr/len(results):>7.2f}s  {total_pr/len(results):>10.2f}s  {avg_speedup:>7.2f}×")
    print("=" * 68)

    return results


if __name__ == "__main__":
    run()
