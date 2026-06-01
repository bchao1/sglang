#!/usr/bin/env python3
"""
Generate fullres vs progressive side-by-side comparisons for the PR.

Uses sglang generate CLI per call, captures the log output, and extracts
DENOISING-ONLY timing from the log lines:
  - Progressive: "Progressive denoising done in X.XXs"
  - Fullres:     "average time per step: X.Xs"  × STEPS

This matches the methodology used in scratch/test_progressive_benchmark.sh
and the integration doc speedup numbers (1.32–1.62×, no model-load overhead).

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
    (
        "landscape",
        "Golden hour over misty mountain peaks, dramatic cinematic light, photorealistic",
    ),
    (
        "architecture",
        "Gothic cathedral interior with stained glass windows, volumetric light shafts, 8K",
    ),
    (
        "portrait",
        "Close-up portrait of an elderly woman with weathered skin, natural outdoor light, film grain",
    ),
    (
        "cityscape",
        "Hong Kong neon-lit street at night, rain reflections, teal and magenta, cinematic",
    ),
    (
        "object",
        "Macro photograph of a vintage pocket watch on velvet, bokeh, studio lighting",
    ),
    (
        "wildlife",
        "Snow leopard mid-leap in Himalayan blizzard, motion blur, National Geographic style",
    ),
    (
        "interior",
        "Minimalist Japanese tea room, morning light, cherry blossoms through paper screen door",
    ),
    (
        "seascape",
        "Aerial view of turquoise sea over coral reef, Maldives, drone shot, midday sun",
    ),
    (
        "desert",
        "Arizona slot canyon, swirling sandstone walls, single beam of orange sunlight from above",
    ),
    (
        "fantasy",
        "Ancient floating islands with waterfalls, lush forests, misty atmosphere, concept art",
    ),
]

BASE_FLAGS = [
    "--model-path",
    FLUX_MODEL,
    "--attention-backend",
    "torch_sdpa",
    "--num-inference-steps",
    str(STEPS),
    "--seed",
    str(SEED),
    "--height",
    str(HEIGHT),
    "--width",
    str(WIDTH),
    "--dit-cpu-offload",
    "false",
    "--save-output",
]

# ---------------------------------------------------------------------------
# Log-based denoising timing (same method as test_progressive_benchmark.sh)
# ---------------------------------------------------------------------------


def parse_denoise_time(log: str, mode: str) -> float | None:
    """Extract denoising-loop seconds from sglang generate log output.

    Progressive: "Progressive denoising done in X.XXs"
    Fullres:     "average time per step: X.Xs"  → multiply by STEPS
    """
    if "progressive" in mode and mode != "fullres":
        m = re.search(r"Progressive denoising done in ([0-9.]+)", log)
        if m:
            return float(m.group(1))
    # Fullres (and fallback): average time per step
    m = re.search(r"average time per step: ([0-9.]+)", log)
    if m:
        return float(m.group(1)) * STEPS
    return None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def make_label_image(
    text: str, width: int, height: int = 36, bg=(20, 20, 20), fg=(255, 255, 255)
) -> Image.Image:
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    for fpath in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(fpath, 18)
            break
        except Exception:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=fg, font=font)
    return img


def make_side_by_side(
    img_fr: Image.Image,
    img_pr: Image.Image,
    label: str,
    denoise_fr: float,
    denoise_pr: float,
    out_path: Path,
    thumb_w: int = 512,
):
    """Side-by-side strip labelled with DENOISING-ONLY time and speedup."""
    img_fr = img_fr.resize((thumb_w, thumb_w), Image.LANCZOS)
    img_pr = img_pr.resize((thumb_w, thumb_w), Image.LANCZOS)
    speedup = denoise_fr / denoise_pr

    lbl_fr = make_label_image(f"Fullres  denoise={denoise_fr:.1f}s", thumb_w)
    lbl_pr = make_label_image(
        f"Progressive δ=0.05  denoise={denoise_pr:.1f}s  ({speedup:.2f}×)",
        thumb_w,
        bg=(0, 60, 30),
    )

    strip_w = thumb_w * 2 + 6
    strip_h = thumb_w + lbl_fr.height + 28
    strip = Image.new("RGB", (strip_w, strip_h), (40, 40, 40))
    title_bar = make_label_image(
        label.upper().replace("_", " "), strip_w, height=28, bg=(60, 60, 60)
    )
    strip.paste(title_bar, (0, 0))
    y_img = 28
    strip.paste(img_fr, (0, y_img))
    strip.paste(img_pr, (thumb_w + 6, y_img))
    strip.paste(lbl_fr, (0, y_img + thumb_w))
    strip.paste(lbl_pr, (thumb_w + 6, y_img + thumb_w))
    strip.save(out_path, quality=95)
    print(f"  Saved: {out_path.name}")


def make_montage(strip_paths: list, out_path: Path, cols: int = 2):
    strips = [Image.open(p) for p in strip_paths if Path(p).exists()]
    if not strips:
        return
    sw, sh = strips[0].size
    rows = (len(strips) + cols - 1) // cols
    montage = Image.new(
        "RGB", (sw * cols + (cols - 1) * 4, sh * rows + (rows - 1) * 4), (20, 20, 20)
    )
    for i, strip in enumerate(strips):
        r, c = divmod(i, cols)
        montage.paste(strip, (c * (sw + 4), r * (sh + 4)))
    montage.save(out_path, quality=92)
    print(f"\nMontage saved: {out_path}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def run_generate(
    prompt: str, outfile: Path, extra_flags: list = None
) -> tuple[float, float | None, bool]:
    """Run sglang generate, return (wall_clock_s, denoise_s | None, success)."""
    cmd = (
        ["sglang", "generate"]
        + BASE_FLAGS
        + ["--prompt", prompt, "--output-file-path", str(outfile)]
        + (extra_flags or [])
    )
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    log = result.stdout + result.stderr
    mode = "progressive" if extra_flags else "fullres"
    denoise = parse_denoise_time(log, mode)
    ok = result.returncode == 0 and outfile.exists()

    if not ok:
        print(f"  WARN: generation failed (rc={result.returncode})")
        print(f"  stderr tail: {result.stderr[-400:]}")

    return wall, denoise, ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    prog_flags = [
        "--progressive-mode",
        "dct_rewind",
        "--progressive-levels",
        str(LEVELS),
        "--progressive-delta",
        str(DELTA),
    ]

    print(f"Model:  {FLUX_MODEL}")
    print(
        f"GPU:    CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
    print(f"Steps:  {STEPS}  Seed: {SEED}  Resolution: {WIDTH}x{HEIGHT}")
    print(f"Mode:   dct_rewind  levels={LEVELS}  delta={DELTA}")
    print(f"Timing: DENOISING LOOP ONLY (model load not counted)")
    print()

    results = []
    strip_paths = []

    for i, (label, prompt) in enumerate(PROMPTS, 1):
        print(f"\n=== [{i:02d}/10] {label} ===")
        print(f"  {prompt[:80]}")

        out_fr = OUT_DIR / f"{i:02d}_{label}_fullres.png"
        out_pr = OUT_DIR / f"{i:02d}_{label}_progressive.png"

        # Fullres
        wall_fr, den_fr, ok_fr = run_generate(prompt, out_fr)
        if not ok_fr:
            print(f"  SKIP — fullres failed")
            continue
        den_fr_str = f"{den_fr:.2f}s" if den_fr is not None else "N/A"
        print(f"  fullres:     wall={wall_fr:.1f}s  denoise={den_fr_str}")

        # Progressive
        wall_pr, den_pr, ok_pr = run_generate(prompt, out_pr, prog_flags)
        if not ok_pr:
            print(f"  SKIP — progressive failed")
            continue
        den_pr_str = f"{den_pr:.2f}s" if den_pr is not None else "N/A"

        if den_fr and den_pr:
            speedup = den_fr / den_pr
            print(
                f"  progressive: wall={wall_pr:.1f}s  denoise={den_pr:.2f}s  ({speedup:.2f}×)"
            )
        else:
            speedup = wall_fr / wall_pr
            print(
                f"  progressive: wall={wall_pr:.1f}s  (fallback wall-clock speedup={speedup:.2f}×)"
            )

        results.append(
            {
                "id": i,
                "label": label,
                "wall_fullres": round(wall_fr, 2),
                "wall_prog": round(wall_pr, 2),
                "denoise_fullres": round(den_fr, 2) if den_fr else None,
                "denoise_prog": round(den_pr, 2) if den_pr else None,
                "speedup_denoise": (
                    round(den_fr / den_pr, 3) if (den_fr and den_pr) else None
                ),
                "speedup_wall": round(wall_fr / wall_pr, 3),
            }
        )

        # Side-by-side labelled with denoising times
        d_fr_plot = den_fr if den_fr else wall_fr
        d_pr_plot = den_pr if den_pr else wall_pr
        strip_path = DOC_DIR / f"{i:02d}_{label}_compare.png"
        make_side_by_side(
            Image.open(out_fr),
            Image.open(out_pr),
            label,
            d_fr_plot,
            d_pr_plot,
            strip_path,
        )
        strip_paths.append(strip_path)

    if not results:
        print("No successful generations.")
        sys.exit(1)

    make_montage(strip_paths, DOC_DIR / "montage_progressive_vs_fullres.png")

    with open(OUT_DIR / "timing.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary — denoising speedup only
    print()
    print("=" * 75)
    print(f"{'Label':<14}  {'Denoise FR':>10}  {'Denoise PR':>10}  {'Speedup':>8}")
    print(f"  (denoising loop only — model load NOT counted)")
    print("-" * 75)
    totals_fr = totals_pr = 0.0
    n = 0
    for r in results:
        if r["denoise_fullres"] and r["denoise_prog"]:
            totals_fr += r["denoise_fullres"]
            totals_pr += r["denoise_prog"]
            n += 1
            print(
                f"{r['label']:<14}  {r['denoise_fullres']:>9.2f}s  {r['denoise_prog']:>9.2f}s  {r['speedup_denoise']:>7.2f}×"
            )
        else:
            print(f"{r['label']:<14}  (denoise timing not parsed)")
    if n > 0:
        print("-" * 75)
        print(
            f"{'AVERAGE':<14}  {totals_fr/n:>9.2f}s  {totals_pr/n:>9.2f}s  {totals_fr/totals_pr:>7.2f}×"
        )
    print("=" * 75)

    return results


if __name__ == "__main__":
    run()
