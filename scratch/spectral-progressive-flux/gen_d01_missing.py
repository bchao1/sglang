#!/usr/bin/env python3
"""Generate missing d01 (delta=0.1) progressive images for wildlife, seascape, desert."""

import json
import subprocess
import sys
import time
from pathlib import Path

FLUX_MODEL = "/miele/brian/modelscope/black-forest-labs/FLUX.1-dev"
STEPS = 50
SEED = 42
HEIGHT = 1024
WIDTH = 1024
DELTA = 0.1
LEVELS = 1
OUT_DIR = Path("scratch/results/pr_images_d01")
DOC_DIR = Path("docs_new/images/progressive_d01")

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
    "--master-port",
    "30305",
    "--scheduler-port",
    "5935",
    "--port",
    "30300",
]

MISSING = [
    (
        "06",
        "wildlife",
        "Snow leopard mid-leap in Himalayan blizzard, motion blur, National Geographic style",
    ),
    (
        "08",
        "seascape",
        "Aerial view of turquoise sea over coral reef, Maldives, drone shot, midday sun",
    ),
    (
        "09",
        "desert",
        "Arizona slot canyon, swirling sandstone walls, single beam of orange sunlight from above",
    ),
]

import re


def parse_denoise(log, mode):
    if mode == "progressive":
        m = re.search(r"Progressive denoising done in ([0-9.]+)", log)
        if m:
            return float(m.group(1))
    m = re.search(r"average time per step: ([0-9.]+)", log)
    return float(m.group(1)) * STEPS if m else None


def run(prompt, outfile, extra=[]):
    cmd = (
        ["sglang", "generate"]
        + BASE_FLAGS
        + ["--prompt", prompt, "--output-file-path", str(outfile)]
        + extra
    )
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    log = r.stdout + r.stderr
    ok = r.returncode == 0 and Path(outfile).exists()
    return wall, parse_denoise(log, "progressive" if extra else "fullres"), ok


def make_strip(label, fr_path, pr_path, den_fr, den_pr, strip_path, i):
    from PIL import Image, ImageDraw, ImageFont

    thumb_w = 512

    def lbl(text, w, bg=(20, 20, 20)):
        img = Image.new("RGB", (w, 36), bg)
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
            )
        except:
            font = ImageFont.load_default()
        bb = d.textbbox((0, 0), text, font=font)
        d.text(
            ((w - bb[2] + bb[0]) // 2, (36 - bb[3] + bb[1]) // 2),
            text,
            fill=(255, 255, 255),
            font=font,
        )
        return img

    fr = Image.open(fr_path).resize((thumb_w, thumb_w), Image.LANCZOS)
    pr = Image.open(pr_path).resize((thumb_w, thumb_w), Image.LANCZOS)
    spd = den_fr / den_pr if den_fr and den_pr else 0
    title = lbl(label.upper(), thumb_w * 2 + 6, bg=(60, 60, 60))
    lf = lbl(f"Fullres  denoise={den_fr:.1f}s" if den_fr else "Fullres", thumb_w)
    lp = lbl(
        (
            f"Progressive d=0.1  denoise={den_pr:.1f}s  ({spd:.2f}x)"
            if den_pr
            else f"Progressive d=0.1"
        ),
        thumb_w,
        bg=(0, 40, 60),
    )
    strip = Image.new("RGB", (thumb_w * 2 + 6, thumb_w + 64), (40, 40, 40))
    strip.paste(title, (0, 0))
    strip.paste(fr, (0, 28))
    strip.paste(pr, (thumb_w + 6, 28))
    strip.paste(lf, (0, 28 + thumb_w))
    strip.paste(lp, (thumb_w + 6, 28 + thumb_w))
    strip.save(strip_path, quality=95)
    print(f"  Strip saved: {strip_path.name}")


OUT_DIR.mkdir(exist_ok=True)
DOC_DIR.mkdir(exist_ok=True)

prog_flags = [
    "--progressive-mode",
    "dct_rewind",
    "--progressive-levels",
    str(LEVELS),
    "--progressive-delta",
    str(DELTA),
]

for idx, label, prompt in MISSING:
    fr_path = OUT_DIR / f"{idx}_{label}_fullres.png"
    pr_path = OUT_DIR / f"{idx}_{label}_progressive.png"

    # Check if fullres exists
    if not fr_path.exists():
        print(f"\n[{idx}] {label} - generating fullres...")
        wall_fr, den_fr, ok = run(prompt, fr_path)
        if not ok:
            print("  SKIP - fullres failed")
            continue
        print(
            f"  fullres: {wall_fr:.1f}s denoise={den_fr:.1f}s"
            if den_fr
            else f"  fullres: {wall_fr:.1f}s"
        )
    else:
        den_fr = None  # will re-read from log below - just use existing
        print(f"\n[{idx}] {label} - fullres exists")

    # Generate progressive
    if not pr_path.exists():
        print(f"  generating progressive d=0.1...")
        wall_pr, den_pr, ok = run(prompt, pr_path, prog_flags)
        if not ok:
            print("  SKIP - progressive failed")
            continue
        print(
            f"  progressive: {wall_pr:.1f}s denoise={den_pr:.1f}s"
            if den_pr
            else f"  progressive: {wall_pr:.1f}s"
        )
    else:
        den_pr = None
        print(f"  progressive exists")

    # Read denoise from existing timing if available
    timing_path = OUT_DIR / "timing.json"
    if timing_path.exists():
        data = json.load(open(timing_path))
        for r in data:
            if r["label"] == label:
                den_fr = r.get("denoise_fullres", den_fr)
                den_pr = r.get("denoise_prog", den_pr)
                break

    # Create strip
    strip_path = DOC_DIR / f"{idx}_{label}_compare.png"
    make_strip(label, fr_path, pr_path, den_fr, den_pr, strip_path, idx)

print("\nDone! Regenerating 3-way montage...")
import subprocess

subprocess.run(["python3", "scratch/gen_3way_comparison.py"], check=False)
