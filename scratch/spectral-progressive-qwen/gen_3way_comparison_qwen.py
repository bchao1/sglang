#!/usr/bin/env python3
"""
Generate 3-way comparison strips for Qwen-Image: fullres | δ=best1 | δ=best2.

Run after gen_delta_images_qwen.sh to produce strips for the two best deltas.
Images expected layout:
  results/pr_images/01_landscape_fullres.png
  results/pr_images/01_landscape_progressive.png   (δ=DELTA_A)
  results/pr_images_d2/01_landscape_progressive.png (δ=DELTA_B)

Edit DELTA_A / DELTA_B / SOURCES below to match your run directories.

Usage:
  python3 scratch/spectral-progressive-qwen/gen_3way_comparison_qwen.py
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRATCH = Path(__file__).parent

# ── Configuration ─────────────────────────────────────────────────────────────
# Set these to the two delta values selected for quality comparison
DELTA_A = "0.05"
DELTA_B = "0.10"

# Directories containing images (edit to match actual result dirs)
SOURCES = {
    "fullres": SCRATCH / "results" / "pr_images",
    DELTA_A: SCRATCH / "results" / "pr_images",
    DELTA_B: SCRATCH / "results" / "pr_images_d10",
}
TIMING = {
    DELTA_A: SCRATCH / "results" / "pr_images" / "timing.json",
    DELTA_B: SCRATCH / "results" / "pr_images_d10" / "timing.json",
}
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = SCRATCH / "pr_visuals" / "3way"
LABELS_ORDER = ["fullres", DELTA_A, DELTA_B]

PROMPTS = [
    "landscape",
    "architecture",
    "portrait",
    "cityscape",
    "object",
    "wildlife",
    "interior",
    "seascape",
    "desert",
    "fantasy",
]

BG_COLORS = {
    "fullres": (40, 40, 40),
    DELTA_A: (0, 50, 20),
    DELTA_B: (0, 40, 60),
}


def make_label(text, width, height=34, bg=(20, 20, 20), fg=(255, 255, 255)):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 15)
            break
        except Exception:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=fg, font=font)
    return img


def find_image(source_dir: Path, prompt_label: str, suffix: str):
    matches = list(source_dir.glob(f"??_{prompt_label}_{suffix}.png"))
    return matches[0] if matches else None


def load_timing(timing_path: Path):
    if not timing_path.exists():
        return {}
    data = json.load(open(timing_path))
    return {r["label"]: r for r in data}


def make_3way_strip(prompt_label, denoise_times, out_path, thumb_w=512):
    imgs = {}
    for mode in LABELS_ORDER:
        src = SOURCES[mode]
        suffix = "fullres" if mode == "fullres" else "progressive"
        p = find_image(src, prompt_label, suffix)
        if not p or not p.exists():
            print(f"  MISSING: {mode} {prompt_label} ({suffix}) in {src}")
            return
        imgs[mode] = Image.open(p).resize((thumb_w, thumb_w), Image.LANCZOS)

    label_h = 34
    title_h = 26
    n = len(LABELS_ORDER)
    strip_w = thumb_w * n + (n - 1) * 3
    strip_h = title_h + thumb_w + label_h
    strip = Image.new("RGB", (strip_w, strip_h), (20, 20, 20))
    strip.paste(
        make_label(prompt_label.upper(), strip_w, title_h, bg=(50, 50, 50)), (0, 0)
    )

    d_fr = denoise_times.get("fullres")
    for i, mode in enumerate(LABELS_ORDER):
        x = i * (thumb_w + 3)
        strip.paste(imgs[mode], (x, title_h))

        if mode == "fullres":
            text = f"Fullres  {d_fr:.1f}s" if d_fr else "Fullres"
            bg = BG_COLORS["fullres"]
        else:
            d_pr = denoise_times.get(mode)
            spd = f"{d_fr/d_pr:.2f}x" if (d_fr and d_pr) else "?"
            text = f"δ={mode}  {d_pr:.1f}s  ({spd})" if d_pr else f"δ={mode}"
            bg = BG_COLORS.get(mode, (30, 30, 60))

        strip.paste(make_label(text, thumb_w, label_h, bg=bg), (x, title_h + thumb_w))

    strip.save(out_path, quality=92)
    print(f"  Saved: {out_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timing = {}
    for mode, tpath in TIMING.items():
        timing[mode] = load_timing(tpath)

    strips = []
    for i, prompt_label in enumerate(PROMPTS, 1):
        denoise_times = {}

        # fullres time — look in both timing files
        for mode in [DELTA_A, DELTA_B]:
            if prompt_label in timing.get(mode, {}):
                r = timing[mode][prompt_label]
                if not denoise_times.get("fullres") and r.get("denoise_fullres"):
                    denoise_times["fullres"] = r["denoise_fullres"]

        # progressive times per delta
        for mode in [DELTA_A, DELTA_B]:
            if prompt_label in timing.get(mode, {}):
                r = timing[mode][prompt_label]
                d = r.get("denoise_s") or r.get("denoise_prog")
                if d:
                    denoise_times[mode] = d

        strip_path = OUT_DIR / f"{i:02d}_{prompt_label}_3way.png"
        make_3way_strip(prompt_label, denoise_times, strip_path)
        if strip_path.exists():
            strips.append(strip_path)

    if strips:
        w, h = Image.open(strips[0]).size
        cols = 2
        rows = (len(strips) + 1) // 2
        montage = Image.new(
            "RGB", (w * cols + 4, h * rows + (rows - 1) * 4), (15, 15, 15)
        )
        for i, p in enumerate(strips):
            r, c = divmod(i, cols)
            montage.paste(Image.open(p), (c * (w + 4), r * (h + 4)))
        montage.save(OUT_DIR / "montage_3way.png", quality=90)
        print(f"\nMontage: {OUT_DIR}/montage_3way.png  ({len(strips)} strips)")


if __name__ == "__main__":
    main()
