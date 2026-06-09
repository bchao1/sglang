#!/usr/bin/env python3
"""
Fix comparison strip labels for delta=0.1 run.
The generation script had the label hardcoded as 'delta=0.05'; this script
re-draws the bottom label bar with the correct delta=0.10 value.
Reads timing from scratch/results/pr_images_d01/timing.json.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DOC_DIR = Path("docs_new/images/progressive_d01")
TIMING = Path("scratch/results/pr_images_d01/timing.json")
DELTA = 0.10


def make_label(text, width, height=36, bg=(20, 20, 20), fg=(255, 255, 255)):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 18)
            break
        except Exception:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=fg, font=font)
    return img


def fix_strip(strip_path: Path, denoise_fr: float, denoise_pr: float):
    strip = Image.open(strip_path)
    w, h = strip.size
    label_h = 36
    # Bottom two rows are the two label bars
    img_area_h = h - label_h - 28  # 28px for title bar
    thumb_w = w // 2 - 3  # 3px divider

    speedup = denoise_fr / denoise_pr
    lbl_fr = make_label(f"Fullres  denoise={denoise_fr:.1f}s", thumb_w)
    lbl_pr = make_label(
        f"Progressive d={DELTA}  denoise={denoise_pr:.1f}s  ({speedup:.2f}x)",
        thumb_w,
        bg=(0, 60, 30),
    )

    # Replace bottom labels in-place
    y_label = h - label_h
    strip.paste(lbl_fr, (0, y_label))
    strip.paste(lbl_pr, (thumb_w + 6, y_label))
    strip.save(strip_path, quality=95)
    print(f"  Fixed: {strip_path.name}  ({denoise_pr:.1f}s → {speedup:.2f}x)")


def main():
    data = json.load(open(TIMING))
    timing = {r["label"]: r for r in data}

    for strip_path in sorted(DOC_DIR.glob("*_compare.png")):
        label = strip_path.stem.split("_", 2)[
            1
        ]  # e.g. "01_landscape_compare" → "landscape"
        # Handle numeric prefix: "01_landscape_compare" → name part after index
        parts = strip_path.stem.split("_")
        label = parts[1] if len(parts) > 1 else parts[0]

        if label not in timing:
            print(f"  SKIP {strip_path.name} — no timing data for '{label}'")
            continue

        r = timing[label]
        fix_strip(strip_path, r["denoise_fullres"], r["denoise_prog"])

    # Regenerate montage
    strips = sorted(DOC_DIR.glob("*_compare.png"))
    strips = [p for p in strips if "montage" not in p.name]
    if strips:
        imgs = [Image.open(p) for p in strips]
        sw, sh = imgs[0].size
        cols = 2
        rows = (len(imgs) + 1) // 2
        montage = Image.new(
            "RGB", (sw * cols + 4, sh * rows + (rows - 1) * 4), (20, 20, 20)
        )
        for i, img in enumerate(imgs):
            r, c = divmod(i, cols)
            montage.paste(img, (c * (sw + 4), r * (sh + 4)))
        montage_path = DOC_DIR / "montage_progressive_vs_fullres.png"
        montage.save(montage_path, quality=92)
        print(f"\nMontage saved: {montage_path}")


if __name__ == "__main__":
    main()
