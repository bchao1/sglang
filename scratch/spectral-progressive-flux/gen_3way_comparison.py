#!/usr/bin/env python3
"""
Generate 3-way comparison strips: fullres | δ=0.05 | δ=0.10
Uses already-generated raw images from pr_images/ and pr_images_d01/.
Reads timing.json to show denoising speedup labels.
Saves to docs_new/images/progressive_3way/
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SOURCES = {
    "fullres": Path("scratch/results/pr_images"),
    "0.05": Path("scratch/results/pr_images"),
    "0.10": Path("scratch/results/pr_images_d01"),
}

TIMING = {
    "0.05": Path("scratch/results/pr_images/timing.json"),
    "0.10": Path("scratch/results/pr_images_d01/timing.json"),
}

OUT_DIR = Path("docs_new/images/progressive_3way")

LABELS_ORDER = ["fullres", "0.05", "0.10"]

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
    "0.05": (0, 50, 20),
    "0.10": (0, 40, 60),
}


def make_label(text, width, height=32, bg=(20, 20, 20), fg=(255, 255, 255)):
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


def find_image(source_dir: Path, label: str, suffix: str) -> Path | None:
    matches = list(source_dir.glob(f"??_{label}_{suffix}.png"))
    return matches[0] if matches else None


def make_3way_strip(label: str, denoise: dict, out_path: Path, thumb_w: int = 512):
    imgs = {}
    for mode in LABELS_ORDER:
        src = SOURCES[mode]
        suffix = "fullres" if mode == "fullres" else "progressive"
        p = find_image(src, label, suffix)
        if not p or not p.exists():
            print(f"  MISSING: {mode} {label} ({suffix})")
            return
        imgs[mode] = Image.open(p).resize((thumb_w, thumb_w), Image.LANCZOS)

    label_h = 32
    title_h = 24
    n = len(LABELS_ORDER)
    strip_w = thumb_w * n + (n - 1) * 3
    strip_h = title_h + thumb_w + label_h

    strip = Image.new("RGB", (strip_w, strip_h), (20, 20, 20))
    title = make_label(label.upper(), strip_w, title_h, bg=(50, 50, 50))
    strip.paste(title, (0, 0))

    for i, mode in enumerate(LABELS_ORDER):
        x = i * (thumb_w + 3)
        strip.paste(imgs[mode], (x, title_h))
        d_fr = denoise.get("fullres")
        if mode == "fullres":
            text = f"Fullres  {d_fr:.1f}s" if d_fr else "Fullres"
            bg = BG_COLORS["fullres"]
        else:
            d_pr = denoise.get(mode)
            spd = f"{d_fr/d_pr:.2f}x" if (d_fr and d_pr) else "?"
            text = f"d={mode}  {d_pr:.1f}s  ({spd})" if d_pr else f"d={mode}"
            bg = BG_COLORS[mode]
        lbl = make_label(text, thumb_w, label_h, bg=bg)
        strip.paste(lbl, (x, title_h + thumb_w))

    strip.save(out_path, quality=92)
    print(f"  Saved: {out_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load timing data
    timing = {}
    for mode, tpath in TIMING.items():
        if tpath.exists():
            data = json.load(open(tpath))
            timing[mode] = {r["label"]: r for r in data}

    strips = []
    for i, label in enumerate(PROMPTS, 1):
        denoise = {}
        # Fullres from any available timing file
        for mode in ["0.05", "0.10"]:
            if mode in timing and label in timing[mode]:
                r = timing[mode][label]
                key = "denoise_fullres" if r.get("denoise_fullres") else None
                if key:
                    denoise["fullres"] = r[key]
                    break

        for mode in ["0.05", "0.10"]:
            if mode in timing and label in timing[mode]:
                r = timing[mode][label]
                key = "denoise_prog" if r.get("denoise_prog") else None
                if key:
                    denoise[mode] = r[key]

        strip_path = OUT_DIR / f"{i:02d}_{label}_3way.png"
        make_3way_strip(label, denoise, strip_path)
        if strip_path.exists():
            strips.append(strip_path)

    # Montage (2 cols, 5 rows)
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
        print(f"\nMontage saved: {OUT_DIR}/montage_3way.png  ({len(strips)} strips)")


if __name__ == "__main__":
    main()
