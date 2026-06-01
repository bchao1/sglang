#!/usr/bin/env python3
"""
Generate 4-way comparison strips: fullres | d=0.05 | d=0.10 | d=0.50
Uses already-generated raw images from pr_images, pr_images_d01, pr_images_d05.
Reads timing from timing.json files to show per-image denoising speedup.
Saves to docs_new/images/progressive_4way/
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS = {
    "fullres": Path("scratch/results/pr_images"),  # baseline
    "0.05": Path("scratch/results/pr_images"),  # same dir, different suffix
    "0.10": Path("scratch/results/pr_images_d01"),
    "0.50": Path("scratch/results/pr_images_d05"),
}

TIMING = {
    "0.05": Path("scratch/results/pr_images/timing.json"),
    "0.10": Path("scratch/results/pr_images_d01/timing.json"),
    "0.50": Path("scratch/results/pr_images_d05/timing.json"),
}

OUT_DIR = Path("docs_new/images/progressive_4way")

LABELS_ORDER = ["fullres", "0.05", "0.10", "0.50"]

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


BG_COLORS = {
    "fullres": (40, 40, 40),
    "0.05": (0, 50, 20),
    "0.10": (0, 40, 60),
    "0.50": (60, 30, 0),
}


def make_4way_strip(label: str, denoise: dict, out_path: Path, thumb_w: int = 384):
    """4-way strip: fullres | d=0.05 | d=0.10 | d=0.50"""
    imgs = {}
    for mode in LABELS_ORDER:
        if mode == "fullres":
            p = RESULTS["fullres"] / f"??_{label}_fullres.png"
        else:
            d_str = mode.replace(".", "")
            p = RESULTS[mode] / f"??_{label}_progressive.png"
        # Find the file
        parent = RESULTS["fullres"] if mode in ("fullres", "0.05") else RESULTS[mode]
        suffix = "fullres" if mode == "fullres" else "progressive"
        matches = list(parent.glob(f"??_{label}_{suffix}.png"))
        if not matches:
            print(f"  MISSING: {mode} {label}")
            return
        imgs[mode] = Image.open(matches[0]).resize((thumb_w, thumb_w), Image.LANCZOS)

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
        # Label
        if mode == "fullres":
            text = f"Fullres  {denoise.get('fullres', '?'):.1f}s  (1.00x)"
            bg = BG_COLORS["fullres"]
        else:
            den = denoise.get(mode)
            fr = denoise.get("fullres")
            spd = f"{fr/den:.2f}x" if den and fr else "?"
            text = f"d={mode}  {den:.1f}s  ({spd})" if den else f"d={mode}"
            bg = BG_COLORS[mode]
        lbl = make_label(text, thumb_w, label_h, bg=bg)
        strip.paste(lbl, (x, title_h + thumb_w))

    strip.save(out_path, quality=92)
    print(f"  Saved: {out_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load timing data
    timing_by_mode = {}
    for mode, tpath in TIMING.items():
        if tpath.exists():
            data = json.load(open(tpath))
            timing_by_mode[mode] = {r["label"]: r for r in data}
        else:
            print(
                f"  WARN: timing.json missing for d={mode} — labels will be incomplete"
            )

    strips = []
    for i, label in enumerate(PROMPTS, 1):
        denoise = {}
        # fullres denoise from any available timing file
        for mode in ["0.05", "0.10", "0.50"]:
            if mode in timing_by_mode and label in timing_by_mode[mode]:
                r = timing_by_mode[mode][label]
                if "denoise_fullres" in r:
                    denoise["fullres"] = r["denoise_fullres"]
                    break

        for mode in ["0.05", "0.10", "0.50"]:
            if mode in timing_by_mode and label in timing_by_mode[mode]:
                r = timing_by_mode[mode][label]
                if "denoise_prog" in r:
                    denoise[mode] = r["denoise_prog"]

        strip_path = OUT_DIR / f"{i:02d}_{label}_4way.png"
        make_4way_strip(label, denoise, strip_path)
        strips.append(strip_path)

    # Montage (2 cols, 5 rows)
    strips = [p for p in strips if p.exists()]
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
        montage.save(OUT_DIR / "montage_4way.png", quality=90)
        print(f"\nMontage saved: {OUT_DIR}/montage_4way.png")


if __name__ == "__main__":
    main()
