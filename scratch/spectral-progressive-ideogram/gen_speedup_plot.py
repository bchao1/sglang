#!/usr/bin/env python3
"""Generate denoising speedup chart from Ideogram 4 delta sweep results.

Usage:
    python scratch/spectral-progressive-ideogram/gen_speedup_plot.py --json <results.json>
    python scratch/spectral-progressive-ideogram/gen_speedup_plot.py --json <results.json> --out /tmp/speedup.png
"""

import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# X-brand purple used across the multi-model speedup charts in final_PR_smoke
IDEOGRAM_COLOR = "#7856FF"
BG = "#FFFFFF"
GRID_COL = "#E7E7E7"
TEXT_COL = "#0F1419"
SUB_COL = "#536471"


def plot(json_path: str, out_path: str) -> None:
    with open(json_path) as f:
        data = json.load(f)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(BG)

    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
        spine.set_linewidth(0.8)

    ax.axhline(y=1.0, color=GRID_COL, linewidth=1.2, zorder=1)

    model_key = "Ideogram 4"
    md = data.get(model_key, {})
    pts = [
        (p["delta"], p["speedup"])
        for p in md.get("points", [])
        if p.get("speedup") is not None
    ]

    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # End-point annotation
        ax.annotate(
            f"{ys[-1]:.2f}×",
            xy=(xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=16,
            fontweight="bold",
            color=IDEOGRAM_COLOR,
            va="center",
        )

        # Glow + line
        ax.plot(xs, ys, color=IDEOGRAM_COLOR, linewidth=8, alpha=0.12,
                solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color=IDEOGRAM_COLOR, marker="D", markersize=11,
                markeredgewidth=2.0, markeredgecolor="white", linewidth=3.0,
                solid_capstyle="round", label=model_key, zorder=3)

        # Per-point labels
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.2f}×",
                xy=(x, y),
                xytext=(0, 11),
                textcoords="offset points",
                fontsize=12,
                color=IDEOGRAM_COLOR,
                ha="center",
            )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(0.008, 0.18)

    all_speedups = [p["speedup"] for p in md.get("points", []) if p.get("speedup")]
    ymax = max(all_speedups) * 1.20 if all_speedups else 2.5
    ax.set_ylim(0.85, ymax)
    yticks = np.arange(1.0, ymax, 0.25)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))

    ax.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=0)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5, linestyle=":", zorder=0)
    ax.tick_params(colors=SUB_COL, labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(SUB_COL)

    ax.set_xlabel("Progressive delta (δ)", fontsize=18, color=SUB_COL, labelpad=10)
    ax.set_ylabel("Denoising speedup (×)", fontsize=18, color=SUB_COL, labelpad=10)

    fullres = md.get("fullres_denoise_s")
    steps = md.get("steps", "?")
    sub = f"Ideogram 4 fp8 · {steps} steps · 1024×1024 · baseline={fullres:.1f}s" if fullres else f"Ideogram 4 fp8 · {steps} steps · 1024×1024"
    fig.suptitle(
        "Spectral Progressive Diffusion — Ideogram 4\nDenoising Speedup vs. δ",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COL,
        y=1.02,
    )
    ax.set_title(sub, fontsize=13, color=SUB_COL, pad=4)

    ax.text(0.0083, 1.006, "1× baseline", fontsize=13, color=SUB_COL, va="bottom")

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")

    # Summary table
    print(f"\nModel: {model_key}  (fullres={fullres:.2f}s, {steps} steps)")
    print(f"{'delta':>8}  {'denoise_s':>10}  {'speedup':>9}")
    print("-" * 32)
    for p in md.get("points", []):
        sp = p.get("speedup")
        ds = p.get("denoise_s")
        sp_str = f"{sp:.4f}×" if sp else "N/A"
        ds_str = f"{ds:.2f}s" if ds else "N/A"
        print(f"  {p['delta']:>6}  {ds_str:>10}  {sp_str:>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "speedup_ideogram4.png"),
    )
    args = parser.parse_args()
    plot(args.json, args.out)


if __name__ == "__main__":
    main()
