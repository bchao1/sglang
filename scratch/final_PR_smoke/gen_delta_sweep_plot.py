#!/usr/bin/env python3
"""Generate combined speedup-vs-delta chart from bench_all_delta_sweep.sh results.

Usage:
    python scratch/final_PR_smoke/gen_delta_sweep_plot.py --json <results.json>
    python scratch/final_PR_smoke/gen_delta_sweep_plot.py --json <results.json> --out /tmp/speedup.png
"""

import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Twitter / X brand palette  (https://developer.twitter.com/en/docs/twitter-for-websites/cards/guides/getting-started)
# ---------------------------------------------------------------------------
# X Blue   #1D9BF0   X Green  #00BA7C   X Purple #7856FF
# X Orange #FF7A00   X Pink   #F91880
MODEL_STYLE = {
    "FLUX.1-dev": {
        "label": "FLUX.1-dev",
        "color": "#1D9BF0",  # X Blue
        "marker": "o",
        "zorder": 4,
    },
    "FLUX.2-klein-4B": {
        "label": "FLUX.2-klein-4B",
        "color": "#00BA7C",  # X Green
        "marker": "s",
        "zorder": 4,
    },
    "Z-Image": {
        "label": "Z-Image",
        "color": "#7856FF",  # X Purple
        "marker": "D",
        "zorder": 4,
    },
    "Wan 2.1 T2V 1.3B": {
        "label": "Wan 2.1 T2V 1.3B",
        "color": "#FF7A00",  # X Orange
        "marker": "^",
        "zorder": 5,
    },
    "Qwen-Image": {
        "label": "Qwen-Image",
        "color": "#F91880",  # X Pink
        "marker": "v",
        "zorder": 4,
    },
}

# Draw order: bottom-to-top visually (Wan ends highest, Qwen lowest).
# Legend is reversed so the top entry matches the top line.
MODEL_ORDER = [
    "Wan 2.1 T2V 1.3B",
    "Z-Image",
    "FLUX.2-klein-4B",
    "FLUX.1-dev",
    "Qwen-Image",
]

BG = "#FFFFFF"
GRID_COL = "#E7E7E7"
TEXT_COL = "#0F1419"  # X near-black
SUB_COL = "#536471"  # X secondary text


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

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
    ax.set_facecolor(BG)

    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
        spine.set_linewidth(0.8)

    # --- baseline reference line ---
    ax.axhline(y=1.0, color=GRID_COL, linewidth=1.2, zorder=1)

    # --- plot each model ---
    for key in MODEL_ORDER:
        if key not in data:
            continue
        cfg = MODEL_STYLE[key]
        pts = [
            (p["delta"], p["speedup"])
            for p in data[key]["points"]
            if p["speedup"] is not None
        ]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # end-point speedup label
        y_nudge = -13 if key == "FLUX.1-dev" else 0
        ax.annotate(
            f"{ys[-1]:.2f}×",
            xy=(xs[-1], ys[-1]),
            xytext=(7, y_nudge),
            textcoords="offset points",
            fontsize=16,
            fontweight="bold",
            color=cfg["color"],
            va="center",
        )

        # subtle glow
        ax.plot(
            xs,
            ys,
            color=cfg["color"],
            linewidth=8,
            alpha=0.12,
            solid_capstyle="round",
            zorder=cfg["zorder"] - 1,
        )
        ax.plot(
            xs,
            ys,
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=11,
            markeredgewidth=2.0,
            markeredgecolor="white",
            linewidth=3.0,
            solid_capstyle="round",
            label=cfg["label"],
            zorder=cfg["zorder"],
        )

    # --- axes ---
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(0.0085, 0.33)  # extra right room for end-point annotations

    all_speedups = [
        p["speedup"]
        for md in data.values()
        for p in md["points"]
        if p["speedup"] is not None
    ]
    ymax = max(all_speedups) * 1.10 if all_speedups else 3.5
    ax.set_ylim(0.88, ymax)
    yticks = np.arange(1.0, ymax, 0.25)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))

    ax.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=0)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5, linestyle=":", zorder=0)
    ax.tick_params(colors=SUB_COL, labelsize=21)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(SUB_COL)

    ax.set_xlabel("Progressive delta (δ)", fontsize=24, color=SUB_COL, labelpad=10)
    ax.set_ylabel("Denoising speedup (×)", fontsize=24, color=SUB_COL, labelpad=10)

    # --- title ---
    fig.suptitle(
        "Spectral Progressive Diffusion\nSpeedup vs. δ",
        fontsize=26,
        fontweight="bold",
        color=TEXT_COL,
        y=1.01,
    )

    # --- legend in upper-left blank space (above the data lines at small δ) ---
    leg = ax.legend(
        loc="upper left",
        fontsize=17,
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID_COL,
        facecolor=BG,
        handlelength=1.8,
        handleheight=1.1,
        labelspacing=0.45,
        borderpad=0.7,
    )
    for text in leg.get_texts():
        text.set_color(TEXT_COL)

    # --- "1× baseline" label ---
    ax.text(
        0.0088,
        1.006,
        "1× baseline",
        fontsize=16,
        color=SUB_COL,
        va="bottom",
    )

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")

    # speedup table
    print("\nSpeedup table:")
    header = f"{'Model':<26} {'Fullres':>9}  " + "  ".join(
        f"δ={d}" for d in [0.01, 0.02, 0.05, 0.1, 0.2]
    )
    print(header)
    print("-" * len(header))
    for key in MODEL_ORDER:
        if key not in data:
            continue
        md = data[key]
        row = f"{key:<26} {md.get('fullres_denoise_s', 'N/A'):>9.2f}s "
        dm = {p["delta"]: p["speedup"] for p in md["points"]}
        for d in [0.01, 0.02, 0.05, 0.1, 0.2]:
            sp = dm.get(d)
            row += f"  {sp:.2f}×" if sp else "   N/A"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "combined_speedup.png"),
    )
    args = parser.parse_args()
    plot(args.json, args.out)


if __name__ == "__main__":
    main()
