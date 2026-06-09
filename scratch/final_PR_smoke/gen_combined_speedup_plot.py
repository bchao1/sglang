#!/usr/bin/env python3
"""Generate a combined speedup-vs-delta chart for all progressive-resolution models.

Data is drawn from per-model benchmarks on RTX A6000 48 GB,
--dit-cpu-offload false, denoising loop only.

Usage:
    python scratch/final_PR_smoke/gen_combined_speedup_plot.py
    python scratch/final_PR_smoke/gen_combined_speedup_plot.py --out /tmp/speedup.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Benchmark data (delta, speedup) per model
# Sources: per-model PR descriptions / individual scratch benchmarks
# All on RTX A6000 48 GB, denoising loop only, L1 distance threshold
# ---------------------------------------------------------------------------
MODEL_DATA = {
    "FLUX.1-dev\n(50 steps, 1024²)": {
        "color": "#1565C0",
        "marker": "o",
        "points": [
            (0.01, 1.32),
            (0.05, 1.63),
            (0.10, 1.83),
        ],
    },
    "FLUX.2-klein-4B\n(30 steps, 1024²)": {
        "color": "#0288D1",
        "marker": "s",
        "points": [
            (0.05, 1.77),
            (0.10, 1.93),
        ],
    },
    "Z-Image\n(50 steps, 1024²)": {
        "color": "#7B1FA2",
        "marker": "D",
        "points": [
            (0.01, 1.53),
            (0.05, 2.03),
            (0.10, 2.33),
            (0.20, 2.66),
            (0.50, 2.97),
        ],
    },
    "Wan 2.1 T2V 1.3B\n(50 steps, 480×832, 81fr)": {
        "color": "#2E7D32",
        "marker": "^",
        "points": [
            (0.01, 1.65),
            (0.02, 1.86),
            (0.05, 2.32),
            (0.10, 2.78),
        ],
    },
    "Qwen-Image\n(30 steps, 1024²)": {
        "color": "#E65100",
        "marker": "v",
        "points": [
            (0.05, 1.29),
            (0.10, 1.27),
            (0.20, 1.69),
        ],
    },
}


def plot_combined(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for label, cfg in MODEL_DATA.items():
        deltas = [p[0] for p in cfg["points"]]
        speedups = [p[1] for p in cfg["points"]]
        ax.plot(
            deltas,
            speedups,
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=7,
            linewidth=1.8,
            label=label,
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Progressive delta (δ)", fontsize=12)
    ax.set_ylabel("Denoising speedup (×)", fontsize=12)
    ax.set_title(
        "Spectral Progressive Resolution — Speedup vs. δ\n"
        "All models, RTX A6000 48 GB, denoising loop only",
        fontsize=12,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlim(0.008, 0.65)
    ax.set_ylim(0.9, 3.1)
    ax.set_yticks(np.arange(1.0, 3.2, 0.25))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}×"))
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(
        fontsize=9,
        loc="upper left",
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=2.2,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "combined_speedup.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()
    plot_combined(args.out)


if __name__ == "__main__":
    main()
