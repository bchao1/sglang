#!/usr/bin/env python3
"""Generate delta vs speedup plot for Wan T2V progressive resolution PR.

Plots measured wall-clock speedups only — smooth log-fit curve through
the 4 measured points, same dark aesthetic as the FLUX.2 plot.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUT = Path("scratch/spectral-progressive-wan/images/speedup_vs_delta.png")

# ── Measured wall-clock speedups (RTX A6000, --dit-cpu-offload false) ───────
# 50 steps · 480×832 · 81 frames · guidance=5.0 · flow_shift=5.0
MEASURED = {
    0.01: 1.65,
    0.02: 1.86,
    0.05: 2.32,
    0.10: 2.78,
}
RECOMMENDED = 0.05

# ── Smooth curve: fit  speedup = a * log(delta / b) + c  to measured data ──
deltas_m = np.array(list(MEASURED.keys()))
speeds_m = np.array(list(MEASURED.values()))


def log_model(d, a, b, c):
    return a * np.log(d / b) + c


popt, _ = curve_fit(log_model, deltas_m, speeds_m, p0=[0.5, 0.005, 2.0])

deltas_curve = np.linspace(0.005, 0.12, 600)
speeds_curve = log_model(deltas_curve, *popt)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

# Smooth curve
ax.plot(
    deltas_curve, speeds_curve,
    color="#e94560", linewidth=2.5, alpha=0.9, zorder=2,
)
ax.fill_between(deltas_curve, 1.0, speeds_curve, alpha=0.15, color="#e94560", zorder=1)

# Measured points
for d, s in MEASURED.items():
    is_rec = d == RECOMMENDED
    ax.scatter(
        d, s,
        s=260 if is_rec else 110,
        color="#f5a623" if is_rec else "#e94560",
        marker="*" if is_rec else "o",
        zorder=6, edgecolors="white", linewidth=1.8,
    )

# Annotations
ann = {
    0.01: dict(xytext=(0.014, 1.48), color="#ccc",  label=f"δ=0.01  {MEASURED[0.01]:.2f}×"),
    0.02: dict(xytext=(0.026, 1.68), color="#ccc",  label=f"δ=0.02  {MEASURED[0.02]:.2f}×"),
    0.05: dict(xytext=(0.062, 2.12), color="#f5a623", label=f"δ=0.05  {MEASURED[0.05]:.2f}×\n★ best tradeoff"),
    0.10: dict(xytext=(0.072, 2.60), color="#ccc",  label=f"δ=0.10  {MEASURED[0.10]:.2f}×"),
}
for d, cfg in ann.items():
    ax.annotate(
        f"  {cfg['label']}",
        xy=(d, MEASURED[d]), xytext=cfg["xytext"],
        fontsize=10, color=cfg["color"],
        fontweight="bold" if d == RECOMMENDED else "normal",
        arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.3),
    )

# Baseline
ax.axhline(y=1.0, color="#888", linewidth=1.0, linestyle="--", alpha=0.5)
ax.text(0.108, 1.02, "fullres baseline", color="#888", fontsize=9, ha="right")

ax.text(0.006, 2.72, "← increasing speedup", color="#888", fontsize=9, style="italic")

ax.set_xlabel("δ (noise-dominated tolerance)", fontsize=12, color="#ddd", labelpad=8)
ax.set_ylabel("Denoising speedup  (×)", fontsize=12, color="#ddd", labelpad=8)
ax.set_title(
    "Wan 2.1 T2V Progressive Generation — Speedup vs δ\n"
    "RTX A6000 · 50 steps · 480×832 · 81 frames · denoising loop only",
    fontsize=12, color="#eee", pad=12,
)

ax.set_xlim(0.005, 0.115)
ax.set_ylim(0.90, 2.90)
ax.set_xticks([0.01, 0.02, 0.05, 0.10])
ax.set_xticklabels(["0.01", "0.02", "0.05", "0.10"])

ax.tick_params(colors="#aaa", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#444")
ax.grid(True, color="#333", linewidth=0.7, alpha=0.6)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
