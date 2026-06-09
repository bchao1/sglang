#!/usr/bin/env python3
"""Generate delta vs speedup plot for Wan T2V progressive resolution PR.

Plots 480p and 720p measured speedups on the same axes — smooth log-fit
curves through measured points, same dark aesthetic as the FLUX.2 plot.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUT = Path("scratch/spectral-progressive-wan/images/speedup_vs_delta.png")

# ── Measured denoising speedups (RTX A6000, --dit-cpu-offload false) ─────────
# 480p: 50 steps · 480×832 · 81 frames · guidance=5.0 · flow_shift=5.0
MEASURED_480 = {
    0.01: 1.65,
    0.02: 1.86,
    0.05: 2.32,
    0.10: 2.78,
}
# 720p: 50 steps · 720×1280 (auto-aligned to 704×1280) · 81 frames
# avg over 10 Chinese cinematic prompts
MEASURED_720 = {
    0.01: 2.19,
    0.05: 3.34,
}
RECOMMENDED = 0.05


# ── Smooth curve: fit  speedup = a * log(delta / b) + c ──────────────────────
def log_model(d, a, b, c):
    return a * np.log(d / b) + c


# 480p: 4-point unconstrained fit
deltas_480 = np.array(list(MEASURED_480.keys()))
speeds_480 = np.array(list(MEASURED_480.values()))
popt_480, _ = curve_fit(log_model, deltas_480, speeds_480, p0=[0.5, 0.005, 2.0])

# 720p: 2-point fit with b fixed to 480p value (same noise spectrum model)
b_shared = popt_480[1]


def log_model_fixed_b(d, a, c):
    return a * np.log(d / b_shared) + c


deltas_720 = np.array(list(MEASURED_720.keys()))
speeds_720 = np.array(list(MEASURED_720.values()))
popt_720_ac, _ = curve_fit(log_model_fixed_b, deltas_720, speeds_720)


def curve_720(d):
    return log_model_fixed_b(d, *popt_720_ac)


deltas_curve = np.linspace(0.005, 0.12, 600)
# 720p curve only drawn within/slightly past measured range (0.01–0.05)
deltas_curve_720 = np.linspace(0.007, 0.065, 400)
speeds_curve_480 = log_model(deltas_curve, *popt_480)
speeds_curve_720 = curve_720(deltas_curve_720)

# ── Colors ────────────────────────────────────────────────────────────────────
C480 = "#e94560"  # coral-red  — 480p
C720 = "#4ecdc4"  # cyan-teal  — 720p
C_REC = "#f5a623"  # gold       — recommended δ

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

# --- 480p series ---
ax.plot(
    deltas_curve,
    speeds_curve_480,
    color=C480,
    linewidth=2.5,
    alpha=0.9,
    zorder=2,
    label="480×832",
)
ax.fill_between(deltas_curve, 1.0, speeds_curve_480, alpha=0.12, color=C480, zorder=1)
for d, s in MEASURED_480.items():
    is_rec = d == RECOMMENDED
    ax.scatter(
        d,
        s,
        s=260 if is_rec else 110,
        color=C_REC if is_rec else C480,
        marker="*" if is_rec else "o",
        zorder=6,
        edgecolors="white",
        linewidth=1.8,
    )

# --- 720p series ---
ax.plot(
    deltas_curve_720,
    speeds_curve_720,
    color=C720,
    linewidth=2.5,
    alpha=0.9,
    zorder=2,
    label="720×1280",
    linestyle="--",
)
ax.fill_between(
    deltas_curve_720, 1.0, speeds_curve_720, alpha=0.10, color=C720, zorder=1
)
for d, s in MEASURED_720.items():
    is_rec = d == RECOMMENDED
    ax.scatter(
        d,
        s,
        s=260 if is_rec else 110,
        color=C_REC if is_rec else C720,
        marker="*" if is_rec else "D",
        zorder=6,
        edgecolors="white",
        linewidth=1.8,
    )

# --- Annotations: 480p (right side) ---
ann_480 = {
    0.01: dict(
        xytext=(0.013, 1.42), color="#ccc", label=f"δ=0.01  {MEASURED_480[0.01]:.2f}×"
    ),
    0.02: dict(
        xytext=(0.026, 1.62), color="#ccc", label=f"δ=0.02  {MEASURED_480[0.02]:.2f}×"
    ),
    0.05: dict(
        xytext=(0.063, 2.10), color=C_REC, label=f"δ=0.05  {MEASURED_480[0.05]:.2f}×  ★"
    ),
    0.10: dict(
        xytext=(0.073, 2.58), color="#ccc", label=f"δ=0.10  {MEASURED_480[0.10]:.2f}×"
    ),
}
for d, cfg in ann_480.items():
    ax.annotate(
        f"  {cfg['label']}",
        xy=(d, MEASURED_480[d]),
        xytext=cfg["xytext"],
        fontsize=9.5,
        color=cfg["color"],
        fontweight="bold" if d == RECOMMENDED else "normal",
        arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.2),
    )

# --- Annotations: 720p (left / above) ---
ann_720 = {
    0.01: dict(
        xytext=(0.0055, 2.38), color=C720, label=f"δ=0.01  {MEASURED_720[0.01]:.2f}×"
    ),
    0.05: dict(
        xytext=(0.062, 3.14), color=C_REC, label=f"δ=0.05  {MEASURED_720[0.05]:.2f}×  ★"
    ),
}
for d, cfg in ann_720.items():
    ax.annotate(
        f"  {cfg['label']}",
        xy=(d, MEASURED_720[d]),
        xytext=cfg["xytext"],
        fontsize=9.5,
        color=cfg["color"],
        fontweight="bold" if d == RECOMMENDED else "normal",
        arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1.2),
    )

# --- Baseline and helpers ---
ax.axhline(y=1.0, color="#888", linewidth=1.0, linestyle="--", alpha=0.5)
ax.text(0.113, 1.02, "fullres baseline", color="#888", fontsize=9, ha="right")
ax.text(
    0.074,
    3.48,
    "← higher δ = more low-res steps",
    color="#888",
    fontsize=9,
    style="italic",
)

# --- Legend ---
legend = ax.legend(
    title="Resolution",
    title_fontsize=10,
    fontsize=10,
    loc="upper left",
    facecolor="#1a1a2e",
    edgecolor="#555",
    labelcolor="#ddd",
)
legend.get_title().set_color("#aaa")

# --- Axes styling ---
ax.set_xlabel("δ (noise-dominated tolerance)", fontsize=12, color="#ddd", labelpad=8)
ax.set_ylabel("Denoising speedup  (×)", fontsize=12, color="#ddd", labelpad=8)
ax.set_title(
    "Wan 2.1 T2V Progressive Generation — Speedup vs δ\n"
    "RTX A6000 · 50 steps · 81 frames · denoising loop only",
    fontsize=12,
    color="#eee",
    pad=12,
)

ax.set_xlim(0.004, 0.115)
ax.set_ylim(0.88, 3.60)
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
