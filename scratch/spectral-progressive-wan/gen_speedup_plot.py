#!/usr/bin/env python3
"""Generate delta vs speedup plot for Wan T2V progressive resolution PR."""

import sys

sys.path.insert(0, "python")

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.scheduler_utils import (
    compute_stage_transitions,
    find_transition_steps,
)

OUT = Path("scratch/spectral-progressive-wan/images/speedup_vs_delta.png")

# ── Wan 2.1 T2V 1.3B at 50 steps, 480×832, L1 progressive ─────────────────
# Latent: 480//8=60 H, 832//8=104 W  → 60×104 full-res, 30×52 half-res
H_LAT, W_LAT = 60, 104
N_STEPS = 50
A, BETA = 219.484718, 2.422687  # WAN 2.1 VAE spectrum (VChitect, 9050 videos)

# Reference sigma schedule: shift=5.0, linspace(1,0,51)[:-1]
raw_sigmas = np.linspace(1.0, 0.0, N_STEPS + 1)[:-1]
shifted = 5.0 * raw_sigmas / (1.0 + 4.0 * raw_sigmas)
sigmas_t = torch.tensor(np.append(shifted, 0.0), dtype=torch.float32)

# T_lat cancels in speedup ratio, so token-step is purely spatial × temporal constant
FULLRES_TS = N_STEPS * H_LAT * W_LAT  # 50 × 60×104 = 312,000


def theoretical_speedup(delta):
    """Token-step speedup for a given delta (L=1)."""
    stage_sigmas = compute_stage_transitions(delta, 1, A, BETA, H_LAT, W_LAT)
    trans = find_transition_steps(sigmas_t, stage_sigmas, N_STEPS)
    t = trans.get(2, N_STEPS)
    token_steps = (
        t * (H_LAT // 2) * (W_LAT // 2)
        + (N_STEPS - t) * H_LAT * W_LAT
    )
    return FULLRES_TS / token_steps if token_steps > 0 else 1.0


# ── Measured wall-clock speedups (RTX A6000, --dit-cpu-offload false) ───────
MEASURED = {
    0.01: 1.65,
    0.02: 1.86,
    0.05: 2.32,
    0.10: 2.78,
}
RECOMMENDED = 0.05

# ── Theoretical curve ────────────────────────────────────────────────────────
deltas_theory = np.linspace(0.005, 0.60, 400)
speedups_theory = [theoretical_speedup(float(d)) for d in deltas_theory]

ANNOTATED_DELTAS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
annotated = [(d, theoretical_speedup(float(d))) for d in ANNOTATED_DELTAS]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

# Theoretical curve
ax.plot(
    deltas_theory,
    speedups_theory,
    color="#e94560",
    linewidth=2.5,
    alpha=0.9,
    zorder=2,
    label="token-step theory",
)
ax.fill_between(
    deltas_theory, 1.0, speedups_theory, alpha=0.15, color="#e94560", zorder=1
)

# Small theory annotation points (non-measured)
for d, s in annotated:
    if d not in MEASURED:
        ax.scatter(
            d, s, s=60, color="#e94560", zorder=4,
            edgecolors="white", linewidth=1.0, alpha=0.7,
        )

# Measured wall-clock points
for d, s in MEASURED.items():
    is_rec = d == RECOMMENDED
    marker = "*" if is_rec else "o"
    sz = 260 if is_rec else 120
    color = "#f5a623" if is_rec else "#4fc3f7"
    ax.scatter(
        d, s, s=sz, color=color, zorder=6,
        edgecolors="white", linewidth=1.8, marker=marker,
    )

# Annotate measured points
offsets = {
    0.01: (0.02, -0.14),
    0.02: (0.03, -0.17),
    0.05: (0.07, -0.12),
    0.10: (0.13, -0.16),
}
labels = {
    0.01: f"δ=0.01  {MEASURED[0.01]:.2f}×",
    0.02: f"δ=0.02  {MEASURED[0.02]:.2f}×",
    0.05: f"δ=0.05  {MEASURED[0.05]:.2f}× (recommended)",
    0.10: f"δ=0.10  {MEASURED[0.10]:.2f}×\nbest throughput",
}
colors = {0.01: "#4fc3f7", 0.02: "#4fc3f7", 0.05: "#f5a623", 0.10: "#aaa"}

for d, s in MEASURED.items():
    ox, oy = offsets[d]
    ax.annotate(
        f"  {labels[d]}",
        xy=(d, s),
        xytext=(d + ox, s + oy),
        fontsize=9,
        color=colors[d],
        fontweight="bold" if d == RECOMMENDED else "normal",
        arrowprops=dict(arrowstyle="->", color=colors[d], lw=1.3),
    )

# Baseline
ax.axhline(y=1.0, color="#888", linewidth=1.0, linestyle="--", alpha=0.5)
ax.text(0.57, 1.02, "fullres baseline", color="#888", fontsize=9, ha="right")

# Legend
legend_elements = [
    plt.Line2D([0], [0], color="#e94560", linewidth=2.5,
               label="theory (token-step model)"),
    plt.scatter([], [], s=120, color="#4fc3f7", edgecolors="white",
                linewidth=1.5, label="measured (wall-clock)"),
    plt.scatter([], [], s=220, color="#f5a623", marker="*",
                edgecolors="white", linewidth=1.5, label="recommended δ"),
]
ax.legend(
    handles=legend_elements, loc="lower right", fontsize=9,
    facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ccc",
)

ax.set_xlabel("δ (noise-dominated tolerance)", fontsize=12, color="#ddd", labelpad=8)
ax.set_ylabel("Denoising speedup  (×)", fontsize=12, color="#ddd", labelpad=8)
ax.set_title(
    "Wan 2.1 T2V Progressive Generation — Speedup vs δ\n"
    "RTX A6000 · 50 steps · 480×832 · 81 frames · denoising loop only",
    fontsize=12, color="#eee", pad=12,
)

ax.set_xlim(0.0, 0.62)
ax.set_ylim(0.90, 3.30)
ax.set_xticks([0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
ax.set_xticklabels(["0.01", "0.02", "0.05", "0.10", "0.20", "0.50"])

ax.tick_params(colors="#aaa", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#444")
ax.grid(True, color="#333", linewidth=0.7, alpha=0.6)

# Theory-only annotation labels
for d, s in annotated:
    if d in MEASURED:
        continue
    ax.annotate(
        f"{s:.2f}×", xy=(d, s), xytext=(d + 0.005, s + 0.06),
        fontsize=9, color="#ccc",
    )

ax.text(0.01, 3.18, "← increasing speedup", color="#888", fontsize=9, style="italic")
ax.text(
    0.01, 3.08,
    "   (quality remains high up to δ≈0.05, verify δ=0.10 on dynamic scenes)",
    color="#888", fontsize=9, style="italic",
)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
