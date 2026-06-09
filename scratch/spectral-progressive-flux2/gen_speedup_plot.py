#!/usr/bin/env python3
"""Generate delta vs speedup plot for FLUX.2 progressive resolution PR."""

import sys

sys.path.insert(0, "python")

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.scheduler_utils import (
    compute_stage_transitions,
    find_transition_steps,
)

OUT = Path("docs_new/images/progressive_flux2/speedup_vs_delta.png")

# ── FLUX.2-klein at 30 steps, 1024×1024, L1 progressive ────────────────────
# Effective latent: 1024//16 = 64 (H_lat = W_lat = 64)
H_LAT, W_LAT = 64, 64
N_STEPS = 30
A, BETA = 203.615097, 1.915461  # FLUX.1 VAE constants (placeholder)

# Approximate FlowMatch sigma schedule for 30 steps (mu shifts it but we use
# linspace as a conservative approximation for the theoretical curve)
sigmas = np.concatenate([np.linspace(1.0, 1.0 / N_STEPS, N_STEPS + 1)])
import torch

sigmas_t = torch.tensor(sigmas, dtype=torch.float32)

FULLRES_TS = N_STEPS * H_LAT * W_LAT  # 30 × 4096 = 122,880


def theoretical_speedup(delta):
    """Compute token-step speedup for a given delta."""
    stage_sigmas = compute_stage_transitions(delta, 1, A, BETA, H_LAT, W_LAT)
    trans = find_transition_steps(sigmas_t, stage_sigmas, N_STEPS)
    t = trans.get(2, N_STEPS)
    token_steps = t * (H_LAT // 2) * (W_LAT // 2) + (N_STEPS - t) * H_LAT * W_LAT
    return FULLRES_TS / token_steps if token_steps > 0 else 1.0


# ── Measured wall-clock speedups ────────────────────────────────────────────
MEASURED = {
    0.05: 1.77,
    0.10: 1.93,
}

# ── Theoretical curve ────────────────────────────────────────────────────────
deltas_theory = np.linspace(0.005, 0.60, 400)
speedups_theory = [theoretical_speedup(float(d)) for d in deltas_theory]

# Annotated points (theory)
ANNOTATED_DELTAS = [0.01, 0.05, 0.10, 0.20, 0.50]
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

# Annotated theory points (small)
for d, s in annotated:
    if d not in (0.10,):
        ax.scatter(
            d,
            s,
            s=60,
            color="#e94560",
            zorder=4,
            edgecolors="white",
            linewidth=1.0,
            alpha=0.7,
        )

# Measured wall-clock points (prominent)
for d, s in MEASURED.items():
    marker = "*" if d == 0.10 else "o"
    sz = 220 if d == 0.10 else 120
    color = "#f5a623" if d == 0.10 else "#4fc3f7"
    ax.scatter(
        d,
        s,
        s=sz,
        color=color,
        zorder=6,
        edgecolors="white",
        linewidth=1.8,
        marker=marker,
    )

# Annotate measured points
ax.annotate(
    f"  δ=0.05  {MEASURED[0.05]:.2f}× (measured)",
    xy=(0.05, MEASURED[0.05]),
    xytext=(0.07, MEASURED[0.05] - 0.10),
    fontsize=10,
    color="#4fc3f7",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=1.3),
)
ax.annotate(
    f"  δ=0.10  {MEASURED[0.10]:.2f}×\n  best quality/speed tradeoff",
    xy=(0.10, MEASURED[0.10]),
    xytext=(0.14, MEASURED[0.10] - 0.14),
    fontsize=10,
    color="#f5a623",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#f5a623", lw=1.5),
)

# Baseline
ax.axhline(y=1.0, color="#888", linewidth=1.0, linestyle="--", alpha=0.5)
ax.text(0.56, 1.02, "fullres baseline", color="#888", fontsize=9, ha="right")

# Legend patches
import matplotlib.patches as mpatches

legend_elements = [
    plt.Line2D(
        [0], [0], color="#e94560", linewidth=2.5, label="theory (token-step model)"
    ),
    plt.scatter(
        [],
        [],
        s=120,
        color="#4fc3f7",
        edgecolors="white",
        linewidth=1.5,
        label="measured (wall-clock)",
    ),
    plt.scatter(
        [],
        [],
        s=200,
        color="#f5a623",
        marker="*",
        edgecolors="white",
        linewidth=1.5,
        label="recommended δ",
    ),
]
ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=9,
    facecolor="#1a1a2e",
    edgecolor="#444",
    labelcolor="#ccc",
)

ax.set_xlabel("δ (noise-dominated tolerance)", fontsize=12, color="#ddd", labelpad=8)
ax.set_ylabel("Denoising speedup  (×)", fontsize=12, color="#ddd", labelpad=8)
ax.set_title(
    "FLUX.2 Progressive Generation — Speedup vs δ\n"
    "RTX A6000 · 30 steps · 1024×1024 · denoising loop only",
    fontsize=12,
    color="#eee",
    pad=12,
)

ax.set_xlim(0.0, 0.62)
ax.set_ylim(0.90, 2.30)
ax.set_xticks([0.01, 0.05, 0.10, 0.20, 0.50])
ax.set_xticklabels(["0.01", "0.05", "0.10", "0.20", "0.50"])

ax.tick_params(colors="#aaa", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#444")
ax.grid(True, color="#333", linewidth=0.7, alpha=0.6)

# Annotate theory points (small labels)
for d, s in annotated:
    if d in MEASURED:
        continue
    ax.annotate(
        f"{s:.2f}×", xy=(d, s), xytext=(d + 0.005, s + 0.05), fontsize=9, color="#ccc"
    )

ax.text(0.01, 2.22, "← increasing speedup", color="#888", fontsize=9, style="italic")
ax.text(
    0.01,
    2.15,
    "   (quality remains high up to δ≈0.10)",
    color="#888",
    fontsize=9,
    style="italic",
)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
