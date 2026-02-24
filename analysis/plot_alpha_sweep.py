"""α-Sweep visualization: Ø Health vs. alpha for hybrid defender policy.

Alle Datenpunkte aus den Simulationsläufen (50 Episoden × 365 Ticks,
Attack=0.8, Defend=0.4, Seed=42, Soja-Szenario S1).

Usage:
    cd palestrai_simulation
    python analysis/plot_alpha_sweep.py
    python analysis/plot_alpha_sweep.py --output analysis/alpha_sweep.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_OUTPUT_DEFAULT = Path(__file__).resolve().parent / "alpha_sweep.png"

# Simulationsergebnisse (Attack=0.8, Defend=0.4, 50 Ep × 365 Ticks, Seed=42)
_ALPHA_VALUES = [0.00, 0.25, 0.50, 0.75, 1.00]
_HEALTH_VALUES = [0.6999, 0.6986, 0.6964, 0.6958, 0.6954]
_LABELS = ["reaktiv\n(α=0.0)", "hybrid\n(α=0.25)", "hybrid\n(α=0.50)", "hybrid\n(α=0.75)", "präventiv\n(α=1.0)"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot alpha sweep results")
    p.add_argument("--output", type=str, default=str(_OUTPUT_DEFAULT),
                   help="Output path for PNG (default: analysis/alpha_sweep.png)")
    return p.parse_args()


def plot_alpha_sweep(output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    alphas = np.array(_ALPHA_VALUES)
    healths = np.array(_HEALTH_VALUES)

    # Linie + Punkte
    ax.plot(alphas, healths, color="#2196F3", linewidth=2, zorder=2)
    ax.scatter(alphas, healths, color="#2196F3", s=80, zorder=3)

    # Endpunkte hervorheben
    ax.scatter([0.00], [0.6999], color="#4CAF50", s=120, zorder=4, label="reaktiv (α=0.0)")
    ax.scatter([1.00], [0.6954], color="#F44336", s=120, zorder=4, label="präventiv (α=1.0)")

    # Wertelabels an Punkten
    offsets = [(0.02, 0.00005), (0.02, 0.00005), (0.02, 0.00005), (0.02, 0.00005), (-0.08, 0.00005)]
    for i, (alpha, health) in enumerate(zip(alphas, healths)):
        dx, dy = offsets[i]
        ax.annotate(
            f"{health:.4f}",
            xy=(alpha, health),
            xytext=(alpha + dx, health + dy),
            fontsize=9,
            color="#333333",
        )

    # Achsen
    ax.set_xlabel("α  (0 = reaktiv, 1 = präventiv)", fontsize=12)
    ax.set_ylabel("Ø Health", fontsize=12)
    ax.set_title(
        "Hybrid-Defender α-Sweep — Ø Health vs. α\n"
        "Attack=0.8  Defend=0.4  50 Ep × 365 Ticks  Soja S1",
        fontsize=12,
    )
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])

    # Y-Achse eng skalieren für sichtbaren Effekt
    y_margin = 0.0005
    ax.set_ylim(min(healths) - y_margin, max(healths) + y_margin)

    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik gespeichert: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    print("Erstelle α-Sweep-Grafik …")
    plot_alpha_sweep(args.output)
