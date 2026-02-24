#!/usr/bin/env python3
"""Plot PPO training curves from palaestrAI log output.

Usage:
    palaestrai experiment-start experiments/soja_arl_ppo.yaml 2>&1 | tee experiments/ppo_training.log
    python experiments/plot_training_curves.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_LOG_DEFAULT = Path(__file__).resolve().parent / "ppo_training.log"
_OUTPUT_DEFAULT = Path(__file__).resolve().parent / "ppo_curves.png"

# Matches: [PPOBrain] Episode 3  steps=365  O reward=0.3142  total=114.6830
_PATTERN = re.compile(
    r"\[PPOBrain\] Episode\s+(\d+)\s+steps=\d+\s+O reward=([\d.]+)\s+total=([\d.]+)"
)


def parse_log(log_path: str) -> dict:
    all_entries: list = []
    with open(log_path) as f:
        for line in f:
            m = _PATTERN.search(line)
            if m:
                all_entries.append(float(m.group(2)))
    attacker = [r for i, r in enumerate(all_entries) if i % 2 == 0]
    defender = [r for i, r in enumerate(all_entries) if i % 2 == 1]
    return {"attacker": attacker, "defender": defender}


def smooth(values: list, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_curves(data: dict, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (agent, rewards) in zip(axes, data.items()):
        if not rewards:
            ax.text(0.5, 0.5, f"Keine Daten fuer {agent}", ha="center", va="center")
            continue
        eps = np.arange(1, len(rewards) + 1)
        ax.plot(eps, rewards, alpha=0.3, color="#2196F3", linewidth=1, label="Roh")
        if len(rewards) >= 5:
            sm = smooth(rewards)
            ax.plot(np.arange(3, 3 + len(sm)), sm, color="#2196F3", linewidth=2, label="Geglaettet (w=5)")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("O Reward pro Tick", fontsize=11)
        ax.set_title(f"{agent.capitalize()} - PPO Trainingskurve", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.suptitle(
        "PPO Adversarial RL - Soja S1  (Attack=0.8, Defend=0.4, 50 Ep x 365 Ticks)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik gespeichert: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot PPO training curves from log")
    p.add_argument("--log", default=str(_LOG_DEFAULT))
    p.add_argument("--output", default=str(_OUTPUT_DEFAULT))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.log).exists():
        print(f"Log nicht gefunden: {args.log}")
        print("Experiment starten: palaestrai experiment-start experiments/soja_arl_ppo.yaml 2>&1 | tee experiments/ppo_training.log")
        sys.exit(1)
    data = parse_log(args.log)
    print(f"Attacker: {len(data['attacker'])} Episoden, Defender: {len(data['defender'])} Episoden")
    plot_curves(data, args.output)
