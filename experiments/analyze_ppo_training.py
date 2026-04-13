"""PPO Training Reward-Kurve visualisieren.

Liest train_ppo_*.log Dateien, extrahiert Attacker/Defender-Rewards
pro Episode und plottet die Lernkurven.

Usage:
    python experiments/analyze_ppo_training.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_EXPERIMENTS = Path(__file__).parent
_ANALYSIS = _EXPERIMENTS.parent / "analysis"
_ANALYSIS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Log-Parsing
# ---------------------------------------------------------------------------
_REWARD_RE = re.compile(r"\[PPOBrain\] Episode 1\s+steps=(\d+)\s+reward=([\d.]+)")


def parse_log(path: Path) -> list[tuple[int, float]]:
    """Gibt Liste von (steps, reward) je Brain-Update zurück."""
    results = []
    with open(path) as f:
        for line in f:
            m = _REWARD_RE.search(line)
            if m:
                results.append((int(m.group(1)), float(m.group(2))))
    return results


def split_agents(entries: list[tuple[int, float]]) -> tuple[list[float], list[float]]:
    """Trennt interleaved Brain-Outputs in zwei Agenten-Serien.

    palaestrAI startet Brain-Subprozesse parallel; im Log erscheinen Attacker
    und Defender abwechselnd (oder geclustert). Da der Defender-Reward > 0.5
    und der Attacker-Reward < 0.5 ist, trennen wir nach diesem Kriterium.
    """
    attacker, defender = [], []
    for _, r in entries:
        if r > 0.5:
            defender.append(r)
        else:
            attacker.append(r)
    return attacker, defender


# ---------------------------------------------------------------------------
# Logs laden und zusammenführen
# ---------------------------------------------------------------------------
log_files = sorted(_EXPERIMENTS.glob("train_ppo_*.log"))
if not log_files:
    print("Keine train_ppo_*.log Dateien in experiments/ gefunden.")
    sys.exit(1)

all_entries: list[tuple[int, float]] = []
for lf in log_files:
    entries = parse_log(lf)
    print(f"  {lf.name}: {len(entries)} Brain-Updates")
    all_entries.extend(entries)

attacker_rewards, defender_rewards = split_agents(all_entries)

# Auf gleiche Länge kürzen (je nach Parallelität können minimal unterschiedlich)
n = min(len(attacker_rewards), len(defender_rewards))
att = np.array(attacker_rewards[:n])
dfd = np.array(defender_rewards[:n])
episodes = np.arange(1, n + 1)

print(f"\nGesamt: {n} Episoden")
print(f"Defender: mean={dfd.mean():.4f}  min={dfd.min():.4f}  max={dfd.max():.4f}")
print(f"Attacker: mean={att.mean():.4f}  min={att.min():.4f}  max={att.max():.4f}")


# ---------------------------------------------------------------------------
# Gleitender Mittelwert
# ---------------------------------------------------------------------------
def rolling_mean(x: np.ndarray, w: int = 10) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


WINDOW = 10
dfd_smooth = rolling_mean(dfd, WINDOW)
att_smooth = rolling_mean(att, WINDOW)
smooth_ep = episodes[WINDOW - 1:]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("PPO Training — S1 Soja-Futtermittel (200 Episoden)", fontsize=14, fontweight="bold")

# --- Oben: Rohe Rewards ---
ax1 = axes[0]
ax1.plot(episodes, dfd, alpha=0.3, color="#2196F3", linewidth=0.8, label="Defender (roh)")
ax1.plot(episodes, att, alpha=0.3, color="#F44336", linewidth=0.8, label="Attacker (roh)")
ax1.plot(smooth_ep, dfd_smooth, color="#1565C0", linewidth=2.0, label=f"Defender (Ø{WINDOW})")
ax1.plot(smooth_ep, att_smooth, color="#B71C1C", linewidth=2.0, label=f"Attacker (Ø{WINDOW})")
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Gleichgewicht")
ax1.set_ylabel("Episode Reward")
ax1.set_ylim(0.3, 0.75)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title("Reward-Verlauf", fontsize=11)

# --- Unten: Defender-Vorteil (dfd - att) ---
ax2 = axes[1]
advantage = dfd - att
adv_smooth = rolling_mean(advantage, WINDOW)
ax2.plot(episodes, advantage, alpha=0.3, color="#9C27B0", linewidth=0.8, label="Vorteil (roh)")
ax2.plot(smooth_ep, adv_smooth, color="#6A1B9A", linewidth=2.0, label=f"Vorteil (Ø{WINDOW})")
ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.fill_between(smooth_ep, adv_smooth, 0,
                 where=(adv_smooth >= 0), alpha=0.15, color="#2196F3", label="Defender führt")
ax2.fill_between(smooth_ep, adv_smooth, 0,
                 where=(adv_smooth < 0), alpha=0.15, color="#F44336", label="Attacker führt")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Defender − Attacker")
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title("Defender-Vorteil (dfd − att)", fontsize=11)

plt.tight_layout()
out = _ANALYSIS / "ppo_training_rewards.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot gespeichert: {out}")
