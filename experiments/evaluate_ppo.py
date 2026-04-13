"""PPO-Benchmark: trainierter Agent vs. Dummy (Zufallsaktionen).

Vergleicht vier Szenarien ueber N Seeds:
  1. PPO-Defender vs. Random-Attacker
  2. PPO-Attacker vs. Random-Defender
  3. PPO-Defender vs. PPO-Attacker  (Self-Play, Referenz)
  4. Random vs. Random              (Baseline)

Usage:
    python experiments/evaluate_ppo.py [attacker_ckpt] [defender_ckpt] [n_seeds]

Defaults:
    attacker_ckpt = experiments/checkpoints/attacker_palaestrai200.pt
    defender_ckpt = experiments/checkpoints/defender_palaestrai200.pt
    n_seeds = 10
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider_sim.env.environment import ProviderEnvironment
from provider_sim.rl.network import PPONet

_BASE = Path(__file__).parent.parent
_PDL = _BASE / "scenarios" / "s1-soja.pdl.yaml"
_ANALYSIS = _BASE / "analysis"
_ANALYSIS.mkdir(exist_ok=True)
_CKPT_DIR = _BASE / "experiments" / "checkpoints"

N_OBS = 99
N_ACT = 20
MAX_TICKS = 365

# Sensor-Keys in der Reihenfolge wie in soja_arl_ppo_vm.yaml (99 Eintraege)
# entity.<id>.<metric> fuer alle 20 Entities (ohne UID-Prefix)
_ENTITY_IDS = [
    "brazil_farms", "argentina_farms", "us_farms", "santos_port",
    "paranagua_port", "rotterdam_port", "hamburg_port", "eu_oil_mills",
    "feed_mills", "poultry_farms", "pig_farms", "dairy_farms",
    "food_retail", "consumers", "alternative_protein_sources",
    "soy_oil_market", "us_gulf_ports", "strategic_feed_reserves",
    "gas_supply", "fertilizer_supply",
]
_EVENT_IDS = [
    "brazil_drought", "soy_export_reduction", "port_congestion",
    "feed_price_spike", "livestock_pressure", "gas_price_spike",
    "ammonia_halt", "oil_mill_slowdown", "fertilizer_demand_spike",
    "consumer_price_increase", "alternative_protein_activation",
    "reserve_release", "port_route_shift", "argentina_supply_increase",
    "us_supply_activated", "soy_oil_shortage", "consumer_substitution",
    "feed_demand_reduction",
]
_SENSOR_KEYS: list[str] = (
    [f"entity.{eid}.{m}" for eid in _ENTITY_IDS for m in ("supply", "demand", "price", "health")]
    + [f"event.{ev}.active" for ev in _EVENT_IDS]
    + ["sim.tick"]
)


# ---------------------------------------------------------------------------
# Agenten-Wrapper
# ---------------------------------------------------------------------------

class PPOAgent:
    """Inference-only PPO-Agent (kein Training)."""

    def __init__(self, checkpoint: Path, budget: float) -> None:
        self._net = PPONet(n_obs=N_OBS, n_act=N_ACT)
        self._net.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self._net.train(False)
        self._budget = budget
        self._name = checkpoint.stem

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            actions_t, _, _, _ = self._net.sample_action(obs_t, self._budget)
        return np.clip(actions_t.numpy(), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"PPO({self._name})"


class RandomAgent:
    """Zufaellige Budget-normierte Aktionen (wie DummyMuscle)."""

    def __init__(
        self, budget: float, n_act: int = N_ACT,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._budget = budget
        self._n_act = n_act
        self._rng = rng or np.random.default_rng()
        self._name = "Random"

    def act(self, obs: np.ndarray) -> np.ndarray:
        raw = self._rng.random(self._n_act).astype(np.float32)
        s = raw.sum()
        if s > 0:
            raw = raw * (self._budget / s)
        return np.clip(raw, 0.0, 1.0)

    def __repr__(self) -> str:
        return "Random"


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _obs_from_dict(obs_dict: dict) -> np.ndarray:
    """Extrahiert die 99 Sensor-Werte in der Trainings-Reihenfolge."""
    return np.array([float(obs_dict.get(k, 0.0)) for k in _SENSOR_KEYS], dtype=np.float32)


def run_episode(attacker, defender, seed: int) -> tuple[float, float]:
    """Fuehrt eine Episode durch; gibt (att_reward, def_reward) zurueck."""
    env = ProviderEnvironment(pdl_source=_PDL, max_ticks=MAX_TICKS, seed=seed)
    obs_dict, _ = env.reset_dict()
    obs = _obs_from_dict(obs_dict)
    entity_ids = env.engine.state.entity_ids

    att_rewards, def_rewards = [], []
    done = False
    while not done:
        att_actions = attacker.act(obs)
        def_actions = defender.act(obs)

        action_dict: dict[str, float] = {}
        for i, eid in enumerate(entity_ids[:N_ACT]):
            action_dict[f"attacker.{eid}"] = float(att_actions[i])
            action_dict[f"defender.{eid}"] = float(def_actions[i])

        obs_dict, rewards, done = env.step_dict(action_dict)
        obs = _obs_from_dict(obs_dict)
        att_rewards.append(rewards.get("reward.attacker", 0.0))
        def_rewards.append(rewards.get("reward.defender", 0.0))

    return float(np.mean(att_rewards)), float(np.mean(def_rewards))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(att_agent, def_agent, n_seeds: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    att_list, def_list = [], []
    for seed in range(n_seeds):
        if isinstance(att_agent, RandomAgent):
            att_agent._rng = np.random.default_rng(seed * 1000)
        if isinstance(def_agent, RandomAgent):
            def_agent._rng = np.random.default_rng(seed * 1000 + 1)
        a, d = run_episode(att_agent, def_agent, seed=seed)
        att_list.append(a)
        def_list.append(d)
        print(f"  {label}  seed={seed}  att={a:.4f}  def={d:.4f}")
    return np.array(att_list), np.array(def_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    att_ckpt = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else _CKPT_DIR / "attacker_palaestrai200.pt"
    )
    def_ckpt = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else _CKPT_DIR / "defender_palaestrai200.pt"
    )
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    for ckpt in (att_ckpt, def_ckpt):
        if not ckpt.exists():
            print(f"Checkpoint nicht gefunden: {ckpt}")
            sys.exit(1)

    ppo_att = PPOAgent(att_ckpt, budget=0.8)
    ppo_def = PPOAgent(def_ckpt, budget=0.4)

    scenarios = [
        ("PPO-Def vs. Random-Att", RandomAgent(0.8), ppo_def),
        ("PPO-Att vs. Random-Def", ppo_att, RandomAgent(0.4)),
        ("PPO-Att vs. PPO-Def",   ppo_att, ppo_def),
        ("Random vs. Random",      RandomAgent(0.8), RandomAgent(0.4)),
    ]

    print(f"\nBenchmark: {n_seeds} Seeds, {MAX_TICKS} Ticks/Episode\n")
    results: dict[str, dict[str, np.ndarray]] = {}
    for label, att, dfd in scenarios:
        print(f"--- {label} ---")
        att_r, def_r = benchmark(att, dfd, n_seeds, label)
        results[label] = {"att": att_r, "def": def_r}
        print(f"  => att: {att_r.mean():.4f} +/- {att_r.std():.4f}  "
              f"def: {def_r.mean():.4f} +/- {def_r.std():.4f}\n")

    # --- Zusammenfassung ---
    print("=" * 60)
    print("  Zusammenfassung (Defender-Reward)")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label:<28s}  {r['def'].mean():.4f} +/- {r['def'].std():.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "PPO-Benchmark: Defender-Reward je Szenario\n"
        "S1 Soja-Futtermittel, 200 Trainings-Episoden",
        fontsize=12, fontweight="bold",
    )

    labels = list(results.keys())
    def_means = [results[l]["def"].mean() for l in labels]
    def_stds  = [results[l]["def"].std()  for l in labels]
    att_means = [results[l]["att"].mean() for l in labels]
    att_stds  = [results[l]["att"].std()  for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    bars_d = ax.bar(x - w/2, def_means, w, yerr=def_stds, capsize=5,
                    color="#2196F3", alpha=0.85, label="Defender-Reward")
    bars_a = ax.bar(x + w/2, att_means, w, yerr=att_stds, capsize=5,
                    color="#F44336", alpha=0.85, label="Attacker-Reward")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Gleichgewicht (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
    ax.set_ylabel("Mittlerer Episode-Reward")
    ax.set_ylim(0.0, 0.90)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars_d, def_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#1565C0",
                fontweight="bold")
    for bar, val in zip(bars_a, att_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#B71C1C",
                fontweight="bold")

    plt.tight_layout()
    out = _ANALYSIS / "ppo_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot gespeichert: {out}")


if __name__ == "__main__":
    main()
