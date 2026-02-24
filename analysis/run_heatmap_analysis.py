"""Multi-episode supply-chain heatmap analysis.

Runs N episodes of the PROVIDER soy scenario with strategic
attacker/defender policies and produces a health heatmap.

Usage:
    cd palestrai_simulation
    python analysis/run_heatmap_analysis.py
    python analysis/run_heatmap_analysis.py --episodes 10 --ticks 100
    python analysis/run_heatmap_analysis.py --policy preventive
    python analysis/run_heatmap_analysis.py --policy hybrid --alpha 0.75
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # kein Display noetig
import matplotlib.pyplot as plt

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

_PDL_PATH = _BASE / "scenarios" / "s1-soja.pdl.yaml"
_OUTPUT_PATH = Path(__file__).resolve().parent / "heatmap_soja.png"


def _alpha_type(value: str) -> float:
    """argparse type for --alpha: must be in [0.0, 1.0]."""
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"--alpha must be in [0.0, 1.0], got {v}")
    return v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PROVIDER soy heatmap analysis")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--ticks",    type=int, default=365)
    p.add_argument("--attack",   type=float, default=0.8,
                   help="Max attack budget (0-1)")
    p.add_argument("--defend",   type=float, default=0.4,
                   help="Max defend budget (0-1)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   type=str, default="",
                   help="Output path for heatmap PNG (default: auto-named by policy)")
    p.add_argument(
        "--policy",
        choices=["reactive", "preventive", "hybrid"],
        default="reactive",
        help="Defender policy: reactive (default), preventive (vulnerability-weighted), or hybrid",
    )
    p.add_argument(
        "--alpha",
        type=_alpha_type,
        default=0.5,
        help="Blend factor for hybrid policy: 1.0=preventive, 0.0=reactive (default: 0.5)",
    )
    return p.parse_args()


def attacker_policy(
    entities: list,
    obs: dict,
    budget: float,
) -> dict:
    """Vulnerability-weighted attack: high-vulnerability entities get more pressure."""
    vulns = np.array([e.vulnerability for e in entities], dtype=np.float32)
    weights = vulns / vulns.sum() if vulns.sum() > 0 else np.ones(len(entities)) / len(entities)
    return {f"attacker.{e.id}": float(budget * w) for e, w in zip(entities, weights)}


def defender_policy(
    entities: list,
    obs: dict,
    budget: float,
) -> dict:
    """Reactive defense: entities with lowest health get strongest defense."""
    healths = np.array(
        [obs.get(f"entity.{e.id}.health", 1.0) for e in entities],
        dtype=np.float32,
    )
    inv = 1.0 - healths
    total = inv.sum()
    weights = inv / total if total > 0 else np.ones(len(entities)) / len(entities)
    return {f"defender.{e.id}": float(budget * w) for e, w in zip(entities, weights)}


def preventive_defender_policy(
    entities: list,
    obs: dict,
    budget: float,
) -> dict:
    """Vulnerability-weighted defense: protect structurally weak nodes pre-emptively."""
    vulns = np.array([e.vulnerability for e in entities], dtype=np.float32)
    weights = vulns / vulns.sum() if vulns.sum() > 0 else np.ones(len(entities), dtype=np.float32) / len(entities)
    return {f"defender.{e.id}": float(budget * w) for e, w in zip(entities, weights)}


def hybrid_defender_policy(
    entities: list,
    obs: dict,
    budget: float,
    alpha: float = 0.5,
) -> dict:
    """Blend of preventive (vulnerability) and reactive (inverse-health) defense.

    alpha=1.0 -> purely preventive (vulnerability-weighted)
    alpha=0.0 -> purely reactive (inverse-health-weighted)
    """
    vulns = np.array([e.vulnerability for e in entities], dtype=np.float32)
    prev_w = vulns / vulns.sum() if vulns.sum() > 0 else np.ones(len(entities), dtype=np.float32) / len(entities)

    healths = np.array([obs.get(f"entity.{e.id}.health", 1.0) for e in entities], dtype=np.float32)
    inv = 1.0 - healths
    react_w = inv / inv.sum() if inv.sum() > 0 else np.ones(len(entities), dtype=np.float32) / len(entities)

    weights = alpha * prev_w + (1.0 - alpha) * react_w
    return {f"defender.{e.id}": float(budget * w) for e, w in zip(entities, weights)}


def run_episodes(
    episodes: int,
    ticks: int,
    attack_budget: float,
    defend_budget: float,
    seed: int,
    defender_policy_name: str = "reactive",
    alpha: float = 0.5,
) -> tuple:
    """Run N episodes and collect health per entity per tick.

    Returns:
        health_data: shape (episodes, n_entities, ticks)
        entity_ids:  list of entity id strings in order
    """
    from provider_sim.env.environment import ProviderEnvironment

    env = ProviderEnvironment(_PDL_PATH, seed=seed, max_ticks=ticks)
    entity_ids = [e.id for e in env.doc.entities]
    n_entities = len(entity_ids)

    health_data = np.zeros((episodes, n_entities, ticks), dtype=np.float32)

    for ep in range(episodes):
        ep_env = ProviderEnvironment(_PDL_PATH, seed=seed + ep, max_ticks=ticks)
        obs, _ = ep_env.reset_dict()

        for tick in range(ticks):
            actions = {}
            actions.update(attacker_policy(ep_env.doc.entities, obs, attack_budget))
            if defender_policy_name == "preventive":
                actions.update(preventive_defender_policy(ep_env.doc.entities, obs, defend_budget))
            elif defender_policy_name == "hybrid":
                actions.update(hybrid_defender_policy(ep_env.doc.entities, obs, defend_budget, alpha=alpha))
            else:
                actions.update(defender_policy(ep_env.doc.entities, obs, defend_budget))

            obs, rewards, done = ep_env.step_dict(actions)

            for i, eid in enumerate(entity_ids):
                health_data[ep, i, tick] = obs.get(f"entity.{eid}.health", 1.0)

            if done:
                if tick + 1 < ticks:
                    health_data[ep, :, tick + 1:] = health_data[ep, :, tick:tick + 1]
                break

        pct = int((ep + 1) / episodes * 40)
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r  Episode {ep + 1:>3}/{episodes}  [{bar}]", end="", flush=True)

    print()
    return health_data, entity_ids


def plot_heatmap(
    health_data: np.ndarray,
    entity_ids: list,
    output_path: str,
    episodes: int,
    ticks: int,
    attack_budget: float,
    defend_budget: float,
    policy_label: str = "reactive",
) -> None:
    """Save health heatmap: entities (Y) x ticks (X), colour = mean health."""
    mean_health = health_data.mean(axis=0)

    sort_idx = np.argsort(mean_health.mean(axis=1))
    mean_health_sorted = mean_health[sort_idx]
    labels_sorted = [entity_ids[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        mean_health_sorted,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=9)
    ax.set_xlabel("Tick (Tag)", fontsize=11)
    ax.set_ylabel("Entity", fontsize=11)
    ax.set_title(
        f"Soja-Lieferkette — Ø Health über {episodes} Episoden\n"
        f"Attacker={attack_budget:.1f}  Defender={defend_budget:.1f}  "
        f"Policy={policy_label}  Ticks={ticks}",
        fontsize=12,
    )

    tick_step = max(1, ticks // 12)
    ax.set_xticks(range(0, ticks, tick_step))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Ø Health (0=kritisch, 1=stabil)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap gespeichert: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.alpha != 0.5 and args.policy != "hybrid":
        print(f"  Hinweis: --alpha={args.alpha} wird bei Policy '{args.policy}' ignoriert.")

    print(f"Starte {args.episodes} Episoden × {args.ticks} Ticks …")
    policy_info = f"hybrid(α={args.alpha:.2f})" if args.policy == "hybrid" else args.policy
    print(f"  Attack={args.attack}  Defend={args.defend}  Seed={args.seed}  Policy={policy_info}")

    health_data, entity_ids = run_episodes(
        args.episodes, args.ticks, args.attack, args.defend, args.seed,
        defender_policy_name=args.policy,
        alpha=args.alpha,
    )

    mean_all = health_data.mean()
    min_entity = entity_ids[health_data.mean(axis=(0, 2)).argmin()]
    print(f"  Ø Health gesamt     : {mean_all:.4f}")
    print(f"  Kritischste Entity  : {min_entity}")

    output = args.output
    if not output:
        if args.policy == "hybrid":
            stem = f"heatmap_soja_hybrid_{args.alpha:.2f}"
        elif args.policy == "preventive":
            stem = "heatmap_soja_preventive"
        else:
            stem = "heatmap_soja"
        output = str(Path(__file__).resolve().parent / f"{stem}.png")

    policy_label = f"hybrid(α={args.alpha:.2f})" if args.policy == "hybrid" else args.policy

    print("Erstelle Heatmap …")
    plot_heatmap(
        health_data, entity_ids, output,
        args.episodes, args.ticks, args.attack, args.defend,
        policy_label=policy_label,
    )
