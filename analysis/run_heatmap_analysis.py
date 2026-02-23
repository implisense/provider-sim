"""Multi-episode supply-chain heatmap analysis.

Runs N episodes of the PROVIDER soy scenario with strategic
attacker/defender policies and produces a health heatmap.

Usage:
    cd palestrai_simulation
    python analysis/run_heatmap_analysis.py
    python analysis/run_heatmap_analysis.py --episodes 10 --ticks 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

_PDL_PATH = _BASE / "scenarios" / "s1-soja.pdl.yaml"
_OUTPUT_PATH = Path(__file__).resolve().parent / "heatmap_soja.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PROVIDER soy heatmap analysis")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--ticks",    type=int, default=365)
    p.add_argument("--attack",   type=float, default=0.8,
                   help="Max attack budget (0-1)")
    p.add_argument("--defend",   type=float, default=0.4,
                   help="Max defend budget (0-1)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   type=str, default=str(_OUTPUT_PATH))
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

if __name__ == "__main__":
    args = parse_args()
    print(f"Episodes: {args.episodes}, Ticks: {args.ticks}")
    print(f"Attack budget: {args.attack}, Defend budget: {args.defend}")
