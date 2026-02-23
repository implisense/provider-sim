#!/usr/bin/env python
"""Generate palaestrAI experiment YAML from a PDL scenario.

Usage:
    python experiments/generate_config.py /path/to/scenario.pdl.yaml \\
        --output experiments/soja_arl_dummy.yaml \\
        --max-ticks 365 --episodes 1 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Ensure provider_sim is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from provider_sim.pdl.parser import load_pdl


def build_config(
    pdl_path: str,
    *,
    max_ticks: int = 365,
    episodes: int = 1,
    seed: int = 42,
    uid: str = "provider_env",
) -> Dict[str, Any]:
    """Build a palaestrAI experiment config dict from a PDL scenario."""
    doc = load_pdl(pdl_path)
    abs_pdl_path = str(Path(pdl_path).resolve())

    # Sensor IDs: entity sensors + event sensors + sim.tick
    sensor_ids: List[str] = []
    for ent in doc.entities:
        for suffix in ("supply", "demand", "price", "health"):
            sensor_ids.append(f"{uid}.entity.{ent.id}.{suffix}")
    for ev in doc.events:
        sensor_ids.append(f"{uid}.event.{ev.id}.active")
    sensor_ids.append(f"{uid}.sim.tick")

    # Actuator IDs: attacker + defender per entity
    attacker_actuators: List[str] = []
    defender_actuators: List[str] = []
    for ent in doc.entities:
        attacker_actuators.append(f"{uid}.attacker.{ent.id}")
        defender_actuators.append(f"{uid}.defender.{ent.id}")

    scenario_id = doc.scenario.id if hasattr(doc, "scenario") and doc.scenario else Path(pdl_path).stem

    config: Dict[str, Any] = {
        "uid": f"provider-{scenario_id}-arl-dummy",
        "seed": seed,
        "version": "3.4.1",
        "schedule": [
            {
                "phase_train": {
                    "environments": [
                        {
                            "environment": {
                                "name": "provider_sim.env.environment:ProviderEnvironment",
                                "uid": uid,
                                "params": {
                                    "pdl_source": abs_pdl_path,
                                    "max_ticks": max_ticks,
                                },
                            },
                            "reward": {
                                "name": "palaestrai.agent.dummy_objective:DummyObjective",
                                "params": {"params": {}},
                            },
                        }
                    ],
                    "agents": [
                        {
                            "name": "attacker",
                            "brain": {
                                "name": "palaestrai.agent.dummy_brain:DummyBrain",
                                "params": {},
                            },
                            "muscle": {
                                "name": "palaestrai.agent.dummy_muscle:DummyMuscle",
                                "params": {},
                            },
                            "objective": {
                                "name": "provider_sim.env.objectives:AttackerObjective",
                                "params": {"reward_id": "reward.attacker"},
                            },
                            "sensors": sensor_ids,
                            "actuators": attacker_actuators,
                        },
                        {
                            "name": "defender",
                            "brain": {
                                "name": "palaestrai.agent.dummy_brain:DummyBrain",
                                "params": {},
                            },
                            "muscle": {
                                "name": "palaestrai.agent.dummy_muscle:DummyMuscle",
                                "params": {},
                            },
                            "objective": {
                                "name": "provider_sim.env.objectives:DefenderObjective",
                                "params": {"reward_id": "reward.defender"},
                            },
                            "sensors": sensor_ids,
                            "actuators": defender_actuators,
                        },
                    ],
                    "simulation": {
                        "name": "palaestrai.simulation.vanilla_sim_controller:VanillaSimController",
                        "conditions": [
                            {
                                "name": "palaestrai.simulation.vanilla_simcontroller_termination_condition:VanillaSimControllerTerminationCondition",
                                "params": {},
                            }
                        ],
                    },
                    "phase_config": {
                        "mode": "train",
                        "worker": 1,
                        "episodes": episodes,
                    },
                }
            }
        ],
        "run_config": {
            "condition": {
                "name": "palaestrai.experiment.vanilla_rungovernor_termination_condition:VanillaRunGovernorTerminationCondition",
                "params": {},
            }
        },
    }

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate palaestrAI experiment config from PDL scenario"
    )
    parser.add_argument("pdl_path", help="Path to PDL scenario YAML")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output YAML path (default: stdout)",
    )
    parser.add_argument("--max-ticks", type=int, default=365)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uid", default="provider_env")

    args = parser.parse_args()
    config = build_config(
        args.pdl_path,
        max_ticks=args.max_ticks,
        episodes=args.episodes,
        seed=args.seed,
        uid=args.uid,
    )

    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if args.output:
        Path(args.output).write_text(yaml_str, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(yaml_str)


if __name__ == "__main__":
    main()
