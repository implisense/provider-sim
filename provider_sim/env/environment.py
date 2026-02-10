"""palaestrAI environment wrapper for the PROVIDER supply-chain simulation.

The palaestrai dependency is optional.  If it is not installed, this module
still imports successfully but ``ProviderEnvironment`` will inherit from a
minimal stub so that unit tests can exercise the logic without palaestrai.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from provider_sim.pdl.model import PdlDocument
from provider_sim.pdl.parser import load_pdl
from provider_sim.sim.engine import SimulationEngine

# ---------------------------------------------------------------------------
# Optional palaestrai import with stub fallback
# ---------------------------------------------------------------------------

try:
    from palaestrai.environment import Environment as _BaseEnv
except ImportError:

    class _BaseEnv:  # type: ignore[no-redef]
        """Minimal stub when palaestrai is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


# ---------------------------------------------------------------------------
# Sensor / Actuator descriptors
# ---------------------------------------------------------------------------


def _box(low: float, high: float) -> Dict[str, Any]:
    return {"type": "Box", "low": low, "high": high}


def _discrete(n: int) -> Dict[str, Any]:
    return {"type": "Discrete", "n": n}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ProviderEnvironment(_BaseEnv):
    """palaestrAI-compatible environment for PROVIDER supply-chain simulation.

    Sensors (per entity 4 + per event 1 + 1 global):
        entity.<id>.supply   — Box(0, 2)
        entity.<id>.demand   — Box(0, 3)
        entity.<id>.price    — Box(0, 10)
        entity.<id>.health   — Box(0, 1)
        event.<id>.active    — Discrete(2)
        sim.tick             — Box(0, max_ticks)

    Actuators (per entity 2):
        attacker.<entity_id> — Box(0, 1)
        defender.<entity_id> — Box(0, 1)

    Rewards (zero-sum):
        reward.attacker = mean(1 - health) over all entities
        reward.defender = mean(health) over all entities
    """

    def __init__(
        self,
        pdl_source: str | PdlDocument,
        seed: int | None = None,
        max_ticks: int = 365,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(pdl_source, PdlDocument):
            self.doc = pdl_source
        else:
            self.doc = load_pdl(pdl_source)

        self._seed = seed
        self._max_ticks = max_ticks
        self.engine = SimulationEngine(self.doc, seed=seed, max_ticks=max_ticks)

        # Build sensor / actuator descriptors
        self._sensor_keys: List[str] = []
        self._sensor_spaces: List[Dict[str, Any]] = []
        self._actuator_keys: List[str] = []
        self._actuator_spaces: List[Dict[str, Any]] = []

        for ent in self.doc.entities:
            for attr, space in [
                ("supply", _box(0, 2)),
                ("demand", _box(0, 3)),
                ("price", _box(0, 10)),
                ("health", _box(0, 1)),
            ]:
                self._sensor_keys.append(f"entity.{ent.id}.{attr}")
                self._sensor_spaces.append(space)

        for ev in self.doc.events:
            self._sensor_keys.append(f"event.{ev.id}.active")
            self._sensor_spaces.append(_discrete(2))

        self._sensor_keys.append("sim.tick")
        self._sensor_spaces.append(_box(0, max_ticks))

        for ent in self.doc.entities:
            self._actuator_keys.append(f"attacker.{ent.id}")
            self._actuator_spaces.append(_box(0, 1))
            self._actuator_keys.append(f"defender.{ent.id}")
            self._actuator_spaces.append(_box(0, 1))

    # ---- palaestrAI interface ----

    @property
    def sensor_names(self) -> List[str]:
        return list(self._sensor_keys)

    @property
    def sensor_spaces(self) -> List[Dict[str, Any]]:
        return list(self._sensor_spaces)

    @property
    def actuator_names(self) -> List[str]:
        return list(self._actuator_keys)

    @property
    def actuator_spaces(self) -> List[Dict[str, Any]]:
        return list(self._actuator_spaces)

    def reset(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Reset environment and return (observations, rewards)."""
        self.engine.reset()
        return self._observe(), self._rewards()

    def step(
        self, actions: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """Execute one tick.  Returns (observations, rewards, done)."""
        attacker_actions: Dict[str, float] = {}
        defender_actions: Dict[str, float] = {}

        for key, value in actions.items():
            if key.startswith("attacker."):
                eid = key[len("attacker."):]
                attacker_actions[eid] = float(value)
            elif key.startswith("defender."):
                eid = key[len("defender."):]
                defender_actions[eid] = float(value)

        self.engine.step(attacker_actions, defender_actions)

        done = self.engine.state.tick >= self._max_ticks
        return self._observe(), self._rewards(), done

    # ---- helpers ----

    def _observe(self) -> Dict[str, float]:
        s = self.engine.state
        obs: Dict[str, float] = {}
        for ent_id in s.entity_ids:
            es = s.entities[ent_id]
            obs[f"entity.{ent_id}.supply"] = es.supply
            obs[f"entity.{ent_id}.demand"] = es.demand
            obs[f"entity.{ent_id}.price"] = es.price
            obs[f"entity.{ent_id}.health"] = es.health

        for ev_id in s.event_ids:
            obs[f"event.{ev_id}.active"] = 1.0 if s.events[ev_id].active else 0.0

        obs["sim.tick"] = float(s.tick)
        return obs

    def _rewards(self) -> Dict[str, float]:
        healths = [
            self.engine.state.entities[eid].health
            for eid in self.engine.state.entity_ids
        ]
        if not healths:
            return {"reward.attacker": 0.0, "reward.defender": 0.0}
        mean_health = sum(healths) / len(healths)
        return {
            "reward.attacker": 1.0 - mean_health,
            "reward.defender": mean_health,
        }
