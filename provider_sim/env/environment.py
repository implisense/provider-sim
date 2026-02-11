"""palaestrAI environment wrapper for the PROVIDER supply-chain simulation.

Implements the palaestrAI ``Environment`` ABC (start_environment / update)
using proper ``SensorInformation``, ``ActuatorInformation``,
``RewardInformation``, ``Box``/``Discrete`` spaces, ``EnvironmentBaseline``,
``EnvironmentState``, and ``SimTime``.

When palaestrai is not installed, a lightweight stub hierarchy is used so
that the environment can still be instantiated and tested without the
full framework.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from provider_sim.pdl.model import PdlDocument
from provider_sim.pdl.parser import load_pdl
from provider_sim.sim.engine import SimulationEngine

# ---------------------------------------------------------------------------
# Conditional palaestrai imports with stub fallback
# ---------------------------------------------------------------------------

_HAS_PALAESTRAI = False

try:
    from palaestrai.environment import Environment as _BaseEnv
    from palaestrai.environment import EnvironmentBaseline, EnvironmentState
    from palaestrai.agent import (
        SensorInformation,
        ActuatorInformation,
        RewardInformation,
    )
    from palaestrai.types import Box, Discrete, SimTime

    _HAS_PALAESTRAI = True
except ImportError:
    # --- Minimal stubs so the module works without palaestrai ---------------

    class _BaseEnv:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.is_terminal = False

    class SimTime:  # type: ignore[no-redef]
        def __init__(
            self,
            simtime_ticks: Optional[int] = None,
            simtime_timestamp: Any = None,
        ) -> None:
            self.simtime_ticks = simtime_ticks
            self.simtime_timestamp = simtime_timestamp

    class Box:  # type: ignore[no-redef]
        def __init__(
            self,
            low: float = 0.0,
            high: float = 1.0,
            shape: Optional[tuple] = None,
            dtype: Any = np.float32,
        ) -> None:
            self.low = np.full(shape or (), low, dtype=dtype)
            self.high = np.full(shape or (), high, dtype=dtype)
            self.shape = shape or ()
            self.dtype = np.dtype(dtype)

        def reshape_to_space(self, value: Any, **kw: Any) -> np.ndarray:
            return np.reshape(np.array(value), self.shape)

    class Discrete:  # type: ignore[no-redef]
        def __init__(self, n: int) -> None:
            self.n = n

        def reshape_to_space(self, value: Any, **kw: Any) -> np.ndarray:
            if np.isscalar(value) or np.ndim(value) == 0:
                return np.array([value])
            return np.array(value)

    class SensorInformation:  # type: ignore[no-redef]
        def __init__(self, sensor_value: Any, observation_space: Any, sensor_id: Any = None) -> None:
            self.sensor_value = sensor_value
            self.observation_space = observation_space
            self.sensor_id = sensor_id

        @property
        def id(self) -> Any:
            return self.sensor_id

        @id.setter
        def id(self, value: Any) -> None:
            self.sensor_id = value

    class ActuatorInformation:  # type: ignore[no-redef]
        def __init__(self, setpoint: Any, action_space: Any, actuator_id: Any = None) -> None:
            self._setpoint = setpoint
            self.action_space = action_space
            self.actuator_id = actuator_id

        @property
        def setpoint(self) -> Any:
            return self._setpoint

        @property
        def id(self) -> Any:
            return self.actuator_id

        @id.setter
        def id(self, value: Any) -> None:
            self.actuator_id = value

    class RewardInformation:  # type: ignore[no-redef]
        def __init__(self, reward_value: Any, observation_space: Any, reward_id: Any = None) -> None:
            self.reward_value = reward_value
            self.observation_space = observation_space
            self.reward_id = reward_id

        def __call__(self) -> Any:
            return self.reward_value

    class EnvironmentBaseline:  # type: ignore[no-redef]
        def __init__(
            self,
            sensors_available: List[Any],
            actuators_available: List[Any],
            simtime: Any = None,
        ) -> None:
            self.sensors_available = sensors_available
            self.actuators_available = actuators_available
            self.simtime = simtime or SimTime(simtime_ticks=1)

    class EnvironmentState:  # type: ignore[no-redef]
        def __init__(
            self,
            sensor_information: List[Any],
            rewards: List[Any],
            done: bool,
            simtime: Any = None,
        ) -> None:
            self.sensor_information = sensor_information
            self.rewards = rewards
            self.done = done
            self.simtime = simtime


# ---------------------------------------------------------------------------
# Space factories
# ---------------------------------------------------------------------------

def _box_space(low: float, high: float) -> Box:
    return Box(low=low, high=high, shape=(1,), dtype=np.float32)


def _discrete_space(n: int) -> Discrete:
    return Discrete(n)


_REWARD_SPACE = _box_space(0.0, 1.0)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ProviderEnvironment(_BaseEnv):
    """palaestrAI Environment for PROVIDER supply-chain simulation.

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
        pdl_source: Union[str, PdlDocument],
        seed: int = 0,
        max_ticks: int = 365,
        uid: str = "provider_env",
        broker_uri: str = "",
        **kwargs: Any,
    ) -> None:
        if _HAS_PALAESTRAI:
            super().__init__(uid=uid, broker_uri=broker_uri, seed=seed)
        else:
            super().__init__()

        if isinstance(pdl_source, PdlDocument):
            self.doc = pdl_source
        else:
            self.doc = load_pdl(pdl_source)

        self._seed = seed
        self._max_ticks = max_ticks
        self.engine = SimulationEngine(self.doc, seed=seed, max_ticks=max_ticks)

        # Pre-build space descriptors (id → Space)
        self._sensor_defs: List[Tuple[str, Any]] = []
        self._actuator_defs: List[Tuple[str, Any]] = []

        for ent in self.doc.entities:
            self._sensor_defs.append((f"entity.{ent.id}.supply", _box_space(0, 2)))
            self._sensor_defs.append((f"entity.{ent.id}.demand", _box_space(0, 3)))
            self._sensor_defs.append((f"entity.{ent.id}.price", _box_space(0, 10)))
            self._sensor_defs.append((f"entity.{ent.id}.health", _box_space(0, 1)))

        for ev in self.doc.events:
            self._sensor_defs.append((f"event.{ev.id}.active", _discrete_space(2)))

        self._sensor_defs.append(("sim.tick", _box_space(0, max_ticks)))

        for ent in self.doc.entities:
            self._actuator_defs.append((f"attacker.{ent.id}", _box_space(0, 1)))
            self._actuator_defs.append((f"defender.{ent.id}", _box_space(0, 1)))

    # ---- Convenience accessors (for tests and non-palaestrAI usage) -------

    @property
    def sensor_names(self) -> List[str]:
        return [sid for sid, _ in self._sensor_defs]

    @property
    def actuator_names(self) -> List[str]:
        return [aid for aid, _ in self._actuator_defs]

    # ---- palaestrAI Environment ABC implementation ------------------------

    def start_environment(self) -> EnvironmentBaseline:
        """Initialise (or re-initialise) the simulation and return baseline."""
        self.engine.reset()
        sensors = self._build_sensors()
        actuators = self._build_actuators()
        return EnvironmentBaseline(
            sensors_available=sensors,
            actuators_available=actuators,
            simtime=SimTime(simtime_ticks=0),
        )

    def update(
        self,
        actuators: List[ActuatorInformation],
    ) -> EnvironmentState:
        """Receive agent actions, step simulation, return new state."""
        attacker_actions: Dict[str, float] = {}
        defender_actions: Dict[str, float] = {}

        for act in actuators:
            aid = act.actuator_id if hasattr(act, "actuator_id") else act.id
            raw = act.setpoint if hasattr(act, "setpoint") else act._setpoint
            setpoint = float(np.asarray(raw).item())
            if aid.startswith("attacker."):
                attacker_actions[aid[len("attacker."):]] = setpoint
            elif aid.startswith("defender."):
                defender_actions[aid[len("defender."):]] = setpoint

        self.engine.step(attacker_actions, defender_actions)

        done = self.engine.state.tick >= self._max_ticks
        return EnvironmentState(
            sensor_information=self._build_sensors(),
            rewards=self._build_rewards(),
            done=done,
            simtime=SimTime(simtime_ticks=self.engine.state.tick),
        )

    # ---- Standalone step (non-palaestrAI usage) ---------------------------

    def step_dict(
        self, actions: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """Simple dict-based step for usage without palaestrAI orchestrator.

        Returns (observations, rewards, done) as plain dicts.
        """
        attacker_actions: Dict[str, float] = {}
        defender_actions: Dict[str, float] = {}

        for key, value in actions.items():
            if key.startswith("attacker."):
                attacker_actions[key[len("attacker."):]] = float(value)
            elif key.startswith("defender."):
                defender_actions[key[len("defender."):]] = float(value)

        self.engine.step(attacker_actions, defender_actions)
        done = self.engine.state.tick >= self._max_ticks
        return self._observe_dict(), self._rewards_dict(), done

    def reset_dict(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Simple dict-based reset. Returns (observations, rewards)."""
        self.engine.reset()
        return self._observe_dict(), self._rewards_dict()

    # ---- Internal helpers -------------------------------------------------

    def _build_sensors(self) -> List[SensorInformation]:
        s = self.engine.state
        values: Dict[str, float] = {}
        for eid in s.entity_ids:
            es = s.entities[eid]
            values[f"entity.{eid}.supply"] = es.supply
            values[f"entity.{eid}.demand"] = es.demand
            values[f"entity.{eid}.price"] = es.price
            values[f"entity.{eid}.health"] = es.health
        for ev_id in s.event_ids:
            values[f"event.{ev_id}.active"] = 1.0 if s.events[ev_id].active else 0.0
        values["sim.tick"] = float(s.tick)

        sensors: List[SensorInformation] = []
        for sid, space in self._sensor_defs:
            val = values[sid]
            if isinstance(space, Discrete):
                sensors.append(SensorInformation(int(val), space, sensor_id=sid))
            else:
                sensors.append(
                    SensorInformation(
                        np.array([val], dtype=np.float32), space, sensor_id=sid
                    )
                )
        return sensors

    def _build_actuators(self) -> List[ActuatorInformation]:
        return [
            ActuatorInformation(
                np.array([0.0], dtype=np.float32), space, actuator_id=aid
            )
            for aid, space in self._actuator_defs
        ]

    def _build_rewards(self) -> List[RewardInformation]:
        healths = [
            self.engine.state.entities[eid].health
            for eid in self.engine.state.entity_ids
        ]
        mean_health = sum(healths) / len(healths) if healths else 0.0
        return [
            RewardInformation(
                np.array([1.0 - mean_health], dtype=np.float32),
                _REWARD_SPACE,
                reward_id="reward.attacker",
            ),
            RewardInformation(
                np.array([mean_health], dtype=np.float32),
                _REWARD_SPACE,
                reward_id="reward.defender",
            ),
        ]

    def _observe_dict(self) -> Dict[str, float]:
        s = self.engine.state
        obs: Dict[str, float] = {}
        for eid in s.entity_ids:
            es = s.entities[eid]
            obs[f"entity.{eid}.supply"] = es.supply
            obs[f"entity.{eid}.demand"] = es.demand
            obs[f"entity.{eid}.price"] = es.price
            obs[f"entity.{eid}.health"] = es.health
        for ev_id in s.event_ids:
            obs[f"event.{ev_id}.active"] = 1.0 if s.events[ev_id].active else 0.0
        obs["sim.tick"] = float(s.tick)
        return obs

    def _rewards_dict(self) -> Dict[str, float]:
        healths = [
            self.engine.state.entities[eid].health
            for eid in self.engine.state.entity_ids
        ]
        mean_health = sum(healths) / len(healths) if healths else 0.0
        return {
            "reward.attacker": 1.0 - mean_health,
            "reward.defender": mean_health,
        }
