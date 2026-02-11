"""Objective classes for Adversarial Resilience Learning (ARL).

Provides AttackerObjective and DefenderObjective that filter the
environment's zero-sum rewards by ``reward_id``.

When palaestrai is not installed, a lightweight stub is used so the
module can be imported and tested without the full framework.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Conditional palaestrai imports with stub fallback
# ---------------------------------------------------------------------------

_HAS_PALAESTRAI = False

try:
    from palaestrai.agent.objective import Objective as _BaseObjective
    from palaestrai.agent import RewardInformation

    _HAS_PALAESTRAI = True
except ImportError:
    from abc import ABC, abstractmethod

    class _BaseObjective(ABC):  # type: ignore[no-redef]
        def __init__(self, params: dict) -> None:
            self.params = params

        @abstractmethod
        def internal_reward(self, rewards: List[Any]) -> float:
            raise NotImplementedError

    class RewardInformation:  # type: ignore[no-redef]
        def __init__(self, reward_value: Any, observation_space: Any, reward_id: Any = None) -> None:
            self.reward_value = reward_value
            self.observation_space = observation_space
            self.reward_id = reward_id


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class AttackerObjective(_BaseObjective):
    """Objective for the attacker agent.

    Extracts the reward identified by ``params["reward_id"]``
    (default ``"reward.attacker"``) from the environment's reward list.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._reward_id: str = params.get("reward_id", "reward.attacker")

    def internal_reward(self, rewards: List[RewardInformation]) -> float:
        for r in rewards:
            if r.reward_id == self._reward_id:
                return float(np.asarray(r.reward_value).item())
        return 0.0


class DefenderObjective(_BaseObjective):
    """Objective for the defender agent.

    Extracts the reward identified by ``params["reward_id"]``
    (default ``"reward.defender"``) from the environment's reward list.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self._reward_id: str = params.get("reward_id", "reward.defender")

    def internal_reward(self, rewards: List[RewardInformation]) -> float:
        for r in rewards:
            if r.reward_id == self._reward_id:
                return float(np.asarray(r.reward_value).item())
        return 0.0
