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
        def internal_reward(self, memory_or_rewards: Any, **kwargs: Any) -> float:
            raise NotImplementedError

    class RewardInformation:  # type: ignore[no-redef]
        def __init__(self, value: Any = None, space: Any = None, uid: Any = None, **kwargs: Any) -> None:
            self.value = value
            self.space = space
            self.uid = uid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_reward(memory_or_rewards: Any, reward_id: str) -> float:
    """Extract a scalar reward value by ID.

    Supports two call conventions:
    - palaestrAI 3.5.8: ``memory`` is a ``Memory`` object; rewards are
      accessed via ``memory.tail(1).rewards`` (a pandas DataFrame keyed by uid).
    - Legacy / test path: ``memory_or_rewards`` is a plain
      ``List[RewardInformation]``; rewards are accessed by ``.uid``.
    """
    if isinstance(memory_or_rewards, list):
        for r in memory_or_rewards:
            rid = getattr(r, "uid", None) or getattr(r, "reward_id", None)
            if rid == reward_id:
                val = getattr(r, "value", None)
                if val is None:
                    val = getattr(r, "reward_value", None)
                return float(np.asarray(val).item())
        return 0.0

    # Memory object (palaestrAI 3.5.8)
    try:
        shard = memory_or_rewards.tail(1)
        if shard.rewards.empty or reward_id not in shard.rewards.columns:
            return 0.0
        val = shard.rewards[reward_id].iloc[-1]
        return float(np.asarray(val).item())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class AttackerObjective(_BaseObjective):
    """Objective for the attacker agent.

    Extracts the reward identified by ``reward_id``
    (default ``"reward.attacker"``) from the environment's reward list.
    """

    def __init__(self, reward_id: str = "reward.attacker", **kwargs: Any) -> None:
        super().__init__({"reward_id": reward_id, **kwargs})
        self._reward_id = reward_id

    def internal_reward(self, memory_or_rewards: Any, **kwargs: Any) -> float:
        return _extract_reward(memory_or_rewards, self._reward_id)


class DefenderObjective(_BaseObjective):
    """Objective for the defender agent.

    Extracts the reward identified by ``reward_id``
    (default ``"reward.defender"``) from the environment's reward list.
    """

    def __init__(self, reward_id: str = "reward.defender", **kwargs: Any) -> None:
        super().__init__({"reward_id": reward_id, **kwargs})
        self._reward_id = reward_id

    def internal_reward(self, memory_or_rewards: Any, **kwargs: Any) -> float:
        return _extract_reward(memory_or_rewards, self._reward_id)
