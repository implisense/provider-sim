"""Tests for AttackerObjective and DefenderObjective."""

from __future__ import annotations

import numpy as np
import pytest

from provider_sim.env.objectives import AttackerObjective, DefenderObjective
from provider_sim.env.environment import RewardInformation, _box_space


_REWARD_SPACE = _box_space(0.0, 1.0)


def _make_rewards(attacker_val: float, defender_val: float):
    return [
        RewardInformation(
            value=np.array([attacker_val], dtype=np.float32),
            space=_REWARD_SPACE,
            uid="reward.attacker",
        ),
        RewardInformation(
            value=np.array([defender_val], dtype=np.float32),
            space=_REWARD_SPACE,
            uid="reward.defender",
        ),
    ]


class TestAttackerObjective:
    def test_filters_attacker_reward(self):
        obj = AttackerObjective(reward_id="reward.attacker")
        rewards = _make_rewards(0.7, 0.3)
        assert obj.internal_reward(rewards) == pytest.approx(0.7)

    def test_returns_zero_when_reward_missing(self):
        obj = AttackerObjective(reward_id="reward.attacker")
        rewards = [
            RewardInformation(
                value=np.array([0.5], dtype=np.float32),
                space=_REWARD_SPACE,
                uid="reward.other",
            )
        ]
        assert obj.internal_reward(rewards) == 0.0

    def test_default_reward_id(self):
        obj = AttackerObjective()
        assert obj._reward_id == "reward.attacker"

    def test_custom_reward_id(self):
        obj = AttackerObjective(reward_id="custom.attacker")
        rewards = [
            RewardInformation(
                value=np.array([0.9], dtype=np.float32),
                space=_REWARD_SPACE,
                uid="custom.attacker",
            )
        ]
        assert obj.internal_reward(rewards) == pytest.approx(0.9)

    def test_empty_rewards_list(self):
        obj = AttackerObjective()
        assert obj.internal_reward([]) == 0.0


class TestDefenderObjective:
    def test_filters_defender_reward(self):
        obj = DefenderObjective(reward_id="reward.defender")
        rewards = _make_rewards(0.7, 0.3)
        assert obj.internal_reward(rewards) == pytest.approx(0.3)

    def test_returns_zero_when_reward_missing(self):
        obj = DefenderObjective(reward_id="reward.defender")
        rewards = [
            RewardInformation(
                value=np.array([0.5], dtype=np.float32),
                space=_REWARD_SPACE,
                uid="reward.other",
            )
        ]
        assert obj.internal_reward(rewards) == 0.0

    def test_default_reward_id(self):
        obj = DefenderObjective()
        assert obj._reward_id == "reward.defender"

    def test_custom_reward_id(self):
        obj = DefenderObjective(reward_id="custom.defender")
        rewards = [
            RewardInformation(
                value=np.array([0.4], dtype=np.float32),
                space=_REWARD_SPACE,
                uid="custom.defender",
            )
        ]
        assert obj.internal_reward(rewards) == pytest.approx(0.4)

    def test_empty_rewards_list(self):
        obj = DefenderObjective()
        assert obj.internal_reward([]) == 0.0

    def test_zero_sum_property(self):
        """Both objectives should sum to 1.0 for a valid zero-sum pair."""
        attacker = AttackerObjective()
        defender = DefenderObjective()
        rewards = _make_rewards(0.6, 0.4)
        total = attacker.internal_reward(rewards) + defender.internal_reward(rewards)
        assert total == pytest.approx(1.0)
