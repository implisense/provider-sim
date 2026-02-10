"""Tests for the palaestrAI environment wrapper."""

from __future__ import annotations

import pytest

from provider_sim.env.environment import ProviderEnvironment


class TestProviderEnvironment:
    def test_sensor_count(self, soja_doc):
        env = ProviderEnvironment(soja_doc)
        # 20 entities * 4 + 18 events + 1 tick = 99
        assert len(env.sensor_names) == 99

    def test_actuator_count(self, soja_doc):
        env = ProviderEnvironment(soja_doc)
        # 20 entities * 2 = 40
        assert len(env.actuator_names) == 40

    def test_reset(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        obs, rewards = env.reset()
        assert "entity.brazil_farms.supply" in obs
        assert obs["entity.brazil_farms.supply"] == 1.0
        assert "reward.attacker" in rewards
        assert "reward.defender" in rewards

    def test_step(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        env.reset()
        obs, rewards, done = env.step({"attacker.brazil_farms": 0.5})
        assert obs["sim.tick"] == 1.0
        assert not done

    def test_rewards_zero_sum(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        _, rewards = env.reset()
        total = rewards["reward.attacker"] + rewards["reward.defender"]
        assert total == pytest.approx(1.0)

    def test_done_after_max_ticks(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42, max_ticks=5)
        env.reset()
        done = False
        for _ in range(10):
            _, _, done = env.step({})
            if done:
                break
        assert done

    def test_load_from_path(self, soja_path):
        env = ProviderEnvironment(str(soja_path))
        assert len(env.sensor_names) == 99
