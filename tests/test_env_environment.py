"""Tests for the palaestrAI environment wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from provider_sim.env.environment import (
    ActuatorInformation,
    EnvironmentBaseline,
    EnvironmentState,
    ProviderEnvironment,
    RewardInformation,
    SensorInformation,
    _box_space,
)


class TestProviderEnvironment:
    def test_sensor_count(self, soja_doc):
        env = ProviderEnvironment(soja_doc)
        # 20 entities * 4 + 18 events + 1 tick = 99
        assert len(env.sensor_names) == 99

    def test_actuator_count(self, soja_doc):
        env = ProviderEnvironment(soja_doc)
        # 20 entities * 2 = 40
        assert len(env.actuator_names) == 40

    def test_load_from_path(self, soja_path):
        env = ProviderEnvironment(str(soja_path))
        assert len(env.sensor_names) == 99


class TestPalaestrAIProtocol:
    """Tests using the real palaestrAI API (start_environment / update)."""

    def test_start_environment_returns_baseline(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        baseline = env.start_environment()
        assert isinstance(baseline, EnvironmentBaseline)
        assert len(baseline.sensors_available) == 99
        assert len(baseline.actuators_available) == 40
        assert baseline.simtime.simtime_ticks == 0

    def test_sensor_information_types(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        baseline = env.start_environment()
        sensor = baseline.sensors_available[0]
        assert isinstance(sensor, SensorInformation)
        assert sensor.sensor_id == "entity.brazil_farms.supply"
        assert isinstance(sensor.sensor_value, np.ndarray)
        assert sensor.sensor_value.dtype == np.float32

    def test_actuator_information_types(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        baseline = env.start_environment()
        actuator = baseline.actuators_available[0]
        assert isinstance(actuator, ActuatorInformation)
        assert actuator.actuator_id == "attacker.brazil_farms"

    def test_discrete_sensor(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        baseline = env.start_environment()
        # First event sensor is at index 20*4 = 80
        event_sensor = baseline.sensors_available[80]
        assert "event." in event_sensor.sensor_id
        assert isinstance(event_sensor.sensor_value, int)

    def test_update_returns_state(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        env.start_environment()

        # Build an actuator action
        actuators = [
            ActuatorInformation(
                np.array([0.5], dtype=np.float32),
                _box_space(0, 1),
                actuator_id="attacker.brazil_farms",
            )
        ]
        state = env.update(actuators)
        assert isinstance(state, EnvironmentState)
        assert len(state.sensor_information) == 99
        assert len(state.rewards) == 2
        assert isinstance(state.done, bool)
        assert state.simtime.simtime_ticks == 1

    def test_reward_information_types(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        env.start_environment()
        state = env.update([])
        assert isinstance(state.rewards[0], RewardInformation)
        assert state.rewards[0].reward_id == "reward.attacker"
        assert state.rewards[1].reward_id == "reward.defender"

    def test_rewards_zero_sum(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        env.start_environment()
        state = env.update([])
        attacker_r = float(np.asarray(state.rewards[0].reward_value).item())
        defender_r = float(np.asarray(state.rewards[1].reward_value).item())
        assert attacker_r + defender_r == pytest.approx(1.0)

    def test_done_after_max_ticks(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42, max_ticks=5)
        env.start_environment()
        done = False
        for _ in range(10):
            state = env.update([])
            done = state.done
            if done:
                break
        assert done

    def test_simtime_increments(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        env.start_environment()
        for tick in range(1, 4):
            state = env.update([])
            assert state.simtime.simtime_ticks == tick


class TestDictInterface:
    """Tests for the standalone dict-based API (no orchestrator)."""

    def test_reset_dict(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        obs, rewards = env.reset_dict()
        assert "entity.brazil_farms.supply" in obs
        assert obs["entity.brazil_farms.supply"] == 1.0
        assert "reward.attacker" in rewards
        assert "reward.defender" in rewards

    def test_step_dict(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        env.reset_dict()
        obs, rewards, done = env.step_dict({"attacker.brazil_farms": 0.5})
        assert obs["sim.tick"] == 1.0
        assert not done

    def test_rewards_zero_sum_dict(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42)
        _, rewards = env.reset_dict()
        total = rewards["reward.attacker"] + rewards["reward.defender"]
        assert total == pytest.approx(1.0)

    def test_done_after_max_ticks_dict(self, soja_doc):
        env = ProviderEnvironment(soja_doc, seed=42, max_ticks=5)
        env.reset_dict()
        done = False
        for _ in range(10):
            _, _, done = env.step_dict({})
            if done:
                break
        assert done


class TestUidPrepending:
    """Verify that IDs work correctly with palaestrAI's _prepend_uid/_remove_uid."""

    def test_sensor_ids_are_bare(self, soja_doc):
        """Environment must return bare sensor IDs (no uid prefix)."""
        env = ProviderEnvironment(soja_doc, uid="test_uid")
        baseline = env.start_environment()
        for s in baseline.sensors_available:
            assert not s.sensor_id.startswith(
                "test_uid."
            ), f"Sensor ID already prefixed: {s.sensor_id}"

    def test_actuator_ids_are_bare(self, soja_doc):
        """Environment must return bare actuator IDs (no uid prefix)."""
        env = ProviderEnvironment(soja_doc, uid="test_uid")
        baseline = env.start_environment()
        for a in baseline.actuators_available:
            assert not a.actuator_id.startswith(
                "test_uid."
            ), f"Actuator ID already prefixed: {a.actuator_id}"

    def test_reward_ids_are_bare(self, soja_doc):
        """Rewards must use bare IDs (framework never prepends rewards)."""
        env = ProviderEnvironment(soja_doc, uid="test_uid")
        env.start_environment()
        state = env.update([])
        for r in state.rewards:
            assert not r.reward_id.startswith(
                "test_uid."
            ), f"Reward ID prefixed: {r.reward_id}"

    def test_update_with_bare_actuators(self, soja_doc):
        """Simulate framework round-trip: prepend uid, remove uid, then update()."""
        env = ProviderEnvironment(soja_doc, uid="test_uid")
        baseline = env.start_environment()
        # Simulate _prepend_uid: prefix all actuator IDs
        for a in baseline.actuators_available:
            a.actuator_id = f"test_uid.{a.actuator_id}"
        # Simulate _remove_uid: strip prefix before calling update
        for a in baseline.actuators_available:
            a.actuator_id = a.actuator_id.removeprefix("test_uid.")
        # update() should work with bare IDs
        state = env.update(baseline.actuators_available)
        assert len(state.sensor_information) == 99

    def test_config_generator_uid_matches(self, soja_path):
        """Config generator must use same uid as environment section."""
        from experiments.generate_config import build_config

        config = build_config(str(soja_path), uid="my_env")
        env_section = config["schedule"][0]["phase_train"]["environments"][0]
        env_uid = env_section["environment"]["uid"]
        sensor_ids = config["schedule"][0]["phase_train"]["agents"][0]["sensors"]
        actuator_ids = config["schedule"][0]["phase_train"]["agents"][0]["actuators"]
        assert env_uid == "my_env"
        assert all(s.startswith("my_env.") for s in sensor_ids)
        assert all(a.startswith("my_env.") for a in actuator_ids)
