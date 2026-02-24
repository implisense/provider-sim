"""PPO Muscle: inference, budget-constrained action proposal."""
from __future__ import annotations

import numpy as np
import torch

from provider_sim.rl.network import PPONet

try:
    from palaestrai.agent.muscle import Muscle
except ImportError:
    class Muscle:  # type: ignore[no-redef]
        """Stub for environments without palaestrAI."""
        def __init__(self, *args, **kwargs):
            pass


class PPOMuscle(Muscle):
    """PPO Muscle for palaestrAI.

    Params (YAML):
        budget: float -- total action budget (0.8 for attacker, 0.4 for defender)
        n_obs:  int   -- observation size (default 99)
        n_act:  int   -- action size (default 20)
    """

    def __init__(self, *args, budget: float = 0.8, n_obs: int = 99, n_act: int = 20, **kwargs) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            pass  # standalone usage without palaestrAI orchestrator
        self._budget = float(budget)
        self._n_obs = int(n_obs)
        self._n_act = int(n_act)
        self._net: PPONet | None = None

    def setup(self) -> None:
        self._net = PPONet(n_obs=self._n_obs, n_act=self._n_act)
        self._net.eval()

    def propose_actions(self, sensors, actuators_available, is_terminal: bool = False):
        """Propose budget-constrained actions via PPO policy.

        Returns:
            4-Tuple (env_actions, last_actions, last_inputs, additional_data)
            additional_data contains sampled_logits, log_prob, value for Brain.
        """
        if self._net is None:
            self.setup()

        obs_list = []
        for s in sensors:
            val = s.sensor_value
            if hasattr(val, "__iter__"):
                obs_list.extend(float(v) for v in np.asarray(val).flatten())
            else:
                obs_list.append(float(val))
        obs = torch.tensor(obs_list, dtype=torch.float32)

        with torch.no_grad():
            actions_t, sampled_logits, log_prob, value = self._net.sample_action(obs, self._budget)

        actions_np = np.array(actions_t.tolist(), dtype=np.float32)
        sampled_logits_np = np.array(sampled_logits.tolist(), dtype=np.float32)

        for actuator, act_val in zip(actuators_available, actions_np):
            try:
                actuator(float(np.clip(act_val, 0.0, 1.0)))
            except Exception:
                actuator(0.0)

        additional_data = {
            "sampled_logits": sampled_logits_np,
            "log_prob": log_prob.item(),
            "value": value.item(),
        }
        return (actuators_available, actuators_available, obs_list, additional_data)

    def update(self, update) -> None:
        """Load new state_dict from Brain after PPO update."""
        if update is None or self._net is None:
            return
        try:
            self._net.load_state_dict(update)
        except Exception as exc:
            print(f"[PPOMuscle] update failed: {exc}")

    def prepare_model(self) -> None:
        if self._net is not None:
            self._net.eval()

    def __repr__(self) -> str:
        return f"PPOMuscle(budget={self._budget}, n_obs={self._n_obs}, n_act={self._n_act})"

    @property
    def parameters(self) -> dict:
        return {"budget": self._budget, "n_obs": self._n_obs, "n_act": self._n_act}
