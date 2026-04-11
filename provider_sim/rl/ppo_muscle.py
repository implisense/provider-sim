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

# Muscle always runs on CPU: palaestrAI passes actuator/sensor data through
# inter-process boundaries (shared memory), and MPS tensors cannot cross them.
# Inference on this small network is negligible on CPU.
_DEVICE = torch.device("cpu")


class PPOMuscle(Muscle):
    """PPO Muscle for palaestrAI.

    Params (YAML):
        budget: float -- total action budget (0.8 for attacker, 0.4 for defender)
        n_obs:  int   -- observation size (default 99)
        n_act:  int   -- action size (default 20)
    """

    def __init__(self, *args, budget: float = 0.8, n_obs: int = 99, n_act: int = 20, checkpoint_path: str = "", **kwargs) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            pass  # standalone usage without palaestrAI orchestrator
        self._budget = float(budget)
        self._n_obs = int(n_obs)
        self._n_act = int(n_act)
        self._checkpoint_path = str(checkpoint_path)
        self._net: PPONet | None = None

    def setup(self) -> None:
        import os
        self._net = PPONet(n_obs=self._n_obs, n_act=self._n_act).to(_DEVICE)
        if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
            self._net.load_state_dict(
                torch.load(self._checkpoint_path, map_location=_DEVICE, weights_only=True)
            )
            print(f"[PPOMuscle] Loaded checkpoint: {self._checkpoint_path}")
        self._net.eval()  # inference mode

    def propose_actions(self, sensors, actuators_available):
        """Propose budget-constrained actions via PPO policy.

        Returns (palaestrAI 3.5.8 API):
            2-Tuple (actuator_setpoints, data_for_brain)
            data_for_brain: dict with obs, sampled_logits, log_prob, value for PPOBrain.
        """
        if self._net is None:
            self.setup()

        obs_list = []
        for s in sensors:
            val = s.value
            if hasattr(val, "__iter__"):
                obs_list.extend(float(v) for v in np.asarray(val).flatten())
            else:
                obs_list.append(float(val))
        obs_t = torch.tensor(obs_list, dtype=torch.float32).to(_DEVICE)

        with torch.no_grad():
            actions_t, sampled_logits, log_prob, value = self._net.sample_action(obs_t, self._budget)

        actions_np = np.array(actions_t.tolist(), dtype=np.float32)
        sampled_logits_np = np.array(sampled_logits.tolist(), dtype=np.float32)

        for actuator, act_val in zip(actuators_available, actions_np):
            # palaestrAI Box(shape=(1,)) requires a numpy array, not a plain float
            setpoint = np.array([float(np.clip(act_val, 0.0, 1.0))], dtype=np.float32)
            actuator(setpoint)

        data_for_brain = {
            "obs": np.array(obs_list, dtype=np.float32),
            "sampled_logits": sampled_logits_np,
            "log_prob": log_prob.item(),
            "value": value.item(),
        }
        return (actuators_available, data_for_brain)

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
