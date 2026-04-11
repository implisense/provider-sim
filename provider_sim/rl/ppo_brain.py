"""PPO Brain: trajectory accumulation + PPO update (palaestrAI 3.5.8)."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from provider_sim.rl.network import PPONet

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

try:
    from palaestrai.agent.brain import Brain
except ImportError:
    class Brain:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self._memory = None


class PPOBrain(Brain):
    """PPO Brain for palaestrAI 3.5.8.

    Accumulates one full episode via thinking() callbacks, then runs PPO
    update on done=True and returns the new state_dict to the Muscle.

    YAML params: n_obs, n_act, reward_id, lr, gamma, gae_lambda,
                 clip_eps, ppo_epochs, value_coef, entropy_coef,
                 min_episode_steps, checkpoint_path
    """

    def __init__(
        self, *args,
        n_obs: int = 136, n_act: int = 20,
        reward_id: str = "reward.defender",
        lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_eps: float = 0.2, ppo_epochs: int = 4,
        value_coef: float = 0.5, entropy_coef: float = 0.01,
        min_episode_steps: int = 10,
        checkpoint_path: str = "",
        **kwargs,
    ) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            pass
        self._n_obs = int(n_obs)
        self._n_act = int(n_act)
        self._reward_id = str(reward_id)
        self._lr = float(lr)
        self._gamma = float(gamma)
        self._gae_lambda = float(gae_lambda)
        self._clip_eps = float(clip_eps)
        self._ppo_epochs = int(ppo_epochs)
        self._value_coef = float(value_coef)
        self._entropy_coef = float(entropy_coef)
        self._min_episode_steps = int(min_episode_steps)
        self._checkpoint_path = str(checkpoint_path)

        # Init on CPU: subprocess spawn requires pickling; MPS tensors cannot
        # cross process boundaries. _ensure_on_device() moves lazily on first update.
        self._net = PPONet(n_obs=self._n_obs, n_act=self._n_act)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        self._on_device = False

        import os
        if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
            self._net.load_state_dict(
                torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
            )
            print(f"[PPOBrain] Loaded checkpoint: {self._checkpoint_path}")

        self._obs_buf: list = []
        self._logits_buf: list = []
        self._reward_buf: list = []
        self._log_prob_buf: list = []
        self._value_buf: list = []
        self._done_buf: list = []
        self._episode = 0
        self._total_reward = 0.0

    # ------------------------------------------------------------------
    # palaestrAI Brain ABC (3.5.8)
    # ------------------------------------------------------------------

    def thinking(self, muscle_id: str, data_from_muscle: Any) -> Any:
        """Called after each environment step.

        data_from_muscle: dict sent by PPOMuscle with obs, sampled_logits,
        log_prob, value. done + reward are read from self._memory (populated
        by the framework before this call).

        Returns state_dict on episode end, None otherwise.
        """
        if data_from_muscle is None:
            return None

        # --- Read done + reward from framework-managed memory ----------
        done = False
        r = 0.0
        memory = getattr(self, "_memory", None)
        if memory is not None:
            try:
                shard = memory.tail(1)
                if shard.dones.size > 0:
                    done = bool(shard.dones[-1])
                rewards_df = shard.rewards
                if self._reward_id in rewards_df.columns:
                    ri = rewards_df[self._reward_id].iloc[-1]
                    r = float(np.asarray(ri.value).item())
            except Exception:
                pass

        # --- PPO-specific data from Muscle -----------------------------
        obs = np.asarray(
            data_from_muscle.get("obs", np.zeros(self._n_obs)), dtype=np.float32
        )
        sampled_logits = np.asarray(
            data_from_muscle.get("sampled_logits", np.zeros(self._n_act)),
            dtype=np.float32,
        )
        log_prob = float(data_from_muscle.get("log_prob", 0.0))
        value = float(data_from_muscle.get("value", 0.0))

        self._total_reward += r
        self._obs_buf.append(obs)
        self._logits_buf.append(sampled_logits)
        self._reward_buf.append(r)
        self._log_prob_buf.append(log_prob)
        self._value_buf.append(value)
        self._done_buf.append(done)

        if not done:
            return None

        steps = len(self._reward_buf)
        if steps < self._min_episode_steps:
            print(
                f"[PPOBrain] Skipped spurious episode "
                f"(steps={steps} < min={self._min_episode_steps})"
            )
            self._clear_buffers()
            return None

        self._episode += 1
        avg_r = self._total_reward / max(steps, 1)
        print(
            f"\n[PPOBrain] Episode {self._episode}  "
            f"steps={steps}  reward={avg_r:.4f}  total={self._total_reward:.4f}"
        )

        state_dict = self._ppo_update(next_value=0.0)

        if self._checkpoint_path:
            import os
            os.makedirs(os.path.dirname(self._checkpoint_path) or ".", exist_ok=True)
            torch.save(state_dict, self._checkpoint_path)
            print(f"[PPOBrain] Checkpoint: {self._checkpoint_path}")

        self._clear_buffers()
        # Return payload directly — framework wraps in MuscleUpdateResponse
        return state_dict

    def store_model(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._net.state_dict(), path)

    def load_model(self, path: str) -> None:
        self._net.load_state_dict(
            torch.load(path, map_location=_DEVICE, weights_only=True)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_buffers(self) -> None:
        self._obs_buf.clear()
        self._logits_buf.clear()
        self._reward_buf.clear()
        self._log_prob_buf.clear()
        self._value_buf.clear()
        self._done_buf.clear()
        self._total_reward = 0.0

    def _ensure_on_device(self) -> None:
        if not self._on_device:
            self._net = self._net.to(_DEVICE)
            self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
            self._on_device = True

    def _compute_gae(self, next_value: float):
        T = len(self._reward_buf)
        values = self._value_buf + [next_value]
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = (
                self._reward_buf[t]
                + self._gamma * values[t + 1] * (1.0 - float(self._done_buf[t]))
                - values[t]
            )
            gae = delta + self._gamma * self._gae_lambda * (1.0 - float(self._done_buf[t])) * gae
            advantages[t] = gae
        returns = advantages + np.array(self._value_buf, dtype=np.float32)
        adv_t = torch.tensor(advantages.tolist(), dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return adv_t, torch.tensor(returns.tolist(), dtype=torch.float32)

    def _ppo_update(self, next_value: float) -> dict:
        self._ensure_on_device()
        obs_t = torch.tensor(np.stack(self._obs_buf).tolist(), dtype=torch.float32).to(_DEVICE)
        logits_t = torch.tensor(np.stack(self._logits_buf).tolist(), dtype=torch.float32).to(_DEVICE)
        old_lp_t = torch.tensor(self._log_prob_buf, dtype=torch.float32).to(_DEVICE)
        advantages, returns = self._compute_gae(next_value)
        advantages = advantages.to(_DEVICE)
        returns = returns.to(_DEVICE)

        self._net.train()
        actor_loss = critic_loss = entropy = torch.tensor(0.0)
        for _ in range(self._ppo_epochs):
            new_lp, new_val, entropy = self._net.recompute_logprob(obs_t, logits_t)
            ratio = torch.exp(new_lp - old_lp_t)
            clipped = torch.clamp(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps)
            actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            critic_loss = F.mse_loss(new_val, returns)
            loss = actor_loss + self._value_coef * critic_loss - self._entropy_coef * entropy
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
            self._optimizer.step()

        self._net.eval()
        print(
            f"[PPOBrain] PPO update  "
            f"actor={actor_loss.item():.4f}  "
            f"critic={critic_loss.item():.4f}  "
            f"entropy={entropy.item():.4f}"
        )
        return {k: v.cpu() for k, v in self._net.state_dict().items()}
