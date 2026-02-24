"""PPO Brain: trajectory accumulation + PPO update."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from provider_sim.rl.network import PPONet

try:
    from palaestrai.agent.brain import Brain
    from palaestrai.core.protocol.muscle_update_rsp import MuscleUpdateResponse
except ImportError:
    class Brain:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    class MuscleUpdateResponse:  # type: ignore[no-redef]
        def __init__(self, is_updated: bool, updates: Any):
            self.is_updated = is_updated
            self.updates = updates


class PPOBrain(Brain):
    """PPO Brain for palaestrAI.

    Accumulates one full episode, then runs PPO update on done=True.

    Params (YAML): n_obs, n_act, lr, gamma, gae_lambda, clip_eps, ppo_epochs, value_coef, entropy_coef
    """

    def __init__(
        self, *args,
        n_obs: int = 99, n_act: int = 20,
        lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_eps: float = 0.2, ppo_epochs: int = 4,
        value_coef: float = 0.5, entropy_coef: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._n_obs = int(n_obs)
        self._n_act = int(n_act)
        self._lr = float(lr)
        self._gamma = float(gamma)
        self._gae_lambda = float(gae_lambda)
        self._clip_eps = float(clip_eps)
        self._ppo_epochs = int(ppo_epochs)
        self._value_coef = float(value_coef)
        self._entropy_coef = float(entropy_coef)

        self._net = PPONet(n_obs=self._n_obs, n_act=self._n_act)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)

        self._obs_buf: list = []
        self._logits_buf: list = []
        self._reward_buf: list = []
        self._log_prob_buf: list = []
        self._value_buf: list = []
        self._done_buf: list = []
        self._episode = 0
        self._total_reward = 0.0

    # ------------------------------------------------------------------
    # palaestrAI Brain ABC
    # ------------------------------------------------------------------

    def thinking(self, muscle_id, readings, actions, reward, next_state, done, additional_data) -> MuscleUpdateResponse:
        """Called after each environment step.

        Appends transition to buffer. On done=True: runs PPO update and returns new state_dict.
        """
        # Extract scalar reward
        if isinstance(reward, (int, float)):
            r = float(reward)
        elif isinstance(reward, list) and len(reward) > 0:
            try:
                r = float(self.objective.internal_reward(reward))
            except Exception:
                r = float(np.mean([float(ri.reward_value) for ri in reward]))
        else:
            r = 0.0

        self._total_reward += r
        obs = self._extract_obs(readings)
        sampled_logits = additional_data.get("sampled_logits", np.zeros(self._n_act))
        log_prob = float(additional_data.get("log_prob", 0.0))
        value = float(additional_data.get("value", 0.0))

        self._obs_buf.append(obs)
        self._logits_buf.append(np.asarray(sampled_logits, dtype=np.float32))
        self._reward_buf.append(r)
        self._log_prob_buf.append(log_prob)
        self._value_buf.append(value)
        self._done_buf.append(bool(done))

        if not done:
            return MuscleUpdateResponse(False, None)

        self._episode += 1
        avg_r = self._total_reward / max(len(self._reward_buf), 1)
        print(
            f"\n[PPOBrain] Episode {self._episode}  "
            f"steps={len(self._reward_buf)}  "
            f"O reward={avg_r:.4f}  "
            f"total={self._total_reward:.4f}"
        )

        state_dict = self._ppo_update(next_value=0.0)

        self._obs_buf.clear()
        self._logits_buf.clear()
        self._reward_buf.clear()
        self._log_prob_buf.clear()
        self._value_buf.clear()
        self._done_buf.clear()
        self._total_reward = 0.0

        return MuscleUpdateResponse(True, state_dict)

    def store_model(self, path: str) -> None:
        torch.save(self._net.state_dict(), path)

    def load_model(self, path: str) -> None:
        self._net.load_state_dict(torch.load(path, weights_only=True))

    # ------------------------------------------------------------------
    # PPO helpers
    # ------------------------------------------------------------------

    def _extract_obs(self, sensors) -> np.ndarray:
        obs_list = []
        for s in sensors:
            val = s.sensor_value
            if hasattr(val, "__iter__"):
                obs_list.extend(float(v) for v in np.asarray(val).flatten())
            else:
                obs_list.append(float(val))
        return np.array(obs_list, dtype=np.float32)

    def _compute_gae(self, next_value: float) -> tuple:
        """Compute GAE advantages and discounted returns."""
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
        adv_t = torch.tensor(advantages)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return adv_t, torch.tensor(returns)

    def _ppo_update(self, next_value: float) -> dict:
        """Run PPO update on accumulated trajectory. Returns new state_dict."""
        obs_t = torch.tensor(np.stack(self._obs_buf), dtype=torch.float32)
        logits_t = torch.tensor(np.stack(self._logits_buf), dtype=torch.float32)
        old_log_prob_t = torch.tensor(self._log_prob_buf, dtype=torch.float32)
        advantages, returns = self._compute_gae(next_value)

        self._net.train()
        actor_loss = critic_loss = entropy = torch.tensor(0.0)
        for _ in range(self._ppo_epochs):
            new_log_prob, new_value, entropy = self._net.recompute_logprob(obs_t, logits_t)
            ratio = torch.exp(new_log_prob - old_log_prob_t)
            clipped = torch.clamp(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps)
            actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            critic_loss = F.mse_loss(new_value, returns)
            loss = actor_loss + self._value_coef * critic_loss - self._entropy_coef * entropy
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
            self._optimizer.step()

        self._net.eval()
        print(
            f"[PPOBrain] PPO update done  "
            f"actor={actor_loss.item():.4f}  "
            f"critic={critic_loss.item():.4f}  "
            f"entropy={entropy.item():.4f}"
        )
        return {k: v.cpu() for k, v in self._net.state_dict().items()}
