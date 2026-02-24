"""PPO shared network: MLP backbone + policy head + value head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PPONet(nn.Module):
    """Shared MLP for PPO actor-critic.

    Architecture: Linear(n_obs, 128) -> ReLU -> Linear(128, 64) -> ReLU
                  +-- policy_head: Linear(64, n_act)  [logit means]
                  +-- value_head:  Linear(64, 1)       [state value]

    Exploration via learnable log_std per action dimension.
    Budget constraint: Softmax(sampled_logits) x budget.
    """

    def __init__(self, n_obs: int = 99, n_act: int = 20) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, n_act)
        self.log_std = nn.Parameter(torch.zeros(n_act))
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns mu (batch, n_act), std (batch, n_act), value (batch,)."""
        hidden = self.shared(x)
        mu = self.policy_head(hidden)
        std = self.log_std.exp().expand_as(mu)
        value = self.value_head(hidden).squeeze(-1)
        return mu, std, value

    def sample_action(self, obs: torch.Tensor, budget: float) -> tuple:
        """Sample action from policy.

        Returns actions (n_act,), sampled_logits (n_act,), log_prob (scalar), value (scalar).
        """
        mu, std, value = self.forward(obs.unsqueeze(0))
        mu, std, value = mu.squeeze(0), std.squeeze(0), value.squeeze(0)
        dist = Normal(mu, std)
        sampled_logits = dist.rsample()
        log_prob = dist.log_prob(sampled_logits).sum()
        weights = F.softmax(sampled_logits, dim=-1)
        actions = weights * budget
        return actions, sampled_logits, log_prob, value

    def recompute_logprob(self, obs: torch.Tensor, sampled_logits: torch.Tensor) -> tuple:
        """Recompute log_prob and value for stored (obs, logits) pairs.

        Returns log_prob (batch,), value (batch,), entropy (scalar).
        """
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(sampled_logits).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_prob, value, entropy
