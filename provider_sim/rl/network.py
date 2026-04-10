"""PPO shared network: MLP backbone + policy head + value head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# AutoRL: PPO Hyperparameters — fair game, agent may modify
# ---------------------------------------------------------------------------
LR: float = 3e-4          # Adam learning rate
GAMMA: float = 0.99       # discount factor
GAE_LAMBDA: float = 0.95  # GAE lambda
CLIP_EPS: float = 0.4     # PPO clip epsilon
PPO_EPOCHS: int = 5       # gradient update passes per episode
VALUE_COEF: float = 0.5   # critic loss weight
ENTROPY_COEF: float = 0.1   # entropy bonus weight
ATTACKER_BUDGET: float = 0.8  # total action budget for attacker
DEFENDER_BUDGET: float = 0.6  # total action budget for defender


class PPONet(nn.Module):
    """Shared MLP for PPO actor-critic with residual connection.

    Architecture: Linear(n_obs, h1) -> GELU -> Linear(h1, h2) -> GELU
                  + skip: Linear(n_obs, h2, bias=False) added to layer2 output
                  +-- policy_head: Linear(h2, n_act)
                  +-- value_head:  Linear(h2, 1)

    Residual + clip_eps=0.3 + ppo_epochs=5 + value double-skip = 0.670 best.
    """

    def __init__(self, n_obs: int = 99, n_act: int = 20) -> None:
        super().__init__()
        h1 = max(256, (n_obs * 2 // 64) * 64)
        h2 = h1 // 2
        self.layer1 = nn.Linear(n_obs, h1)
        self.act1 = nn.GELU()
        self.layer2 = nn.Linear(h1, h2)
        self.act2 = nn.GELU()
        self.skip = nn.Linear(n_obs, h2, bias=False)
        self.policy_head = nn.Linear(h2, n_act)
        self.log_std = nn.Parameter(torch.zeros(n_act))
        self.value_head = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns mu (batch, n_act), std (batch, n_act), value (batch,)."""
        skip_out = self.skip(x)
        hidden = self.act2(self.layer2(self.act1(self.layer1(x)))) + skip_out
        mu = self.policy_head(hidden)
        std = self.log_std.exp().expand_as(mu)
        value = self.value_head(hidden + skip_out).squeeze(-1)
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
