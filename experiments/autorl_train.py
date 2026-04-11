"""Standalone PPO training harness for PROVIDER AutoRL. DO NOT MODIFY.

Trains attacker and defender PPO agents via self-play against the S1-Soja
supply-chain scenario for a fixed 5-minute wall-clock budget, then validates
on 3 fixed seeds.

Metric: val_reward (mean defender health over validation episodes) — higher = better.

The agent modifies provider_sim/rl/network.py ONLY. This file is frozen.

Usage:
    python experiments/autorl_train.py > run.log 2>&1
    grep "^val_reward:" run.log

Checkpoint strategy:
    .pt files are backed up as .bak at the start of each run.
    On discard: restore the .bak files alongside git reset --hard HEAD~1.
    On keep:    .bak files remain as previous-good-state reference.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from provider_sim.rl.network import (
    PPONet,
    LR,
    GAMMA,
    GAE_LAMBDA,
    CLIP_EPS,
    PPO_EPOCHS,
    VALUE_COEF,
    ENTROPY_COEF,
    ATTACKER_BUDGET,
    DEFENDER_BUDGET,
)
from provider_sim.env.environment import ProviderEnvironment

# ---------------------------------------------------------------------------
# Constants (frozen — do not edit)
# ---------------------------------------------------------------------------

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENARIO = os.path.normpath(
    os.path.join(_BASE, "..", "..", "06_Szenarien", "scenarios", "s1-soja.pdl.yaml")
)
_CKPT_DIR = os.path.join(_BASE, "experiments", "checkpoints")
_CKPT_ATK  = os.path.join(_CKPT_DIR, "autorl_attacker.pt")
_CKPT_DEF  = os.path.join(_CKPT_DIR, "autorl_defender.pt")
_CKPT_NORM = os.path.join(_CKPT_DIR, "autorl_obs_norm.npz")

# Backup paths — restored on discard
_CKPT_ATK_BAK  = _CKPT_ATK  + ".bak"
_CKPT_DEF_BAK  = _CKPT_DEF  + ".bak"
_CKPT_NORM_BAK = _CKPT_NORM + ".bak"

_MAX_TICKS       = 365
_TRAINING_SECONDS = 300       # 5-minute wall-clock training budget
_VAL_SEEDS        = (42, 43, 44)  # fixed for reproducible val_reward

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Observation normalizer (Welford online mean/std, persisted across runs)
# ---------------------------------------------------------------------------

class _ObsNorm:
    """Incremental mean/std normalizer using Welford's algorithm."""

    def __init__(self, n_obs: int) -> None:
        self.n    = 0
        self.mean = np.zeros(n_obs, dtype=np.float64)
        self.M2   = np.zeros(n_obs, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta      = x.astype(np.float64) - self.mean
        self.mean += delta / self.n
        self.M2   += delta * (x.astype(np.float64) - self.mean)

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1) + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x.astype(np.float64) - self.mean) / self.std, -10.0, 10.0
        ).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, n=np.array(self.n), mean=self.mean, M2=self.M2)

    @classmethod
    def load(cls, path: str, n_obs: int) -> "_ObsNorm":
        norm = cls(n_obs)
        if os.path.isfile(path):
            data     = np.load(path)
            norm.n    = int(data["n"])
            norm.mean = data["mean"]
            norm.M2   = data["M2"]
        return norm


# ---------------------------------------------------------------------------
# Trajectory buffer
# ---------------------------------------------------------------------------

class _Buf:
    def __init__(self) -> None:
        self.obs:      List[np.ndarray] = []
        self.logits:   List[np.ndarray] = []
        self.rewards:  List[float]      = []
        self.log_probs: List[float]     = []
        self.values:   List[float]      = []
        self.dones:    List[bool]       = []

    def push(self, obs: np.ndarray, logits: np.ndarray, reward: float,
             log_prob: float, value: float, done: bool) -> None:
        self.obs.append(obs)
        self.logits.append(np.asarray(logits, dtype=np.float32))
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.obs.clear(); self.logits.clear(); self.rewards.clear()
        self.log_probs.clear(); self.values.clear(); self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sample(net: PPONet, obs: np.ndarray, budget: float,
            greedy: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
    obs_t = torch.tensor(obs.tolist(), dtype=torch.float32, device=_DEVICE).unsqueeze(0)
    mu, std, value = net(obs_t)
    mu, std, value = mu.squeeze(0), std.squeeze(0), value.squeeze(0)
    sampled_logits = mu if greedy else Normal(mu, std).rsample()
    log_prob = Normal(mu, std).log_prob(sampled_logits).sum().item()
    actions = (F.softmax(sampled_logits, dim=-1) * budget).cpu().numpy()
    return actions, sampled_logits.cpu().numpy(), log_prob, value.item()


def _ppo_update(net: PPONet, opt: torch.optim.Optimizer,
                buf: _Buf) -> Tuple[float, float, float]:
    T = len(buf)
    values_ext = buf.values + [0.0]
    advantages  = np.zeros(T, dtype=np.float32)
    gae         = 0.0
    for t in reversed(range(T)):
        not_done   = 1.0 - float(buf.dones[t])
        delta      = buf.rewards[t] + GAMMA * values_ext[t + 1] * not_done - values_ext[t]
        gae        = delta + GAMMA * GAE_LAMBDA * not_done * gae
        advantages[t] = gae
    returns = advantages + np.array(buf.values, dtype=np.float32)

    obs_t    = torch.tensor(np.stack(buf.obs).tolist(),    dtype=torch.float32, device=_DEVICE)
    logits_t = torch.tensor(np.stack(buf.logits).tolist(), dtype=torch.float32, device=_DEVICE)
    old_lp_t = torch.tensor(buf.log_probs,                 dtype=torch.float32, device=_DEVICE)
    adv_t    = torch.tensor(advantages.tolist(),            dtype=torch.float32, device=_DEVICE)
    adv_t    = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t    = torch.tensor(returns.tolist(),               dtype=torch.float32, device=_DEVICE)

    al = cl = ent_val = 0.0
    net.train()
    for _ in range(PPO_EPOCHS):
        new_lp, new_val, entropy = net.recompute_logprob(obs_t, logits_t)
        ratio   = torch.exp(new_lp - old_lp_t)
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        al_t    = -torch.min(ratio * adv_t, clipped * adv_t).mean()
        cl_t    = F.mse_loss(new_val, ret_t)
        loss    = al_t + VALUE_COEF * cl_t - ENTROPY_COEF * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
        al, cl, ent_val = al_t.item(), cl_t.item(), entropy.item()
    net.train(False)  # inference mode
    buf.clear()
    return al, cl, ent_val


def _run_episode(
    env: ProviderEnvironment,
    atk_net: PPONet, def_net: PPONet,
    atk_names: List[str], def_names: List[str],
    obs_names: List[str],
    norm: _ObsNorm,
    seed: int, greedy: bool = False,
) -> Tuple[float, _Buf, _Buf, int]:
    obs_dict, _ = env.reset_dict()
    env.engine.rng = np.random.default_rng(seed)

    atk_buf, def_buf = _Buf(), _Buf()
    total_def_r = 0.0
    steps       = 0

    while True:
        raw_obs = np.array([obs_dict[k] for k in obs_names], dtype=np.float32)
        if not greedy:
            norm.update(raw_obs)
        obs = norm.normalize(raw_obs)

        atk_a, atk_lg, atk_lp, atk_v = _sample(atk_net, obs, ATTACKER_BUDGET, greedy)
        def_a, def_lg, def_lp, def_v = _sample(def_net, obs, DEFENDER_BUDGET, greedy)

        actions: Dict[str, float] = {}
        for i, n in enumerate(atk_names):
            actions[n] = float(atk_a[i])
        for i, n in enumerate(def_names):
            actions[n] = float(def_a[i])

        obs_dict, rewards, done = env.step_dict(actions)

        def_r = rewards["reward.defender"]
        atk_r = rewards["reward.attacker"]
        total_def_r += def_r
        steps += 1

        if not greedy:
            atk_buf.push(obs, atk_lg, atk_r, atk_lp, atk_v, done)
            def_buf.push(obs, def_lg, def_r, def_lp, def_v, done)

        if done:
            break

    return total_def_r / max(steps, 1), atk_buf, def_buf, steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    if not os.path.isfile(_SCENARIO):
        print(f"[autorl] ERROR: Scenario not found: {_SCENARIO}")
        sys.exit(1)

    os.makedirs(_CKPT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Backup checkpoints before this run.
    # On discard: restore .bak alongside git reset --hard HEAD~1.
    # ------------------------------------------------------------------
    backed_up = []
    for src, dst in [(_CKPT_ATK, _CKPT_ATK_BAK),
                     (_CKPT_DEF, _CKPT_DEF_BAK),
                     (_CKPT_NORM, _CKPT_NORM_BAK)]:
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            backed_up.append(os.path.basename(dst))
    if backed_up:
        print(f"[autorl] Backed up: {', '.join(backed_up)}")
    else:
        print("[autorl] No prior checkpoints — starting from scratch")

    # ------------------------------------------------------------------
    # Environment dimensions
    # ------------------------------------------------------------------
    env       = ProviderEnvironment(pdl_source=_SCENARIO, seed=42, max_ticks=_MAX_TICKS)
    obs_names = env.sensor_names
    atk_names = [n for n in env.actuator_names if n.startswith("attacker.")]
    def_names = [n for n in env.actuator_names if n.startswith("defender.")]
    n_obs, n_atk, n_def = len(obs_names), len(atk_names), len(def_names)

    print(f"[autorl] S1-Soja | obs={n_obs} atk_act={n_atk} def_act={n_def}")
    print(f"[autorl] device={_DEVICE} | budget={ATTACKER_BUDGET}/{DEFENDER_BUDGET} "
          f"| lr={LR} gamma={GAMMA} clip={CLIP_EPS} epochs={PPO_EPOCHS}")

    # ------------------------------------------------------------------
    # Networks + normalizer
    # ------------------------------------------------------------------
    norm    = _ObsNorm.load(_CKPT_NORM, n_obs)
    atk_net = PPONet(n_obs=n_obs, n_act=n_atk).to(_DEVICE)
    def_net = PPONet(n_obs=n_obs, n_act=n_def).to(_DEVICE)
    atk_opt = torch.optim.Adam(atk_net.parameters(), lr=LR)
    def_opt = torch.optim.Adam(def_net.parameters(), lr=LR)

    if os.path.isfile(_CKPT_ATK):
        atk_net.load_state_dict(torch.load(_CKPT_ATK, map_location=_DEVICE, weights_only=True))
        print("[autorl] Loaded attacker checkpoint")
    if os.path.isfile(_CKPT_DEF):
        def_net.load_state_dict(torch.load(_CKPT_DEF, map_location=_DEVICE, weights_only=True))
        print("[autorl] Loaded defender checkpoint")
    if norm.n > 0:
        print(f"[autorl] Loaded obs normalizer (n={norm.n} samples)")

    # ------------------------------------------------------------------
    # Training loop — fixed 5-minute wall-clock budget
    # ------------------------------------------------------------------
    ep            = 0
    train_rewards: List[float] = []
    t_train_start = time.time()

    while time.time() - t_train_start < _TRAINING_SECONDS:
        seed = 200 + ep
        def_r, atk_buf, def_buf, steps = _run_episode(
            env, atk_net, def_net, atk_names, def_names, obs_names, norm, seed=seed
        )
        al, cl, ent = _ppo_update(atk_net, atk_opt, atk_buf)
        _ppo_update(def_net, def_opt, def_buf)

        train_rewards.append(def_r)
        ep += 1
        elapsed = time.time() - t_train_start
        print(f"[autorl] ep={ep:3d} steps={steps} def_r={def_r:.4f} "
              f"actor={al:.4f} critic={cl:.4f} ent={ent:.4f} t={elapsed:.0f}s")

    training_seconds = time.time() - t_train_start

    torch.save({k: v.cpu() for k, v in atk_net.state_dict().items()}, _CKPT_ATK)
    torch.save({k: v.cpu() for k, v in def_net.state_dict().items()}, _CKPT_DEF)
    norm.save(_CKPT_NORM)

    # ------------------------------------------------------------------
    # Validation — greedy policy on fixed seeds
    # ------------------------------------------------------------------
    print(f"\n[autorl] Validation on {len(_VAL_SEEDS)} fixed seeds (greedy)...")
    val_rewards: List[float] = []
    for s in _VAL_SEEDS:
        r, _, _, st = _run_episode(
            env, atk_net, def_net, atk_names, def_names, obs_names, norm,
            seed=s, greedy=True
        )
        val_rewards.append(r)
        print(f"[autorl] val seed={s} def_r={r:.4f} steps={st}")

    val_reward   = float(np.mean(val_rewards))
    mean_train_r = float(np.mean(train_rewards[-10:])) if train_rewards else 0.0
    total_seconds = time.time() - t0

    peak_mem_mb = 0.0
    try:
        if _DEVICE.type == "mps":
            peak_mem_mb = torch.mps.current_allocated_memory() / 1024 / 1024
        elif _DEVICE.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass

    n_params = sum(p.numel() for p in atk_net.parameters())

    print(f"\n---")
    print(f"val_reward:       {val_reward:.6f}")
    print(f"mean_train_r:     {mean_train_r:.6f}")
    print(f"train_episodes:   {ep}")
    print(f"norm_samples:     {norm.n}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_memory_mb:   {peak_mem_mb:.1f}")
    print(f"n_params:         {n_params}")
    print(f"n_obs:            {n_obs}")
    print(f"n_atk_act:        {n_atk}")
    print(f"restore_cmd:      cp {_CKPT_ATK_BAK} {_CKPT_ATK} && cp {_CKPT_DEF_BAK} {_CKPT_DEF} && cp {_CKPT_NORM_BAK} {_CKPT_NORM}")


if __name__ == "__main__":
    main()
