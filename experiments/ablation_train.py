"""Ablation study harness for PROVIDER data-source experiment.

Runs N_SEEDS independent fresh training runs for a given combination
of BACI capacity constraints and ICIO flow weights.

Usage:
    python experiments/ablation_train.py [--baci] [--icio] [--n-seeds N] [--budget SECS]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from provider_sim.rl.network import (
    PPONet, LR, GAMMA, GAE_LAMBDA, CLIP_EPS, PPO_EPOCHS,
    VALUE_COEF, ENTROPY_COEF, ATTACKER_BUDGET, DEFENDER_BUDGET,
)
from provider_sim.env.environment import ProviderEnvironment

_BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENARIO = os.path.normpath(
    os.path.join(_BASE, "experiments", "configs", "s1-soja_icio.pdl.yaml")
)
_CKPT_DIR = os.path.join(_BASE, "experiments", "checkpoints")
_RESULTS  = os.path.join(_CKPT_DIR, "ablation_results.tsv")
_MAX_TICKS = 365
_VAL_SEEDS = (42, 43, 44, 45, 46)
_DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class _ObsNorm:
    def __init__(self, n: int) -> None:
        self.n    = 0
        self.mean = np.zeros(n, dtype=np.float64)
        self.M2   = np.zeros(n, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        d = x.astype(np.float64) - self.mean
        self.mean += d / self.n
        self.M2   += d * (x.astype(np.float64) - self.mean)

    @property
    def std(self) -> np.ndarray:
        return (np.sqrt(self.M2 / (self.n - 1) + 1e-8)
                if self.n >= 2 else np.ones_like(self.mean))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x.astype(np.float64) - self.mean) / self.std, -10.0, 10.0
        ).astype(np.float32)


class _Buf:
    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.logits: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(self, obs, logits, reward, log_prob, value, done) -> None:
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


@torch.no_grad()
def _sample(net: PPONet, obs: np.ndarray, budget: float,
            greedy: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
    obs_t = torch.tensor(obs.tolist(), dtype=torch.float32, device=_DEVICE).unsqueeze(0)
    mu, std, value = net(obs_t)
    mu, std, value = mu.squeeze(0), std.squeeze(0), value.squeeze(0)
    logits = mu if greedy else Normal(mu, std).rsample()
    log_prob = Normal(mu, std).log_prob(logits).sum().item()
    actions = (F.softmax(logits, dim=-1) * budget).cpu().numpy()
    return actions, logits.cpu().numpy(), log_prob, value.item()


def _ppo_update(net: PPONet, opt: torch.optim.Optimizer, buf: _Buf) -> None:
    T = len(buf)
    values_ext = buf.values + [0.0]
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nd    = 1.0 - float(buf.dones[t])
        delta = buf.rewards[t] + GAMMA * values_ext[t + 1] * nd - values_ext[t]
        gae   = delta + GAMMA * GAE_LAMBDA * nd * gae
        advantages[t] = gae
    returns = advantages + np.array(buf.values, dtype=np.float32)

    obs_t    = torch.tensor(np.stack(buf.obs).tolist(),    dtype=torch.float32, device=_DEVICE)
    logits_t = torch.tensor(np.stack(buf.logits).tolist(), dtype=torch.float32, device=_DEVICE)
    old_lp_t = torch.tensor(buf.log_probs,                 dtype=torch.float32, device=_DEVICE)
    adv_t    = torch.tensor(advantages.tolist(),            dtype=torch.float32, device=_DEVICE)
    adv_t    = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t    = torch.tensor(returns.tolist(),               dtype=torch.float32, device=_DEVICE)

    net.train()
    for _ in range(PPO_EPOCHS):
        new_lp, new_val, entropy = net.recompute_logprob(obs_t, logits_t)
        ratio   = torch.exp(new_lp - old_lp_t)
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        loss    = (-torch.min(ratio * adv_t, clipped * adv_t).mean()
                   + VALUE_COEF * F.mse_loss(new_val, ret_t)
                   - ENTROPY_COEF * entropy)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
    net.train(False)
    buf.clear()


def _run_episode(
    env: ProviderEnvironment,
    atk_net: PPONet, def_net: PPONet,
    atk_names: List[str], def_names: List[str],
    obs_names: List[str],
    norm: _ObsNorm,
    seed: int,
    greedy: bool = False,
) -> Tuple[float, _Buf, _Buf]:
    obs_dict, _ = env.reset_dict()
    env.engine.rng = np.random.default_rng(seed)
    atk_buf, def_buf = _Buf(), _Buf()
    total_def_r = 0.0; steps = 0

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
        total_def_r += def_r; steps += 1

        if not greedy:
            atk_buf.push(obs, atk_lg, atk_r, atk_lp, atk_v, done)
            def_buf.push(obs, def_lg, def_r, def_lp, def_v, done)

        if done:
            break

    return total_def_r / max(steps, 1), atk_buf, def_buf


def _run_single_seed(baci: bool, icio: bool, seed: int, budget_secs: int,
                     baci_scale: float = 1.0, icio_norm: str = "linear") -> float:
    """Train from scratch for budget_secs, return mean val_reward over VAL_SEEDS."""
    env = ProviderEnvironment(
        pdl_source=_SCENARIO, seed=seed, max_ticks=_MAX_TICKS,
        use_baci_capacity=baci, use_icio_weights=icio,
        baci_capacity_scale=baci_scale, icio_norm=icio_norm,
    )
    obs_names = env.sensor_names
    atk_names = [n for n in env.actuator_names if n.startswith("attacker.")]
    def_names  = [n for n in env.actuator_names if n.startswith("defender.")]
    n_obs = len(obs_names)

    norm    = _ObsNorm(n_obs)
    atk_net = PPONet(n_obs=n_obs, n_act=len(atk_names)).to(_DEVICE)
    def_net = PPONet(n_obs=n_obs, n_act=len(def_names)).to(_DEVICE)
    atk_opt = torch.optim.Adam(atk_net.parameters(), lr=LR)
    def_opt = torch.optim.Adam(def_net.parameters(), lr=LR)

    ep = 0; t0 = time.time()
    while time.time() - t0 < budget_secs:
        _, atk_buf, def_buf = _run_episode(
            env, atk_net, def_net, atk_names, def_names, obs_names, norm, seed=200 + ep
        )
        _ppo_update(atk_net, atk_opt, atk_buf)
        _ppo_update(def_net, def_opt, def_buf)
        ep += 1

    val_rewards = []
    for vs in _VAL_SEEDS:
        r, _, _ = _run_episode(
            env, atk_net, def_net, atk_names, def_names, obs_names, norm,
            seed=vs, greedy=True,
        )
        val_rewards.append(r)

    mean_val = float(np.mean(val_rewards))
    print(f"  seed={seed} episodes={ep} val_reward={mean_val:.6f}", flush=True)
    return mean_val


def _append_result(variant: str, rewards: List[float]) -> None:
    os.makedirs(_CKPT_DIR, exist_ok=True)
    write_header = not os.path.isfile(_RESULTS)
    with open(_RESULTS, "a") as f:
        if write_header:
            f.write("variant\tmean_val\tstd_val\tmin_val\tmax_val\tn_seeds\trewards\n")
        rewards_str = ",".join(f"{r:.6f}" for r in rewards)
        f.write(
            f"{variant}\t{np.mean(rewards):.6f}\t{np.std(rewards):.6f}\t"
            f"{np.min(rewards):.6f}\t{np.max(rewards):.6f}\t"
            f"{len(rewards)}\t{rewards_str}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PROVIDER Ablationsstudie Datenquellen")
    parser.add_argument("--baci",    action="store_true", help="BACI-Kapazitaetsgrenzen")
    parser.add_argument("--icio",    action="store_true", help="ICIO-Ghosh-Gewichte")
    parser.add_argument("--n-seeds", type=int, default=5,   metavar="N")
    parser.add_argument("--budget",  type=int, default=300, metavar="SECS")
    parser.add_argument("--baci-scale", type=float, default=1.0,
                        metavar="S", help="BACI-Cap-Multiplikator (default: 1.0)")
    parser.add_argument("--icio-norm", type=str, default="linear",
                        choices=["linear", "sqrt", "softmax", "uniform"],
                        help="ICIO-Normalisierungsschema (default: linear)")
    args = parser.parse_args()

    if not os.path.isfile(_SCENARIO):
        print(f"[ablation] ERROR: Szenario nicht gefunden: {_SCENARIO}")
        sys.exit(1)

    variant = (f"baci={'1' if args.baci else '0'}"
               f"_icio={'1' if args.icio else '0'}"
               f"_scale={args.baci_scale}"
               f"_norm={args.icio_norm}")
    print(f"[ablation] Variante : {variant}")
    print(f"[ablation] Szenario : {_SCENARIO}")
    print(f"[ablation] Seeds    : {args.n_seeds} x {args.budget}s Training")
    print(f"[ablation] Device   : {_DEVICE}")

    rewards = []
    for seed in range(args.n_seeds):
        print(f"\n[ablation] --- Seed {seed + 1}/{args.n_seeds} ---")
        r = _run_single_seed(
            baci=args.baci, icio=args.icio, seed=seed,
            budget_secs=args.budget,
            baci_scale=args.baci_scale, icio_norm=args.icio_norm,
        )
        rewards.append(r)

    print(f"\n[ablation] === Ergebnis {variant} ===")
    print(f"mean_val: {np.mean(rewards):.6f}")
    print(f"std_val:  {np.std(rewards):.6f}")
    print(f"min/max:  {np.min(rewards):.6f} / {np.max(rewards):.6f}")

    _append_result(variant, rewards)
    print(f"[ablation] Ergebnis gespeichert: {_RESULTS}")


if __name__ == "__main__":
    main()
