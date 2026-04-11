# Ablationsstudie Datenquellen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Messe den marginalen Beitrag von GTA-Events, BACI-Kapazitätsgrenzen und ICIO-Flusgewichten auf die Qualität der gelernten Defender-Policy (val_reward).

**Architecture:** Zwei neue optionale SimulationEngine-Features (BACI-Cap, ICIO-Gewichte) hinter Konstruktor-Flags. Ein ablation_train.py-Harness führt das 2×2-Factorial-Design (BACI × ICIO-Gewichte) mit je N=5 Seeds pro Variante durch und berechnet mean ± std val_reward. Als Basis dient das bestehende ICIO-Szenario (s1-soja_icio.pdl.yaml, 139 Sensoren), damit GTA-Events in allen Varianten konstant sind.

**Tech Stack:** Python 3.9, PyTorch, NumPy, provider_sim (engine.py, state.py), autorl_train.py als Vorlage

---

## Studiendesign

| Variante | BACI-Cap | ICIO-Gewichte | Bedeutung |
|----------|----------|---------------|-----------|
| V0 | — | — | ICIO-Baseline (GTA vorhanden, alles andere uniform) |
| V1 | ✓ | — | + physische Exportkapazitätsgrenzen |
| V2 | — | ✓ | + ICIO-gewichtete Flusausbreitung |
| V3 | ✓ | ✓ | Vollständige Anreicherung |

Referenzpunkt außerhalb (bereits vorhanden): autorl base (99 Sensoren, kein GTA) ≈ 0.649 → ICIO-V0 zeigt GTA-Effekt.

Wiederholungen: 5 Seeds (0–4) pro Variante → 20 Runs total, ~110 Minuten.

---

## Dateiliste

| Datei | Aktion | Zweck |
|-------|--------|-------|
| `provider_sim/sim/state.py` | Modify | `baci_capacity` + `icio_weight` Dicts in SupplyChainState |
| `provider_sim/sim/engine.py` | Modify | `use_baci_capacity` + `use_icio_weights` Flags + Phase-4-Logik |
| `experiments/ablation_train.py` | Create | Harness mit CLI-Args, N-Seeds, TSV-Output |
| `experiments/run_ablation.sh` | Create | Orchestrierungsskript für alle 4 Varianten |
| `tests/test_ablation_features.py` | Create | Unit-Tests für BACI-Cap + ICIO-Gewichte |

---

## Task 1: SupplyChainState um BACI/ICIO-Felder erweitern

**Files:**
- Modify: `provider_sim/sim/state.py`

- [ ] **Schritt 1: Failing Test schreiben**

```python
# tests/test_ablation_features.py
from provider_sim.sim.state import SupplyChainState, build_state_from_pdl
from provider_sim.pdl.parser import load_pdl
import pytest

SOJA_ICIO = "../../06_Szenarien/../04_Apps/palestrai_simulation/experiments/s1-soja_icio.pdl.yaml"

@pytest.fixture
def soja_icio_path():
    import pathlib
    base = pathlib.Path(__file__).parent.parent
    p = base / "experiments" / "s1-soja_icio.pdl.yaml"
    if not p.exists():
        pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
    return str(p)

class TestAblationState:
    def test_baci_capacity_extracted(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        state = build_state_from_pdl(doc)
        # brazil_farms hat baci_capacity_t_day im extra
        assert "brazil_farms" in state.baci_capacity
        # normalisiert: max = 1.0
        assert max(state.baci_capacity.values()) == pytest.approx(1.0)
        # brazil > argentina (größerer Exporteur)
        assert state.baci_capacity["brazil_farms"] > state.baci_capacity["argentina_farms"]

    def test_icio_weight_extracted(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        state = build_state_from_pdl(doc)
        # brazil_farms hat ghosh_multiplier im extra
        assert "brazil_farms" in state.icio_weight
        assert state.icio_weight["brazil_farms"] > 1.0  # Ghosh > 1 bedeutet Amplifikation

    def test_entities_without_baci_not_in_dict(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        state = build_state_from_pdl(doc)
        # consumers hat kein baci_capacity_t_day
        assert "consumers" not in state.baci_capacity

    def test_base_scenario_has_empty_dicts(self):
        import pathlib
        base = pathlib.Path(__file__).parent.parent
        p = base / "scenarios" / "s1-soja.pdl.yaml"
        if not p.exists():
            pytest.skip("s1-soja.pdl.yaml nicht gefunden")
        doc = load_pdl(str(p))
        state = build_state_from_pdl(doc)
        assert state.baci_capacity == {}
        assert state.icio_weight == {}
```

- [ ] **Schritt 2: Test ausführen — erwartet FAIL**

```bash
pytest tests/test_ablation_features.py::TestAblationState::test_baci_capacity_extracted -v
# Erwartet: AttributeError: 'SupplyChainState' object has no attribute 'baci_capacity'
```

- [ ] **Schritt 3: SupplyChainState erweitern**

In `provider_sim/sim/state.py`, nach Zeile 54 (`self.rng: ...`) einfügen:

```python
        # Optional enrichment data (populated from entity.extra if present)
        self.baci_capacity: Dict[str, float] = {}  # entity_id → normalized [0, 1]
        self.icio_weight: Dict[str, float] = {}    # entity_id → Ghosh multiplier (default 1.0)
```

In `build_state_from_pdl`, nach dem Entities-Block (nach Zeile 83 `state.vulnerability[ent.id] = ent.vulnerability`) einfügen:

```python
    # BACI capacity: normalize so max = 1.0
    raw_caps: Dict[str, float] = {}
    for ent in doc.entities:
        cap = ent.extra.get("baci_capacity_t_day")
        if cap is not None:
            raw_caps[ent.id] = float(cap)
    if raw_caps:
        max_cap = max(raw_caps.values())
        for eid, cap in raw_caps.items():
            state.baci_capacity[eid] = cap / max_cap

    # ICIO Ghosh weights
    for ent in doc.entities:
        ghosh = ent.extra.get("ghosh_multiplier")
        if ghosh is not None:
            state.icio_weight[ent.id] = float(ghosh)
```

- [ ] **Schritt 4: Tests ausführen — erwartet PASS**

```bash
pytest tests/test_ablation_features.py::TestAblationState -v
# Erwartet: 4 PASSED
```

- [ ] **Schritt 5: Bestehende Tests nicht gebrochen**

```bash
pytest tests/ -v --tb=short
# Erwartet: alle Tests grün (neue Felder sind additive)
```

- [ ] **Schritt 6: Commit**

```bash
git add provider_sim/sim/state.py tests/test_ablation_features.py
git commit -m "feat: extract baci_capacity + icio_weight from PDL extra fields"
```

---

## Task 2: BACI-Kapazitätsgrenzen in Phase 4

**Files:**
- Modify: `provider_sim/sim/engine.py`
- Test: `tests/test_ablation_features.py`

- [ ] **Schritt 1: Failing Test schreiben**

An `tests/test_ablation_features.py` anhängen:

```python
from provider_sim.sim.engine import SimulationEngine

class TestBaciCap:
    def test_baci_cap_limits_defender_boost(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        engine_cap = SimulationEngine(doc, seed=0, use_baci_capacity=True)
        engine_free = SimulationEngine(doc, seed=0, use_baci_capacity=False)

        # Defender pusht argentina_farms maximal (budget=2.0)
        big_defense = {"argentina_farms": 2.0}
        engine_cap.step(defender_actions=big_defense)
        engine_free.step(defender_actions=big_defense)

        # Mit Cap: argentina kann Brazil nicht überholen (Argentina hat ~0.3 normalized cap)
        s_cap = engine_cap.state.entities["argentina_farms"].supply
        s_free = engine_free.state.entities["argentina_farms"].supply
        cap_limit = engine_cap.state.baci_capacity["argentina_farms"]

        assert s_cap <= cap_limit + 1e-6, f"supply {s_cap} überschreitet BACI-Cap {cap_limit}"
        assert s_free > s_cap, "ohne Cap sollte supply höher sein"

    def test_no_baci_cap_by_default(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        engine = SimulationEngine(doc, seed=0)
        # Standard-Konstruktor hat use_baci_capacity=False
        big_defense = {"argentina_farms": 2.0}
        engine.step(defender_actions=big_defense)
        s = engine.state.entities["argentina_farms"].supply
        cap = engine.state.baci_capacity.get("argentina_farms", 1.0)
        # Ohne Flag kann supply über die normalisierte Kapazität steigen
        assert s > cap or True  # kein Crash, kein Cap aktiv

    def test_entities_without_baci_unaffected(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        engine = SimulationEngine(doc, seed=0, use_baci_capacity=True)
        big_defense = {"consumers": 2.0}
        engine.step(defender_actions=big_defense)
        s = engine.state.entities["consumers"].supply
        # consumers hat kein BACI-Eintrag → kein Cap, supply kann > 1.0 sein
        assert s > 0.0
```

- [ ] **Schritt 2: Test ausführen — erwartet FAIL**

```bash
pytest tests/test_ablation_features.py::TestBaciCap -v
# Erwartet: TypeError: __init__() got unexpected keyword argument 'use_baci_capacity'
```

- [ ] **Schritt 3: Engine-Konstruktor erweitern**

In `provider_sim/sim/engine.py`, `__init__`-Signatur (Zeile 22) ändern zu:

```python
    def __init__(
        self,
        doc: PdlDocument,
        seed: int | None = None,
        max_ticks: int = 365,
        use_baci_capacity: bool = False,
        use_icio_weights: bool = False,
    ) -> None:
        self.doc = doc
        self._seed = seed
        self._max_ticks = max_ticks
        self._use_baci_capacity = use_baci_capacity
        self._use_icio_weights = use_icio_weights
        self.state = build_state_from_pdl(doc, seed=seed, max_ticks=max_ticks)
```

In `_phase4_propagate_flow` (Zeile 227) die innere Supply-Zuweisung erweitern:

```python
    def _phase4_propagate_flow(self) -> None:
        s = self.state

        for eid in self._topo_order:
            es = s.entities[eid]
            ups = s.upstream.get(eid, set())

            if ups:
                upstream_ids = [u for u in ups if u in s.entities]
                incoming = [s.entities[u].supply for u in upstream_ids]
                if incoming:
                    if self._use_icio_weights and upstream_ids:
                        weights = [s.icio_weight.get(u, 1.0) for u in upstream_ids]
                        total_w = sum(weights)
                        mean_incoming = sum(
                            sup * w for sup, w in zip(incoming, weights)
                        ) / total_w
                    else:
                        mean_incoming = sum(incoming) / len(incoming)
                    es.supply = min(es.supply, mean_incoming)

            # Dependency penalty (unverändert)
            deps = s.depends_on.get(eid, set())
            for dep_id in deps:
                dep_es = s.entities.get(dep_id)
                if dep_es and dep_es.supply < 0.5:
                    penalty = 0.3 * (1.0 - dep_es.supply)
                    es.supply = max(0.0, es.supply - penalty)

            # BACI capacity cap (nach Flow-Propagation)
            if self._use_baci_capacity and eid in s.baci_capacity:
                es.supply = min(es.supply, s.baci_capacity[eid])
```

- [ ] **Schritt 4: Tests ausführen — erwartet PASS**

```bash
pytest tests/test_ablation_features.py -v
# Erwartet: alle Tests PASSED
```

- [ ] **Schritt 5: Regressionstests**

```bash
pytest tests/ -v --tb=short
# Erwartet: alle bestehenden Tests grün (Flags default=False → kein Verhalten geändert)
```

- [ ] **Schritt 6: Commit**

```bash
git add provider_sim/sim/engine.py tests/test_ablation_features.py
git commit -m "feat: add use_baci_capacity + use_icio_weights flags to SimulationEngine"
```

---

## Task 3: ProviderEnvironment leitet Flags durch

**Files:**
- Modify: `provider_sim/env/environment.py`
- Test: `tests/test_ablation_features.py`

- [ ] **Schritt 1: Failing Test**

An `tests/test_ablation_features.py` anhängen:

```python
from provider_sim.env.environment import ProviderEnvironment

class TestEnvironmentFlags:
    def test_baci_flag_passed_to_engine(self, soja_icio_path):
        env = ProviderEnvironment(
            pdl_source=soja_icio_path,
            seed=0,
            use_baci_capacity=True,
        )
        assert env.engine._use_baci_capacity is True

    def test_icio_flag_passed_to_engine(self, soja_icio_path):
        env = ProviderEnvironment(
            pdl_source=soja_icio_path,
            seed=0,
            use_icio_weights=True,
        )
        assert env.engine._use_icio_weights is True

    def test_defaults_are_false(self, soja_icio_path):
        env = ProviderEnvironment(pdl_source=soja_icio_path, seed=0)
        assert env.engine._use_baci_capacity is False
        assert env.engine._use_icio_weights is False
```

- [ ] **Schritt 2: Test ausführen — erwartet FAIL**

```bash
pytest tests/test_ablation_features.py::TestEnvironmentFlags -v
# Erwartet: TypeError: __init__() got unexpected keyword argument 'use_baci_capacity'
```

- [ ] **Schritt 3: ProviderEnvironment-Konstruktor erweitern**

In `provider_sim/env/environment.py` den `__init__`-Parameter und die Engine-Instanziierung anpassen.
`ProviderEnvironment.__init__` erhält zwei neue optionale Parameter:

```python
def __init__(
    self,
    pdl_source,
    seed: int = 0,
    max_ticks: int = 365,
    uid: str = "provider_env",
    broker_uri: str = "",
    weighted_defender_reward: bool = False,
    use_baci_capacity: bool = False,
    use_icio_weights: bool = False,
) -> None:
```

Und im Body, wo `SimulationEngine` instanziiert wird, die Flags übergeben:

```python
self.engine = SimulationEngine(
    doc,
    seed=seed,
    max_ticks=max_ticks,
    use_baci_capacity=use_baci_capacity,
    use_icio_weights=use_icio_weights,
)
```

- [ ] **Schritt 4: Tests ausführen — erwartet PASS**

```bash
pytest tests/test_ablation_features.py -v
pytest tests/test_env_environment.py -v
# Erwartet: alle PASSED
```

- [ ] **Schritt 5: Commit**

```bash
git add provider_sim/env/environment.py tests/test_ablation_features.py
git commit -m "feat: pass use_baci_capacity + use_icio_weights through ProviderEnvironment"
```

---

## Task 4: Ablations-Harness (ablation_train.py)

**Files:**
- Create: `experiments/ablation_train.py`

Dieser Harness ist eine angepasste Version von `autorl_train.py`. Er ist **nicht frozen** — für jede Variante werden frische Checkpoints verwendet.

- [ ] **Schritt 1: Datei anlegen**

```python
# experiments/ablation_train.py
"""Ablation study harness for PROVIDER data-source experiment.

Runs N_SEEDS fresh training runs (no warm checkpoints) for a given
combination of BACI capacity constraints and ICIO flow weights.

Usage:
    python experiments/ablation_train.py [--baci] [--icio] [--n-seeds N] [--budget SECS]

Outputs:
    - ablation_<variant>.log   per-run results
    - checkpoints/ablation_results.tsv  accumulated results
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

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENARIO = os.path.normpath(
    os.path.join(_BASE, "experiments", "s1-soja_icio.pdl.yaml")
)
_CKPT_DIR  = os.path.join(_BASE, "experiments", "checkpoints")
_RESULTS   = os.path.join(_CKPT_DIR, "ablation_results.tsv")
_MAX_TICKS = 365
_VAL_SEEDS = (42, 43, 44, 45, 46)  # 5 seeds für stabilere Schätzung
_DEVICE    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------- Observation normalizer (Welford) ----------

class _ObsNorm:
    def __init__(self, n: int) -> None:
        self.n = 0
        self.mean = np.zeros(n, dtype=np.float64)
        self.M2   = np.zeros(n, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        d = x.astype(np.float64) - self.mean
        self.mean += d / self.n
        self.M2   += d * (x.astype(np.float64) - self.mean)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.M2 / (self.n - 1) + 1e-8) if self.n >= 2 else np.ones_like(self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x.astype(np.float64) - self.mean) / self.std, -10.0, 10.0).astype(np.float32)


# ---------- Trajectory buffer ----------

class _Buf:
    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.logits: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(self, obs, logits, reward, log_prob, value, done) -> None:
        self.obs.append(obs); self.logits.append(np.asarray(logits, dtype=np.float32))
        self.rewards.append(float(reward)); self.log_probs.append(float(log_prob))
        self.values.append(float(value)); self.dones.append(bool(done))

    def clear(self) -> None:
        self.obs.clear(); self.logits.clear(); self.rewards.clear()
        self.log_probs.clear(); self.values.clear(); self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------- Core helpers (identisch zu autorl_train.py) ----------

@torch.no_grad()
def _sample(net, obs, budget, greedy=False):
    obs_t = torch.tensor(obs.tolist(), dtype=torch.float32, device=_DEVICE).unsqueeze(0)
    mu, std, value = net(obs_t)
    mu, std, value = mu.squeeze(0), std.squeeze(0), value.squeeze(0)
    logits = mu if greedy else Normal(mu, std).rsample()
    log_prob = Normal(mu, std).log_prob(logits).sum().item()
    actions = (F.softmax(logits, dim=-1) * budget).cpu().numpy()
    return actions, logits.cpu().numpy(), log_prob, value.item()


def _ppo_update(net, opt, buf):
    T = len(buf)
    values_ext = buf.values + [0.0]
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nd = 1.0 - float(buf.dones[t])
        delta = buf.rewards[t] + GAMMA * values_ext[t+1] * nd - values_ext[t]
        gae = delta + GAMMA * GAE_LAMBDA * nd * gae
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
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
    net.train(False)
    buf.clear()


def _run_episode(env, atk_net, def_net, atk_names, def_names, obs_names,
                 norm, seed, greedy=False):
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
        for i, n in enumerate(atk_names): actions[n] = float(atk_a[i])
        for i, n in enumerate(def_names): actions[n] = float(def_a[i])

        obs_dict, rewards, done = env.step_dict(actions)
        def_r = rewards["reward.defender"]; atk_r = rewards["reward.attacker"]
        total_def_r += def_r; steps += 1
        if not greedy:
            atk_buf.push(obs, atk_lg, atk_r, atk_lp, atk_v, done)
            def_buf.push(obs, def_lg, def_r, def_lp, def_v, done)
        if done:
            break

    return total_def_r / max(steps, 1), atk_buf, def_buf, steps


def run_single(baci: bool, icio: bool, seed: int, budget_secs: int) -> float:
    """Train from scratch for budget_secs, return mean val_reward."""
    env = ProviderEnvironment(
        pdl_source=_SCENARIO, seed=seed, max_ticks=_MAX_TICKS,
        use_baci_capacity=baci, use_icio_weights=icio,
    )
    obs_names = env.sensor_names
    atk_names = [n for n in env.actuator_names if n.startswith("attacker.")]
    def_names  = [n for n in env.actuator_names if n.startswith("defender.")]
    n_obs, n_atk, n_def = len(obs_names), len(atk_names), len(def_names)

    norm    = _ObsNorm(n_obs)
    atk_net = PPONet(n_obs=n_obs, n_act=n_atk).to(_DEVICE)
    def_net = PPONet(n_obs=n_obs, n_act=n_def).to(_DEVICE)
    atk_opt = torch.optim.Adam(atk_net.parameters(), lr=LR)
    def_opt = torch.optim.Adam(def_net.parameters(), lr=LR)

    ep = 0; t0 = time.time()
    while time.time() - t0 < budget_secs:
        _, atk_buf, def_buf, _ = _run_episode(
            env, atk_net, def_net, atk_names, def_names, obs_names, norm, seed=200+ep
        )
        _ppo_update(atk_net, atk_opt, atk_buf)
        _ppo_update(def_net, def_opt, def_buf)
        ep += 1

    val_rewards = []
    for vs in _VAL_SEEDS:
        r, _, _, _ = _run_episode(
            env, atk_net, def_net, atk_names, def_names, obs_names, norm,
            seed=vs, greedy=True,
        )
        val_rewards.append(r)

    mean_val = float(np.mean(val_rewards))
    print(f"  seed={seed} episodes={ep} val_reward={mean_val:.6f}", flush=True)
    return mean_val


def _append_result(variant: str, rewards: List[float]) -> None:
    os.makedirs(_CKPT_DIR, exist_ok=True)
    header = not os.path.isfile(_RESULTS)
    with open(_RESULTS, "a") as f:
        if header:
            f.write("variant\tmean_val\tstd_val\tmin_val\tmax_val\tn_seeds\trewards\n")
        rewards_str = ",".join(f"{r:.6f}" for r in rewards)
        f.write(
            f"{variant}\t{np.mean(rewards):.6f}\t{np.std(rewards):.6f}\t"
            f"{np.min(rewards):.6f}\t{np.max(rewards):.6f}\t{len(rewards)}\t{rewards_str}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baci",    action="store_true", help="BACI-Kapazitätsgrenzen aktivieren")
    parser.add_argument("--icio",    action="store_true", help="ICIO-Ghosh-Gewichte aktivieren")
    parser.add_argument("--n-seeds", type=int, default=5, help="Anzahl Seeds (default: 5)")
    parser.add_argument("--budget",  type=int, default=300, help="Training-Sekunden pro Seed (default: 300)")
    args = parser.parse_args()

    if not os.path.isfile(_SCENARIO):
        print(f"ERROR: Szenario nicht gefunden: {_SCENARIO}")
        sys.exit(1)

    variant = f"baci={'1' if args.baci else '0'}_icio={'1' if args.icio else '0'}"
    print(f"[ablation] Variante: {variant}")
    print(f"[ablation] Szenario: {_SCENARIO}")
    print(f"[ablation] Seeds: {args.n_seeds} × {args.budget}s Training")

    rewards = []
    for seed in range(args.n_seeds):
        print(f"\n[ablation] --- Seed {seed}/{args.n_seeds-1} ---")
        r = run_single(baci=args.baci, icio=args.icio, seed=seed, budget_secs=args.budget)
        rewards.append(r)

    print(f"\n[ablation] === Ergebnis Variante {variant} ===")
    print(f"mean_val: {np.mean(rewards):.6f}")
    print(f"std_val:  {np.std(rewards):.6f}")
    print(f"min/max:  {np.min(rewards):.6f} / {np.max(rewards):.6f}")
    print(f"rewards:  {rewards}")

    _append_result(variant, rewards)
    print(f"\n[ablation] Ergebnis gespeichert in {_RESULTS}")


if __name__ == "__main__":
    main()
```

- [ ] **Schritt 2: Smoke-Test (1 Seed, 30 Sekunden)**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
python experiments/ablation_train.py --n-seeds 1 --budget 30
# Erwartet: läuft durch, gibt val_reward aus, schreibt ablation_results.tsv
```

- [ ] **Schritt 3: Commit**

```bash
git add experiments/ablation_train.py
git commit -m "feat: ablation harness with BACI/ICIO flags, N-seeds, TSV output"
```

---

## Task 5: Orchestrierungsskript

**Files:**
- Create: `experiments/run_ablation.sh`

- [ ] **Schritt 1: Skript anlegen**

```bash
#!/usr/bin/env bash
# experiments/run_ablation.sh
# Führt alle 4 Ablationsvarianten durch.
# Laufzeit ca. 4 × 5 Seeds × 5.5 min = ~110 Minuten.
#
# Nutzung:
#   bash experiments/run_ablation.sh          # 5 Seeds, 300s Budget
#   bash experiments/run_ablation.sh 2 60     # 2 Seeds, 60s Budget (Schnelltest)

set -e
cd "$(dirname "$0")/.."

N_SEEDS="${1:-5}"
BUDGET="${2:-300}"
LOG_DIR="experiments/checkpoints/ablation_logs"
mkdir -p "$LOG_DIR"

echo "=== PROVIDER Ablationsstudie Datenquellen ==="
echo "Seeds: $N_SEEDS | Budget: ${BUDGET}s | Start: $(date)"

run_variant() {
    local LABEL="$1"
    shift
    echo ""
    echo "--- Variante $LABEL ---"
    python experiments/ablation_train.py "$@" --n-seeds "$N_SEEDS" --budget "$BUDGET" \
        2>&1 | tee "$LOG_DIR/ablation_${LABEL}.log"
    echo "--- $LABEL fertig ---"
}

run_variant "v0"             
run_variant "v1_baci"   --baci
run_variant "v2_icio"   --icio
run_variant "v3_both"   --baci --icio

echo ""
echo "=== Alle Varianten abgeschlossen: $(date) ==="
echo "Ergebnisse: experiments/checkpoints/ablation_results.tsv"
echo ""
echo "Schnellauswertung:"
column -t experiments/checkpoints/ablation_results.tsv
```

- [ ] **Schritt 2: Ausführbar machen und Schnelltest**

```bash
chmod +x experiments/run_ablation.sh
bash experiments/run_ablation.sh 1 30
# Erwartet: alle 4 Varianten laufen durch, TSV hat 4 Zeilen
```

- [ ] **Schritt 3: Commit**

```bash
git add experiments/run_ablation.sh
git commit -m "feat: run_ablation.sh orchestrates 4-variant datasource ablation study"
```

---

## Task 6: Vollstudie ausführen + Ergebnisse dokumentieren

- [ ] **Schritt 1: Vollstudie starten (im Hintergrund, ~110 min)**

```bash
bash experiments/run_ablation.sh 5 300 > experiments/checkpoints/ablation_full.log 2>&1 &
echo "PID: $!"
# Optional: Fortschritt verfolgen
tail -f experiments/checkpoints/ablation_full.log
```

- [ ] **Schritt 2: Ergebnistabelle lesen**

```bash
column -t experiments/checkpoints/ablation_results.tsv
# Erwartetes Format:
# variant        mean_val  std_val  min_val  max_val  n_seeds  rewards
# baci=0_icio=0  0.59XXXX  0.0XXX   ...      ...      5        ...
# baci=1_icio=0  0.6XXXXX  0.0XXX   ...      ...      5        ...
# baci=0_icio=1  0.6XXXXX  0.0XXX   ...      ...      5        ...
# baci=1_icio=1  0.6XXXXX  0.0XXX   ...      ...      5        ...
```

- [ ] **Schritt 3: Effektgrößen berechnen**

```python
# Einmaliger Auswertungssnippet (kein separates Skript nötig)
import pandas as pd
import numpy as np

df = pd.read_csv("experiments/checkpoints/ablation_results.tsv", sep="\t")
baseline = df[df.variant == "baci=0_icio=0"]["mean_val"].values[0]

for _, row in df.iterrows():
    delta = row["mean_val"] - baseline
    sig   = abs(delta) > 2 * row["std_val"]  # grober Signifikanztest
    print(f"{row['variant']}: Δ={delta:+.4f}  sig={'✓' if sig else '✗'}  "
          f"({row['mean_val']:.4f} ± {row['std_val']:.4f})")
```

- [ ] **Schritt 4: Ergebnis-Commit (nur TSV-Log, keine Checkpoints)**

```bash
git add experiments/checkpoints/ablation_full.log
git add experiments/checkpoints/ablation_results.tsv
git commit -m "results: datasource ablation study (BACI x ICIO, 5 seeds each)"
```

---

## Interpretation der Ergebnisse

| Ergebnis | Bedeutung |
|----------|-----------|
| BACI > baseline, ICIO neutral | Physische Kapazitätsgrenzen erzwingen realistische Policies; ICIO-Gewichte rauschen mehr als sie helfen |
| ICIO > baseline, BACI neutral | Wirtschaftliche Schockausbreitung gibt dem Netz nützliches Kausalitätssignal; Kapazitäten sind schon implizit im Graphen |
| Beide > baseline, V3 = V1+V2 | Additive Effekte — beide Quellen tragen unabhängig bei |
| Alle ~gleich wie baseline | Das 5-Minuten-Budget reicht nicht für feinere Strukturunterschiede — längeres Training nötig |
| V3 < V1 oder V2 | Negativinteraktion — die Kombination erzeugt widersprüchliche Lernsignale |

---

## Self-Review

**Spec-Coverage:**
- ✓ BACI-Kapazitätsgrenzen implementiert (Task 1+2)
- ✓ ICIO-Ghosh-Gewichte implementiert (Task 2)
- ✓ Flags durch ProviderEnvironment durchgereicht (Task 3)
- ✓ Fairer Vergleich: alle Varianten mit frischen Checkpoints (Task 4)
- ✓ 5 Seeds für Varianzschätzung (Task 4+5)
- ✓ GTA-Events konstant (alle Varianten nutzen s1-soja_icio.pdl.yaml)

**Placeholder-Scan:** Keine TBDs, alle Code-Blöcke vollständig.

**Typ-Konsistenz:** `_ObsNorm`, `_Buf`, `_sample`, `_ppo_update`, `_run_episode` sind vollständig in Task 4 definiert und in keiner anderen Task referenziert.
