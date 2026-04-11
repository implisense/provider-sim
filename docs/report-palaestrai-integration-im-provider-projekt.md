# palaestrAI-Integration im PROVIDER-Projekt

**OFFIS e.V. — Arbeitspakt AP3: Dynamisch instanziierbare Simulationsumgebung**
**Stand: April 2026 | Autor: Andreas Schäfer**

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
2. [Systemarchitektur](#2-systemarchitektur)
3. [PROVIDER Domain Language (PDL)](#3-provider-domain-language-pdl)
4. [Simulationsengine: Das 5-Phasen-Modell](#4-simulationsengine-das-5-phasen-modell)
5. [palaestrAI-Environment: ProviderEnvironment](#5-palaestrai-environment-providerenvironment)
6. [Objectives: AttackerObjective & DefenderObjective](#6-objectives-attackerobjective--defenderobjective)
7. [PPO-Implementierung: Brain, Muscle, Network](#7-ppo-implementierung-brain-muscle-network)
8. [Experiment-Konfiguration und YAML-Generierung](#8-experiment-konfiguration-und-yaml-generierung)
9. [Externer Trainings-Loop](#9-externer-trainings-loop)
10. [Bekannte Eigenheiten und Lösungen](#10-bekannte-eigenheiten-und-lösungen)
11. [Test-Abdeckung](#11-test-abdeckung)
12. [Aktueller Stand und Ergebnisse](#12-aktueller-stand-und-ergebnisse)

---

## 1. Überblick

Das PROVIDER-Projekt (BMWK-Verbundprojekt, 36 Monate, ~4,25 Mio. EUR) entwickelt ein Frühwarnsystem für Versorgungsengpässe in nicht-KRITIS-Lieferketten. Kernstück von AP3 ist eine agentenbasierte Simulation mit **Adversarial Resilience Learning (ARL)**: Ein Angreifer-Agent stört die Lieferkette, ein Verteidiger-Agent versucht Ausfälle zu verhindern. Beide Agenten lernen durch Selbstspiel.

palaestrAI dient als **Orchestrierungsrahmen** für dieses ARL-Szenario. PROVIDER stellt:
- Eine domänenspezifische Beschreibungssprache (**PDL**) für Lieferkettenstörungen
- Eine 5-Phasen-Simulationsengine als palaestrAI-Environment
- Eigene PPO-Brain/Muscle-Implementierungen für das RL-Training

### Verzeichnisstruktur

```
palestrai_simulation/
├── provider_sim/
│   ├── pdl/           PDL-Parser (standalone, nur PyYAML)
│   ├── sim/           Simulationsengine (PyYAML + NumPy)
│   ├── env/           palaestrAI Environment + Objectives
│   └── rl/            PPO Brain/Muscle/Network
├── experiments/
│   ├── generate_config.py     PDL → Experiment-YAML Generator
│   ├── train_ppo.py           Externer Trainings-Loop
│   ├── soja_arl_dummy.yaml    Baseline (DummyBrain/Muscle)
│   ├── soja_arl_ppo.yaml      PPO-Training
│   └── checkpoints/           attacker.pt, defender.pt, progress.json
└── tests/                     95 Tests (8 Dateien)
```

---

## 2. Systemarchitektur

### Schichtenmodell

```
┌───────────────────────────────────────────────────────┐
│  palaestrAI Orchestrator                              │
│  (VanillaSimController, Brain-/Muscle-Subprozesse)    │
└───────────────┬───────────────────────────────────────┘
                │ Environment ABC
┌───────────────▼───────────────────────────────────────┐
│  ProviderEnvironment  (provider_sim/env/environment.py)│
│  – start_environment() → EnvironmentBaseline          │
│  – update(actuators)  → EnvironmentState              │
└───────────────┬───────────────────────────────────────┘
                │ step()
┌───────────────▼───────────────────────────────────────┐
│  SimulationEngine  (provider_sim/sim/engine.py)        │
│  5-Phasen-Modell: Actions → Events → Impacts →        │
│                   Flow-Propagation → Health            │
└───────────────┬───────────────────────────────────────┘
                │ load_pdl()
┌───────────────▼───────────────────────────────────────┐
│  PDL-Parser  (provider_sim/pdl/)                      │
│  YAML → PdlDocument (Entities, Events, Dependencies)  │
└───────────────────────────────────────────────────────┘
```

### Abhängigkeitsrichtung

| Schicht | Imports | Externe Deps |
|---------|---------|--------------|
| `pdl/` | — | PyYAML |
| `sim/` | `pdl` | NumPy |
| `env/` | `pdl`, `sim` | palaestrAI (optional, Stub-Fallback) |
| `rl/` | `env` | PyTorch, palaestrAI |

Der Stub-Fallback (`_HAS_PALAESTRAI`-Flag) erlaubt es, `env/` und `rl/` ohne installiertes palaestrAI zu importieren und zu testen.

---

## 3. PROVIDER Domain Language (PDL)

PDL (PROVIDER Domain Language) ist eine YAML-basierte DSL zur deklarativen Beschreibung von Lieferkettenstörungsszenarien. Aktuell existieren 9 Szenarien (S1–S9).

### Datenmodell

```
PdlDocument
├── scenario: Scenario        (id, name, sector, criticality)
├── entities: List[Entity]    (Akteure: manufacturer, commodity, infrastructure, ...)
├── supply_chains: List[SupplyChain]  (Stufenmodell + Dependencies)
├── events: List[Event]       (Störereignisse mit Trigger + Impact)
└── cascades: List[Cascade]   (Zeitliche Ereignissequenzen mit Validation)
```

**Entity** (`provider_sim/pdl/model.py`):

```python
@dataclass
class Entity:
    id: str
    type: EntityType          # manufacturer | commodity | infrastructure | service | region
    name: str
    sector: str
    location: str
    vulnerability: float = 0.5
    extra: Dict[str, Any] = field(default_factory=dict)
    # extra enthält domänenspezifische Felder: market_share, tier,
    # diesel_reserve_hours, substitution_potential, etc.
```

**Event** mit Trigger-Typen:

```python
@dataclass
class Event:
    id: str
    name: str
    type: EventType           # natural_disaster | market_shock | cyber_attack | ...
    trigger: Trigger
    impact: Impact
    causes: List[str]         # Event-IDs für Kaskadenlogik

    @property
    def is_root_event(self) -> bool:
        # Root-Event: probabilistisch ausgelöst (keine Condition)
        return self.trigger.probability is not None and \
               self.trigger.condition is None
```

**Impact** mit Percentage-Werten:

```python
@dataclass
class Impact:
    supply: Optional[Percentage]   # z.B. Percentage(raw="-40%", decimal=-0.4)
    demand: Optional[Percentage]
    price: Optional[Percentage]
    duration: Optional[Duration]   # z.B. Duration(raw="90d", days=90.0)
    severity: Severity             # critical | high | medium | low
```

### Condition-Grammatik

Conditional Events werden per AST-Parser ausgewertet. Die Grammatik kennt vier Atomtypen:

```
expr       := or_expr
or_expr    := and_expr (" OR " and_expr)*
and_expr   := atom (" AND " atom)*
atom       := EVENT_ID ".active"
            | EVENT_ID ".duration > " DURATION
```

Beispiele aus S1 (Soja):
```
brazil_drought.active
oil_mill_slowdown.active OR soy_export_reduction.active
livestock_pressure.active AND consumer_substitution.active
soy_export_reduction.active AND soy_export_reduction.duration > 30d
```

### S1-Soja-Szenario (Referenz)

Das Soja-Szenario (S1) dient als primäres Entwicklungs- und Testszenario:

| Kennzahl | Wert |
|----------|------|
| Entities | 20 |
| Events | 18 |
| Dependencies | ~15 |
| Sensoren (palaestrAI) | 99 |
| Aktuatoren pro Agent | 20 |
| Episode-Länge | 365 Ticks |

---

## 4. Simulationsengine: Das 5-Phasen-Modell

`SimulationEngine.step()` (`provider_sim/sim/engine.py`) führt pro Tick fünf Phasen aus:

### Phase 1 — Agent-Actions

```python
# Attacker reduziert Entity-Supply (gewichtet mit Vulnerability)
entity.supply -= action_strength * entity.vulnerability

# Defender addiert Supply-Verstärkung
entity.supply += action_strength * (1 - entity.vulnerability)
```

Aktionen beider Agenten werden **gleichzeitig** angewendet. Das Attacker-Budget (0.8) übersteigt das Defender-Budget (0.4), was asymmetrisches Spiel modelliert.

### Phase 2 — Event-Evaluierung

- **Root-Events** (probabilistisch): `rng.random() < event.trigger.probability`
- **Conditional Events** (AST-basiert): `condition_ast.evaluate(state)` → True/False
- Der **Attacker** kann Root-Events mit erhöhter Wahrscheinlichkeit erzwingen

### Phase 3 — Impact-Stack

Modifier-basiert, **kein** Per-Tick-Compounding:

```python
effective_supply = Π(1 + modifier_i)  # Produkt aller aktiven Impacts
```

Demand und Price werden direkt gesetzt (nicht mulitplikativ). Dies verhindert, dass sich Schocks bei langen Events exponentiell aufschaukeln.

### Phase 4 — Flow-Propagation

Topologische Sortierung (Kahn-Algorithmus) über den Dependency-Graph:

```python
# Supply fließt von upstream nach downstream
supply[node] = min(intrinsic_supply, mean(supply[upstream_nodes]))

# Dependency-Penalty bei gestörter Abhängigkeit
if dependency.criticality == "high" and upstream_supply < threshold:
    supply[node] *= penalty_factor
```

### Phase 5 — Health-Berechnung

```python
health = clip(
    0.5 * supply
    + 0.3 * (1.0 / max(price, 1e-9))
    + 0.2 * min(demand, 1.0),
    0.0,
    1.0
)
```

Natürliche Erholung: +2% pro Tick in Richtung 1.0 wenn keine aktiven Events.

### Reward-Berechnung (Zero-Sum)

```python
mean_health = mean([entity.health for entity in all_entities])

reward_defender = mean_health          # Verteidiger will Health maximieren
reward_attacker = 1.0 - mean_health   # Angreifer will Health minimieren

# Invariante: reward_attacker + reward_defender == 1.0
```

---

## 5. palaestrAI-Environment: ProviderEnvironment

`ProviderEnvironment` (`provider_sim/env/environment.py`) implementiert das palaestrAI `Environment` ABC und bietet zusätzlich eine standalone Dict-API für Tests ohne Orchestrator.

### Konstruktor

```python
class ProviderEnvironment(_BaseEnv):
    def __init__(
        self,
        pdl_source: Union[str, Path, PdlDocument],
        seed: int = 0,
        max_ticks: int = 365,
        uid: str = "provider_env",
        broker_uri: str = "",
        **kwargs: Any,
    ) -> None
```

`pdl_source` akzeptiert Dateipfad, Path-Objekt oder bereits geparstes `PdlDocument`. Das Environment wird lazy initialisiert.

### palaestrAI-API

| Methode | Rückgabe | Beschreibung |
|---------|----------|--------------|
| `start_environment()` | `EnvironmentBaseline` | Reset + vollständige Sensor-/Aktuator-Beschreibung |
| `update(actuators)` | `EnvironmentState` | Führt einen Step aus, liefert neuen Zustand |

`EnvironmentBaseline` enthält:
- `sensors_available`: vollständige Liste aller `SensorInformation`-Objekte
- `actuators_available`: vollständige Liste aller `ActuatorInformation`-Objekte
- `simtime`: `SimTime(ticks=0)`

`EnvironmentState` enthält:
- `sensor_information`: aktualisierte Sensorwerte
- `rewards`: `[RewardInformation("reward.attacker", ...), RewardInformation("reward.defender", ...)]`
- `done`: `True` wenn `tick >= max_ticks`
- `simtime`: `SimTime(ticks=current_tick)`

### Sensor-Schema (S1-Soja: 99 Sensoren)

```
# Pro Entity (20 × 4 = 80 Sensoren):
provider_env.entity.brazil_farms.supply   → Box(low=0, high=2, shape=(1,), dtype=float32)
provider_env.entity.brazil_farms.demand   → Box(low=0, high=3, shape=(1,), dtype=float32)
provider_env.entity.brazil_farms.price    → Box(low=0, high=10, shape=(1,), dtype=float32)
provider_env.entity.brazil_farms.health   → Box(low=0, high=1, shape=(1,), dtype=float32)
...

# Pro Event (18 × 1 = 18 Sensoren):
provider_env.event.brazil_drought.active  → Discrete(2)  # Wert: int (0 oder 1)
provider_env.event.soy_export_reduction.active
...

# Global (1 Sensor):
provider_env.sim.tick                     → Box(low=0, high=max_ticks, shape=(1,))
```

### Aktuator-Schema (S1-Soja: 40 Aktuatoren)

```
# Pro Entity (20 × 2 = 40 Aktuatoren):
provider_env.attacker.brazil_farms        → Box(low=0, high=1, shape=(1,), dtype=float32)
provider_env.defender.brazil_farms        → Box(low=0, high=1, shape=(1,), dtype=float32)
...
```

**Wichtig:** palaestrAI entfernt/ergänzt automatisch den `uid`-Prefix. Die Environment-Implementierung arbeitet intern mit "bare IDs" (`entity.brazil_farms.supply`), das Framework prepended den konfigurierten UID (`provider_env.entity.brazil_farms.supply`).

### Standalone Dict-API

Für Unit-Tests und Analyse-Skripte ohne Orchestrator:

```python
env = ProviderEnvironment("s1-soja.pdl.yaml", seed=42)

# Reset
obs_dict, reward_dict = env.reset_dict()
# obs_dict: {"entity.brazil_farms.supply": 1.0, "event.brazil_drought.active": 0, ...}

# Step mit actions als flaches Dict
obs_dict, reward_dict, done = env.step_dict({
    "attacker.brazil_farms": 0.3,
    "defender.brazil_farms": 0.1,
    ...
})
```

---

## 6. Objectives: AttackerObjective & DefenderObjective

`provider_sim/env/objectives.py` implementiert das palaestrAI `_BaseObjective`-Interface.

```python
class AttackerObjective(_BaseObjective):
    def __init__(self, reward_id: str = "reward.attacker", **kwargs):
        ...

    def internal_reward(self, rewards: List[RewardInformation]) -> float:
        for r in rewards:
            if r.reward_id == self._reward_id:
                return float(np.asarray(r.reward_value).item())
        return 0.0
```

`DefenderObjective` ist identisch aufgebaut, filtert aber nach `"reward.defender"`.

Die `reward_id` ist konfigurierbar, was mehrere parallele Environments mit unterschiedlichen IDs ermöglicht. Der Stub-Fallback-Mechanismus ist konsistent mit `environment.py` implementiert.

---

## 7. PPO-Implementierung: Brain, Muscle, Network

### 7.1 PPONet — Shared Actor-Critic MLP

`provider_sim/rl/network.py`

```python
class PPONet(nn.Module):
    """
    Architektur: Input → Linear(h1) → ReLU → Linear(h2) → ReLU
                          ├── policy_head: Linear(h2, n_act)  [Logit-Mittelwerte]
                          └── value_head:  Linear(h2, 1)       [Zustandswert]

    h1 = max(256, (n_obs * 2 // 64) * 64)   # Auto-Scaling
    h2 = h1 // 2
    """
    def __init__(self, n_obs: int = 99, n_act: int = 20) -> None
```

**Auto-Scaling der Hidden-Layer:** Für 99 Sensoren ergibt sich h1=256, h2=128. Bei größeren Szenarien (mehr Entities/Events) skaliert das Netz automatisch.

**Learnable `log_std`-Parameter** für Exploration — wird über Backpropagation trainiert.

**Schlüsselmethoden:**

| Methode | Beschreibung |
|---------|--------------|
| `forward(x)` | → `(mu, std, value)` |
| `sample_action(obs, budget)` | Samplet aus `Normal(mu, std)`, Budget-Constraint per Softmax |
| `recompute_logprob(obs, logits)` | Recomputed `log_prob`, `value`, `entropy` für PPO-Update |

**Budget-Constraint in `sample_action`:**
```python
raw_actions = Normal(mu, std).rsample()         # Sampling
bounded     = softmax(raw_actions) * budget     # Normierung auf Budget
# → Summe der Aktionen ≤ budget (0.8 Attacker, 0.4 Defender)
```

### 7.2 PPOMuscle — Inference-Schicht

`provider_sim/rl/ppo_muscle.py` läuft als palaestrAI **Muscle-Subprocess**.

**Gerät: immer CPU** — palaestrAI übergibt Daten über `aiomultiprocess`-Shared-Memory; MPS/GPU-Tensoren können nicht serialisiert werden.

```python
class PPOMuscle(Muscle):
    def __init__(
        self, *args,
        budget: float = 0.8,
        n_obs: int = 99,
        n_act: int = 20,
        checkpoint_path: str = "",
        topk: int = 0,          # Top-K-Masking (0 = deaktiviert)
        **kwargs
    ) -> None
```

**`propose_actions`-Ablauf pro Tick:**

```python
def propose_actions(self, sensors, actuators_available, is_terminal=False):
    # 1. Observation aus SensorInformation extrahieren
    obs = self._extract_obs(sensors)        # np.ndarray, shape (99,)

    # 2. Forward pass (kein Gradient)
    with torch.no_grad():
        actions, sampled_logits, log_prob, value = \
            self._net.sample_action(obs, budget=self._budget)

    # 3. Optional: Top-K-Masking
    if self._topk > 0:
        # Konzentriere Budget auf die topk stärksten Aktionen
        mask = actions < torch.topk(actions, self._topk).values[-1]
        actions[mask] = 0.0
        if actions.sum() > 0:
            actions = actions / actions.sum() * self._budget

    # 4. Setpoints schreiben
    for actuator, val in zip(actuators_available, actions.tolist()):
        actuator.setpoint = np.array([val], dtype=np.float32)

    # 5. Zusatzdaten für Brain (Trajektorienpuffer)
    return (actuators_available, actuators_available, obs, {
        "sampled_logits": sampled_logits,
        "log_prob": log_prob.item(),
        "value": value.item(),
    })
```

**`update(state_dict)`:** Wird aufgerufen, wenn Brain nach PPO-Update einen neuen `state_dict` liefert. Muscle lädt die Gewichte sofort: `self._net.load_state_dict(state_dict)`.

### 7.3 PPOBrain — Learning-Schicht

`provider_sim/rl/ppo_brain.py` läuft als palaestrAI **Brain-Subprocess**.

**Gerät: lazy MPS/GPU** — Brain initialisiert das Netz auf CPU (sicheres Pickling beim Spawn), verschiebt es beim ersten PPO-Update auf MPS per `_ensure_on_device()`.

```python
class PPOBrain(Brain):
    def __init__(
        self, *args,
        n_obs: int = 99, n_act: int = 20,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_decay: float = 1.0,
        entropy_coef_min: float = 0.001,
        min_episode_steps: int = 10,    # Spurious-Episode-Filter
        checkpoint_path: str = "",
        best_checkpoint_path: str = "",
        **kwargs,
    ) -> None
```

**Trajektorienpuffer** (wächst über 365 Ticks):

```
_obs_buf      List[torch.Tensor]  # Beobachtungen
_logits_buf   List[torch.Tensor]  # Gesamplete Logits (für PPO ratio)
_reward_buf   List[float]         # Skalare Rewards
_log_prob_buf List[float]         # Log-Wahrscheinlichkeiten
_value_buf    List[float]         # State-Values
_done_buf     List[bool]          # Done-Flags
```

**`thinking()`-Ablauf:**

```python
def thinking(self, muscle_id, readings, actions, reward, next_state, done, additional_data):
    # 1. Reward extrahieren (via Objective)
    r = self._objective.internal_reward(reward)

    # 2. Transition puffern
    self._buffer_append(readings, additional_data, r, done)

    if not done:
        return MuscleUpdateResponse(False, None)

    # 3. Spurious-Episoden filtern (palaestrAI-Quirk, siehe Abschnitt 10)
    if len(self._reward_buf) < self._min_episode_steps:
        self._clear_buffers()
        return MuscleUpdateResponse(False, None)

    # 4. GAE + PPO-Update
    state_dict = self._ppo_update(next_value=0.0)

    # 5. Entropy-Decay nach jeder Episode
    self._entropy_coef = max(
        self._entropy_coef * self._entropy_decay,
        self._entropy_coef_min
    )

    # 6. Checkpoint speichern
    torch.save(state_dict, self._checkpoint_path)

    # 7. Muscle mit neuen Gewichten aktualisieren
    return MuscleUpdateResponse(True, state_dict)
```

**GAE (Generalized Advantage Estimation):**

```python
def _compute_gae(self, next_value: float):
    advantages = []
    gae = 0.0
    values = self._value_buf + [next_value]

    for t in reversed(range(len(self._reward_buf))):
        delta = self._reward_buf[t] + self.gamma * values[t+1] - values[t]
        gae   = delta + self.gamma * self.gae_lambda * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, self._value_buf)]

    # Advantage-Normalisierung
    adv_t = torch.tensor(advantages)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    return adv_t, torch.tensor(returns)
```

**PPO-Update (4 Epochen):**

```python
def _ppo_update(self, next_value):
    adv, returns = self._compute_gae(next_value)
    obs_t    = torch.stack(self._obs_buf)
    logits_t = torch.stack(self._logits_buf)
    old_lp   = torch.tensor(self._log_prob_buf)

    for _ in range(self._ppo_epochs):
        new_lp, new_val, entropy = self._net.recompute_logprob(obs_t, logits_t)

        ratio   = (new_lp - old_lp).exp()
        clipped = ratio.clamp(1 - self._clip_eps, 1 + self._clip_eps)

        actor_loss  = -torch.min(ratio * adv, clipped * adv).mean()
        critic_loss = F.mse_loss(new_val, returns)
        loss = actor_loss + self._value_coef * critic_loss \
                          - self._entropy_coef * entropy

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
        self._optimizer.step()

    self._clear_buffers()
    return self._net.state_dict()
```

### Zusammenfassung: Brain ↔ Muscle Interaktion

```
palaestrAI Orchestrator
        │
        │ start_environment()
        ▼
ProviderEnvironment.start_environment()
        │
        │ 365× update(actuators)
        ▼
┌──────────────────────────────────────┐
│ Tick-Loop                            │
│                                      │
│  PPOMuscle.propose_actions()         │
│    → Forward pass (CPU)              │
│    → Budget-Constraint               │
│    → Setpoints + additional_data     │
│            │                         │
│            ▼                         │
│  ProviderEnvironment.update()        │
│    → 5-Phasen-Step                   │
│    → Rewards berechnen               │
│            │                         │
│            ▼                         │
│  PPOBrain.thinking()                 │
│    → Transition puffern              │
│    → Bei done=True: PPO-Update (MPS) │
│    → state_dict → Muscle.update()    │
└──────────────────────────────────────┘
```

---

## 8. Experiment-Konfiguration und YAML-Generierung

### `generate_config.py`

Der Generator liest ein PDL-Szenario und erzeugt automatisch ein vollständiges palaestrAI-Experiment-YAML:

```bash
python experiments/generate_config.py scenarios/s1-soja.pdl.yaml \
    --output experiments/soja_arl_dummy.yaml \
    --max-ticks 365 --episodes 1 --seed 42
```

**Generierungslogik:**
```python
doc = load_pdl(pdl_path)

# Sensoren aus PDL
sensors = []
for ent in doc.entities:
    sensors += [f"{uid}.entity.{ent.id}.supply",
                f"{uid}.entity.{ent.id}.demand",
                f"{uid}.entity.{ent.id}.price",
                f"{uid}.entity.{ent.id}.health"]
for ev in doc.events:
    sensors.append(f"{uid}.event.{ev.id}.active")
sensors.append(f"{uid}.sim.tick")

# Aktuatoren aus Entities
actuators = [f"{uid}.attacker.{ent.id}" for ent in doc.entities]
actuators += [f"{uid}.defender.{ent.id}" for ent in doc.entities]
```

### Experiment-YAML-Struktur (Auszug)

```yaml
uid: provider-soy_feed_disruption-arl-ppo
seed: 42
version: 3.4.1

schedule:
  - phase_train:
      environments:
        - environment:
            name: provider_sim.env.environment:ProviderEnvironment
            uid: provider_env
            params:
              pdl_source: /abs/path/to/s1-soja.pdl.yaml
              max_ticks: 365
          reward:
            name: palaestrai.agent.dummy_objective:DummyObjective

      agents:
        - name: attacker
          brain:
            name: provider_sim.rl.ppo_brain:PPOBrain
            params:
              n_obs: 99
              n_act: 20
              lr: 0.0003
              clip_eps: 0.2
              checkpoint_path: experiments/checkpoints/attacker.pt
          muscle:
            name: provider_sim.rl.ppo_muscle:PPOMuscle
            params:
              budget: 0.8
              n_obs: 99
              n_act: 20
              checkpoint_path: experiments/checkpoints/attacker.pt
          objective:
            name: provider_sim.env.objectives:AttackerObjective
            params:
              reward_id: reward.attacker
          sensors: [provider_env.entity.brazil_farms.supply, ...]
          actuators: [provider_env.attacker.brazil_farms, ...]

        - name: defender
          # ... analog, budget: 0.4
          objective:
            name: provider_sim.env.objectives:DefenderObjective

      simulation:
        - name: palaestrai.simulation.vanilla_sim_controller:VanillaSimController
          params: {}

      phase_config:
        mode: train
        episodes: 1         # Pro Aufruf genau 1 Episode
        worker: 1

run_config:
  condition: VanillaRunGovernorTerminationCondition
```

---

## 9. Externer Trainings-Loop

### Motivation

palaestrAI unterstützt kein echtes Multi-Episode-Training über mehrere `experiment-start`-Aufrufe hinweg — jeder Aufruf startet das Experiment von vorn. Um 50 Trainingsepisoden durchzuführen, wurde `experiments/train_ppo.py` entwickelt: ein externer Python-Skript, das 50 Mal `palaestrai experiment-start` aufruft und Fortschritt sowie Checkpoints verwaltet.

### Architektur

```
train_ppo.py (Manager-Prozess)
│
├── Liest: experiments/checkpoints/progress.json
│            {"completed_episodes": 15, "total_episodes": 50}
│
├── For episode in [16, 17, ..., 50]:
│   │
│   ├── Bereinige stale Prozesse
│   │   pkill -f "palaestrai" && pkill -f "spawn_main"
│   │
│   ├── subprocess.run(["palaestrai", "experiment-start", config.yaml])
│   │   → PPOBrain und PPOMuscle starten als Subprozesse
│   │   → 365 Ticks werden durchgeführt
│   │   → Bei done=True: PPO-Update, Checkpoint gespeichert
│   │
│   ├── Bei Erfolg:
│   │   └── Schreibe progress.json, bereinige palaestrai.db
│   │
│   └── Bei Fehler:
│       └── Max. 3 Versuche, dann Abbruch
│
└── Abschluss: Report über avg_reward Verlauf
```

### Checkpoint-Kontinuität

```
Episode N:
  PPOBrain.thinking() [done=True]
    → PPO-Update
    → torch.save(state_dict, "checkpoints/attacker.pt")
    → MuscleUpdateResponse(True, state_dict)

Episode N+1 (nächster experiment-start):
  PPOMuscle.setup()
    → if os.path.isfile("checkpoints/attacker.pt"):
    →     self._net.load_state_dict(torch.load("checkpoints/attacker.pt"))
  PPOBrain.__init__()
    → if os.path.isfile("checkpoints/attacker.pt"):
    →     self._net.load_state_dict(torch.load("checkpoints/attacker.pt"))
```

Damit ist das Training **mit Ctrl+C unterbrechbar** und mit `python train_ppo.py 50` **fortsetzbar** — ohne Datenverlust.

**Neustart:** `rm experiments/checkpoints/*.pt experiments/checkpoints/progress.json`

### Laufzeitcharakteristik

| Metrik | Wert |
|--------|------|
| Ticks pro Episode | 365 |
| Dauer pro Episode | ~3–5 Minuten |
| Trainings-Episoden bisher | 25 / 50 |
| Startup-Overhead (palaestrAI) | ~15–20 Sekunden |
| Checkpoint-Größe | ~180 KB je Agent |

---

## 10. Bekannte Eigenheiten und Lösungen

### Eigenheit 1: Spurious Micro-Episoden

**Problem:** `VanillaSimController` sendet `ex_termination=True` sofort nach einem Restart. Das erzeugt pro Aufruf mit `episodes: N` exakt **(N−1) Micro-Episoden** von 2 Steps, bevor die echte Episode startet.

**Lösung:** `PPOBrain` filtert Episoden unter `min_episode_steps=10`:
```python
if len(self._reward_buf) < self._min_episode_steps:
    self._clear_buffers()
    return MuscleUpdateResponse(False, None)   # Kein Update
```

**Empfehlung:** `episodes: 1` in Experiment-YAMLs verwenden — dann gibt es nur 1 Micro-Episode (sofort gefiltert) + 1 echte Episode.

---

### Eigenheit 2: `store_model()` wird bei jedem Tick aufgerufen

**Problem:** palaestrAI ruft `Brain.store_model(path)` bei **jedem Tick** auf, nicht nur am Episode-Ende. Checkpoint-Speicherung darf daher **nur** im `done=True`-Zweig von `thinking()` erfolgen.

**Lösung:** Checkpoint-Logik ausschließlich in `_ppo_update()`, nicht in `store_model()`.

---

### Eigenheit 3: Sensor-Werte als plain floats beim ersten Tick

**Problem:** Die ersten Sensor-Daten aus `propose_actions()` können plain `float`-Objekte statt `SensorInformation`-Instanzen sein.

**Lösung:** Guard in `_extract_obs()`:
```python
for s in sensors:
    if hasattr(s, "sensor_value"):
        val = float(np.asarray(s.sensor_value).item())
    else:
        val = float(s)   # Fallback für raw floats
```

---

### Eigenheit 4: Aktuator-Setpoint muss np.array sein

**Problem:** palaestrAI erwartet `np.array([float], dtype=np.float32)` als Setpoint — ein plain `float` führt zu Serialisierungsfehlern.

**Lösung:**
```python
actuator.setpoint = np.array([val], dtype=np.float32)   # korrekt
# NICHT: actuator.setpoint = float(val)                 # falsch
```

---

### Eigenheit 5: MPS vs. CPU für Brain und Muscle

**Problem:** Brain und Muscle laufen in separaten `aiomultiprocess`-Prozessen. MPS-Tensoren (Apple Silicon) können nicht über Prozessgrenzen via Shared Memory übertragen werden.

**Lösung (Lazy Device Assignment):**
- **Muscle:** Immer CPU. Inference auf kleinem MLP ist vernachlässigbar schnell.
- **Brain:** Initialisierung auf CPU (sicheres Pickling beim Spawn), lazy Move nach MPS beim ersten PPO-Update:

```python
def _ensure_on_device(self) -> None:
    if self._device_moved:
        return
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    self._net = self._net.to(device)
    # Optimizer-States ebenfalls verschieben
    for state in self._optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    self._device_moved = True
```

---

### Eigenheit 6: Kein `-vv` bei `palaestrai experiment-start`

Verbose-Logging (`-vv`) erzeugt binäre DB-Daten im Stdout-Stream, was zu >500 MB Output und I/O-Stagnation führt. Standard-Verbosity ist ausreichend.

---

### Eigenheit 7: ZMQ-Port-Konflikte

palaestrAI nutzt ZMQ-Messaging auf Ports 4242 und 4243. Nach Abbrüchen (Ctrl+C) verbleiben Prozesse, die diese Ports belegen.

```bash
# Vor jedem Training prüfen:
lsof -i :4242 -i :4243

# Bereinigen:
pkill -f "palaestrai" && pkill -f "spawn_main" && pkill -f "resource_tracker"
```

`train_ppo.py` führt diese Bereinigung **automatisch** vor jeder Episode durch.

---

## 11. Test-Abdeckung

Alle 95 Tests (`pytest tests/ -v`) laufen ohne palaestrAI-Installation (Stub-Fallback).

### Übersicht

| Testdatei | Tests | Abdeckung |
|-----------|-------|-----------|
| `test_pdl_parser.py` | ~20 | PDL-Parsing, Duration, Percentage, Validierung |
| `test_condition.py` | ~15 | AST-Konstruktion, Auswertung mit MockState |
| `test_sim_state.py` | ~12 | State-Init, Entity-Count, Adjazenz-Graph |
| `test_sim_engine.py` | ~15 | Tick, Reset, Attacker/Defender, Kaskaden, 365-Tick |
| `test_env_environment.py` | 22 | ProviderEnvironment, palaestrAI-Protokoll, Dict-API |
| `test_objectives.py` | 11 | Reward-Filterung, Zero-Sum, Default-Werte |
| `test_integration.py` | ~9 | Alle 9 Szenarien × 365 Ticks ohne Fehler |

### Environment-Tests im Detail (`test_env_environment.py`)

**Klasse TestProviderEnvironment:**
- Sensor-Count: 136 Sensoren (für S1 mit allen Events: 20×4 + 55 + 1)
- Actuator-Count: 40 (20×2)

**Klasse TestPalaestrAIProtocol:**
- `start_environment()` → `EnvironmentBaseline` mit korrekten Datentypen
- `SensorInformation.sensor_value` → `np.float32`
- Event-Sensoren → `Discrete(2)`, Entity-Sensoren → `Box(...)`
- `update(actuators)` → `EnvironmentState` mit Rewards, SimTime, Done
- Zero-Sum-Invariante: `attacker_r + defender_r ≈ 1.0`

**Klasse TestDictInterface:**
- `reset_dict()` / `step_dict()` mit konsistenten Ergebnissen
- Gleiche Zero-Sum- und Done-Logik wie palaestrAI-API

**Klasse TestUidPrepending:**
- Bare IDs intern, Framework ergänzt/entfernt UID-Prefix

### Integrations-Test

```python
@pytest.mark.parametrize("path", all_scenario_paths())
def test_full_simulation_365_ticks(path):
    doc    = load_pdl(path)
    engine = SimulationEngine(doc, seed=0)
    for _ in range(365):
        engine.step({}, {})  # Kein Agent → Baseline-Drift
    assert engine.state.tick == 365
    for ent in engine.state.entities.values():
        assert 0.0 <= ent.health <= 1.0
        assert ent.supply >= 0.0
```

---

## 12. Aktueller Stand und Ergebnisse

### Trainingsfortschritt

| Agent | Episoden abgeschlossen | Gesamt-Ziel |
|-------|------------------------|-------------|
| Attacker + Defender | 25 | 50 |

Training ist pausiert (Checkpoints vorhanden), kann mit `python experiments/train_ppo.py 50` fortgesetzt werden.

### Implementierte Szenarien

| Szenario | Entitäten | Events | Status |
|----------|-----------|--------|--------|
| S1 Soja | 20 | 18 | Vollständig trainiert (25 EP) |
| S2 Halbleiter | — | — | PDL vorhanden, Training ausstehend |
| S3 Pharma | — | — | PDL vorhanden |
| S4 Düngemittel | — | — | PDL vorhanden |
| S5 Wasser | — | — | PDL vorhanden |
| S6 Rechenzentren | — | — | PDL vorhanden |
| S7 Seltene Erden | — | — | PDL vorhanden |
| S8 Seefracht | — | — | PDL vorhanden |
| S9 Unterwasserkabel | — | — | PDL vorhanden |

Alle 9 Szenarien werden im Integrationstest erfolgreich mit 365 Ticks ausgeführt.

### Nächste Schritte

1. **Training abschließen:** Restliche 25 Episoden für S1-Soja
2. **Weitere Szenarien:** Experiment-YAMLs für S2–S9 per `generate_config.py` generieren
3. **arsenAI-Integration:** Design of Experiments für systematische Hyperparameter-Suche
4. **AP5 (ARL):** Self-Play-Evaluierung — Attacker vs. trainierter Defender und umgekehrt
5. **Multi-Szenario-Training:** Curriculum Learning über mehrere PDL-Szenarien

---

## Anhang: Schnellstart

```bash
# Installation
pip install -e ".[dev]"

# Tests
pytest tests/ -v

# Baseline (kein RL)
palaestrai experiment-start experiments/soja_arl_dummy.yaml

# PPO-Training (50 Episoden, unterbrechbar)
python experiments/train_ppo.py 50

# Training fortsetzen
python experiments/train_ppo.py 50   # liest progress.json automatisch

# Neustart (Checkpoints löschen)
rm experiments/checkpoints/*.pt experiments/checkpoints/progress.json

# Portbereinigung nach Abbruch
pkill -f "palaestrai" && pkill -f "spawn_main" && pkill -f "resource_tracker"
```

---

*Generiert mit Claude Code — PROVIDER Projekt / OFFIS e.V. AP3*
