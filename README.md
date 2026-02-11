# provider_sim

PDL-Parser und Lieferketten-Simulationsengine für das PROVIDER-Projekt. Brücke zwischen den PDL-Szenarien (YAML-basierte Lieferketten-Beschreibungen) und dem [palaestrAI](https://2.2.2.2/2)-Framework für Adversarial Resilience Learning.

## Überblick

```
PDL-YAML laden → typisierte Python-Objekte → Simulation ausführen → palaestrAI-Environment
```

Das Paket besteht aus drei unabhängigen Schichten:

| Modul | Abhängigkeiten | Zweck |
|---|---|---|
| `provider_sim.pdl` | PyYAML | Standalone PDL-Parser: YAML → Dataclasses |
| `provider_sim.sim` | PyYAML, NumPy | 5-Phasen-Simulationsengine |
| `provider_sim.env` | PyYAML, NumPy, palaestrAI (optional) | palaestrAI-Environment-Wrapper |

## Installation

```bash
pip install -e ".[dev]"
```

Für palaestrAI-Integration:

```bash
pip install -e ".[rl,dev]"
```

## Quickstart

### PDL-Szenario laden

```python
from provider_sim.pdl.parser import load_pdl

doc = load_pdl("path/to/s1-soja.pdl.yaml")
print(f"{len(doc.entities)} Entities, {len(doc.events)} Events")
# 20 Entities, 18 Events
```

### Simulation ausführen

```python
from provider_sim.pdl.parser import load_pdl
from provider_sim.sim.engine import SimulationEngine

doc = load_pdl("path/to/s1-soja.pdl.yaml")
engine = SimulationEngine(doc, seed=42)

for tick in range(365):
    engine.step(
        attacker_actions={"brazil_farms": 0.3},
        defender_actions={"santos_port": 0.5},
    )

for eid in engine.state.entity_ids[:5]:
    es = engine.state.entities[eid]
    print(f"{eid}: health={es.health:.2f}, supply={es.supply:.2f}")
```

### palaestrAI-Environment (Orchestrator-API)

```python
import numpy as np
from provider_sim.env.environment import (
    ProviderEnvironment, ActuatorInformation, _box_space,
)

env = ProviderEnvironment("path/to/s1-soja.pdl.yaml", seed=42, max_ticks=365)
baseline = env.start_environment()

print(f"{len(baseline.sensors_available)} Sensoren, {len(baseline.actuators_available)} Aktuatoren")
# 99 Sensoren, 40 Aktuatoren

# Agent-Aktion als ActuatorInformation
actuators = [
    ActuatorInformation(
        np.array([0.5], dtype=np.float32),
        _box_space(0, 1),
        actuator_id="attacker.brazil_farms",
    )
]
state = env.update(actuators)

print(f"Tick: {state.simtime.simtime_ticks}, Done: {state.done}")
print(f"Attacker-Reward: {float(state.rewards[0].reward_value):.3f}")
print(f"Defender-Reward: {float(state.rewards[1].reward_value):.3f}")
```

### Standalone-Nutzung (Dict-API, ohne Orchestrator)

```python
from provider_sim.env.environment import ProviderEnvironment

env = ProviderEnvironment("path/to/s1-soja.pdl.yaml", seed=42, max_ticks=365)
obs, rewards = env.reset_dict()

obs, rewards, done = env.step_dict({"attacker.brazil_farms": 0.5})
print(f"Attacker: {rewards['reward.attacker']:.3f}, Defender: {rewards['reward.defender']:.3f}")
```

## Architektur

### PDL-Parser (`provider_sim.pdl`)

- **model.py** — Dataclasses: `PdlDocument`, `Entity`, `Event`, `Cascade`, `SupplyChain`, Enums (`EntityType`, `EventType`, `Criticality`, `Severity`), Value Objects (`Duration`, `Percentage`)
- **parser.py** — `load_pdl(source)` akzeptiert Dateipfade, YAML-Strings oder Dicts. Parst Durations (`90d`, `2w`, `6m`), Percentages (`-40%`, `+60%`), validiert alle ID-Referenzen
- **condition.py** — AST-basierter Parser für Trigger-Conditions (`event.active`, `event.duration > 30d`, `AND`/`OR`)
- **errors.py** — `PdlParseError`, `PdlValidationError`

### Simulationsengine (`provider_sim.sim`)

**5-Phasen-Step pro Tick:**

1. **Agent-Actions** — Attacker reduziert Entity-Supply (gewichtet mit Vulnerability), Defender kompensiert
2. **Event-Evaluierung** — Root-Events probabilistisch, Condition-Events per AST-Auswertung
3. **Impact-Stack** — Modifier-basiert: `effective_supply = base × Π(1 + modifier_i)`, kein Per-Tick-Compounding
4. **Flow-Propagation** — Topologische Sortierung, Downstream durch Upstream-Supply beschränkt, Dependency-Penalties
5. **Health** — `health = 0.5 × supply + 0.3 × (1/price) + 0.2 × min(demand, 1.0)`, clipped [0, 1]

Natürliche Recovery: 2%/Tick Richtung 1.0 bei inaktiven Events.

### palaestrAI-Environment (`provider_sim.env`)

Implementiert das palaestrAI `Environment` ABC mit zwei APIs:

**Orchestrator-API** (für palaestrAI-Agenten):

| Methode | Rückgabe | Beschreibung |
|---|---|---|
| `start_environment()` | `EnvironmentBaseline` | Reset + Baseline (`sensors_available`, `actuators_available`, `SimTime`) |
| `update(actuators)` | `EnvironmentState` | Step mit `List[ActuatorInformation]` → `sensor_information`, `rewards`, `done`, `SimTime` |

**Dict-API** (Standalone, ohne Orchestrator):

| Methode | Rückgabe | Beschreibung |
|---|---|---|
| `reset_dict()` | `(obs, rewards)` | Reset, flache Dicts |
| `step_dict(actions)` | `(obs, rewards, done)` | Step mit `{"attacker.<id>": float}` |

**Sensoren, Aktuatoren, Rewards:**

| Typ | Schema | Beispiel (Soja) |
|---|---|---|
| Sensoren | `SensorInformation` mit `Box`/`Discrete` Spaces, 4 pro Entity + 1 pro Event + 1 global | 20×4 + 18 + 1 = **99** |
| Aktuatoren | `ActuatorInformation` mit `Box(0, 1)`, 2 pro Entity (Attacker + Defender) | 20×2 = **40** |
| Rewards | `RewardInformation`, Zero-Sum: `attacker = mean(1−health)`, `defender = mean(health)` | Summe = 1.0 |

palaestrAI ist eine optionale Abhängigkeit — ohne sie werden Stub-Klassen für alle Typen (`SensorInformation`, `ActuatorInformation`, `RewardInformation`, `Box`, `Discrete`, `SimTime`, `EnvironmentBaseline`, `EnvironmentState`) verwendet.

## Unterstützte PDL-Szenarien

Alle 9 PROVIDER-Szenarien werden geladen und simuliert:

| # | Szenario | Entities | Events |
|---|---|---|---|
| S1 | Soja-Futtermittel | 20 | 18 |
| S2 | Halbleiter | 25 | 23 |
| S3 | Pharma | 24 | 18 |
| S4 | Düngemittel/AdBlue | 28 | 25 |
| S5 | Wasseraufbereitung | 22 | 23 |
| S6 | Rechenzentren | 38 | 30 |
| S7 | Seltene Erden | 25 | 24 |
| S8 | Seefracht | 22 | 23 |
| S9 | Unterwasserkabel | 21 | 23 |

## Tests

```bash
pytest tests/ -v
```

79 Tests:
- `test_pdl_parser.py` — Duration/Percentage-Parsing, alle 9 Szenarien laden, Validierung, Extra-Felder
- `test_condition.py` — AST-Konstruktion und -Auswertung mit Mock-State
- `test_sim_state.py` — State-Initialisierung, Adjazenz-Graph, ConditionState-Protokoll
- `test_sim_engine.py` — Tick-Inkrement, Reset, Attacker/Defender-Effekte, Kaskaden
- `test_env_environment.py` — 3 Testklassen: Basis (Counts, Pfad-Laden), palaestrAI-Protokoll (Baseline, State, Typen, Zero-Sum, SimTime), Dict-Interface (Reset, Step, Rewards)
- `test_integration.py` — PDL laden → 365 Ticks simulieren → alle 9 Szenarien

## Projektkontext

Teil des BMFTR-geförderten Verbundprojekts **PROVIDER** — Proaktive Versorgungssicherheit durch dynamische Simulation mit selbstlernenden LLM-Agenten. PDL (PROVIDER Domain Language) beschreibt Versorgungsszenarien maschinenlesbar als YAML-Dateien.

## Lizenz

Projektintern (PROVIDER-Konsortium).
