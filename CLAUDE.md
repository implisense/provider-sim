# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Dieses Paket ist der Python-Kern der PROVIDER-Simulation: PDL-Parser, Simulationsengine, palaestrAI-Environment und PPO-RL-Training.

## Sprache

Alle Kommunikation auf Deutsch. Code-Bezeichner, Docstrings und Kommentare bleiben Englisch.

## Sprache

Alle Kommunikation auf Deutsch. Code-Bezeichner, Docstrings und Kommentare bleiben Englisch.

## Kommandos

```bash
# Paket installieren (editable, mit Test-Deps)
pip install -e ".[dev]"

# Tests ausfuehren (95 Tests)
pytest tests/ -v

# Einzelnen Test ausfuehren
pytest tests/test_sim_engine.py::TestSimulationEngine::test_step -v

# PPO-Training starten (externer Loop, fortsetzbar mit Ctrl+C)
python experiments/train_ppo.py 50

# Einzelne palaestrAI-Episode (NICHT mit -vv!)
palaestrai experiment-start experiments/soja_arl_ppo.yaml

# Baseline-Experiment (DummyBrain, kein RL)
palaestrai experiment-start experiments/soja_arl_dummy.yaml

# Stale Prozesse bereinigen (nach Abbruechen noetig)
pkill -f "palaestrai" && pkill -f "spawn_main" && pkill -f "resource_tracker"
```

## Architektur

Drei Schichten mit steigenden Abhaengigkeiten:

```
provider_sim/
├── pdl/           Standalone PDL-Parser (nur PyYAML)
│   ├── model.py       Dataclasses: PdlDocument, Entity, Event, Cascade, ...
│   ├── parser.py      YAML → Dataclasses, Duration/Percentage-Parsing, Validierung
│   ├── condition.py   Condition-Expression-Parser (AST mit AND/OR)
│   └── errors.py      PdlParseError, PdlValidationError
│
├── sim/           Simulationsengine (PyYAML + NumPy)
│   ├── state.py       SupplyChainState, EntityState, EventState, build_state_from_pdl()
│   └── engine.py      SimulationEngine: 5-Phasen-Step
│
├── env/           palaestrAI Environment ABC (optionale Dep, Stub-Fallback)
│   ├── environment.py ProviderEnvironment(Environment): start_environment/update + reset_dict/step_dict
│   └── objectives.py  AttackerObjective / DefenderObjective (reward_id-basiert)
│
└── rl/            PPO-Agenten fuer ARL-Training
    ├── network.py     PPONet: shared MLP (99→128→64), policy head + value head
    ├── ppo_brain.py   PPOBrain: Trajektorie-Puffer + GAE + PPO-Update (Brain-Subprocess)
    └── ppo_muscle.py  PPOMuscle: Inference + Budget-Constraint (Muscle-Subprocess)
```

### Abhaengigkeitsrichtung

`pdl` → standalone, keine numpy/palaestrai-Imports
`sim` → importiert `pdl`, benoetigt numpy
`env` → importiert `pdl` + `sim`, palaestrai optional (Stub-Fallback)

### Schluesselklassen

| Klasse | Datei | Beschreibung |
|---|---|---|
| `PdlDocument` | `pdl/model.py` | Root-Dataclass, Index-Lookups (`entity_by_id()`, `event_by_id()`) |
| `load_pdl()` | `pdl/parser.py` | Haupteinstiegspunkt: str/Path/dict → PdlDocument |
| `parse_condition()` | `pdl/condition.py` | Condition-String → AST (`ActiveCheck`, `DurationCheck`, `AndExpr`, `OrExpr`) |
| `SupplyChainState` | `sim/state.py` | Simulationszustand: Entity/Event-States, Graph-Adjazenz, ConditionState-Protokoll |
| `SimulationEngine` | `sim/engine.py` | 5-Phasen-Step, topologische Sortierung, Impact-Stack |
| `ProviderEnvironment` | `env/environment.py` | palaestrAI Environment ABC: `start_environment()` → `EnvironmentBaseline`, `update()` → `EnvironmentState` |
| `AttackerObjective` | `env/objectives.py` | Filtert `reward.attacker` aus RewardInformation-Liste |
| `DefenderObjective` | `env/objectives.py` | Filtert `reward.defender` aus RewardInformation-Liste |

## Simulationsengine — 5-Phasen-Modell

`SimulationEngine.step()` fuehrt pro Tick aus:

1. **Phase 1 — Agent-Actions**: Attacker reduziert `entity.supply` (gewichtet mit `vulnerability`), Defender addiert
2. **Phase 2 — Event-Evaluierung**: Root-Events probabilistisch (`rng.random() < prob`), Condition-Events per AST. Attacker kann Root-Events forcieren
3. **Phase 3 — Impact-Stack**: Modifier-basiert, kein Per-Tick-Compounding. `effective_supply = Π(1 + modifier_i)`. Demand/Price direkt gesetzt
4. **Phase 4 — Flow-Propagation**: Topologische Sortierung (Kahn), `supply = min(intrinsic, mean(incoming))`. Dependency-Penalty bei gestoerter Abhaengigkeit
5. **Phase 5 — Health**: `health = 0.5*supply + 0.3*(1/price) + 0.2*min(demand, 1.0)`, clipped [0,1]

Natuerliche Recovery: 2%/Tick Richtung 1.0 wenn keine aktiven Events.

## PDL-Datenmodell

### Enums

- `EntityType`: manufacturer, commodity, infrastructure, service, region
- `EventType`: natural_disaster, market_shock, infrastructure_failure, regulatory, geopolitical, pandemic, cyber_attack
- `Criticality`: high, medium, low
- `Severity`: critical, high, medium, low

### Value Objects (frozen)

- `Duration(raw="90d", days=90.0)` — Einheiten: h (1/24), d (1), w (7), m (30), y (365)
- `Percentage(raw="-40%", decimal=-0.4)`

### Entity.extra

Entities haben ein `extra: Dict[str, Any]` fuer domainspezifische Zusatzfelder (z.B. `market_share`, `tier`, `diesel_reserve_hours`, `substitution_potential`). Diese sind nicht typisiert — Parser sammelt alle Felder ausserhalb der bekannten Menge.

### Event.is_root_event

Property: `True` wenn `trigger.probability` gesetzt und `trigger.condition` ist `None`. Root-Events werden probabilistisch evaluiert, alle anderen per Condition-AST.

## Condition-Grammatik

```
expr       := or_expr
or_expr    := and_expr (" OR " and_expr)*
and_expr   := atom (" AND " atom)*
atom       := EVENT_ID ".active" | EVENT_ID ".duration > " DURATION
```

Beispiele aus den Szenarien:
- `brazil_drought.active`
- `oil_mill_slowdown.active OR soy_export_reduction.active`
- `livestock_pressure.active AND consumer_substitution.active`
- `soy_export_reduction.active AND soy_export_reduction.duration > 30d`

Parsing via String-Split (`" OR "`, `" AND "`), dann Regex-Match pro Atom. Kein rekursiver Descent noetig — die PDL-Grammatik hat keine Klammerung.

## palaestrAI-Environment

`ProviderEnvironment` implementiert das palaestrAI `Environment` ABC. palaestrai ist eine optionale Abhaengigkeit — ohne Installation werden Stub-Klassen verwendet (`_HAS_PALAESTRAI`-Flag).

### API (palaestrAI-Orchestrator)

| Methode | Rueckgabe | Beschreibung |
|---|---|---|
| `start_environment()` | `EnvironmentBaseline` | Reset + Baseline mit `sensors_available`, `actuators_available`, `SimTime(ticks=0)` |
| `update(actuators: List[ActuatorInformation])` | `EnvironmentState` | Step: `sensor_information`, `rewards`, `done`, `SimTime` |

Konstruktor: `ProviderEnvironment(pdl_source, seed=0, max_ticks=365, uid="provider_env", broker_uri="")`

### API (Standalone, ohne Orchestrator)

| Methode | Rueckgabe | Beschreibung |
|---|---|---|
| `reset_dict()` | `(obs: Dict, rewards: Dict)` | Reset, gibt flache Dicts zurueck |
| `step_dict(actions: Dict)` | `(obs: Dict, rewards: Dict, done: bool)` | Step mit `{"attacker.<id>": float, "defender.<id>": float}` |

### palaestrAI-Typen

Sensoren liefern `SensorInformation(sensor_value, observation_space, sensor_id)`, Aktuatoren verwenden `ActuatorInformation(setpoint, action_space, actuator_id)`, Rewards sind `RewardInformation(reward_value, observation_space, reward_id)`. Spaces: `Box(low, high, shape=(1,), dtype=np.float32)` und `Discrete(n)`.

### Sensoren

Pro Entity 4 + pro Event 1 + 1 global:
- `entity.<id>.supply` — `Box(0, 2)`, Wert: `np.array([float], dtype=np.float32)`
- `entity.<id>.demand` — `Box(0, 3)`, Wert: `np.array([float], dtype=np.float32)`
- `entity.<id>.price` — `Box(0, 10)`, Wert: `np.array([float], dtype=np.float32)`
- `entity.<id>.health` — `Box(0, 1)`, Wert: `np.array([float], dtype=np.float32)`
- `event.<id>.active` — `Discrete(2)`, Wert: `int` (0 oder 1)
- `sim.tick` — `Box(0, max_ticks)`, Wert: `np.array([float], dtype=np.float32)`

### Aktuatoren

Pro Entity 2:
- `attacker.<entity_id>` — `Box(0, 1)` — Disruptionsstaerke
- `defender.<entity_id>` — `Box(0, 1)` — Verteidigungsstaerke

### Rewards (Zero-Sum)

- `reward.attacker` = `RewardInformation(np.array([mean(1 − health)]), Box(0,1))`
- `reward.defender` = `RewardInformation(np.array([mean(health)]), Box(0,1))`
- Summe ist immer 1.0

### Soja-Szenario: 99 Sensoren, 40 Aktuatoren

### Objectives (ARL)

`AttackerObjective` und `DefenderObjective` in `env/objectives.py` implementieren `_BaseObjective.internal_reward()`. Sie filtern die Reward-Liste nach `reward_id` (default `"reward.attacker"` bzw. `"reward.defender"`). Beide nutzen den gleichen Stub-Fallback-Mechanismus wie `environment.py`.

## Experimente (palaestrAI)

### Config-Generierung

```bash
# YAML-Konfiguration aus PDL-Szenario generieren
python experiments/generate_config.py <pdl_path> \
    --output experiments/soja_arl_dummy.yaml \
    --max-ticks 365 --episodes 1 --seed 42
```

`experiments/generate_config.py` liest ein PDL-Szenario, extrahiert alle Entities/Events und erzeugt ein vollstaendiges palaestrAI-Experiment-YAML mit Attacker/Defender-Agenten (DummyBrain/DummyMuscle fuer Baseline-Tests).

### Experiment ausfuehren

```bash
# Experiment starten (OHNE -vv!)
palaestrai experiment-start experiments/soja_arl_dummy.yaml

# Ergebnisse liegen in palaestrai.db (SQLite)
```

### Operationelle Hinweise

- **Kein `-vv`**: Verbose-Logging erzeugt 500+ MB Output mit binaeren DB-Daten und fuehrt zu I/O-Stagnation. Standard-Verbosity reicht.
- **Ports 4242/4243**: palaestrAI nutzt ZMQ-Messaging. Vor dem Start pruefen: `lsof -i :4242 -i :4243`. Alte Prozesse ggf. beenden.
- **Laufzeit**: Soja-Szenario mit 365 Ticks und DummyBrain dauert ca. 3 Minuten (~0.57s/Tick).
- **Kein Mid-Episode-Resume**: Abgebrochene Episoden starten von vorn. Nur abgeschlossene Episoden bleiben in der DB.
- **Ergebnis-DB**: `palaestrai.db` enthaelt `world_states` (Sensor-Dumps als JSON in `state_dump`), `muscle_actions`, `brain_states` etc.

## RL-Training (PPO)

### Architektur

`PPONet` ist ein geteiltes Actor-Critic-MLP. `PPOMuscle` laeuft als palaestrAI-Muscle-Subprocess und macht Inference. `PPOBrain` laeuft als separater Brain-Subprocess, puffert eine Episode und fuehrt dann PPO-Update durch.

**Warum Brain=MPS (lazy), Muscle=CPU (immer):**
Muscle-Tensors muessen ueber `aiomultiprocess`-Prozessgrenzen via Shared Memory. MPS-Tensors koennen das nicht. Brain initialisiert das Netz auf CPU (sicheres Pickling beim Spawn), verschiebt es lazy per `_ensure_on_device()` erst beim ersten PPO-Update nach dem Spawn.

### palaestrAI-Eigenheiten (kritisch)

- `episodes: N` liefert **nur 1 echte Trainings-Episode** + (N-1) spurious (2 Steps je). Ursache: `VanillaSimController` sendet `ex_termination=True` sofort nach Restart. Fix: `min_episode_steps=10` in PPOBrain ueberspringt Micro-Episoden.
- `store_model(path)` wird bei **jedem Tick** aufgerufen, nicht nur am Episode-Ende.
- `propose_actions()` Sensoren: erste Sensor-Daten koennen plain floats sein → `hasattr(s, 'sensor_value')` Guard noetig.
- Aktuator-Setpoint muss `np.array([float], dtype=np.float32)` sein, kein plain float.
- **Ports 4242/4243**: ZMQ-Broker. Nach Abbruechen immer `pkill -f palaestrai && pkill -f spawn_main` ausfuehren.

### Externer Trainings-Loop

Da palaestrAI kein echtes Multi-Episode-Training unterstuetzt, laeuft jede Episode als separater `palaestrai experiment-start`-Prozess:

```bash
python experiments/train_ppo.py 50   # 50 echte Trainings-Episoden
python experiments/train_ppo.py 50   # setzt automatisch fort (liest progress.json)
```

`experiments/checkpoints/progress.json` speichert den Fortschritt — Training ist mit Ctrl+C unterbrechbar und mit demselben Befehl fortsetzbar. Checkpoints: `checkpoints/attacker.pt` und `checkpoints/defender.pt` (werden nach jeder Episode ueberschrieben).

**Zum Neustart:** `rm experiments/checkpoints/*.pt experiments/checkpoints/progress.json`

### Soja-Szenario: Dimensionen

- 99 Sensoren (20 Entities × 4 + 18 Events + 1 Tick), 20 Aktuatoren pro Agent
- Attacker-Budget: 0.8, Defender-Budget: 0.4 (Softmax-normiert)
- Reward zero-sum: `attacker + defender = 1.0` immer

## Tests (95 Tests)

```
tests/
├── conftest.py              Fixtures: soja_path, soja_doc, soja_engine, any_scenario_path
├── test_pdl_parser.py       Duration/Percentage, Szenario-Laden, Validierung, Extra-Felder
├── test_condition.py        AST-Konstruktion, Auswertung mit MockState
├── test_sim_state.py        State-Init, Entity-Count, Adjazenz, ConditionState-Protokoll
├── test_sim_engine.py       Tick, Reset, Attacker/Defender, Kaskaden, 365-Tick
├── test_env_environment.py  4 Testklassen: ProviderEnvironment, PalaestrAIProtocol, DictInterface, UidPrepending (22 Tests)
├── test_objectives.py       AttackerObjective, DefenderObjective: Filterung, Default-Werte, Zero-Sum (11 Tests)
└── test_integration.py      Alle 9 Szenarien × 365 Ticks ohne Crash
```

`any_scenario_path` ist ein parametrisiertes Fixture ueber alle 9 PDL-YAML-Dateien in `04_Apps/pdl viewer/scenarios/`.

## Referenzdateien (ausserhalb dieses Repos)

- PDL-Szenarien: `04_Apps/pdl viewer/scenarios/s[1-9]-*.pdl.yaml`
- Szenario-Dokumentation: `04_Apps/pdl viewer/scenarios/PROVIDER-Szenarien-Dokumentation.md`

## Konventionen

- `from __future__ import annotations` in jeder Datei (Python 3.9-Kompatibilitaet)
- Dataclasses statt Pydantic — minimale Deps
- `_PRIVATE` Konstanten mit Underscore-Prefix
- Keine Docstrings an interne Hilfsfunktionen (`_parse_entity`, etc.) — nur an Public API
- Tests nutzen Klassen (`class TestFoo`) fuer Gruppierung, Methoden (`test_*`) fuer Cases
- Keine Mocks ausser `MockState` in `test_condition.py` — alle Tests laufen gegen echte PDL-Dateien
