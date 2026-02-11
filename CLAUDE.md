# CLAUDE.md — provider_sim

Kontextdatei fuer Claude Code. Dieses Paket ist der Python-Kern der PROVIDER-Simulation: PDL-Parser, Simulationsengine und palaestrAI-Environment.

## Sprache

Alle Kommunikation auf Deutsch. Code-Bezeichner, Docstrings und Kommentare bleiben Englisch.

## Kommandos

```bash
# Paket installieren (editable, mit Test-Deps)
pip install -e ".[dev]"

# Tests ausfuehren
pytest tests/ -v

# Schnelltest: Soja-Szenario laden
python -c "from provider_sim.pdl.parser import load_pdl; doc = load_pdl('/Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/pdl viewer/scenarios/s1-soja.pdl.yaml'); print(len(doc.entities))"

# 365-Tick-Simulation
python -c "from provider_sim.sim.engine import SimulationEngine; from provider_sim.pdl.parser import load_pdl; e = SimulationEngine(load_pdl('/Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/pdl viewer/scenarios/s1-soja.pdl.yaml')); [e.step() for _ in range(365)]; print('OK')"
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
└── env/           palaestrAI Environment ABC (optionale Dep, Stub-Fallback)
    └── environment.py ProviderEnvironment(Environment): start_environment/update + reset_dict/step_dict
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

## Tests

```
tests/
├── conftest.py              Fixtures: soja_path, soja_doc, soja_engine, any_scenario_path
├── test_pdl_parser.py       Duration/Percentage, Szenario-Laden, Validierung, Extra-Felder
├── test_condition.py        AST-Konstruktion, Auswertung mit MockState
├── test_sim_state.py        State-Init, Entity-Count, Adjazenz, ConditionState-Protokoll
├── test_sim_engine.py       Tick, Reset, Attacker/Defender, Kaskaden, 365-Tick
├── test_env_environment.py  4 Testklassen: ProviderEnvironment, PalaestrAIProtocol, DictInterface, UidPrepending (22 Tests)
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
