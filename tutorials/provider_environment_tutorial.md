# palaestrAI Tutorial: Custom Environment Integration

Dieses Tutorial zeigt, wie eine externe Simulation als palaestrAI-`Environment`
eingebunden wird. Als Referenzimplementierung dient die **PROVIDER Supply-Chain
Simulation** mit dem Soja-Futtermittel-Szenario (S1).

---

## 1. Überblick

Das PROVIDER-Projekt entwickelt ein Frühwarnsystem für Versorgungsengpässe in
Lieferketten ausserhalb klassischer KRITIS-Sektoren. Kernkomponente ist eine
agentenbasierte Simulation mit **Adversarial Resilience Learning (ARL)**: Ein
Attacker-Agent versucht, die Lieferkette zu destabilisieren; ein Defender-Agent
versucht, sie zu schützen.

**Was dieses Tutorial zeigt:**

1. Wie die palaestrAI `Environment`-ABC implementiert wird
2. Wie Sensoren, Aktuatoren und Rewards definiert werden
3. Wie ein Experiment-YAML für den palaestrAI-Orchestrator generiert wird
4. Wie das Zero-Sum-ARL-Muster mit Attacker/Defender-Objectives funktioniert

---

## 2. Voraussetzungen

```bash
# Paket installieren (editierbar, mit Test-Abhängigkeiten)
pip install -e ".[dev]"

# Optional: palaestrAI installieren (für Orchestrator-Betrieb)
pip install palaestrai
```

Ohne palaestrai läuft das Environment im **Standalone-Modus** mit Stub-Klassen
(kein Orchestrator, keine ZMQ-Kommunikation). Alle Tests und das Tutorial-Skript
funktionieren in beiden Modi.

---

## 3. Das palaestrAI Environment-ABC

Die palaestrAI-Basisklasse `Environment` definiert zwei Pflichtmethoden:

| Methode | Rückgabe | Beschreibung |
|---|---|---|
| `start_environment()` | `EnvironmentBaseline` | Reset der Simulation; liefert verfügbare Sensoren/Aktuatoren |
| `update(actuators)` | `EnvironmentState` | Empfängt Agenten-Aktionen, führt einen Simulations-Schritt aus, liefert neuen Zustand |

```python
from palaestrai.environment import Environment, EnvironmentBaseline, EnvironmentState
from palaestrai.agent import ActuatorInformation

class MyEnvironment(Environment):

    def start_environment(self) -> EnvironmentBaseline:
        # Simulation resetten, Sensoren/Aktuatoren registrieren
        ...

    def update(self, actuators: list[ActuatorInformation]) -> EnvironmentState:
        # Aktionen auswerten, Schritt ausführen, Zustand zurückgeben
        ...
```

Die `ProviderEnvironment`-Klasse in `provider_sim/env/environment.py` ist die
vollständige Referenzimplementierung dieses Musters.

---

## 4. Das Soja-Szenario (S1)

Das Szenario `scenarios/s1-soja.pdl.yaml` modelliert die europäische
Soja-Futtermittel-Lieferkette:

- **20 Entities** (Produzenten, Häfen, Verarbeiter, Endverbraucher)
- **18 Events** (Dürren, Marktschocks, Infrastrukturausfälle, ...)
- **Cascades** (Ereignisketten, z. B. Dürre → Ernteausfall → Exportreduktion)

```yaml
pdl_version: "1.0"
scenario:
  id: soy_feed_disruption
  name: "Soja-Futtermittel-Lieferkette"
  sector: agriculture
  criticality: high

entities:
  - id: brazil_farms
    type: region
    vulnerability: 0.6
    ...

events:
  - id: brazil_drought
    type: natural_disaster
    trigger:
      probability: 0.05
    impacts:
      - entity: brazil_farms
        supply_modifier: -40%
    duration: 90d
```

Die PDL-Sprache (Provider Domain Language) ist eine YAML-basierte DSL. Der
Parser `provider_sim.pdl.parser.load_pdl()` wandelt YAML in typisierte
Python-Dataclasses um.

---

## 5. Sensor / Aktuator / Reward-Design

### Sensoren

Pro Entity werden **4 kontinuierliche Sensoren** registriert, pro Event **1
diskreter Sensor**, plus ein globaler Tick-Sensor:

| Sensor-ID | Space | Bedeutung |
|---|---|---|
| `entity.<id>.supply` | `Box(0, 2)` | Normiertes Angebot (1.0 = Normalzustand) |
| `entity.<id>.demand` | `Box(0, 3)` | Normierte Nachfrage |
| `entity.<id>.price` | `Box(0, 10)` | Normierter Preis |
| `entity.<id>.health` | `Box(0, 1)` | Gesundheitsindex (composite score) |
| `event.<id>.active` | `Discrete(2)` | Ist das Ereignis aktiv? (0/1) |
| `sim.tick` | `Box(0, max_ticks)` | Aktueller Simulations-Tick |

Für das Soja-Szenario ergibt das: 20 × 4 + 18 + 1 = **99 Sensoren**.

### Aktuatoren

Pro Entity werden **2 Aktuatoren** registriert:

| Aktuator-ID | Space | Bedeutung |
|---|---|---|
| `attacker.<entity_id>` | `Box(0, 1)` | Disruptions-Intensität (0 = keine Aktion) |
| `defender.<entity_id>` | `Box(0, 1)` | Verteidigungs-Intensität |

Für das Soja-Szenario: 20 × 2 = **40 Aktuatoren**.

### Rewards (Zero-Sum)

| Reward-ID | Formel | Wert |
|---|---|---|
| `reward.attacker` | `mean(1 - health)` über alle Entities | hoch wenn Lieferkette gestört |
| `reward.defender` | `mean(health)` über alle Entities | hoch wenn Lieferkette stabil |

Summe ist immer **1.0** → echtes Zero-Sum-Spiel.

```python
# Beispiel: Rewards auslesen
state = env.update(actuators)
for reward in state.rewards:
    print(f"{reward.reward_id}: {float(reward.reward_value[0]):.4f}")
# Ausgabe:
# reward.attacker: 0.1823
# reward.defender: 0.8177
```

---

## 6. Standalone-Ausführung (ohne Orchestrator)

Das Skript `tutorials/run_provider_tutorial.py` demonstriert alle vier Schritte:

```bash
cd palestrai_simulation
python tutorials/run_provider_tutorial.py
```

**Erwartete Ausgabe (gekürzt):**

```
============================================================
ABSCHNITT 1 — PDL-Szenario laden
============================================================
Szenario-ID  : soy_feed_disruption
Entities     : 20
Events       : 18

============================================================
ABSCHNITT 2 — ProviderEnvironment instanziieren
============================================================
Sensoren    : 99
Aktuatoren  : 40

============================================================
ABSCHNITT 3 — Standalone-Demo: 5 Ticks
============================================================
Tick  attacker_reward  defender_reward  mean_supply  mean_health
   1         0.0959          0.9041       0.9878       0.9041
   ...

============================================================
ABSCHNITT 4 — palaestrAI-ABC-Demo
============================================================
start_environment(): sensors_available=99, actuators_available=40
update(): done=False, rewards=['reward.attacker', 'reward.defender']
Zero-Sum-Check: 1.0
```

### Manuelles Testen mit der Dict-API

```python
from provider_sim.env.environment import ProviderEnvironment

env = ProviderEnvironment("scenarios/s1-soja.pdl.yaml", seed=42, max_ticks=365)
obs, rewards = env.reset_dict()

for tick in range(10):
    actions = {f"attacker.{e.id}": 0.2 for e in env.doc.entities}
    actions.update({f"defender.{e.id}": 0.1 for e in env.doc.entities})
    obs, rewards, done = env.step_dict(actions)
    print(f"Tick {tick+1}: health={rewards['reward.defender']:.3f}")
    if done:
        break
```

---

## 7. Mit dem palaestrAI-Orchestrator

### Experiment-Config generieren

```bash
python experiments/generate_config.py scenarios/s1-soja.pdl.yaml \
    --output experiments/soja_arl_dummy.yaml \
    --max-ticks 365 --episodes 1 --seed 42
```

Der Generator liest das PDL-Szenario und erzeugt ein vollständiges
palaestrAI-Experiment-YAML mit:
- `ProviderEnvironment` als Environment
- `DummyBrain` / `DummyMuscle` für Baseline-Tests (Attacker + Defender)
- Alle 99 Sensoren und 40 Aktuatoren korrekt verdrahtet

### Experiment starten

```bash
# Ports prüfen (palaestrAI nutzt ZMQ auf 4242/4243)
lsof -i :4242 -i :4243

# Experiment starten (NICHT mit -vv! → I/O-Stagnation)
palaestrai experiment-start experiments/soja_arl_dummy.yaml
```

Das Experiment läuft ~3 Minuten für 365 Ticks. Ergebnisse landen in
`palaestrai.db` (SQLite):
- `world_states` — Sensor-Dumps (JSON in `state_dump`)
- `muscle_actions` — Aktuator-Werte pro Tick
- `brain_states` — Lernzustand

---

## 8. Zero-Sum ARL: Attacker vs. Defender

Das ARL-Muster in PROVIDER folgt dem **Minmax-Prinzip**:

```
Attacker maximiert:  mean(1 - health)  [Lieferkette stören]
Defender maximiert:  mean(health)      [Lieferkette schützen]
```

Die palaestrAI-Objectives in `provider_sim/env/objectives.py` filtern die
Reward-Liste nach `reward_id`:

```python
from provider_sim.env.objectives import AttackerObjective, DefenderObjective

attacker_obj = AttackerObjective()  # filtert "reward.attacker"
defender_obj = DefenderObjective()  # filtert "reward.defender"

# Intern in palaestrAI:
reward = attacker_obj.internal_reward(reward_information_list)
```

### Trainings-Loop (konzeptionell)

```
Episode 1:
  start_environment() → Baseline
  for tick in range(max_ticks):
      attacker_actions = attacker_brain.act(observations)
      defender_actions = defender_brain.act(observations)
      state = env.update(attacker_actions + defender_actions)
      attacker_brain.learn(attacker_reward)
      defender_brain.learn(defender_reward)

Episode 2, 3, ...:
  → Beide Agenten werden besser in ihren gegensätzlichen Zielen
```

### Interpretierbarkeit

Der `health`-Index fasst den Zustand einer Entity zusammen:

```
health = 0.5 × supply + 0.3 × (1/price) + 0.2 × min(demand, 1.0)
```

Ein `mean(health) > 0.8` bedeutet: Die Lieferkette ist weitgehend stabil.
Ein `mean(health) < 0.5` signalisiert einen kritischen Versorgungsengpass.

---

## Weiterführende Informationen

- **Architektur:** `CLAUDE.md` im Stammverzeichnis des Pakets
- **PDL-Grammatik:** Condition-Ausdrücke, Duration-Parsing, alle 9 Szenarien
- **Tests:** `tests/` — 95 Tests für Parser, Engine und Environment
- **Referenz-Szenarien:** `scenarios/` (Soja, Halbleiter, Pharma, Düngemittel, ...)
