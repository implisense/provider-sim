# S1 KG-Schocks → PDL-Events Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Einen Adapter implementieren, der KG-Schocks aus `coypu-kg-analyser` (JSON `list[dict]`) in synthetische PDL-Events für `palestrai_simulation` umwandelt.

**Architecture:** Der Adapter lebt in `provider_sim/adapters/kg_shocks.py` und importiert nur aus `provider_sim.pdl.model` — keine Abhängigkeit zu `coypu_kg_analyser`. KG-Schocks mit `magnitude=1.0` (kein Schock) werden übersprungen. Die Funktion `apply_kg_shocks()` injiziert die generierten Events an den Anfang des `PdlDocument.events`-Liste, damit sie bei Tick 0 mit `probability=1.0` aktiviert werden.

**Tech Stack:** Python 3.9, `provider_sim.pdl.model` (Event, Trigger, Impact, Percentage, Duration, EventType), `copy.deepcopy`, pytest

**Entity-ID-Mapping S1 (KG → PDL):**

| KG `target_id` | PDL `entity_id` | Hinweis |
|---|---|---|
| `bra_soy_farm` | `brazil_farms` | |
| `arg_soy_farm` | `argentina_farms` | |
| `usa_soy_farm` | `us_farms` | |
| `santos_port` | `santos_port` | ✓ gleich |
| `paranagua_port` | `paranagua_port` | ✓ gleich |
| `rotterdam_port` | `rotterdam_port` | ✓ gleich |
| `hamburg_port` | `hamburg_port` | ✓ gleich |
| `us_gulf_ports` | `us_gulf_ports` | ✓ gleich |
| `eu_oil_mills` | `eu_oil_mills` | ✓ gleich |
| `feed_mills` | `feed_mills` | ✓ gleich |
| `fertilizer_input` | `fertilizer_supply` | Namen verschieden |
| `energy_input` | `gas_supply` | Namen verschieden |
| `deu_soy_farm` | — | **kein PDL-Äquivalent, überspringen** |
| `rosario_port` | — | **kein PDL-Äquivalent, überspringen** |

---

### Task 1: `kg_shocks_to_events()` — Kernfunktion

**Files:**
- Create: `provider_sim/adapters/__init__.py`
- Create: `provider_sim/adapters/kg_shocks.py`
- Test: `tests/test_kg_shocks_adapter.py`

**Step 1: Verzeichnis anlegen und Test schreiben**

Erstelle `provider_sim/adapters/__init__.py` (leer):
```python
```

Erstelle `tests/test_kg_shocks_adapter.py`:
```python
"""Tests für KG-Schocks → PDL-Events Adapter."""
from __future__ import annotations

import pytest

from provider_sim.adapters.kg_shocks import kg_shocks_to_events, S1_KG_TO_PDL
from provider_sim.pdl.model import Event, EventType


def test_capacity_shock_maps_to_supply_event():
    """capacity-Schock mit magnitude=0.6 → Event mit supply=-40%."""
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.6}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert len(events) == 1
    ev = events[0]
    assert ev.id == "kg_shock_brazil_farms"
    assert ev.trigger.target == "brazil_farms"
    assert ev.trigger.probability == 1.0
    assert ev.trigger.condition is None
    assert ev.impact.supply is not None
    assert abs(ev.impact.supply.decimal - (-0.4)) < 1e-9
    assert ev.impact.price is None


def test_price_shock_maps_to_price_event():
    """price-Schock mit magnitude=1.5 → Event mit price=+50%."""
    shocks = [{"target_id": "fertilizer_input", "shock_type": "price", "magnitude": 1.5}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert len(events) == 1
    ev = events[0]
    assert ev.id == "kg_shock_fertilizer_supply"
    assert ev.trigger.target == "fertilizer_supply"
    assert ev.impact.price is not None
    assert abs(ev.impact.price.decimal - 0.5) < 1e-9
    assert ev.impact.supply is None


def test_magnitude_one_skipped_for_capacity():
    """capacity-Schock mit magnitude=1.0 → kein Event (kein Schock)."""
    shocks = [{"target_id": "santos_port", "shock_type": "capacity", "magnitude": 1.0}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert events == []


def test_unmapped_target_skipped():
    """target_id ohne PDL-Äquivalent → kein Event."""
    shocks = [{"target_id": "deu_soy_farm", "shock_type": "capacity", "magnitude": 0.6}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert events == []


def test_multiple_shocks_all_mapped():
    """Vollständige S1-Schockliste: 12 mappable, 2 nicht."""
    shocks = [
        {"target_id": "bra_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},
        {"target_id": "arg_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},
        {"target_id": "usa_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},
        {"target_id": "deu_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},   # skip
        {"target_id": "fertilizer_input", "shock_type": "price",    "magnitude": 1.5},
        {"target_id": "energy_input",     "shock_type": "price",    "magnitude": 1.5},
        {"target_id": "santos_port",      "shock_type": "capacity", "magnitude": 1.0},   # skip (=1.0)
        {"target_id": "rosario_port",     "shock_type": "capacity", "magnitude": 0.85},  # skip (unmapped)
        {"target_id": "paranagua_port",   "shock_type": "capacity", "magnitude": 0.85},
        {"target_id": "rotterdam_port",   "shock_type": "capacity", "magnitude": 1.0},   # skip (=1.0)
        {"target_id": "hamburg_port",     "shock_type": "capacity", "magnitude": 1.0},   # skip (=1.0)
        {"target_id": "us_gulf_ports",    "shock_type": "capacity", "magnitude": 1.0},   # skip (=1.0)
        {"target_id": "eu_oil_mills",     "shock_type": "capacity", "magnitude": 0.8},
        {"target_id": "feed_mills",       "shock_type": "capacity", "magnitude": 0.95},
    ]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    # deu_soy_farm: kein PDL-Mapping → skip
    # rosario_port: kein PDL-Mapping → skip
    # santos_port, rotterdam_port, hamburg_port, us_gulf_ports: magnitude=1.0 → skip
    # Verbleibend: bra, arg, usa, fertilizer, energy, paranagua, eu_oil_mills, feed_mills = 8
    assert len(events) == 8
    ids = {ev.id for ev in events}
    assert "kg_shock_brazil_farms" in ids
    assert "kg_shock_fertilizer_supply" in ids
    assert "kg_shock_paranagua_port" in ids


def test_event_has_correct_type_and_duration():
    """Events sind vom Typ MARKET_SHOCK und haben duration=365d."""
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.7}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    ev = events[0]
    assert ev.type == EventType.MARKET_SHOCK
    assert ev.impact.duration is not None
    assert ev.impact.duration.days == 365


def test_custom_duration():
    """duration_days-Parameter wirkt sich auf Event-Duration aus."""
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.7}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL, duration_days=90)
    assert events[0].impact.duration.days == 90
```

**Step 2: Test laufen lassen (erwartet: ImportError)**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
python -m pytest tests/test_kg_shocks_adapter.py -v 2>&1 | head -20
```

Erwartet: `ModuleNotFoundError: No module named 'provider_sim.adapters'`

**Step 3: Implementierung schreiben**

Erstelle `provider_sim/adapters/kg_shocks.py`:

```python
"""Adapter: KG-Schocks (JSON dict) → PDL-Events für palestrai_simulation."""
from __future__ import annotations

from provider_sim.pdl.model import (
    Duration,
    Event,
    EventType,
    Impact,
    Percentage,
    Trigger,
)

# Mapping: coypu-kg-analyser target_id → provider_sim entity_id (S1 Soja)
S1_KG_TO_PDL: dict[str, str] = {
    "bra_soy_farm":     "brazil_farms",
    "arg_soy_farm":     "argentina_farms",
    "usa_soy_farm":     "us_farms",
    "santos_port":      "santos_port",
    "paranagua_port":   "paranagua_port",
    "rotterdam_port":   "rotterdam_port",
    "hamburg_port":     "hamburg_port",
    "us_gulf_ports":    "us_gulf_ports",
    "eu_oil_mills":     "eu_oil_mills",
    "feed_mills":       "feed_mills",
    "fertilizer_input": "fertilizer_supply",
    "energy_input":     "gas_supply",
    # deu_soy_farm: kein PDL-Äquivalent
    # rosario_port:  kein PDL-Äquivalent
}


def kg_shocks_to_events(
    shocks: list[dict],
    id_mapping: dict[str, str] | None = None,
    duration_days: int = 365,
    reference: str = "CoyPu KG Live-Abfrage",
) -> list[Event]:
    """Wandelt KG-Schocks in synthetische PDL-Events um.

    Regeln:
    - target_ids ohne Mapping werden übersprungen.
    - capacity-Schocks mit magnitude == 1.0 werden übersprungen (kein Schock).
    - price-Schocks mit magnitude == 1.0 werden übersprungen.
    - Alle generierten Events haben trigger.probability = 1.0 (immer aktiv).
    """
    mapping = id_mapping if id_mapping is not None else S1_KG_TO_PDL
    duration = Duration(raw=f"{duration_days}d", days=float(duration_days))
    events: list[Event] = []

    for shock in shocks:
        target_id = shock.get("target_id", "")
        shock_type = shock.get("shock_type", "")
        magnitude: float = shock.get("magnitude", 1.0)

        pdl_id = mapping.get(target_id)
        if pdl_id is None:
            continue

        if magnitude == 1.0:
            continue

        delta = magnitude - 1.0  # 0.6 → -0.4; 1.5 → +0.5
        sign = "+" if delta >= 0 else ""
        pct_raw = f"{sign}{int(round(delta * 100))}%"
        pct = Percentage(raw=pct_raw, decimal=delta)

        if shock_type == "capacity":
            impact = Impact(supply=pct, duration=duration)
        elif shock_type == "price":
            impact = Impact(price=pct, duration=duration)
        else:
            continue

        events.append(Event(
            id=f"kg_shock_{pdl_id}",
            name=f"KG-Schock: {pdl_id}",
            type=EventType.MARKET_SHOCK,
            trigger=Trigger(target=pdl_id, probability=1.0),
            impact=impact,
            reference=reference,
        ))

    return events
```

**Step 4: Tests laufen lassen**

```bash
python -m pytest tests/test_kg_shocks_adapter.py -v
```

Erwartet: 7/7 PASSED

**Step 5: Commit**

```bash
git add provider_sim/adapters/__init__.py provider_sim/adapters/kg_shocks.py tests/test_kg_shocks_adapter.py
git commit -m "feat: KG-Schocks → PDL-Events Adapter (S1 Mapping)"
```

---

### Task 2: `apply_kg_shocks()` — PDL-Dokument-Integration

**Files:**
- Modify: `provider_sim/adapters/kg_shocks.py` (Funktion ergänzen)
- Test: `tests/test_kg_shocks_adapter.py` (Klasse ergänzen)

**Step 1: Zusätzliche Tests in `tests/test_kg_shocks_adapter.py` ergänzen**

Am Ende der Datei anhängen:

```python
# --- apply_kg_shocks Tests ---

from copy import deepcopy
from provider_sim.adapters.kg_shocks import apply_kg_shocks
from provider_sim.pdl.parser import load_pdl

SOJA_PDL = "/Users/aschaefer/Projekte/Forschung/PROVIDER/06_Szenarien/scenarios/s1-soja.pdl.yaml"


@pytest.fixture
def soja_doc():
    return load_pdl(SOJA_PDL)


def test_apply_kg_shocks_prepends_events(soja_doc):
    """KG-Events werden vor die bestehenden PDL-Events eingefügt."""
    original_event_count = len(soja_doc.events)
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.6}]
    new_doc = apply_kg_shocks(soja_doc, shocks, id_mapping=S1_KG_TO_PDL)
    assert len(new_doc.events) == original_event_count + 1
    assert new_doc.events[0].id == "kg_shock_brazil_farms"


def test_apply_kg_shocks_does_not_mutate_original(soja_doc):
    """Originaldokument wird nicht verändert (deepcopy)."""
    original_count = len(soja_doc.events)
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.6}]
    apply_kg_shocks(soja_doc, shocks, id_mapping=S1_KG_TO_PDL)
    assert len(soja_doc.events) == original_count


def test_apply_kg_shocks_engine_runs(soja_doc):
    """SimulationEngine läuft 5 Ticks mit KG-injizierten Events ohne Fehler."""
    from provider_sim.sim.engine import SimulationEngine
    shocks = [
        {"target_id": "bra_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},
        {"target_id": "fertilizer_input", "shock_type": "price",    "magnitude": 1.5},
    ]
    new_doc = apply_kg_shocks(soja_doc, shocks, id_mapping=S1_KG_TO_PDL)
    engine = SimulationEngine(new_doc, seed=42)
    for _ in range(5):
        engine.step()
    assert engine.state.tick == 5
    # KG-Schock aktiv → supply von brazil_farms kleiner als 1.0
    assert engine.state.entities["brazil_farms"].supply < 1.0
```

**Step 2: Test laufen lassen (erwartet: ImportError auf apply_kg_shocks)**

```bash
python -m pytest tests/test_kg_shocks_adapter.py::test_apply_kg_shocks_prepends_events -v
```

Erwartet: `ImportError: cannot import name 'apply_kg_shocks'`

**Step 3: `apply_kg_shocks()` in `provider_sim/adapters/kg_shocks.py` ergänzen**

Am Ende der Datei anhängen:

```python
from copy import deepcopy
from provider_sim.pdl.model import PdlDocument


def apply_kg_shocks(
    doc: PdlDocument,
    shocks: list[dict],
    id_mapping: dict[str, str] | None = None,
    duration_days: int = 365,
    reference: str = "CoyPu KG Live-Abfrage",
) -> PdlDocument:
    """Gibt eine Kopie von doc zurück, in der KG-Schocks als Events vorne stehen.

    Das Originaldokument wird nicht verändert.
    """
    kg_events = kg_shocks_to_events(
        shocks,
        id_mapping=id_mapping,
        duration_days=duration_days,
        reference=reference,
    )
    new_doc = deepcopy(doc)
    new_doc.events = kg_events + new_doc.events
    return new_doc
```

**Step 4: Alle Tests laufen lassen**

```bash
python -m pytest tests/test_kg_shocks_adapter.py -v
```

Erwartet: 10/10 PASSED

**Step 5: Gesamte Testsuite prüfen**

```bash
python -m pytest --tb=short -q
```

Erwartet: Alle bestehenden Tests weiterhin PASSED, keine Regressions.

**Step 6: Commit**

```bash
git add provider_sim/adapters/kg_shocks.py tests/test_kg_shocks_adapter.py
git commit -m "feat: apply_kg_shocks() injiziert KG-Events in PDL-Dokument"
```

---

### Task 3: Nutzungsbeispiel in README/Kommentar

**Files:**
- Modify: `provider_sim/adapters/kg_shocks.py` (Modulheader mit Beispiel)

**Step 1: Docstring am Dateianfang ergänzen**

Den Modulkommentar in `provider_sim/adapters/kg_shocks.py` erweitern (nach der bestehenden Docstring-Zeile, vor den Imports):

```python
"""Adapter: KG-Schocks (JSON dict) → PDL-Events für palestrai_simulation.

Typischer Workflow:

    # 1. KG-Schocks abrufen (coypu-kg-analyser)
    import json, subprocess
    raw = subprocess.run(
        ["python", "-m", "coypu_kg_analyser", "parametrize-s1"],
        capture_output=True, text=True,
    )
    shocks = json.loads(raw.stdout)["shocks"]

    # 2. PDL-Dokument laden und Schocks injizieren
    from provider_sim.pdl.parser import load_pdl
    from provider_sim.adapters.kg_shocks import apply_kg_shocks, S1_KG_TO_PDL

    doc = load_pdl("scenarios/s1-soja.pdl.yaml")
    parametrized_doc = apply_kg_shocks(doc, shocks, id_mapping=S1_KG_TO_PDL)

    # 3. Simulation starten
    from provider_sim.sim.engine import SimulationEngine
    engine = SimulationEngine(parametrized_doc, seed=42)
    for _ in range(365):
        engine.step()

Nicht gemappte target_ids (z.B. 'deu_soy_farm', 'rosario_port') werden
stillschweigend übersprungen. Magnitude == 1.0 erzeugt ebenfalls kein Event.
"""
```

**Step 2: Doctest im Modul-Header prüfen (kein Test nötig — nur Lesbarkeit)**

```bash
python -c "from provider_sim.adapters.kg_shocks import apply_kg_shocks; print('OK')"
```

Erwartet: `OK`

**Step 3: Commit**

```bash
git add provider_sim/adapters/kg_shocks.py
git commit -m "docs: Nutzungsbeispiel in kg_shocks.py Modulheader"
```

---

## Hinweise für den Implementierenden

**Wichtig: `palestrai_simulation`-Paket muss installiert sein:**
```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
pip install -e ".[dev]" --user
```

**Pfad zur PDL-Szenario-Datei in Tests:**
`/Users/aschaefer/Projekte/Forschung/PROVIDER/06_Szenarien/scenarios/s1-soja.pdl.yaml`
(außerhalb des palestrai_simulation-Repos — Pfad ist absolut hardcodiert im Test, ist OK für Integrationstests)

**Keine Abhängigkeit zu `coypu_kg_analyser`** — der Adapter nimmt nur `list[dict]`, nicht `S1ParametrizerResult`. Das erlaubt unabhängige Nutzung und Tests.

**Skalierung auf weitere Szenarien:** Später können `S10_KG_TO_PDL` etc. analog in derselben Datei ergänzt werden.
