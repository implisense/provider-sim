# PROVIDER — AP3 Simulation & KG-Anreicherung
## Technischer Report | März 2026

---

## 1. Überblick

Dieser Report dokumentiert den aktuellen Stand von **Arbeitspaket 3 (Dynamisch instanziierbare Simulationsumgebung)** und die Integration empirischer Handelsdaten aus dem **CoyPu Knowledge Graph** in die PROVIDER-Simulation. Schwerpunkte:

- Architektur der Simulationsengine (`provider_sim`)
- Methodik und Ergebnisse der KG-Anreicherung (BACI + GTA)
- Verifikation am Soja-Szenario (s1-soja_enriched.pdl.yaml)
- Nächste Schritte

---

## 2. Simulationsarchitektur (AP3)

### 2.1 Paketstruktur

```
provider_sim/
├── pdl/       PDL-Parser (nur PyYAML, keine externen Deps)
├── sim/       5-Phasen-Simulationsengine (NumPy)
├── env/       palaestrAI-Environment (optionale Dep, Stub-Fallback)
└── rl/        PPO-Agenten (PyTorch)
```

Die Abhängigkeitsrichtung ist strikt: `pdl` → `sim` → `env` → `rl`. Der `pdl`-Layer ist standalone lauffähig.

### 2.2 PDL — PROVIDER Domain Language

Szenarien werden als YAML-Dateien (`.pdl.yaml`) beschrieben. Das Modell umfasst:

| Konzept | Beschreibung |
|---------|-------------|
| `Entity` | Knoten der Lieferkette (Farms, Ports, Mühlen, …) mit `supply`, `demand`, `price`, `vulnerability` |
| `SupplyChain` | Gerichtete Kante zwischen Entities mit `volume` und `reliability` |
| `Event` | Probabilistisches oder konditioniertes Störungsereignis |
| `Substitution` | Alternativer Lieferant/Produkt mit `coverage`, `cost_delta`, `ramp_up` |
| `Cascade` | Vordefinierte Ereignisketten (zeitlich gestaffelt, mit Konfidenzwert) |

**Condition-Grammatik** (kein Klammer-Parsing nötig):

```
event_a.active AND event_b.active
event_a.active OR event_b.active
event_a.active AND event_a.duration > 30d
```

### 2.3 5-Phasen-Simulationsengine

`SimulationEngine.step()` führt pro Tick folgende Phasen aus:

| Phase | Beschreibung |
|-------|-------------|
| **1 — Agent-Actions** | Attacker reduziert `entity.supply` (gewichtet mit `vulnerability`), Defender addiert |
| **2 — Event-Evaluierung** | Root-Events probabilistisch (`rng.random() < prob`), Condition-Events per AST |
| **3 — Impact-Stack** | `effective_supply = Π(1 + modifier_i)` — kein Per-Tick-Compounding |
| **4 — Flow-Propagation** | Topologische Sortierung (Kahn), `supply = min(intrinsic, mean(incoming))` |
| **5 — Health** | `health = 0.5·supply + 0.3·(1/price) + 0.2·min(demand, 1.0)`, clipped [0, 1] |

**Natürliche Recovery:** +2 %/Tick Richtung 1.0, wenn keine aktiven Events.

### 2.4 palaestrAI-Integration

`ProviderEnvironment` implementiert das palaestrAI `Environment`-ABC und stellt zwei Schnittstellen bereit:

**Standalone (ohne Orchestrator):**
```python
env = ProviderEnvironment(pdl_source, seed=0, max_ticks=365)
obs, rewards = env.reset_dict()
obs, rewards, done = env.step_dict({"attacker.<id>": 0.5, ...})
```

**palaestrAI-Orchestrator:**
```python
baseline = env.start_environment()   # → EnvironmentBaseline
state    = env.update(actuators)     # → EnvironmentState
```

### 2.5 Soja-Szenario — Dimensionen

| Metrik | Original | Angereichert |
|--------|----------|-------------|
| Entities | 20 | 20 |
| Events | 18 | 55 (+37 GTA) |
| Sensoren | 99 | 136 |
| Aktuatoren | 40 | 40 |
| Kaskaden | 3 | 3 |
| Substitutionen | 7 | 7 (Werte aktualisiert) |

**Sensor-Zusammensetzung (angereichert):**
- 80 Entity-Sensoren (20 × supply, demand, price, health)
- 55 Event-Sensoren (18 original + 37 GTA)
- 1 globaler Tick-Sensor

**Reward (Zero-Sum):**
- `reward.attacker = mean(1 − health)` ∈ [0, 1]
- `reward.defender = mean(health)` ∈ [0, 1]
- Summe ist immer 1,0

---

## 3. KG-Anreicherung (BACI + GTA)

### 3.1 Datenquellen

Beide Datensätze wurden via SPARQL aus dem **CoyPu Knowledge Graph** (`copper.coypu.org/coypu`) abgefragt:

| Datensatz | KG-Pfad | Inhalt |
|-----------|---------|--------|
| **BACI** | `https://data.coypu.org/trade/baci/` | Bilaterale Exportvolumina (t/Jahr, kUSD) je (Land, HS-4-Code, Jahr) |
| **GTA** | GTA-Schema (`schema.coypu.org/gta#`) | Handelspolitische Interventionen mit Typ, Evaluierung (Red/Amber/Green), Jahr |

**Abdeckung der SPARQL-Abfrage:**

| Dimension | Scope |
|-----------|-------|
| Länder | ARG, BRA, USA, RUS, CHN, IND, IDN |
| HS-Kapitel | 10 (Getreide), 12 (Ölsaaten), 15 (Fette/Öle), 23 (Ölkuchenrückstände) |
| GTA-Interventionstypen | ExportBan, ExportQuota, ExportTax, ExportLicensingRequirement, ImportTariff, ImportBan, ImportLicensingRequirement, ImportQuota |
| BACI-Zeitraum | 2021 (HS-12 vollständig verfügbar) |
| GTA-Zeitraum | 2021–2024 |

**Recall-Metriken:**
- 449 BACI-Einträge im Scope
- 236 Einträge mit GTA-Match (53 % Entry-Recall, 82 % Volumen-Recall)

### 3.2 Anreicherungs-Pipeline (`enrich_pdl_from_kg.py`)

Das Skript ist unabhängig von `provider_sim` (nur PyYAML + csv aus Standardbibliothek) und führt drei Anreicherungsschritte durch:

#### Schritt 1: Entity.extra — BACI-Exportvolumina

Für jede kartierte Entity werden reale Handelsvolumina aus BACI 2021 eingetragen:

```yaml
extra:
  baci_export_value_kusd:    48212779998.0   # Exportwert in kUSD
  baci_export_volume_t_year: 105630580.0     # Jahresvolumen in Tonnen
  baci_capacity_t_day:       289398.8        # Tageskapazität (÷ 365)
  baci_year: "2021"
  baci_country: BRA
  baci_hs4_codes: [1201, 2304]
```

**Kartierung (Entity → Land, HS-4-Codes):**

| PDL-Entity | Land | HS-4-Codes |
|-----------|------|-----------|
| `brazil_farms` | BRA | 1201, 2304 |
| `argentina_farms` | ARG | 1201, 1202, 1507, 2304 |
| `us_farms` | USA | 1201, 2304 |

#### Schritt 2: Substitution.coverage — BACI-Quotienten

Die Substitutionsdeckung wird aus dem Verhältnis der BACI-Exportvolumina berechnet:

```
coverage = min(dst_export_tons / src_export_tons, 1.0)
```

**Ergebnisse (Hauptprodukt HS-1201 Soybeans):**

| Substitution | Formel | Neu | Alt |
|-------------|--------|-----|-----|
| `sub_supplier_argentina` | ARG/1201 ÷ BRA/1201 = 4,8 Mio. ÷ 88,2 Mio. | **0.055** | 0.15 |
| `sub_supplier_usa` | USA/1201 ÷ BRA/1201 = 54,1 Mio. ÷ 88,2 Mio. | **0.613** | 0.25 |

**Interpretation:** Argentinien kann im Störungsfall nur ~5,5 % des brasilianischen Soja-Exports ersetzen, die USA hingegen ~61 %. Dies ist eine deutliche Korrektur der ursprünglichen Schätzwerte.

#### Schritt 3: GTA-Events — Regulatorische Störungsereignisse

Für jede (Land, HS-4, Interventionstyp)-Kombination mit P(Red) ≥ 0,15 wird ein neues PDL-Event erzeugt:

```
P(Red) = Anzahl Jahre mit ≥1 Red-Bewertung / Gesamtjahre (2021–2024)
```

**Supply-Schocks je Interventionstyp:**

| Typ | Impact |
|----|--------|
| ExportBan | −30 % |
| ImportBan | −20 % |
| ExportQuota | −15 % |
| ImportQuota | −10 % |
| ExportTax | −10 % |
| ExportLicensingRequirement | −8 % |
| ImportLicensingRequirement | −5 % |
| ImportTariff | +5 % |

Alle GTA-Events haben eine Dauer von 180 Tagen (`duration: 180d`).

**Schwellenwert:** `--min-prob 0.15` (parametrisierbar via CLI)

### 3.3 Erzeugte GTA-Events (Soja-Szenario)

37 neue regulatorische Events wurden generiert:

| Land | Anzahl | Haupttypen |
|------|--------|-----------|
| ARG | 28 | ExportQuota, ExportTax, ExportLicensingRequirement |
| USA | 9 | ImportTariff, ExportLicensingRequirement |

**Ausgewählte Events mit höchsten Wahrscheinlichkeiten:**

| Event-ID | P(Red) | Supply-Impact | Produkt |
|----------|--------|--------------|---------|
| `gta_arg_exportquota_1005` | 0.50 | −15 % | Mais |
| `gta_arg_exporttax_1202` | 0.50 | −10 % | Erdnüsse |
| `gta_arg_exportlicensingrequirement_1001` | 0.25 | −8 % | Weizen |
| `gta_usa_importtariff_1201` | 0.25 | +5 % | Soybeans |

---

## 4. Simulationsvalidierung

### 4.1 Episode: s1-soja_enriched, 365 Ticks, DummyBrain

**Konfiguration:**
- PDL: `s1-soja_enriched.pdl.yaml` (20 Entities, 55 Events)
- Agenten: DummyBrain/DummyMuscle (Zufallsaktionen)
- Seed: 42, max_ticks: 365

**Ergebnisse:**

| Metrik | Wert |
|--------|------|
| Mean Health (Episode) | **0.657** |
| Min Health | **0.596** |
| Final Health (Tick 365) | **0.656** |
| GTA-Events ausgelöst | **37 / 37 (100 %)** |
| Attacker Reward Ø | **0.343** |

**Health-Verlauf (ausgewählte Ticks):**

| Tick | Mean Health | Attacker Reward | Aktive Events |
|------|------------|----------------|---------------|
| 1 | 0.973 | 0.027 | 10 |
| 30 | 0.652 | 0.348 | 54 |
| 60 | 0.656 | 0.344 | 55 |
| 120 | 0.647 | 0.353 | 55 |
| 180 | 0.656 | 0.344 | 53 |
| 300 | 0.602 | 0.398 | 52 |
| 365 | 0.656 | 0.344 | 44 |

**Beobachtung:** Ab Tick 30 sind nahezu alle Events dauerhaft aktiv. Der Gesundheitswert stabilisiert sich bei ~0.65 — das System kommt nicht unter 0.60 ohne gezielten Angreifer. Dies ist plausibel für ein Dummy-Baseline-Experiment.

### 4.2 Empirische Kalibrierung der BACI-Volumina

| Entity | BACI-Volumen (t/Jahr) | Tageskapazität (t/Tag) | Exportwert (kUSD) |
|--------|----------------------|----------------------|------------------|
| brazil_farms (BRA) | 105.630.580 | 289.399 | 48.212.780.000 |
| us_farms (USA) | 64.119.820 | 175.671 | 34.393.554.000 |
| argentina_farms (ARG) | 32.021.576 | 87.730 | 18.280.821.000 |

Brasilien dominiert den globalen Sojamarkt (2021) mit ~51 % des Exportvolumens dieser drei Länder, USA mit ~31 %, Argentinien mit ~16 %.

---

## 5. Technischer Hinweis: Bekannte Einschränkungen

| Thema | Status |
|-------|--------|
| BACI-Jahresabdeckung | Nur 2021 vollständig für HS-12 im KG; 2018–2020 haben andere HS-Kapitel |
| GTA-Zeitraum | 4 Jahre (2021–2024) — P(Red)-Schätzungen mit geringer statistischer Basis |
| Entity-Mapping | Aktuell nur 3 von 20 Entities kartiert (brazil_farms, argentina_farms, us_farms) |
| palaestrai CLI | Click-Version-Konflikt (8.1.8 vs. 8.0.4) — workaround: Python-API direkt |
| Event-Persistenz | GTA-Events mit hoher Wahrscheinlichkeit bleiben dauerhaft aktiv (keine Abklingzeit) |

---

## 6. CLI-Referenz

```bash
# PDL-Szenario mit KG-Daten anreichern
python experiments/enrich_pdl_from_kg.py \
    /path/to/scenario.pdl.yaml \
    /path/to/baci_gta_kg.csv \
    --output /path/to/scenario_enriched.pdl.yaml \
    --min-prob 0.15

# palaestrAI-Experiment-Config generieren
python experiments/generate_config.py \
    scenarios/s1-soja_enriched.pdl.yaml \
    --output experiments/soja_enriched_dummy.yaml \
    --max-ticks 365 --episodes 1 --seed 42

# Episode starten (Python-API, ohne palaestrai CLI)
python -c "
from provider_sim.env.environment import ProviderEnvironment
env = ProviderEnvironment('scenarios/s1-soja_enriched.pdl.yaml', seed=42, max_ticks=365)
obs, _ = env.reset_dict()
for _ in range(365):
    obs, rewards, done = env.step_dict({})
    if done: break
"
```

---

## 7. Nächste Schritte

| Priorität | Aufgabe | AP |
|-----------|---------|---|
| Hoch | Entity-Mapping auf alle 20 Entities ausweiten (Ports, Mühlen, Verbraucher) | AP3 |
| Hoch | BACI-Daten für weitere 8 PDL-Szenarien (Halbleiter, Pharma, Düngemittel, …) | AP2/AP3 |
| Mittel | GTA-Events mit Abklingzeit / Recovery-Logik versehen | AP3 |
| Mittel | PPO-Training auf angereichertem Soja-Szenario (136 Sensoren) | AP5 |
| Niedrig | BACI-Jahresabdeckung 2018–2020 im KG schließen | AP2 |
| Niedrig | Click-Version-Konflikt in palaestrai beheben | AP3 |

---

## 8. Artefakte

| Datei | Beschreibung |
|-------|-------------|
| `experiments/enrich_pdl_from_kg.py` | KG-Anreicherungs-Skript (BACI + GTA → PDL) |
| `scenarios/s1-soja_enriched.pdl.yaml` | Angereichertes Soja-Szenario (55 Events, 136 Sensoren) |
| `experiments/soja_enriched_dummy.yaml` | palaestrAI-Experiment-Config (DummyBrain/DummyMuscle) |
| `/Users/aschaefer/baci_gta_kg.csv` | KG-Export: 753 Zeilen, BACI+GTA kombiniert |
| `/Users/aschaefer/baci_gta_kg.json` | KG-Export: 449 Einträge als JSON mit Metadaten |

---

*Erstellt: 2026-03-01 | Autor: AP3-Simulation | Paket: provider_sim*
