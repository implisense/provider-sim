# Hybrid Defender Policy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ergänze `analysis/run_heatmap_analysis.py` um eine `hybrid_defender_policy`, die vulnerability-gewichtetes und reaktives Signal per `--alpha`-Parameter mischt, und führe drei Vergleichsläufe durch.

**Architecture:** `hybrid_defender_policy(alpha)` berechnet `weight = alpha * vuln_weight + (1-alpha) * reactive_weight`. `--policy hybrid` aktiviert sie; `--alpha FLOAT` steuert den Mix. Der Auto-Dateiname kodiert α (`heatmap_soja_hybrid_0.50.png`), der Heatmap-Titel zeigt `Policy=hybrid(α=0.50)`.

**Tech Stack:** Python 3.9+, NumPy, argparse — keine neuen Abhängigkeiten.

---

### Task 1: `hybrid_defender_policy` Funktion hinzufügen

**Files:**
- Modify: `analysis/run_heatmap_analysis.py:85` (nach `preventive_defender_policy`)

**Kontext:** Datei hat aktuell drei Policy-Funktionen (Zeilen 50–85). Die neue kommt als vierte direkt nach `preventive_defender_policy` (nach Zeile 85).

**Step 1: Funktion nach Zeile 85 einfügen**

```python
def hybrid_defender_policy(
    entities: list,
    obs: dict,
    budget: float,
    alpha: float = 0.5,
) -> dict:
    """Blend of preventive (vulnerability) and reactive (inverse-health) defense.

    alpha=1.0 -> purely preventive (vulnerability-weighted)
    alpha=0.0 -> purely reactive (inverse-health-weighted)
    """
    vulns = np.array([e.vulnerability for e in entities], dtype=np.float32)
    prev_w = vulns / vulns.sum() if vulns.sum() > 0 else np.ones(len(entities), dtype=np.float32) / len(entities)

    healths = np.array([obs.get(f"entity.{e.id}.health", 1.0) for e in entities], dtype=np.float32)
    inv = 1.0 - healths
    react_w = inv / inv.sum() if inv.sum() > 0 else np.ones(len(entities), dtype=np.float32) / len(entities)

    weights = alpha * prev_w + (1.0 - alpha) * react_w
    return {f"defender.{e.id}": float(budget * w) for e, w in zip(entities, weights)}
```

**Step 2: Smoke-Verifikation**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
python -c "
import sys; sys.path.insert(0, '.')
from analysis.run_heatmap_analysis import hybrid_defender_policy
from provider_sim.pdl.parser import load_pdl
doc = load_pdl('scenarios/s1-soja.pdl.yaml')
obs = {f'entity.{e.id}.health': 0.7 for e in doc.entities}

# alpha=0.5 Test
r = hybrid_defender_policy(doc.entities, obs, budget=0.4, alpha=0.5)
print('Keys:', len(r))
print('Sum:', round(sum(r.values()), 6))

# alpha=1.0 muss identisch mit preventive_defender_policy sein
from analysis.run_heatmap_analysis import preventive_defender_policy
p = preventive_defender_policy(doc.entities, obs, budget=0.4)
h1 = hybrid_defender_policy(doc.entities, obs, budget=0.4, alpha=1.0)
diffs = {k: abs(h1[k] - p[k.replace('defender.','defender.')]) for k in p}
print('Max diff alpha=1.0 vs preventive:', max(diffs.values()))

# alpha=0.0 muss identisch mit defender_policy sein (bei uniform health)
from analysis.run_heatmap_analysis import defender_policy
d = defender_policy(doc.entities, obs, budget=0.4)
h0 = hybrid_defender_policy(doc.entities, obs, budget=0.4, alpha=0.0)
diffs0 = {k: abs(h0[k] - d[k]) for k in d}
print('Max diff alpha=0.0 vs reactive:', max(diffs0.values()))
"
```

Erwartete Ausgabe:
```
Keys: 20
Sum: 0.4
Max diff alpha=1.0 vs preventive: 0.0
Max diff alpha=0.0 vs reactive: 0.0
```

**Step 3: Commit**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add hybrid_defender_policy with alpha blending"
```

---

### Task 2: CLI-Erweiterungen, Dispatch, Dateiname, Titel

**Files:**
- Modify: `analysis/run_heatmap_analysis.py` — `parse_args()`, `run_episodes()`, `__main__`, `plot_heatmap()`

**Kontext:** Aktueller Stand der relevanten Stellen:
- `parse_args()` Zeile 41–46: `--policy choices=["reactive", "preventive"]`
- `run_episodes()` Zeile 94: `defender_policy_name: str = "reactive"`
- `run_episodes()` Zeile 117–120: if/else für reactive/preventive
- `plot_heatmap()` Zeile 148: `policy: str = "reactive"`
- `plot_heatmap()` Zeile 174: `f"Policy={policy}"`
- `__main__` Zeile 206–208: Auto-Dateiname-Logik

**Step 1: `parse_args()` — `--policy` um `hybrid` erweitern + `--alpha` hinzufügen**

Zeile 43 ändern von:
```python
        choices=["reactive", "preventive"],
```
zu:
```python
        choices=["reactive", "preventive", "hybrid"],
```

Nach dem `--policy`-Block (nach Zeile 46, vor `return p.parse_args()`) einfügen:
```python
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend factor for hybrid policy: 1.0=preventive, 0.0=reactive (default: 0.5)",
    )
```

**Step 2: `run_episodes()` — Signatur und Dispatch**

Signatur: `defender_policy_name: str = "reactive"` bleibt, aber einen neuen Parameter hinzufügen direkt danach:
```python
    defender_policy_name: str = "reactive",
    alpha: float = 0.5,
```

Den if/else-Block (Zeile 117–120) ersetzen durch:
```python
            if defender_policy_name == "preventive":
                actions.update(preventive_defender_policy(ep_env.doc.entities, obs, defend_budget))
            elif defender_policy_name == "hybrid":
                actions.update(hybrid_defender_policy(ep_env.doc.entities, obs, defend_budget, alpha=alpha))
            else:
                actions.update(defender_policy(ep_env.doc.entities, obs, defend_budget))
```

**Step 3: `__main__` — `run_episodes`-Aufruf und Auto-Dateiname**

`run_episodes`-Aufruf (Zeile 195–198) auf:
```python
    health_data, entity_ids = run_episodes(
        args.episodes, args.ticks, args.attack, args.defend, args.seed,
        defender_policy_name=args.policy,
        alpha=args.alpha,
    )
```

Auto-Dateiname-Block (Zeile 206–208) ersetzen durch:
```python
    output = args.output
    if not output:
        if args.policy == "hybrid":
            stem = f"heatmap_soja_hybrid_{args.alpha:.2f}"
        elif args.policy == "preventive":
            stem = "heatmap_soja_preventive"
        else:
            stem = "heatmap_soja"
        output = str(Path(__file__).resolve().parent / f"{stem}.png")
```

**Step 4: `plot_heatmap()` — `policy_label`-Parameter für Titel**

Signatur ändern: `policy: str = "reactive"` → `policy_label: str = "reactive"` (nur den Namen ändern, nicht den Typ).

Titel-String (Zeile 174) ändern von:
```python
        f"Policy={policy}  Ticks={ticks}",
```
zu:
```python
        f"Policy={policy_label}  Ticks={ticks}",
```

Im `__main__`-Block, vor `plot_heatmap(...)`:
```python
    policy_label = f"hybrid(α={args.alpha:.2f})" if args.policy == "hybrid" else args.policy
```

`plot_heatmap`-Aufruf anpassen:
```python
    plot_heatmap(
        health_data, entity_ids, output,
        args.episodes, args.ticks, args.attack, args.defend,
        policy=policy_label,
    )
```

**Step 5: Smoke-Test**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation

# hybrid alpha=0.5
python analysis/run_heatmap_analysis.py --episodes 2 --ticks 10 --policy hybrid --alpha 0.5 2>&1 | grep -E "Health|gespeichert|Policy"

# hybrid alpha=0.25
python analysis/run_heatmap_analysis.py --episodes 2 --ticks 10 --policy hybrid --alpha 0.25 --output /tmp/test_hybrid_025.png 2>&1 | grep "gespeichert"

# reactive bleibt unverändert
python analysis/run_heatmap_analysis.py --episodes 2 --ticks 10 --policy reactive --output /tmp/test_reactive.png 2>&1 | grep "gespeichert"
```

Erwartete Auto-Dateinamen-Ausgaben:
- `hybrid 0.5` → `heatmap_soja_hybrid_0.50.png`
- `reactive` → `heatmap_soja.png`

**Step 6: Commit**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add --policy hybrid and --alpha CLI arguments"
```

---

### Task 3: Drei Vergleichsläufe (50 Ep × 365 Ticks)

**Files:** Keine Code-Änderungen — nur Ausführung.

**Kontext:** Alle Läufe mit Attack=0.8, Defend=0.4, Seed=42. Vergleichswerte:
- Reaktiv (Lauf 1): Ø Health 0.6999
- Präventiv (Lauf 4): Ø Health 0.6954

**Step 1: α=0.25 (näher an reaktiv)**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
python analysis/run_heatmap_analysis.py \
    --episodes 50 --ticks 365 \
    --attack 0.8 --defend 0.4 \
    --policy hybrid --alpha 0.25 --seed 42
```

Ausgabe → `analysis/heatmap_soja_hybrid_0.25.png`. Ø Health notieren.

**Step 2: α=0.50 (Mitte)**

```bash
python analysis/run_heatmap_analysis.py \
    --episodes 50 --ticks 365 \
    --attack 0.8 --defend 0.4 \
    --policy hybrid --alpha 0.50 --seed 42
```

Ausgabe → `analysis/heatmap_soja_hybrid_0.50.png`. Ø Health notieren.

**Step 3: α=0.75 (näher an präventiv)**

```bash
python analysis/run_heatmap_analysis.py \
    --episodes 50 --ticks 365 \
    --attack 0.8 --defend 0.4 \
    --policy hybrid --alpha 0.75 --seed 42
```

Ausgabe → `analysis/heatmap_soja_hybrid_0.75.png`. Ø Health notieren.

**Step 4: Alle drei Heatmaps committen**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
git add analysis/heatmap_soja_hybrid_0.25.png analysis/heatmap_soja_hybrid_0.50.png analysis/heatmap_soja_hybrid_0.75.png
git commit -m "feat: add hybrid defender heatmaps alpha=0.25/0.50/0.75 (Attack=0.8, Defend=0.4, 50ep×365t)"
```

---

### Task 4: Report-Erweiterung

**Files:**
- Modify: `analysis/simulation_report.md`

**Kontext:** Report hat aktuell 4 Einträge in der Übersichtstabelle (Läufe 1–4). Die drei Hybrid-Läufe kommen als Läufe 5–7. Die tatsächlichen Ø-Health-Werte aus Task 3 einsetzen.

**Step 1: Tabelle erweitern**

Nach der Zeile für Lauf 4 (`| Präventiv ...`) drei Zeilen einfügen:
```markdown
| Hybrid α=0.25 | 0.8 | 0.4 | **0.XXXX** | `XXX` | `heatmap_soja_hybrid_0.25.png` |
| Hybrid α=0.50 | 0.8 | 0.4 | **0.XXXX** | `XXX` | `heatmap_soja_hybrid_0.50.png` |
| Hybrid α=0.75 | 0.8 | 0.4 | **0.XXXX** | `XXX` | `heatmap_soja_hybrid_0.75.png` |
```

**Step 2: Abschnitt "Läufe 5–7: Hybrid-Policy" einfügen**

Nach dem Lauf-4-Abschnitt und vor `---` einfügen:

```markdown
### Läufe 5–7: Hybrid-Policy (α=0.25 / 0.50 / 0.75, Attack 0.8 / Defend 0.4)

![Hybrid α=0.25](heatmap_soja_hybrid_0.25.png)
![Hybrid α=0.50](heatmap_soja_hybrid_0.50.png)
![Hybrid α=0.75](heatmap_soja_hybrid_0.75.png)

Ø Health: α=0.25 → **0.XXXX**, α=0.50 → **0.XXXX**, α=0.75 → **0.XXXX**.

[Beobachtungen nach Läufen eintragen: Welches α ist optimal? Wie verhält sich der Bottleneck?]
```

**Step 3: Abschnitt 4 "Übergreifende Befunde" erweitern**

Nach dem Paragraphen "Präventiv vs. reaktiv" einfügen:

```markdown
**Hybrid-Policy α-Sweep:**
[Befunde nach Läufen eintragen: optimales α, Bottleneck-Verlauf, Vergleich mit Lauf 1 und 4.]
```

**Step 4: Commit**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
git add analysis/simulation_report.md
git commit -m "docs: extend report with hybrid defender results (Läufe 5-7)"
```
