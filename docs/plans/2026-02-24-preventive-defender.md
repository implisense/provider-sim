# Preventive Defender Policy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ergänze `analysis/run_heatmap_analysis.py` um eine vulnerability-gewichtete präventive Defender-Policy und ein `--policy`-CLI-Argument, das reaktive und präventive Policy umschaltet.

**Architecture:** Die neue `preventive_defender_policy`-Funktion ist strukturell identisch mit `attacker_policy` — sie gewichtet das Defender-Budget proportional zur statischen PDL-Vulnerability. Ein `--policy {reactive,preventive}`-Argument steuert, welche Policy in `run_episodes` verwendet wird. Der Output-Dateiname enthält den Policy-Namen.

**Tech Stack:** Python 3.9+, NumPy, argparse — keine neuen Abhängigkeiten.

---

### Task 1: Funktion `preventive_defender_policy` hinzufügen

**Files:**
- Modify: `analysis/run_heatmap_analysis.py:53-66` (nach `defender_policy`)

**Kontext:** Die bestehende `defender_policy` (Zeile 53–66) ist das reaktive Gegenstück. Die neue Funktion kommt direkt danach.

**Step 1: Funktion einfügen**

In `run_heatmap_analysis.py` nach `defender_policy` (nach Zeile 66) einfügen:

```python
def preventive_defender_policy(
    entities: list,
    obs: dict,
    budget: float,
) -> dict:
    """Vulnerability-weighted defense: protect structurally weak nodes pre-emptively."""
    vulns = np.array([e.vulnerability for e in entities], dtype=np.float32)
    weights = vulns / vulns.sum() if vulns.sum() > 0 else np.ones(len(entities)) / len(entities)
    return {f"defender.{e.id}": float(budget * w) for e, w in zip(entities, weights)}
```

**Step 2: Manuell prüfen**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
python -c "
import sys; sys.path.insert(0, '.')
from analysis.run_heatmap_analysis import preventive_defender_policy
from provider_sim.pdl.parser import load_pdl
doc = load_pdl('scenarios/s1-soja.pdl.yaml')
obs = {}
result = preventive_defender_policy(doc.entities, obs, budget=0.4)
print('Keys:', len(result), 'entries')
print('Sum:', sum(result.values()))
print('Sample:', list(result.items())[:3])
"
```

Erwartete Ausgabe:
```
Keys: 20 entries
Sum: 0.4
Sample: [('defender.brazil_farms', ...), ...]
```

**Step 3: Commit**

```bash
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add preventive_defender_policy function"
```

---

### Task 2: `--policy` CLI-Argument und Policy-Dispatch in `run_episodes`

**Files:**
- Modify: `analysis/run_heatmap_analysis.py` — `parse_args()` + `run_episodes()` + `__main__`

**Step 1: `parse_args` erweitern**

In `parse_args()` nach dem `--output`-Argument einfügen:

```python
p.add_argument(
    "--policy",
    choices=["reactive", "preventive"],
    default="reactive",
    help="Defender policy: reactive (default) or preventive (vulnerability-weighted)",
)
```

**Step 2: `run_episodes` Signatur und Dispatch anpassen**

Signatur ändern von:
```python
def run_episodes(
    episodes: int,
    ticks: int,
    attack_budget: float,
    defend_budget: float,
    seed: int,
) -> tuple:
```

zu:
```python
def run_episodes(
    episodes: int,
    ticks: int,
    attack_budget: float,
    defend_budget: float,
    seed: int,
    defender_policy_name: str = "reactive",
) -> tuple:
```

Innerhalb von `run_episodes`, die Zeile:
```python
actions.update(defender_policy(ep_env.doc.entities, obs, defend_budget))
```

ersetzen durch:
```python
if defender_policy_name == "preventive":
    actions.update(preventive_defender_policy(ep_env.doc.entities, obs, defend_budget))
else:
    actions.update(defender_policy(ep_env.doc.entities, obs, defend_budget))
```

**Step 3: `__main__`-Block anpassen**

Den `run_episodes`-Aufruf erweitern:

```python
health_data, entity_ids = run_episodes(
    args.episodes, args.ticks, args.attack, args.defend, args.seed,
    defender_policy_name=args.policy,
)
```

**Step 4: Default-Output-Dateiname anpassen**

In `parse_args()` den `--output`-Default anpassen:

```python
p.add_argument("--output", type=str, default="",
               help="Output path for heatmap PNG (default: auto-named by policy)")
```

Im `__main__`-Block vor `plot_heatmap`:

```python
output = args.output
if not output:
    stem = f"heatmap_soja_{args.policy}" if args.policy != "reactive" else "heatmap_soja"
    output = str(Path(__file__).resolve().parent / f"{stem}.png")
```

Und den `plot_heatmap`-Aufruf auf `output` umstellen (statt `args.output`).

**Step 5: Smoke-Test**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
python analysis/run_heatmap_analysis.py --episodes 2 --ticks 10 --policy preventive --output /tmp/test_preventive.png
```

Erwartete Ausgabe:
```
Starte 2 Episoden × 10 Ticks …
  Attack=0.8  Defend=0.4  Seed=42
  Episode   2/2  [████████████████████████████████████████]
  Ø Health gesamt     : 0.XXXX
  Kritischste Entity  : feed_mills
Erstelle Heatmap …
  Heatmap gespeichert: /tmp/test_preventive.png
```

```bash
# Reaktive Policy als Baseline (gleiche Parameter)
python analysis/run_heatmap_analysis.py --episodes 2 --ticks 10 --policy reactive --output /tmp/test_reactive.png
```

Beide Läufe müssen fehlerfrei durchlaufen. Die Health-Werte dürfen sich unterscheiden.

**Step 6: Commit**

```bash
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add --policy flag to switch between reactive and preventive defender"
```

---

### Task 3: Vollständiger Vergleichslauf (50 Ep × 365 Ticks)

**Files:**
- Keine Code-Änderungen — nur Ausführung

**Step 1: Präventiven Lauf starten**

```bash
cd /Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/palestrai_simulation
python analysis/run_heatmap_analysis.py \
    --episodes 50 --ticks 365 \
    --attack 0.8 --defend 0.4 \
    --policy preventive \
    --seed 42
```

Erwartete Ausgabe:
```
Starte 50 Episoden × 365 Ticks …
  Attack=0.8  Defend=0.4  Seed=42
  Episode  50/50  [████████████████████████████████████████]
  Ø Health gesamt     : 0.XXXX
  Kritischste Entity  : XXXX
Erstelle Heatmap …
  Heatmap gespeichert: .../analysis/heatmap_soja_preventive.png
```

**Step 2: Ergebnis mit Lauf 1 (reaktiv, gleiche Parameter) vergleichen**

Lauf 1 (reaktiv, Attack 0.8 / Defend 0.4) hatte Ø Health = **0.6999**.
Wenn die präventive Policy besser abschneidet, liegt Ø Health > 0.6999.

**Step 3: Heatmap committen**

```bash
git add analysis/heatmap_soja_preventive.png
git commit -m "feat: add preventive defender heatmap (Attack=0.8, Defend=0.4, 50ep×365t)"
```

---

### Task 4: Report-Erweiterung

**Files:**
- Modify: `analysis/simulation_report.md`

**Step 1: Tabelle erweitern**

Zeile in der Ergebnistabelle hinzufügen:

```markdown
| Präventiv (Attack 0.8 / Defend 0.4) | 0.8 | 0.4 | **0.XXXX** | XXXX | `heatmap_soja_preventive.png` |
```

**Step 2: Abschnitt "Lauf 4" einfügen**

Nach dem Lauf-3-Abschnitt:

```markdown
### Lauf 4: Präventive Policy (Attack 0.8 / Defend 0.4)

![Heatmap Lauf 4](heatmap_soja_preventive.png)

Ø Health: **0.XXXX** — identische Budget-Parameter wie Lauf 1, aber vulnerability-gewichteter
Defender. [Beobachtungen nach Lauf eintragen.]
```

**Step 3: Commit**

```bash
git add analysis/simulation_report.md
git commit -m "docs: extend report with preventive defender results"
```
