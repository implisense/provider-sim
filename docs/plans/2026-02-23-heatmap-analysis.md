# Heatmap-Analyse Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Multi-Episode-Simulation mit strategischen Attacker/Defender-Policies, Auswertung als Entity-Health-Heatmap über Zeit.

**Architecture:** Standalone-Schleife über N Episoden via `reset_dict`/`step_dict` ohne palaestrAI-Orchestrator. Attacker greift vulnerability-gewichtet an, Defender reagiert auf aktuellen Health-Zustand. Ergebnisse werden in NumPy-Arrays gesammelt und als Matplotlib-Heatmap gespeichert.

**Tech Stack:** Python 3.9, NumPy, Matplotlib, provider_sim (bereits installiert)

---

### Task 1: `analysis/`-Verzeichnis und Einstiegsskript anlegen

**Files:**
- Create: `analysis/run_heatmap_analysis.py`

**Step 1: Datei mit Grundstruktur anlegen**

```python
"""Multi-episode supply-chain heatmap analysis.

Runs N episodes of the PROVIDER soy scenario with strategic
attacker/defender policies and produces a health heatmap.

Usage:
    cd palestrai_simulation
    python analysis/run_heatmap_analysis.py
    python analysis/run_heatmap_analysis.py --episodes 10 --ticks 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

_PDL_PATH = _BASE / "scenarios" / "s1-soja.pdl.yaml"
_OUTPUT_PATH = Path(__file__).resolve().parent / "heatmap_soja.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PROVIDER soy heatmap analysis")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--ticks",    type=int, default=365)
    p.add_argument("--attack",   type=float, default=0.8,
                   help="Max attack budget (0-1)")
    p.add_argument("--defend",   type=float, default=0.4,
                   help="Max defend budget (0-1)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   type=str, default=str(_OUTPUT_PATH))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Episodes: {args.episodes}, Ticks: {args.ticks}")
    print(f"Attack budget: {args.attack}, Defend budget: {args.defend}")
```

**Step 2: Skript ausführen und prüfen dass es ohne Fehler startet**

```bash
cd palestrai_simulation
python analysis/run_heatmap_analysis.py --episodes 2 --ticks 5
```

Erwartete Ausgabe:
```
Episodes: 2, Ticks: 5
Attack budget: 0.8, Defend budget: 0.4
```

**Step 3: Commit**

```bash
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add heatmap analysis scaffold"
```

---

### Task 2: Policy-Funktionen implementieren

**Files:**
- Modify: `analysis/run_heatmap_analysis.py`

Die Policies berechnen pro Tick die Aktions-Werte für alle Entities.

**Step 1: Policy-Funktionen nach `parse_args()` einfügen**

```python
def attacker_policy(
    entities: list,
    obs: dict[str, float],
    budget: float,
) -> dict[str, float]:
    """Vulnerability-weighted attack: high-vulnerability entities get more pressure."""
    vulns = np.array([e.vulnerability for e in entities], dtype=np.float32)
    weights = vulns / vulns.sum() if vulns.sum() > 0 else np.ones(len(entities)) / len(entities)
    return {f"attacker.{e.id}": float(budget * w) for e, w in zip(entities, weights)}


def defender_policy(
    entities: list,
    obs: dict[str, float],
    budget: float,
) -> dict[str, float]:
    """Reactive defense: entities with lowest health get strongest defense."""
    healths = np.array(
        [obs.get(f"entity.{e.id}.health", 1.0) for e in entities],
        dtype=np.float32,
    )
    # Invert: low health → high defense weight
    inv = 1.0 - healths
    total = inv.sum()
    weights = inv / total if total > 0 else np.ones(len(entities)) / len(entities)
    return {f"defender.{e.id}": float(budget * w) for e, w in zip(entities, weights)}
```

**Step 2: Smoke-Test der Policies isoliert**

```python
# In einer Python-REPL prüfen:
import sys; sys.path.insert(0, ".")
from provider_sim.pdl.parser import load_pdl
doc = load_pdl("scenarios/s1-soja.pdl.yaml")
obs = {f"entity.{e.id}.health": 1.0 for e in doc.entities}

# Attacker-Weights müssen sich zu ~budget summieren
from analysis.run_heatmap_analysis import attacker_policy, defender_policy
att = attacker_policy(doc.entities, obs, 0.8)
print(sum(att.values()))   # ≈ 0.8
def_ = defender_policy(doc.entities, obs, 0.4)
print(sum(def_.values()))  # ≈ 0.4
```

**Step 3: Commit**

```bash
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add vulnerability-weighted attacker and reactive defender policies"
```

---

### Task 3: Episoden-Schleife mit Datensammlung

**Files:**
- Modify: `analysis/run_heatmap_analysis.py`

**Step 1: `run_episodes()`-Funktion nach den Policy-Funktionen einfügen**

```python
def run_episodes(
    episodes: int,
    ticks: int,
    attack_budget: float,
    defend_budget: float,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Run N episodes and collect health per entity per tick.

    Returns:
        health_data: shape (episodes, n_entities, ticks) — health values
        entity_ids:  list of entity id strings in order
    """
    from provider_sim.env.environment import ProviderEnvironment

    env = ProviderEnvironment(_PDL_PATH, seed=seed, max_ticks=ticks)
    entity_ids = [e.id for e in env.doc.entities]
    n_entities = len(entity_ids)

    health_data = np.zeros((episodes, n_entities, ticks), dtype=np.float32)

    for ep in range(episodes):
        # New seed per episode for varied event triggers
        ep_env = ProviderEnvironment(_PDL_PATH, seed=seed + ep, max_ticks=ticks)
        obs, _ = ep_env.reset_dict()

        for tick in range(ticks):
            actions = {}
            actions.update(attacker_policy(ep_env.doc.entities, obs, attack_budget))
            actions.update(defender_policy(ep_env.doc.entities, obs, defend_budget))

            obs, rewards, done = ep_env.step_dict(actions)

            for i, eid in enumerate(entity_ids):
                health_data[ep, i, tick] = obs.get(f"entity.{eid}.health", 1.0)

            if done:
                # Fill remaining ticks with last value
                if tick + 1 < ticks:
                    health_data[ep, :, tick + 1:] = health_data[ep, :, tick:tick + 1]
                break

        pct = int((ep + 1) / episodes * 40)
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r  Episode {ep + 1:>3}/{episodes}  [{bar}]", end="", flush=True)

    print()  # newline after progress bar
    return health_data, entity_ids
```

**Step 2: `__main__`-Block erweitern um Episoden-Aufruf**

```python
if __name__ == "__main__":
    args = parse_args()
    print(f"Starte {args.episodes} Episoden × {args.ticks} Ticks …")
    health_data, entity_ids = run_episodes(
        args.episodes, args.ticks, args.attack, args.defend, args.seed
    )
    print(f"health_data shape: {health_data.shape}")
    print(f"mean health (alle Entities, alle Episoden): {health_data.mean():.4f}")
```

**Step 3: Kurzen Lauf testen**

```bash
cd palestrai_simulation
python analysis/run_heatmap_analysis.py --episodes 3 --ticks 10
```

Erwartete Ausgabe:
```
Starte 3 Episoden × 10 Ticks …
  Episode   3/3  [████████████████████████████████████████]
health_data shape: (3, 20, 10)
mean health (alle Entities, alle Episoden): 0.XXXX
```

**Step 4: Commit**

```bash
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add multi-episode loop with health data collection"
```

---

### Task 4: Heatmap-Visualisierung

**Files:**
- Modify: `analysis/run_heatmap_analysis.py`

**Step 1: Matplotlib-Import am Dateianfang ergänzen**

Nach `import numpy as np` einfügen:
```python
import matplotlib
matplotlib.use("Agg")  # kein Display noetig
import matplotlib.pyplot as plt
```

**Step 2: `plot_heatmap()`-Funktion einfügen**

```python
def plot_heatmap(
    health_data: np.ndarray,
    entity_ids: list[str],
    output_path: str,
    episodes: int,
    ticks: int,
    attack_budget: float,
    defend_budget: float,
) -> None:
    """Save health heatmap: entities (Y) x ticks (X), colour = mean health."""
    # Mean über alle Episoden: shape (n_entities, ticks)
    mean_health = health_data.mean(axis=0)

    # Entities nach mittlerer Health sortieren (kritischste oben)
    sort_idx = np.argsort(mean_health.mean(axis=1))
    mean_health_sorted = mean_health[sort_idx]
    labels_sorted = [entity_ids[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        mean_health_sorted,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # Achsenbeschriftung
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=9)
    ax.set_xlabel("Tick (Tag)", fontsize=11)
    ax.set_ylabel("Entity", fontsize=11)
    ax.set_title(
        f"Soja-Lieferkette — Ø Health über {episodes} Episoden\n"
        f"Attacker={attack_budget:.1f}  Defender={defend_budget:.1f}  "
        f"Ticks={ticks}",
        fontsize=12,
    )

    # X-Achse: nur jeden 30. Tick beschriften
    tick_step = max(1, ticks // 12)
    ax.set_xticks(range(0, ticks, tick_step))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Ø Health (0=kritisch, 1=stabil)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap gespeichert: {output_path}")
```

**Step 3: `__main__`-Block finalisieren**

Den bisherigen `__main__`-Block ersetzen durch:
```python
if __name__ == "__main__":
    args = parse_args()
    print(f"Starte {args.episodes} Episoden × {args.ticks} Ticks …")
    print(f"  Attack={args.attack}  Defend={args.defend}  Seed={args.seed}")

    health_data, entity_ids = run_episodes(
        args.episodes, args.ticks, args.attack, args.defend, args.seed
    )

    mean_all = health_data.mean()
    min_entity = entity_ids[health_data.mean(axis=(0, 2)).argmin()]
    print(f"  Ø Health gesamt     : {mean_all:.4f}")
    print(f"  Kritischste Entity  : {min_entity}")

    print("Erstelle Heatmap …")
    plot_heatmap(
        health_data, entity_ids, args.output,
        args.episodes, args.ticks, args.attack, args.defend,
    )
```

**Step 4: Vollständigen Lauf ausführen und PNG prüfen**

```bash
cd palestrai_simulation
python analysis/run_heatmap_analysis.py --episodes 5 --ticks 50
```

Erwartete Ausgabe:
```
Starte 5 Episoden × 50 Ticks …
  Attack=0.8  Defend=0.4  Seed=42
  Episode   5/5  [████████████████████████████████████████]
  Ø Health gesamt     : 0.XXXX
  Kritischste Entity  : poultry_farms   (oder ähnlich)
Erstelle Heatmap …
  Heatmap gespeichert: .../analysis/heatmap_soja.png
```

PNG prüfen:
```bash
ls -lh analysis/heatmap_soja.png
```

**Step 5: Commit**

```bash
git add analysis/run_heatmap_analysis.py
git commit -m "feat: add health heatmap visualisation with entity sorting"
```

---

### Task 5: Vollständiger Lauf (50 Episoden × 365 Ticks)

**Step 1: Produktionslauf starten**

```bash
cd palestrai_simulation
python analysis/run_heatmap_analysis.py
```

Laufzeit ca. 2–4 Minuten. Erwartete Ausgabe:
```
Starte 50 Episoden × 365 Ticks …
  Attack=0.8  Defend=0.4  Seed=42
  Episode  50/50  [████████████████████████████████████████]
  Ø Health gesamt     : 0.XXXX
  Kritischste Entity  : poultry_farms
Erstelle Heatmap …
  Heatmap gespeichert: .../analysis/heatmap_soja.png
```

**Step 2: Heatmap öffnen und inhaltlich prüfen**

```bash
open analysis/heatmap_soja.png
```

Erwartungen:
- Entities mit hoher PDL-`vulnerability` (poultry_farms=0.8, santos_port=0.7, pig_farms=0.75) sollten oben (rot) stehen
- `strategic_feed_reserves` (vulnerability=0.2) sollte unten (grün) stehen
- Zeitliche Degradation ab ca. Tick 30–60 sichtbar

**Step 3: Commit**

```bash
git add analysis/heatmap_soja.png
git commit -m "feat: add heatmap output for 50-episode soy scenario analysis"
```
