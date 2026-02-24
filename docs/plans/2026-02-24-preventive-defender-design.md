# Design: Präventive Defender-Policy

**Datum:** 2026-02-24
**Status:** Genehmigt
**Datei:** `analysis/run_heatmap_analysis.py`

## Problem

Die reaktive Defender-Policy verteilt das Budget invers zur aktuellen Health — d. h. sie reagiert auf bereits eingetretene Schäden. Das führt zu abnehmendem Grenznutzen bei höherem Budget (Lauf 2 → 3: +0,6 pp) und kann kaskadierende Schäden nicht verhindern.

## Lösung

Neue Funktion `preventive_defender_policy`: Schützt Entities proportional zu ihrer statischen PDL-Vulnerability, bevor Schäden entstehen. Symmetrisch zur bestehenden `attacker_policy`.

## Policy-Funktion

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

## CLI-Erweiterung

```
--policy {reactive,preventive}   Defender-Policy (default: reactive)
```

Der Default-Output-Dateiname enthält den Policy-Namen:
- `heatmap_soja.png` (reactive, bisheriges Verhalten)
- `heatmap_soja_preventive.png` (preventive)

## Änderungsumfang

| Datei | Art |
|---|---|
| `analysis/run_heatmap_analysis.py` | Neue Funktion + `--policy` Argument |

Keine Änderungen an Engine, State, Environment, PDL-Modell oder Tests.

## Vergleich

| Policy | Logik | Budget-Quelle |
|---|---|---|
| `attacker_policy` | `weight = vulnerability / Σvulnerability` | attack_budget |
| `defender_policy` (reaktiv) | `weight = (1 - health) / Σ(1-health)` | defend_budget |
| `preventive_defender_policy` | `weight = vulnerability / Σvulnerability` | defend_budget |

Gleicher Vergleichslauf wie Lauf 1 (Attack 0.8 / Defend 0.4), um den Unterschied reaktiv vs. präventiv bei identischem Budget zu isolieren.
