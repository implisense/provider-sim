# Design: Hybrid Defender-Policy

**Datum:** 2026-02-24
**Status:** Genehmigt
**Datei:** `analysis/run_heatmap_analysis.py`

## Motivation

Lauf 4 (präventiv, Attack=0.8/Defend=0.4) ergibt Ø Health 0.6954 — marginal schlechter als reaktiv (0.6999). Die präventive Policy verteilt Budget breit auf vulnerable Nodes, verliert aber Konzentration auf akute Schäden. Hypothesis: Eine lineare Kombination beider Signale (Vulnerability + inverse Health) ist optimal.

## Policy-Funktion

```python
def hybrid_defender_policy(
    entities: list,
    obs: dict,
    budget: float,
    alpha: float = 0.5,
) -> dict:
    """Weighted blend of preventive (vulnerability) and reactive (inverse-health) defense.

    alpha=1.0 → rein präventiv (vulnerability-gewichtet)
    alpha=0.0 → rein reaktiv (inverse-health-gewichtet)
    """
    vulns = np.array([e.vulnerability for e in entities], dtype=np.float32)
    prev_w = vulns / vulns.sum() if vulns.sum() > 0 else np.ones(len(entities), dtype=np.float32) / len(entities)

    healths = np.array([obs.get(f"entity.{e.id}.health", 1.0) for e in entities], dtype=np.float32)
    inv = 1.0 - healths
    react_w = inv / inv.sum() if inv.sum() > 0 else np.ones(len(entities), dtype=np.float32) / len(entities)

    weights = alpha * prev_w + (1.0 - alpha) * react_w
    return {f"defender.{e.id}": float(budget * w) for e, w in zip(entities, weights)}
```

## CLI-Erweiterungen

| Argument | Werte | Default |
|---|---|---|
| `--policy` | `reactive`, `preventive`, `hybrid` | `reactive` |
| `--alpha` | `0.0`–`1.0` (float) | `0.5` |

`--alpha` ist nur relevant bei `--policy hybrid`.

## Output-Dateiname

| Policy | Dateiname |
|---|---|
| `reactive` | `heatmap_soja.png` |
| `preventive` | `heatmap_soja_preventive.png` |
| `hybrid` mit α=0.50 | `heatmap_soja_hybrid_0.50.png` |

## Heatmap-Titel

```
Soja-Lieferkette — Ø Health über N Episoden
Attacker=X.X  Defender=X.X  Policy=hybrid(α=0.50)  Ticks=365
```

Bei `reactive` und `preventive` bleibt der Titel unverändert: `Policy=reactive` / `Policy=preventive`.

## Änderungsumfang

| Datei | Art |
|---|---|
| `analysis/run_heatmap_analysis.py` | Neue Funktion + `--alpha` Argument + Dispatch + Dateiname + Titel |

Keine Änderungen an Engine, Tests, PDL-Modell.

## Vergleichsläufe (geplant)

Alle mit Attack=0.8, Defend=0.4, 50 Ep × 365 Ticks, Seed=42:

| α | Dateiname | Erwartung |
|---|---|---|
| 0.25 | `heatmap_soja_hybrid_0.25.png` | Näher an reaktiv |
| 0.50 | `heatmap_soja_hybrid_0.50.png` | Mitte |
| 0.75 | `heatmap_soja_hybrid_0.75.png` | Näher an präventiv |
