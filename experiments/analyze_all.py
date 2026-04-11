"""
Batch-Simulation und Vergleich aller 9 PROVIDER-Szenarien (Dummy-Baseline).
365 Ticks, Seed 42, keine Agentensteuerung.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider_sim.env.environment import ProviderEnvironment

PDL_DIR = Path(
    "/Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/pdl-ontology-web-viewer/scenarios"
)
MAX_TICKS = 365
SEED = 42

SCENARIOS = [
    ("S1", "s1-soja.pdl.yaml",              "Soja / Tierfutter"),
    ("S2", "s2-halbleiter.pdl.yaml",         "Halbleiter"),
    ("S3", "s3-pharma.pdl.yaml",             "Pharma / Wirkstoffe"),
    ("S4", "s4-duengemittel-adblue.pdl.yaml","Düngemittel / AdBlue"),
    ("S5", "s5-wasseraufbereitung.pdl.yaml", "Wasseraufbereitung"),
    ("S6", "s6-rechenzentren.pdl.yaml",      "Rechenzentren"),
    ("S7", "s7-seltene-erden.pdl.yaml",      "Seltene Erden"),
    ("S8", "s8-seefracht.pdl.yaml",          "Seefracht"),
    ("S9", "s9-unterwasserkabel.pdl.yaml",   "Unterwasserkabel"),
]

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def run_scenario(pdl_path: Path) -> dict:
    env = ProviderEnvironment(pdl_source=pdl_path, max_ticks=MAX_TICKS, seed=SEED)
    obs, rewards = env.reset_dict()

    entity_ids = env.engine.state.entity_ids
    event_ids  = env.engine.state.event_ids

    health_series: list[list[float]] = []   # pro Tick: Liste aller Entity-Healths
    event_active: dict[str, list[int]] = {ev: [] for ev in event_ids}
    reward_def: list[float] = []
    tick_history: list[int] = [0]

    def snapshot(obs: dict, rew: dict) -> None:
        health_series.append([obs[f"entity.{e}.health"] for e in entity_ids])
        for ev in event_ids:
            event_active[ev].append(int(obs.get(f"event.{ev}.active", 0)))
        reward_def.append(rew.get("reward.defender", 0.0))

    snapshot(obs, rewards)

    done = False
    while not done:
        obs, rewards, done = env.step_dict({})
        tick_history.append(int(obs["sim.tick"]))
        snapshot(obs, rewards)

    health_arr = np.array(health_series)          # shape: (ticks, entities)
    mean_health = health_arr.mean(axis=1)
    final_health_per_entity = {
        e: float(health_arr[-1, i]) for i, e in enumerate(entity_ids)
    }
    event_ticks = {ev: sum(v) for ev, v in event_active.items()}

    # Einbruchgeschwindigkeit: erster Tick wo mean_health < 0.9
    drop_tick = next((t for t, h in enumerate(mean_health) if h < 0.9), MAX_TICKS)

    # Erholung: letzter Teil (Ticks 270-365) vs. Minimum
    recovery_window = mean_health[270:]
    recovery = float(recovery_window.mean() - mean_health.min())

    return {
        "n_entities":    len(entity_ids),
        "n_events":      len(event_ids),
        "mean_health_t0":  float(mean_health[0]),
        "mean_health_t90": float(mean_health[min(90, len(mean_health)-1)]),
        "mean_health_t180":float(mean_health[min(180, len(mean_health)-1)]),
        "mean_health_t365":float(mean_health[-1]),
        "health_min":    float(mean_health.min()),
        "health_min_tick":int(mean_health.argmin()),
        "drop_tick":     drop_tick,
        "recovery":      recovery,
        "defender_reward_mean": float(np.mean(reward_def)),
        "defender_reward_min":  float(np.min(reward_def)),
        "final_health":  final_health_per_entity,
        "worst_entity":  min(final_health_per_entity.items(), key=lambda x: x[1]),
        "best_entity":   max(final_health_per_entity.items(), key=lambda x: x[1]),
        "event_ticks":   event_ticks,
        "events_always_active": sum(1 for v in event_ticks.values() if v >= 350),
        "events_never_active":  sum(1 for v in event_ticks.values() if v == 0),
        "mean_health_series": mean_health.tolist(),
    }


# ---------------------------------------------------------------------------
# Alle Szenarien durchlaufen
# ---------------------------------------------------------------------------
results: dict[str, dict] = {}
print(f"Starte Batch-Simulation ({len(SCENARIOS)} Szenarien × {MAX_TICKS} Ticks)\n")

for sid, fname, label in SCENARIOS:
    pdl_path = PDL_DIR / fname
    t0 = time.time()
    print(f"  [{sid}] {label:<25s} ...", end=" ", flush=True)
    try:
        r = run_scenario(pdl_path)
        results[sid] = {"label": label, "file": fname, **r}
        print(f"OK  ({time.time()-t0:.1f}s)  "
              f"Entities={r['n_entities']}  Events={r['n_events']}  "
              f"Health_final={r['mean_health_t365']:.4f}")
    except Exception as e:
        print(f"FEHLER: {e}")
        results[sid] = {"label": label, "file": fname, "error": str(e)}

print(f"\nFertig. {sum(1 for r in results.values() if 'error' not in r)}/9 Szenarien erfolgreich.\n")

# ---------------------------------------------------------------------------
# Vergleichstabelle
# ---------------------------------------------------------------------------
ok = [(sid, r) for sid, r in results.items() if "error" not in r]

print("=" * 95)
print("  SZENARIO-VERGLEICH — Alle 9 PROVIDER-Szenarien (365 Ticks, Dummy-Baseline, Seed 42)")
print("=" * 95)

# Haupttabelle
HDR = f"  {'ID':<4} {'Szenario':<28} {'E':>3} {'Ev':>3}  {'H@90':>6} {'H@180':>6} {'H@365':>6}  {'H_min':>6} {'@Tick':>5}  {'Drop':>5}  {'DefRew':>7}"
print(HDR)
print("  " + "-" * 91)
for sid, r in ok:
    drop = f"T{r['drop_tick']}" if r['drop_tick'] < MAX_TICKS else "—"
    print(f"  {sid:<4} {r['label']:<28} {r['n_entities']:>3} {r['n_events']:>3}  "
          f"{r['mean_health_t90']:>6.4f} {r['mean_health_t180']:>6.4f} {r['mean_health_t365']:>6.4f}  "
          f"{r['health_min']:>6.4f} {r['health_min_tick']:>5}  {drop:>5}  {r['defender_reward_mean']:>7.4f}")

# Ranking
print("\n--- Ranking: stabilste Systeme (Health@365, absteigend) ---")
ranked = sorted(ok, key=lambda x: -x[1]["mean_health_t365"])
for rank, (sid, r) in enumerate(ranked, 1):
    bar = "█" * int(r['mean_health_t365'] * 30)
    print(f"  {rank}. {sid} {r['label']:<28}  {r['mean_health_t365']:.4f}  {bar}")

print("\n--- Ranking: schnellster Einbruch (Drop-Tick, aufsteigend) ---")
ranked_drop = sorted(ok, key=lambda x: x[1]["drop_tick"])
for rank, (sid, r) in enumerate(ranked_drop, 1):
    drop = f"Tick {r['drop_tick']}" if r['drop_tick'] < MAX_TICKS else "kein Einbruch"
    print(f"  {rank}. {sid} {r['label']:<28}  {drop}")

print("\n--- Kritischste Einzelentities (finale Health) ---")
all_worst = [(sid, r["worst_entity"][0], r["worst_entity"][1]) for sid, r in ok]
all_worst.sort(key=lambda x: x[2])
for sid, eid, h in all_worst:
    bar = "█" * int(h * 20)
    print(f"  {sid}  {eid:<40s}  {h:.4f}  {bar}")

print("\n--- Event-Persistenz: dauerhaft aktive Events (≥350/365 Ticks) ---")
for sid, r in ok:
    always = r["events_always_active"]
    never  = r["events_never_active"]
    total  = r["n_events"]
    pct = always / total * 100
    print(f"  {sid} {r['label']:<28}  {always:>2}/{total} dauerhaft ({pct:4.0f}%)  "
          f"  {never:>2} nie aktiv")

print("\n--- Health-Zeitverlauf (ASCII-Sparkline, Ticks 0-365 in 15 Schritten) ---")
steps = list(range(0, 366, 26))
for sid, r in ok:
    series = r["mean_health_series"]
    line = ""
    chars = " ▁▂▃▄▅▆▇█"
    for t in steps:
        idx = min(t, len(series) - 1)
        h = series[idx]
        ci = int(h * 8)
        line += chars[ci]
    print(f"  {sid} {r['label']:<28}  {line}  [{series[0]:.3f}→{series[-1]:.3f}]")

# ---------------------------------------------------------------------------
# JSON speichern
# ---------------------------------------------------------------------------
out = Path(__file__).parent / "data" / "all_scenarios_results.json"
# Entferne mean_health_series aus JSON (zu groß für Übersicht)
save = {}
for sid, r in results.items():
    save[sid] = {k: v for k, v in r.items() if k != "mean_health_series"}
out.write_text(json.dumps(save, indent=2, ensure_ascii=False))
print(f"\nErgebnisse gespeichert: {out}")
