#!/usr/bin/env python
"""S1-Soja Simulation mit KG-Parametrisierung (standalone, kein palaestrAI-Orchestrator).

Ablauf:
  1. KG-Schocks laden (aus JSON-Datei oder live von coypu-kg-analyser)
  2. PDL-Szenario anreichern (apply_kg_shocks)
  3. Simulation 365 Ticks mit ProviderEnvironment.step_dict()
  4. Gesundheits- und Event-Statistiken ausgeben

Verwendung:
    # Mit vorgespeichertem Snapshot (kein SPARQL-Endpoint nötig):
    python experiments/run_s1_kg_simulation.py

    # Mit Live-Abfrage vom KG:
    python experiments/run_s1_kg_simulation.py --live

    # Mit eigenem Schocks-File:
    python experiments/run_s1_kg_simulation.py --shocks experiments/shocks_2026-03-04.json

    # Eigene PDL-Datei:
    python experiments/run_s1_kg_simulation.py --pdl scenarios/s1-soja.pdl.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# provider_sim importierbar machen wenn Skript direkt ausgeführt wird
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from provider_sim.adapters.kg_shocks import S1_KG_TO_PDL, apply_kg_shocks
from provider_sim.env.environment import ProviderEnvironment
from provider_sim.pdl.parser import load_pdl

# Standardpfade
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCENARIO_DIR = Path("/Users/aschaefer/Projekte/Forschung/PROVIDER/06_Szenarien/scenarios")
_DEFAULT_PDL = _SCENARIO_DIR / "s1-soja.pdl.yaml"
_DEFAULT_SHOCKS = _REPO_ROOT / "experiments" / "data" / "shocks_2026-03-04.json"


def fetch_live_shocks() -> list[dict]:
    """Ruft KG-Schocks live von coypu-kg-analyser ab."""
    print("Rufe Live-KG-Daten ab (copper.coypu.org)...")
    result = subprocess.run(
        [sys.executable, "-m", "coypu_kg_analyser", "parametrize-s1"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FEHLER beim KG-Abruf: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(result.stdout)
    print(f"KG-Abfrage erfolgreich: {data.get('summary', '')}")
    return data["shocks"]


def load_shocks(path: Path) -> list[dict]:
    """Lädt Schocks aus JSON-Datei."""
    data = json.loads(path.read_text(encoding="utf-8"))
    generated_at = data.get("generated_at", "unbekannt")
    print(f"Schocks geladen aus {path.name} (generiert: {generated_at})")
    return data["shocks"]


def run_simulation(
    pdl_path: Path,
    shocks: list[dict],
    max_ticks: int = 365,
    seed: int = 42,
) -> dict:
    """Führt Simulation durch und gibt Statistiken zurück."""
    doc = load_pdl(pdl_path)
    enriched_doc = apply_kg_shocks(doc, shocks, id_mapping=S1_KG_TO_PDL)

    kg_events = [ev for ev in enriched_doc.events if ev.id.startswith("kg_shock_")]
    print(f"\nSzenario: {doc.scenario.id}")
    print(f"Entitäten: {len(doc.entities)}")
    print(f"PDL-Events: {len(doc.events) - len(kg_events)}")
    print(f"KG-Events injiziert: {len(kg_events)}")
    for ev in kg_events:
        impact = ev.impact
        impact_str = ""
        if impact.supply:
            impact_str = f"supply {impact.supply.raw}"
        elif impact.price:
            impact_str = f"price {impact.price.raw}"
        print(f"  → {ev.id}: {impact_str}")

    env = ProviderEnvironment(pdl_source=enriched_doc, seed=seed, max_ticks=max_ticks)
    obs, _ = env.reset_dict()

    entity_ids = env.engine.state.entity_ids
    event_ids = env.engine.state.event_ids
    n_ticks = 0

    # Zeitreihen sammeln
    health_series: list[list[float]] = []
    event_active: dict[str, list[int]] = {eid: [] for eid in event_ids}

    def _snapshot() -> None:
        health_series.append([obs[f"entity.{e}.health"] for e in entity_ids])
        for eid in event_ids:
            event_active[eid].append(int(obs.get(f"event.{eid}.active", 0)))

    _snapshot()
    done = False
    while not done:
        obs, _, done = env.step_dict({})
        _snapshot()
        n_ticks += 1

    arr = np.array(health_series)  # (ticks+1, entities)
    mean_health = arr.mean(axis=1)

    # Aktivste Events
    active_ticks = {
        eid: sum(event_active[eid]) for eid in event_ids if sum(event_active[eid]) > 0
    }
    top_events = sorted(active_ticks.items(), key=lambda x: -x[1])[:10]

    return {
        "n_ticks": n_ticks,
        "n_entities": len(entity_ids),
        "n_events": len(event_ids),
        "n_kg_events": len(kg_events),
        "health_t0": float(mean_health[0]),
        "health_min": float(mean_health.min()),
        "health_min_tick": int(mean_health.argmin()),
        "health_t365": float(mean_health[-1]),
        "entity_health_final": {
            e: float(arr[-1, i]) for i, e in enumerate(entity_ids)
        },
        "top_active_events": top_events,
    }


def print_report(stats: dict) -> None:
    """Gibt formatierten Report aus."""
    print("\n" + "=" * 60)
    print("S1-SOJA SIMULATION — ERGEBNISSE")
    print("=" * 60)
    print(f"Ticks:          {stats['n_ticks']}")
    print(f"Entitäten:      {stats['n_entities']}")
    print(f"Events total:   {stats['n_events']} ({stats['n_kg_events']} KG-injiziert)")
    print(f"\nDurchschn. Health:")
    print(f"  Tick   0: {stats['health_t0']:.3f}")
    print(f"  Minimum:  {stats['health_min']:.3f} (Tick {stats['health_min_tick']})")
    print(f"  Tick 365: {stats['health_t365']:.3f}")

    print(f"\nEntitys — finale Health:")
    for entity, h in sorted(stats["entity_health_final"].items(), key=lambda x: x[1]):
        bar = "█" * int(h * 20)
        print(f"  {entity:<35} {h:.3f} {bar}")

    if stats["top_active_events"]:
        print(f"\nAktivste Events (Tick-Aktivierungen):")
        for eid, ticks in stats["top_active_events"]:
            print(f"  {eid:<45} {ticks:>4} Ticks")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S1-Soja Simulation mit KG-Parametrisierung"
    )
    parser.add_argument(
        "--pdl", default=str(_DEFAULT_PDL),
        help=f"Pfad zur PDL-YAML-Datei (default: {_DEFAULT_PDL.name})",
    )
    parser.add_argument(
        "--shocks", default=str(_DEFAULT_SHOCKS),
        help="Pfad zur KG-Schocks-JSON-Datei",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="KG-Schocks live von coypu-kg-analyser abrufen (ignoriert --shocks)",
    )
    parser.add_argument("--max-ticks", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    shocks = fetch_live_shocks() if args.live else load_shocks(Path(args.shocks))
    stats = run_simulation(
        Path(args.pdl), shocks, max_ticks=args.max_ticks, seed=args.seed
    )
    print_report(stats)


if __name__ == "__main__":
    main()
