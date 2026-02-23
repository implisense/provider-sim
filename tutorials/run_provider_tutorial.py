"""Tutorial: PROVIDER ProviderEnvironment als palaestrAI-Environment.

Dieses Skript demonstriert Schritt fuer Schritt, wie die PROVIDER-Simulation
als palaestrAI-Environment genutzt wird -- von der PDL-Szenario-Ladung bis
zu einem vollstaendigen Ablauf mit Attacker- und Defender-Aktionen.

Ausfuehren:
    cd palestrai_simulation
    python tutorials/run_provider_tutorial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Stelle sicher, dass provider_sim im Pfad ist (fuer Ausfuehren aus dem
# tutorials/-Unterverzeichnis heraus)
_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

# ===========================================================================
# Abschnitt 1: PDL-Szenario laden
# ===========================================================================
print("=" * 60)
print("ABSCHNITT 1 — PDL-Szenario laden")
print("=" * 60)

from provider_sim.pdl.parser import load_pdl

PDL_PATH = _BASE / "scenarios" / "s1-soja.pdl.yaml"

doc = load_pdl(PDL_PATH)
print(f"Szenario-ID  : {doc.scenario.id}")
print(f"Szenario-Name: {doc.scenario.name}")
print(f"Entities     : {len(doc.entities)}")
print(f"Events       : {len(doc.events)}")
print()
print("Erste 5 Entities:")
for ent in doc.entities[:5]:
    print(f"  [{ent.type.value:15s}] {ent.id:30s}  vulnerability={ent.vulnerability:.1f}")
print("  ...")
print()

# ===========================================================================
# Abschnitt 2: ProviderEnvironment instanziieren
# ===========================================================================
print("=" * 60)
print("ABSCHNITT 2 — ProviderEnvironment instanziieren")
print("=" * 60)

from provider_sim.env.environment import ProviderEnvironment

env = ProviderEnvironment(PDL_PATH, seed=42, max_ticks=365)

print(f"Sensoren    : {len(env.sensor_names)}")
print(f"Aktuatoren  : {len(env.actuator_names)}")
print()
print("Sensor-Kategorien:")
supply_sensors = [s for s in env.sensor_names if s.endswith(".supply")]
event_sensors  = [s for s in env.sensor_names if s.startswith("event.")]
print(f"  entity.*.supply  : {len(supply_sensors)}")
print(f"  entity.*.demand  : {len(supply_sensors)}")
print(f"  entity.*.price   : {len(supply_sensors)}")
print(f"  entity.*.health  : {len(supply_sensors)}")
print(f"  event.*.active   : {len(event_sensors)}")
print(f"  sim.tick         : 1")
print()
print("Beispiel-Aktuatoren (erste 4):")
for aid in env.actuator_names[:4]:
    print(f"  {aid}")
print("  ...")
print()

# ===========================================================================
# Abschnitt 3: Standalone-Demo (ohne Orchestrator) — 5 Ticks
# ===========================================================================
print("=" * 60)
print("ABSCHNITT 3 — Standalone-Demo: 5 Ticks (reset_dict / step_dict)")
print("=" * 60)

obs, rewards = env.reset_dict()
print(f"Nach reset_dict(): {len(obs)} Beobachtungen, Rewards: {rewards}")
print()
print(f"{'Tick':>4}  {'attacker_reward':>15}  {'defender_reward':>15}  "
      f"{'mean_supply':>11}  {'mean_health':>11}")
print("-" * 65)

for tick in range(5):
    # Attacker greift schwach an (Staerke 0.1), Defender passiv
    actions: dict = {}
    for ent in doc.entities:
        actions[f"attacker.{ent.id}"] = 0.1
        actions[f"defender.{ent.id}"] = 0.0

    obs, rewards, done = env.step_dict(actions)

    mean_supply = float(np.mean([
        v for k, v in obs.items() if k.endswith(".supply")
    ]))
    mean_health = float(np.mean([
        v for k, v in obs.items() if k.endswith(".health")
    ]))

    print(
        f"{tick + 1:>4}  "
        f"{rewards['reward.attacker']:>15.4f}  "
        f"{rewards['reward.defender']:>15.4f}  "
        f"{mean_supply:>11.4f}  "
        f"{mean_health:>11.4f}"
    )

    if done:
        print("  -> Episode beendet (max_ticks erreicht)")
        break

print()

# ===========================================================================
# Abschnitt 4: palaestrAI-ABC-Demo (start_environment / update)
# ===========================================================================
print("=" * 60)
print("ABSCHNITT 4 — palaestrAI-ABC-Demo: start_environment / update")
print("=" * 60)

from provider_sim.env.environment import ActuatorInformation, Box

baseline = env.start_environment()
print(f"start_environment() abgeschlossen")
print(f"  sensors_available  : {len(baseline.sensors_available)}")
print(f"  actuators_available: {len(baseline.actuators_available)}")
print(f"  simtime.ticks      : {baseline.simtime.simtime_ticks}")
print()

# Null-Aktionen (alle Aktuatoren auf 0)
null_actions = [
    ActuatorInformation(
        np.array([0.0], dtype=np.float32),
        Box(0, 1, shape=(1,), dtype=np.float32),
        actuator_id=aid,
    )
    for aid in env.actuator_names
]

state = env.update(null_actions)
print(f"update() (ein Tick mit Null-Aktionen)")
print(f"  sensor_information : {len(state.sensor_information)} Sensoren")
print(f"  rewards            : {[r.reward_id for r in state.rewards]}")
print(f"  done               : {state.done}")
print(f"  simtime.ticks      : {state.simtime.simtime_ticks}")
print()
print("Reward-Werte:")
for r in state.rewards:
    val = float(np.asarray(r.reward_value).item())
    print(f"  {r.reward_id:20s} = {val:.4f}")
print()
print("Zero-Sum-Check: attacker + defender =",
      sum(float(np.asarray(r.reward_value).item()) for r in state.rewards))
print()
print("=" * 60)
print("Tutorial abgeschlossen.")
print("=" * 60)
