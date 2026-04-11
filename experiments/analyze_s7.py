"""
Standalone-Simulation und Auswertung: S7 Seltene Erden (Dummy-Baseline)
Entspricht DummyBrain/DummyMuscle — keine Agentenaktionen (Nullsteuerung).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider_sim.env.environment import ProviderEnvironment

PDL_PATH = Path(
    "/Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/pdl-ontology-web-viewer/scenarios/s7-seltene-erden.pdl.yaml"
)
MAX_TICKS = 365
SEED = 42

# ---------------------------------------------------------------------------
# Simulation starten
# ---------------------------------------------------------------------------
env = ProviderEnvironment(pdl_source=PDL_PATH, max_ticks=MAX_TICKS, seed=SEED)
obs, rewards = env.reset_dict()

entity_ids = env.engine.state.entity_ids
event_ids = env.engine.state.event_ids

# Zeitreihen-Puffer
history: dict[str, list[float]] = {
    eid: {"supply": [], "demand": [], "price": [], "health": []}
    for eid in entity_ids
}
event_history: dict[str, list[int]] = {ev: [] for ev in event_ids}
reward_history = {"attacker": [], "defender": []}
tick_history: list[int] = []

def collect(obs: dict, rewards: dict, tick: int) -> None:
    tick_history.append(tick)
    for eid in entity_ids:
        for metric in ("supply", "demand", "price", "health"):
            history[eid][metric].append(obs[f"entity.{eid}.{metric}"])
    for ev in event_ids:
        event_history[ev].append(int(obs.get(f"event.{ev}.active", 0)))
    reward_history["attacker"].append(rewards.get("reward.attacker", 0.0))
    reward_history["defender"].append(rewards.get("reward.defender", 0.0))

collect(obs, rewards, 0)

# DummyBrain = Nullsteuerung (keine Aktionen)
done = False
while not done:
    obs, rewards, done = env.step_dict({})
    tick = int(obs["sim.tick"])
    collect(obs, rewards, tick)

print(f"Simulation abgeschlossen: {len(tick_history)} Ticks")

# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------
ticks = np.array(tick_history)

# Mittlere Health aller Entities über Zeit
mean_health = np.array([
    np.mean([history[eid]["health"][t] for eid in entity_ids])
    for t in range(len(ticks))
])

# Finale Health-Werte
final_health = {
    eid: history[eid]["health"][-1] for eid in entity_ids
}
sorted_health = sorted(final_health.items(), key=lambda x: x[1])

# Event-Aktivierungen (Anzahl Ticks aktiv)
event_active_ticks = {
    ev: sum(event_history[ev]) for ev in event_ids
}
sorted_events = sorted(event_active_ticks.items(), key=lambda x: -x[1])

# ---------------------------------------------------------------------------
# Konsolen-Report
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("  S7 SELTENE ERDEN — Auswertung (365 Ticks, Dummy-Baseline)")
print("="*65)

print("\n--- Mittlere System-Health ---")
print(f"  Tick 0:    {mean_health[0]:.4f}")
print(f"  Tick 90:   {mean_health[90]:.4f}")
print(f"  Tick 180:  {mean_health[180]:.4f}")
print(f"  Tick 270:  {mean_health[270]:.4f}")
print(f"  Tick 365:  {mean_health[-1]:.4f}")
print(f"  Minimum:   {mean_health.min():.4f} (Tick {int(mean_health.argmin())})")
print(f"  Maximum:   {mean_health.max():.4f} (Tick {int(mean_health.argmax())})")

print("\n--- Finale Health (aufsteigend, kritischste zuerst) ---")
for eid, h in sorted_health[:7]:
    bar = "█" * int(h * 20)
    print(f"  {eid:<35s} {h:.4f}  {bar}")
print("  ...")
for eid, h in sorted_health[-3:]:
    bar = "█" * int(h * 20)
    print(f"  {eid:<35s} {h:.4f}  {bar}")

print("\n--- Event-Aktivierung (Ticks aktiv, absteigend) ---")
for ev, n in sorted_events[:10]:
    bar = "█" * int(n / 3.65)
    print(f"  {ev:<40s} {n:3d}/365  {bar}")

print("\n--- Reward-Statistik (Null-Steuerung, kein Angreifer) ---")
att = np.array(reward_history["attacker"])
dfd = np.array(reward_history["defender"])
print(f"  Attacker-Reward:  mean={att.mean():.4f}  min={att.min():.4f}  max={att.max():.4f}")
print(f"  Defender-Reward:  mean={dfd.mean():.4f}  min={dfd.min():.4f}  max={dfd.max():.4f}")
print(f"  Zero-Sum-Check:   {(att + dfd).mean():.6f} (soll 1.0)")

print("\n--- Supply-Entwicklung: Rohstoff-Quelle vs. Endabnahme ---")
key_entities = [
    "china_rare_earth_mines",
    "ndfeb_magnet_producers",
    "ev_motor_producers_de",
    "ev_market_de",
    "energiewende_wind",
]
for eid in key_entities:
    if eid in history:
        s = history[eid]["supply"]
        print(f"  {eid:<35s}  Δ supply: {s[0]:.3f} → {s[-1]:.3f}  (min={min(s):.3f})")

# ---------------------------------------------------------------------------
# Ergebnisse als JSON speichern
# ---------------------------------------------------------------------------
out = Path(__file__).parent / "data" / "s7_results.json"
results = {
    "ticks": tick_history,
    "mean_health": mean_health.tolist(),
    "final_health": final_health,
    "event_active_ticks": event_active_ticks,
    "reward_attacker": reward_history["attacker"],
    "reward_defender": reward_history["defender"],
    "entity_history": {
        eid: {m: history[eid][m] for m in ("supply", "demand", "price", "health")}
        for eid in entity_ids
    },
}
out.write_text(json.dumps(results, indent=2))
print(f"\nErgebnisse gespeichert: {out}")
