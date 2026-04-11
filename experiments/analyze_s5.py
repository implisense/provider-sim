"""
Standalone-Simulation und Auswertung: S5 Wasseraufbereitung (Dummy-Baseline)
Entspricht DummyBrain/DummyMuscle — keine Agentenaktionen (Nullsteuerung).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from provider_sim.env.environment import ProviderEnvironment

PDL_PATH = Path(__file__).parent / "configs" / "s5-wasseraufbereitung.pdl.yaml"
MAX_TICKS = 365
SEED = 42

# ---------------------------------------------------------------------------
# Simulation starten
# ---------------------------------------------------------------------------
env = ProviderEnvironment(pdl_source=PDL_PATH, max_ticks=MAX_TICKS, seed=SEED)
obs, rewards = env.reset_dict()

entity_ids = env.engine.state.entity_ids
event_ids = env.engine.state.event_ids

history: dict[str, dict[str, list[float]]] = {
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

mean_health = np.array([
    np.mean([history[eid]["health"][t] for eid in entity_ids])
    for t in range(len(ticks))
])

final_health = {eid: history[eid]["health"][-1] for eid in entity_ids}
sorted_health = sorted(final_health.items(), key=lambda x: x[1])

event_active_ticks = {ev: sum(event_history[ev]) for ev in event_ids}
sorted_events = sorted(event_active_ticks.items(), key=lambda x: -x[1])

# Health-Verlauf pro Entity: Zeitpunkt des stärksten Einbruchs
worst_tick = {
    eid: int(np.argmin(history[eid]["health"]))
    for eid in entity_ids
}

# ---------------------------------------------------------------------------
# Konsolen-Report
# ---------------------------------------------------------------------------
print("\n" + "="*68)
print("  S5 WASSERAUFBEREITUNG — Auswertung (365 Ticks, Dummy-Baseline)")
print("="*68)

print("\n--- Mittlere System-Health ---")
for t in (0, 30, 60, 90, 180, 270, 365):
    idx = min(t, len(mean_health) - 1)
    print(f"  Tick {t:3d}:  {mean_health[idx]:.4f}")
print(f"  Minimum:  {mean_health.min():.4f} (Tick {int(mean_health.argmin())})")
print(f"  Maximum:  {mean_health.max():.4f} (Tick {int(mean_health.argmax())})")

print("\n--- Finale Health (aufsteigend, kritischste zuerst) ---")
for eid, h in sorted_health[:8]:
    bar = "█" * int(h * 20)
    wt = worst_tick[eid]
    wh = min(history[eid]["health"])
    print(f"  {eid:<30s} {h:.4f}  {bar}  (Min={wh:.3f}@T{wt})")
print("  ---")
for eid, h in sorted_health[-3:]:
    bar = "█" * int(h * 20)
    print(f"  {eid:<30s} {h:.4f}  {bar}")

print("\n--- Event-Aktivierung (Ticks aktiv, absteigend) ---")
for ev, n in sorted_events:
    bar = "█" * int(n / 3.65)
    print(f"  {ev:<35s} {n:3d}/365  {bar}")

print("\n--- Reward-Statistik (Null-Steuerung, kein Angreifer) ---")
att = np.array(reward_history["attacker"])
dfd = np.array(reward_history["defender"])
print(f"  Attacker-Reward:  mean={att.mean():.4f}  min={att.min():.4f}  max={att.max():.4f}")
print(f"  Defender-Reward:  mean={dfd.mean():.4f}  min={dfd.min():.4f}  max={dfd.max():.4f}")
print(f"  Zero-Sum-Check:   {(att + dfd).mean():.6f} (soll 1.0)")

print("\n--- Supply-Kaskade: Chlor-Lieferkette → Bevölkerung ---")
cascade = [
    "electricity_grid",
    "chlorine_supply",
    "small_rural_utilities",
    "small_urban_utilities",
    "medium_utilities",
    "large_utilities",
    "hospitals",
    "population_rural",
    "population_urban",
]
for eid in cascade:
    if eid in history:
        s = history[eid]["supply"]
        h = history[eid]["health"]
        print(f"  {eid:<30s}  supply: {s[0]:.3f}→{s[-1]:.3f} (min={min(s):.3f})  "
              f"health: {h[0]:.3f}→{h[-1]:.3f}")

# ---------------------------------------------------------------------------
# Ergebnisse als JSON speichern
# ---------------------------------------------------------------------------
out = Path(__file__).parent / "data" / "s5_results.json"
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
