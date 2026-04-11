"""
Visualisierungen aller 9 PROVIDER-Szenarien aus all_scenarios_results.json
+ rohen Zeitreihen aus erneuter Simulation (für Heatmap und Verlaufsplots).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

sys.path.insert(0, str(Path(__file__).parent.parent))
from provider_sim.env.environment import ProviderEnvironment

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
PDL_DIR = Path(
    "/Users/aschaefer/Projekte/Forschung/PROVIDER/04_Apps/pdl-ontology-web-viewer/scenarios"
)
OUT_DIR = Path(__file__).parent.parent / "analysis"
OUT_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    ("S1", "s1-soja.pdl.yaml",               "Soja"),
    ("S2", "s2-halbleiter.pdl.yaml",          "Halbleiter"),
    ("S3", "s3-pharma.pdl.yaml",              "Pharma"),
    ("S4", "s4-duengemittel-adblue.pdl.yaml", "Düngemittel"),
    ("S5", "s5-wasseraufbereitung.pdl.yaml",  "Wasser"),
    ("S6", "s6-rechenzentren.pdl.yaml",       "Rechenzentren"),
    ("S7", "s7-seltene-erden.pdl.yaml",       "Seltene Erden"),
    ("S8", "s8-seefracht.pdl.yaml",           "Seefracht"),
    ("S9", "s9-unterwasserkabel.pdl.yaml",    "Unterwasserkabel"),
]
LABELS = {sid: lbl for sid, _, lbl in SCENARIOS}

# Farben je Szenario
COLORS = {
    "S1": "#4CAF50", "S2": "#2196F3", "S3": "#FF5722",
    "S4": "#FF9800", "S5": "#00BCD4", "S6": "#9C27B0",
    "S7": "#F44336", "S8": "#009688", "S9": "#795548",
}

MAX_TICKS = 365
SEED = 42

# ---------------------------------------------------------------------------
# Daten laden / simulieren
# ---------------------------------------------------------------------------
results_file = Path(__file__).parent / "data" / "all_scenarios_results.json"
with open(results_file) as f:
    meta = json.load(f)

print("Lade Zeitreihendaten (alle 9 Szenarien)...")
series_data: dict[str, dict] = {}

for sid, fname, label in SCENARIOS:
    pdl_path = PDL_DIR / fname
    print(f"  [{sid}] {label} ...", end=" ", flush=True)
    env = ProviderEnvironment(pdl_source=pdl_path, max_ticks=MAX_TICKS, seed=SEED)
    obs, rewards = env.reset_dict()

    entity_ids = env.engine.state.entity_ids
    event_ids  = env.engine.state.event_ids

    health_mat = []   # (ticks, entities)
    supply_mat = []
    event_mat  = []   # (ticks, events)

    def snap(obs):
        health_mat.append([obs[f"entity.{e}.health"] for e in entity_ids])
        supply_mat.append([obs[f"entity.{e}.supply"] for e in entity_ids])
        event_mat.append([int(obs.get(f"event.{ev}.active", 0)) for ev in event_ids])

    snap(obs)
    done = False
    while not done:
        obs, _, done = env.step_dict({})
        snap(obs)

    series_data[sid] = {
        "entity_ids": entity_ids,
        "event_ids":  event_ids,
        "health":     np.array(health_mat),   # (T, E)
        "supply":     np.array(supply_mat),   # (T, E)
        "events":     np.array(event_mat),    # (T, Ev)
        "mean_health": np.array(health_mat).mean(axis=1),
    }
    print("OK")

ticks = np.arange(MAX_TICKS + 1)

# ---------------------------------------------------------------------------
# Plot 1: Health-Zeitverläufe — alle 9 Szenarien überlagert
# ---------------------------------------------------------------------------
print("\nErstelle Plot 1: Health-Zeitverläufe...")
fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle("Mittlere System-Health — Alle 9 Szenarien (365 Ticks, Dummy-Baseline)",
             fontsize=14, fontweight="bold", y=0.98)

for ax, (sid, _, label) in zip(axes.flat, SCENARIOS):
    h = series_data[sid]["mean_health"]
    color = COLORS[sid]
    ax.fill_between(ticks, h, alpha=0.25, color=color)
    ax.plot(ticks, h, color=color, lw=2)
    ax.axhline(h[-1], color=color, lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0.7, color="gray", lw=0.5, ls=":", alpha=0.5)

    # Drop-Tick markieren
    drop = next((t for t, v in enumerate(h) if v < 0.9), None)
    if drop:
        ax.axvline(drop, color="red", lw=0.8, ls=":", alpha=0.6)

    ax.set_title(f"{sid} — {label}", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 365)
    ax.text(350, h[-1] + 0.02, f"{h[-1]:.3f}", ha="right", fontsize=8, color=color)
    ax.set_ylabel("Health", fontsize=8)
    ax.set_xlabel("Tick", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = OUT_DIR / "01_health_timeseries.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Vergleichsübersicht — Balkendiagramm finale Health + Drop-Ticks
# ---------------------------------------------------------------------------
print("Erstelle Plot 2: Vergleichsübersicht...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Szenario-Vergleich: Resilienzkennzahlen", fontsize=13, fontweight="bold")

sids = [s for s, _, _ in SCENARIOS]
labels_short = [LABELS[s] for s in sids]
finals = [series_data[s]["mean_health"][-1] for s in sids]
mins_  = [series_data[s]["mean_health"].min() for s in sids]
drops  = [next((t for t, v in enumerate(series_data[s]["mean_health"]) if v < 0.9), 365)
          for s in sids]
colors = [COLORS[s] for s in sids]

# Finale Health
bars = ax1.barh(labels_short[::-1], finals[::-1], color=colors[::-1], edgecolor="white", lw=0.5)
ax1.axvline(0.7, color="gray", ls="--", lw=1, alpha=0.7, label="0.7-Schwelle")
ax1.set_xlabel("Mittlere Health @ Tick 365")
ax1.set_title("Finale System-Health", fontweight="bold")
ax1.set_xlim(0, 1)
for bar, val in zip(bars, finals[::-1]):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)
ax1.grid(True, axis="x", alpha=0.3)

# Minimum Health
order = np.argsort(mins_)
bars2 = ax2.barh([labels_short[i] for i in order],
                 [mins_[i] for i in order],
                 color=[colors[i] for i in order], edgecolor="white", lw=0.5)
ax2.axvline(0.7, color="gray", ls="--", lw=1, alpha=0.7)
ax2.set_xlabel("Minimale System-Health (schlechtester Tick)")
ax2.set_title("Minimale Health (Worst Case)", fontweight="bold")
ax2.set_xlim(0, 1)
for bar, i in zip(bars2, order):
    ax2.text(mins_[i] + 0.005, bar.get_y() + bar.get_height()/2,
             f"{mins_[i]:.3f}", va="center", fontsize=8)
ax2.grid(True, axis="x", alpha=0.3)

# Drop-Tick
order3 = np.argsort(drops)
bar_colors3 = []
for i in order3:
    d = drops[i]
    if d <= 5:   bar_colors3.append("#F44336")
    elif d <= 15: bar_colors3.append("#FF9800")
    else:         bar_colors3.append("#4CAF50")

bars3 = ax3.barh([labels_short[i] for i in order3],
                 [drops[i] for i in order3],
                 color=bar_colors3, edgecolor="white", lw=0.5)
ax3.set_xlabel("Ticks bis Health < 0.9 (Drop-Tick)")
ax3.set_title("Einbruchgeschwindigkeit", fontweight="bold")
for bar, i in zip(bars3, order3):
    ax3.text(drops[i] + 0.3, bar.get_y() + bar.get_height()/2,
             f"T{drops[i]}", va="center", fontsize=8)
ax3.grid(True, axis="x", alpha=0.3)

red_p = mpatches.Patch(color="#F44336", label="Sofort (≤T5)")
org_p = mpatches.Patch(color="#FF9800", label="Schnell (T6–T15)")
grn_p = mpatches.Patch(color="#4CAF50", label="Verzögert (>T15)")
ax3.legend(handles=[red_p, org_p, grn_p], fontsize=8, loc="lower right")

plt.tight_layout()
out2 = OUT_DIR / "02_resilience_comparison.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out2}")

# ---------------------------------------------------------------------------
# Plot 3: Health-Heatmap — alle Szenarien × Zeit
# ---------------------------------------------------------------------------
print("Erstelle Plot 3: Health-Heatmap...")
fig, ax = plt.subplots(figsize=(16, 5))

health_matrix = np.array([series_data[s]["mean_health"] for s, _, _ in SCENARIOS])

cmap = LinearSegmentedColormap.from_list(
    "health", ["#B71C1C", "#FF5722", "#FFC107", "#8BC34A", "#1B5E20"]
)
im = ax.imshow(health_matrix, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0,
               interpolation="nearest")

ax.set_yticks(range(len(SCENARIOS)))
ax.set_yticklabels([f"{sid} {lbl}" for sid, _, lbl in SCENARIOS], fontsize=9)
ax.set_xlabel("Tick", fontsize=10)
ax.set_title("System-Health über Zeit — Alle Szenarien (rot=kritisch, grün=stabil)",
             fontsize=12, fontweight="bold")

# x-Achse: Monatsbeschriftungen
month_ticks = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
ax.set_xticks(month_ticks)
ax.set_xticklabels([f"T{t}" for t in month_ticks], fontsize=7)

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label("Mittlere Health", fontsize=9)

# Finale Werte annotieren
for i, (sid, _, _) in enumerate(SCENARIOS):
    h_final = health_matrix[i, -1]
    ax.text(368, i, f"{h_final:.3f}", va="center", ha="left", fontsize=8,
            color=COLORS[sid], fontweight="bold")

plt.tight_layout()
out3 = OUT_DIR / "03_health_heatmap.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out3}")

# ---------------------------------------------------------------------------
# Plot 4: Event-Aktivierungsmuster (Heatmap pro Szenario, 3×3)
# ---------------------------------------------------------------------------
print("Erstelle Plot 4: Event-Aktivierungsmuster...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Event-Aktivierungsmuster — Alle Szenarien", fontsize=13, fontweight="bold")

cmap_ev = LinearSegmentedColormap.from_list("ev", ["#ECEFF1", "#B71C1C"])

for ax, (sid, _, label) in zip(axes.flat, SCENARIOS):
    ev_mat = series_data[sid]["events"].T   # (events, ticks)
    ev_ids = series_data[sid]["event_ids"]

    # Events nach Aktivierungshäufigkeit sortieren
    order = np.argsort(-ev_mat.sum(axis=1))
    ev_mat_sorted = ev_mat[order]
    ev_labels = [ev_ids[i] for i in order]

    # Namen kürzen
    short = [e.replace("_", " ")[:28] for e in ev_labels]

    ax.imshow(ev_mat_sorted, aspect="auto", cmap=cmap_ev, vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_yticks(range(len(ev_labels)))
    ax.set_yticklabels(short, fontsize=5)
    ax.set_xlabel("Tick", fontsize=7)
    ax.set_title(f"{sid} — {label}", fontsize=9, fontweight="bold", color=COLORS[sid])
    ax.set_xticks([0, 90, 180, 270, 365])
    ax.set_xticklabels(["0", "90", "180", "270", "365"], fontsize=6)

plt.tight_layout()
out4 = OUT_DIR / "04_event_patterns.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out4}")

# ---------------------------------------------------------------------------
# Plot 5: Entity-Health-Heatmap je Szenario (3×3)
# ---------------------------------------------------------------------------
print("Erstelle Plot 5: Entity-Health-Heatmaps...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Entity-Health-Verlauf — Alle Szenarien", fontsize=13, fontweight="bold")

cmap_ent = LinearSegmentedColormap.from_list(
    "ent", ["#B71C1C", "#FF5722", "#FFC107", "#8BC34A", "#1B5E20"]
)

for ax, (sid, _, label) in zip(axes.flat, SCENARIOS):
    health_ent = series_data[sid]["health"].T  # (entities, ticks)
    entity_ids = series_data[sid]["entity_ids"]

    # nach finaler Health sortieren
    final_h = health_ent[:, -1]
    order = np.argsort(final_h)
    mat_sorted = health_ent[order]
    ent_labels = [entity_ids[i] for i in order]
    short = [e.replace("_", " ")[:25] for e in ent_labels]

    im = ax.imshow(mat_sorted, aspect="auto", cmap=cmap_ent, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(ent_labels)))
    ax.set_yticklabels(short, fontsize=5)
    ax.set_xlabel("Tick", fontsize=7)
    ax.set_title(f"{sid} — {label}", fontsize=9, fontweight="bold", color=COLORS[sid])
    ax.set_xticks([0, 90, 180, 270, 365])
    ax.set_xticklabels(["0", "90", "180", "270", "365"], fontsize=6)

plt.tight_layout()
out5 = OUT_DIR / "05_entity_health_heatmaps.png"
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out5}")

# ---------------------------------------------------------------------------
# Plot 6: Überlagerter Zeitverlauf aller Szenarien (ein Plot)
# ---------------------------------------------------------------------------
print("Erstelle Plot 6: Überlagerung aller Szenarien...")
fig, ax = plt.subplots(figsize=(14, 7))

ax.fill_between(ticks,
                [min(series_data[s]["mean_health"][t] for s, _, _ in SCENARIOS) for t in range(len(ticks))],
                [max(series_data[s]["mean_health"][t] for s, _, _ in SCENARIOS) for t in range(len(ticks))],
                alpha=0.1, color="gray", label="Min–Max-Band")

for sid, _, label in SCENARIOS:
    h = series_data[sid]["mean_health"]
    ax.plot(ticks, h, color=COLORS[sid], lw=2, label=f"{sid} {label}", alpha=0.85)

ax.axhline(0.7, color="black", ls="--", lw=1, alpha=0.4, label="0.7-Referenz")
ax.set_xlabel("Tick (Simulationstage)", fontsize=11)
ax.set_ylabel("Mittlere System-Health", fontsize=11)
ax.set_title("Alle 9 Szenarien im Vergleich — System-Health über 365 Ticks",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 365)
ax.set_ylim(0.4, 1.05)
ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Endbeschriftungen rechts
for sid, _, label in SCENARIOS:
    h = series_data[sid]["mean_health"]
    ax.annotate(f"{sid}: {h[-1]:.3f}", xy=(365, h[-1]),
                xytext=(367, h[-1]), fontsize=7.5, color=COLORS[sid],
                va="center", fontweight="bold")

ax.set_xlim(0, 400)
plt.tight_layout()
out6 = OUT_DIR / "06_all_scenarios_overlay.png"
plt.savefig(out6, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out6}")

# ---------------------------------------------------------------------------
# Plot 7: Radar-Chart — Resilienzdimensionen
# ---------------------------------------------------------------------------
print("Erstelle Plot 7: Radar-Chart...")
dims = ["Health\n@365", "Health\nMin", "Einbruch\n(invertiert)", "Event-\nPersistenz\n(inv.)", "Defender\nReward"]
n_dims = len(dims)
angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.set_title("Resilienzdimensionen aller Szenarien\n(normiert, höher = resilienter)",
             fontsize=12, fontweight="bold", pad=20)

for sid, _, label in SCENARIOS:
    r = meta[sid]
    h_final = series_data[sid]["mean_health"][-1]
    h_min   = series_data[sid]["mean_health"].min()
    drop    = next((t for t, v in enumerate(series_data[sid]["mean_health"]) if v < 0.9), 365)
    ev_pers = 1.0 - (r["events_always_active"] / r["n_events"])
    def_rew = r["defender_reward_mean"]

    # Normierung auf [0,1]
    drop_norm  = min(drop / 30.0, 1.0)   # 30 Ticks = maximale Robustheit
    vals = [h_final, h_min, drop_norm, ev_pers, def_rew]
    vals += vals[:1]

    ax.plot(angles, vals, color=COLORS[sid], lw=2, label=f"{sid} {label}")
    ax.fill(angles, vals, color=COLORS[sid], alpha=0.07)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims, fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

plt.tight_layout()
out7 = OUT_DIR / "07_radar_resilience.png"
plt.savefig(out7, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out7}")

# ---------------------------------------------------------------------------
# Plot 8: Kritischste Entity je Szenario — Detailverläufe
# ---------------------------------------------------------------------------
print("Erstelle Plot 8: Kritischste Entities...")
fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
fig.suptitle("Kritischste Entity je Szenario — Health-Verlauf",
             fontsize=13, fontweight="bold")

for ax, (sid, _, label) in zip(axes.flat, SCENARIOS):
    entity_ids = series_data[sid]["entity_ids"]
    health_ent = series_data[sid]["health"]      # (T, E)
    final_h = health_ent[-1, :]
    worst_idx = int(np.argmin(final_h))
    worst_id  = entity_ids[worst_idx]
    worst_h   = health_ent[:, worst_idx]

    # Top-3 kritischste
    top3_idx = np.argsort(final_h)[:3]
    for i, idx in enumerate(top3_idx):
        alpha = [1.0, 0.5, 0.3][i]
        lw    = [2.5, 1.5, 1.0][i]
        name  = entity_ids[idx].replace("_", " ")
        ax.plot(ticks, health_ent[:, idx], lw=lw, alpha=alpha,
                color=COLORS[sid], label=name)

    ax.fill_between(ticks, worst_h, alpha=0.15, color=COLORS[sid])
    ax.axhline(0.5, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.set_title(f"{sid} — {label}", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 365)
    ax.set_ylabel("Health", fontsize=7)
    ax.set_xlabel("Tick", fontsize=7)
    ax.legend(fontsize=5.5, loc="lower left")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out8 = OUT_DIR / "08_worst_entities.png"
plt.savefig(out8, dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {out8}")

# ---------------------------------------------------------------------------
# Zusammenfassung
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("  Alle Visualisierungen gespeichert in:")
print(f"  {OUT_DIR}")
print("="*55)
for f in sorted(OUT_DIR.glob("0*.png")):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name}  ({size_kb} KB)")
