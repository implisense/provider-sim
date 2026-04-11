"""Analyse und Visualisierung der ICIO-Simulationsergebnisse aus palaestrai.db."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DB_PATH = Path(__file__).parent / "data" / "palaestrai.db"
OUT_DIR = Path(__file__).parent.parent / "analysis"
OUT_DIR.mkdir(exist_ok=True)

# ── Gruppen für farbliche Kodierung ─────────────────────────────────────────
GROUPS = {
    "Produktion (BR/AR/US)": ["brazil_farms", "argentina_farms", "us_farms"],
    "Logistik": ["santos_port", "paranagua_port", "rotterdam_port", "hamburg_port", "us_gulf_ports"],
    "Verarbeitung EU": ["eu_oil_mills", "feed_mills", "soy_oil_market"],
    "Nachfrage EU": ["poultry_farms", "pig_farms", "dairy_farms", "food_retail", "consumers"],
    "Puffer/Input": ["alternative_protein_sources", "strategic_feed_reserves", "fertilizer_supply", "gas_supply"],
}

GROUP_COLORS = {
    "Produktion (BR/AR/US)": "#2ecc71",
    "Logistik":              "#3498db",
    "Verarbeitung EU":       "#e67e22",
    "Nachfrage EU":          "#e74c3c",
    "Puffer/Input":          "#9b59b6",
}

ENTITY_LABELS = {
    "brazil_farms":                "Sojaanbau Brasilien",
    "argentina_farms":             "Sojaanbau Argentinien",
    "us_farms":                    "Sojaanbau USA",
    "santos_port":                 "Hafen Santos",
    "paranagua_port":              "Hafen Paranaguá",
    "rotterdam_port":              "Hafen Rotterdam",
    "hamburg_port":                "Hafen Hamburg",
    "us_gulf_ports":               "US-Golfhäfen",
    "eu_oil_mills":                "EU Ölmühlen",
    "feed_mills":                  "Futtermittelhersteller",
    "soy_oil_market":              "Sojaöl-Markt",
    "poultry_farms":               "Geflügelbetriebe",
    "pig_farms":                   "Schweinebetriebe",
    "dairy_farms":                 "Milchwirtschaft",
    "food_retail":                 "Lebensmitteleinzelhandel",
    "consumers":                   "Verbraucher",
    "alternative_protein_sources": "Alt. Proteinquellen",
    "strategic_feed_reserves":     "Futtermittel-Lager",
    "fertilizer_supply":           "Düngemittelversorgung",
    "gas_supply":                  "Erdgasversorgung",
}


def load_health_timeseries(db_path: Path) -> dict[str, list[float]]:
    """Liest alle Health-Sensoren aus palaestrai.db; gibt {entity_id: [tick1..tick365]}."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT state_dump FROM world_states ORDER BY id").fetchall()
    conn.close()

    series: dict[str, list[float]] = {}
    for (dump,) in rows:
        data = json.loads(dump)
        for s in data:
            sid: str = s.get("sensor_id", "")
            if ".health" not in sid:
                continue
            # sensor_id = "provider_env.entity.<id>.health"
            entity_id = sid.split(".entity.")[-1].replace(".health", "")
            val = s["sensor_value"]["values"][0]
            series.setdefault(entity_id, []).append(float(val))
    return series


def entity_group(eid: str) -> str:
    for grp, members in GROUPS.items():
        if eid in members:
            return grp
    return "Sonstige"


def plot_overview(series: dict[str, list[float]]) -> Path:
    """Plot 1: Alle 20 Entitäten, nach Gruppe eingefärbt."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ticks = np.arange(1, len(next(iter(series.values()))) + 1)

    for eid, vals in sorted(series.items()):
        grp = entity_group(eid)
        color = GROUP_COLORS.get(grp, "#aaa")
        lw = 2.0 if grp in ("Nachfrage EU", "Verarbeitung EU") else 1.2
        ax.plot(ticks, vals, color=color, linewidth=lw, alpha=0.85,
                label=ENTITY_LABELS.get(eid, eid))

    # Legende: Gruppen
    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower left", fontsize=9)
    ax.set_xlabel("Simulationstick (Tage)")
    ax.set_ylabel("Health [0–1]")
    ax.set_title("ICIO-Szenario S1 Soja — Health aller Entitäten (DummyBrain, 365 Ticks)")
    ax.set_xlim(1, len(ticks))
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#aaa", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / "icio_01_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_by_group(series: dict[str, list[float]]) -> Path:
    """Plot 2: 2×2-Grid nach Gruppen (ohne Puffer)."""
    main_groups = ["Produktion (BR/AR/US)", "Logistik", "Verarbeitung EU", "Nachfrage EU"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    ticks = np.arange(1, len(next(iter(series.values()))) + 1)

    for ax, grp in zip(axes.flat, main_groups):
        color = GROUP_COLORS[grp]
        members = GROUPS[grp]
        for eid in members:
            if eid not in series:
                continue
            ax.plot(ticks, series[eid], label=ENTITY_LABELS.get(eid, eid),
                    linewidth=1.8, alpha=0.9)
        ax.set_title(grp, fontsize=11, color=color)
        ax.legend(fontsize=7.5, loc="lower left")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Health")

    for ax in axes[1]:
        ax.set_xlabel("Tick (Tage)")

    fig.suptitle("ICIO-Soja — Health nach Gruppen", fontsize=13, fontweight="bold")
    out = OUT_DIR / "icio_02_by_group.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_heatmap(series: dict[str, list[float]]) -> Path:
    """Plot 3: Heatmap Entity × Tick."""
    entities = sorted(series.keys(), key=lambda e: entity_group(e))
    labels = [ENTITY_LABELS.get(e, e) for e in entities]
    matrix = np.array([series[e] for e in entities])

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1,
                   cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Health")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Tick (Tage)")
    ax.set_title("ICIO-Soja — Health-Heatmap (rot=kritisch, grün=gesund)")
    out = OUT_DIR / "icio_03_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_cascade_critical(series: dict[str, list[float]]) -> Path:
    """Plot 4: Kritische Kettenentitäten (Kaskadeneffekt Brasilien → Verbraucher)."""
    chain = [
        "brazil_farms", "santos_port", "rotterdam_port",
        "eu_oil_mills", "feed_mills", "poultry_farms", "consumers",
    ]
    ticks = np.arange(1, len(next(iter(series.values()))) + 1)
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(chain) - 1)) for i in range(len(chain))]

    for eid, col in zip(chain, colors):
        if eid not in series:
            continue
        ax.plot(ticks, series[eid], color=col, linewidth=2.2,
                label=ENTITY_LABELS.get(eid, eid))

    ax.set_xlabel("Tick (Tage)")
    ax.set_ylabel("Health [0–1]")
    ax.set_title("ICIO-Soja — Kaskadeneffekt entlang der Hauptlieferkette")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#aaa", linestyle="--", linewidth=0.8, label="Kritische Schwelle")
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / "icio_04_cascade.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_final_bar(series: dict[str, list[float]]) -> Path:
    """Plot 5: Balkendiagramm Health T1 vs T365."""
    entities = sorted(series.keys(), key=lambda e: series[e][-1])
    labels = [ENTITY_LABELS.get(e, e) for e in entities]
    start = [series[e][0] for e in entities]
    end = [series[e][-1] for e in entities]

    x = np.arange(len(entities))
    fig, ax = plt.subplots(figsize=(13, 7))
    w = 0.38
    bars_start = ax.barh(x + w / 2, start, w, label="T1 (Start)", color="#3498db", alpha=0.8)
    bars_end = ax.barh(x - w / 2, end, w, label="T365 (Ende)", color="#e74c3c", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Health [0–1]")
    ax.set_title("ICIO-Soja — Health: Start vs. Ende (365 Tage, DummyBrain)")
    ax.axvline(0.5, color="#aaa", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", alpha=0.3)
    out = OUT_DIR / "icio_05_start_vs_end.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_summary(series: dict[str, list[float]]) -> None:
    print("\n=== Zusammenfassung (T1 → T365) ===")
    print(f"{'Entity':<35} {'T1':>6} {'T90':>6} {'T180':>6} {'T365':>6} {'Δ':>7}")
    print("-" * 65)
    for eid in sorted(series, key=lambda e: series[e][-1]):
        v = series[eid]
        t1, t90 = v[0], v[89] if len(v) > 89 else v[-1]
        t180 = v[179] if len(v) > 179 else v[-1]
        t365 = v[-1]
        delta = t365 - t1
        label = ENTITY_LABELS.get(eid, eid)
        print(f"{label:<35} {t1:>6.3f} {t90:>6.3f} {t180:>6.3f} {t365:>6.3f} {delta:>+7.3f}")


if __name__ == "__main__":
    print("Lade Zeitreihen aus palaestrai.db …")
    series = load_health_timeseries(DB_PATH)
    print(f"  {len(series)} Entitäten, {len(next(iter(series.values())))} Ticks")

    print_summary(series)

    print("\nErstelle Plots …")
    plots = [
        plot_overview(series),
        plot_by_group(series),
        plot_heatmap(series),
        plot_cascade_critical(series),
        plot_final_bar(series),
    ]
    for p in plots:
        print(f"  → {p}")
    print("Fertig.")
