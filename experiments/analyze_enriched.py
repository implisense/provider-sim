#!/usr/bin/env python
"""Auswertung der s1-soja_enriched Simulation (environment_id=50)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DB_PATH = Path(__file__).parent / "data" / "palaestrai.db"
ENV_ID = 50
OUT_DIR = Path(__file__).parent.parent / "analysis"
OUT_DIR.mkdir(exist_ok=True)


def load_timeseries(env_id: int) -> dict[str, list[float]]:
    """Lädt alle Sensor-Zeitreihen für ein Environment aus der DB."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT simtime_ticks, state_dump FROM world_states "
        "WHERE environment_id=? ORDER BY simtime_ticks",
        (env_id,),
    )
    rows = cur.fetchall()
    con.close()

    series: dict[str, list[float]] = {}
    ticks: list[int] = []

    for tick, raw in rows:
        ticks.append(tick)
        sensors = json.loads(raw)
        for s in sensors:
            sid = s.get("sensor_id", "")
            sv = s["sensor_value"]
            val = sv["values"][0] if isinstance(sv, dict) else float(sv)
            series.setdefault(sid, []).append(val)

    series["_ticks"] = ticks
    return series


def entity_ids(series: dict) -> list[str]:
    seen, result = set(), []
    for key in series:
        if ".entity." in key and key.endswith(".health"):
            eid = key.split(".entity.")[1].replace(".health", "")
            if eid not in seen:
                seen.add(eid)
                result.append(eid)
    return result


def event_ids(series: dict) -> list[str]:
    return [
        k.split(".event.")[1].replace(".active", "")
        for k in series
        if ".event." in k and k.endswith(".active")
    ]


def plot_health_timeseries(series: dict, ticks: list[int]) -> None:
    entities = entity_ids(series)
    n = len(entities)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5), sharey=True)
    axes = np.array(axes).flatten()

    for i, eid in enumerate(entities):
        key = f"provider_env.entity.{eid}.health"
        health = series.get(key, [])
        axes[i].plot(ticks, health, linewidth=1.2)
        axes[i].axhline(0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[i].set_title(eid.replace("_", "\n"), fontsize=7)
        axes[i].set_ylim(0, 1.05)
        axes[i].tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Health-Zeitreihen — s1-soja_enriched (Dummy)", fontsize=12, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "enriched_health_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {path}")


def plot_mean_health_and_events(series: dict, ticks: list[int]) -> None:
    entities = entity_ids(series)
    events = event_ids(series)

    # Mittlere Health
    health_matrix = np.array([
        series[f"provider_env.entity.{e}.health"] for e in entities
    ])
    mean_health = health_matrix.mean(axis=0)
    min_health = health_matrix.min(axis=0)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(ticks, min_health, mean_health, alpha=0.2, label="Min–Mean Spanne")
    ax1.plot(ticks, mean_health, linewidth=1.5, label="Mittlere Health", color="steelblue")
    ax1.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7, label="Schwelle 0.5")
    ax1.set_ylabel("Health")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.set_title("Mittlere System-Health — s1-soja_enriched (DummyBrain)", fontsize=11)

    # Event-Aktivierungen als Heatmap
    ax2 = fig.add_subplot(gs[1])
    event_matrix = np.array([
        series.get(f"provider_env.event.{e}.active", [0] * len(ticks))
        for e in events
    ])
    ax2.imshow(
        event_matrix, aspect="auto", cmap="Reds",
        extent=[ticks[0], ticks[-1], -0.5, len(events) - 0.5],
        interpolation="none",
    )
    ax2.set_yticks(range(len(events)))
    ax2.set_yticklabels([e.replace("_", " ") for e in events], fontsize=5)
    ax2.set_xlabel("Tick")
    ax2.set_title("Event-Aktivierungen", fontsize=9)

    path = OUT_DIR / "enriched_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {path}")


def plot_supply_vs_health(series: dict, ticks: list[int]) -> None:
    entities = entity_ids(series)
    # Top-5 schlechteste mittlere Health
    mean_healths = {
        e: np.mean(series[f"provider_env.entity.{e}.health"])
        for e in entities
    }
    worst = sorted(mean_healths, key=mean_healths.get)[:5]

    fig, axes = plt.subplots(len(worst), 3, figsize=(13, len(worst) * 2.2))
    metrics = ["supply", "price", "health"]
    colors = ["steelblue", "darkorange", "green"]

    for i, eid in enumerate(worst):
        for j, (metric, color) in enumerate(zip(metrics, colors)):
            key = f"provider_env.entity.{eid}.{metric}"
            axes[i, j].plot(ticks, series.get(key, []), color=color, linewidth=1)
            if j == 0:
                axes[i, j].set_ylabel(eid.replace("_", "\n"), fontsize=7)
            axes[i, j].set_title(metric if i == 0 else "", fontsize=8)
            axes[i, j].tick_params(labelsize=6)

    fig.suptitle("Top-5 am stärksten betroffene Entities (Supply / Price / Health)", fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "enriched_worst_entities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {path}")


def print_summary(series: dict, ticks: list[int]) -> None:
    entities = entity_ids(series)
    events = event_ids(series)

    print(f"\n=== Simulation Summary (env_id={ENV_ID}) ===")
    print(f"Ticks: {len(ticks)}  |  Entities: {len(entities)}  |  Events: {len(events)}")

    mean_healths = {
        e: np.mean(series[f"provider_env.entity.{e}.health"])
        for e in entities
    }
    print("\nMittlere Health pro Entity (aufsteigend):")
    for e, h in sorted(mean_healths.items(), key=lambda x: x[1]):
        bar = "#" * int(h * 30)
        print(f"  {e:35s} {h:.3f}  |{bar}")

    event_freq = {
        e: np.mean(series.get(f"provider_env.event.{e}.active", [0]))
        for e in events
    }
    print("\nEvent-Aktivierungsrate (häufigste zuerst):")
    for e, f in sorted(event_freq.items(), key=lambda x: -x[1]):
        if f > 0:
            print(f"  {e:45s} {f*100:.1f}%")


if __name__ == "__main__":
    print("Lade Daten aus palaestrai.db ...")
    series = load_timeseries(ENV_ID)
    ticks = series.pop("_ticks")

    print_summary(series, ticks)

    print("\nErstelle Plots ...")
    plot_health_timeseries(series, ticks)
    plot_mean_health_and_events(series, ticks)
    plot_supply_vs_health(series, ticks)

    print("\nFertig. Plots in analysis/")
