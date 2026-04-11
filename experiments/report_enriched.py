#!/usr/bin/env python
"""Erstellt einen Markdown-Bericht für die s1-soja_enriched Simulation."""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "data" / "palaestrai.db"
ENV_ID = 50
OUT_DIR = Path(__file__).parent.parent / "analysis"
REPORT_PATH = OUT_DIR / "enriched_report.md"


def load_timeseries(env_id: int) -> tuple[list[int], dict[str, list[float]]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT simtime_ticks, state_dump FROM world_states "
        "WHERE environment_id=? ORDER BY simtime_ticks",
        (env_id,),
    )
    rows = cur.fetchall()
    con.close()

    ticks, series = [], {}
    for tick, raw in rows:
        ticks.append(tick)
        for s in json.loads(raw):
            sid = s.get("sensor_id", "")
            sv = s["sensor_value"]
            val = sv["values"][0] if isinstance(sv, dict) else float(sv)
            series.setdefault(sid, []).append(val)
    return ticks, series


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


def health_status(h: float) -> str:
    if h < 0.4:
        return "kritisch"
    if h < 0.5:
        return "gefährdet"
    if h < 0.7:
        return "eingeschränkt"
    return "stabil"


def health_emoji(h: float) -> str:
    if h < 0.4:
        return "🔴"
    if h < 0.5:
        return "🟠"
    if h < 0.7:
        return "🟡"
    return "🟢"


def build_report(ticks: list[int], series: dict) -> str:
    entities = entity_ids(series)
    events = event_ids(series)

    mean_health = {
        e: float(np.mean(series[f"provider_env.entity.{e}.health"]))
        for e in entities
    }
    min_health = {
        e: float(np.min(series[f"provider_env.entity.{e}.health"]))
        for e in entities
    }
    mean_supply = {
        e: float(np.mean(series[f"provider_env.entity.{e}.supply"]))
        for e in entities
    }
    mean_price = {
        e: float(np.mean(series[f"provider_env.entity.{e}.price"]))
        for e in entities
    }

    event_freq = {
        e: float(np.mean(series.get(f"provider_env.event.{e}.active", [0])))
        for e in events
    }

    system_health = float(np.mean(list(mean_health.values())))
    n_critical = sum(1 for h in mean_health.values() if h < 0.5)
    gta_events = [e for e in events if e.startswith("gta_")]
    pdl_events = [e for e in events if not e.startswith("gta_")]

    # Entities sortiert nach Health (aufsteigend)
    ranked = sorted(mean_health.items(), key=lambda x: x[1])
    worst5 = ranked[:5]
    best5 = ranked[-5:]

    lines = []

    # Titel
    lines += [
        f"# Simulationsbericht: S1 Soja-Lieferkette (enriched)",
        f"",
        f"**Szenario:** `s1-soja_enriched.pdl.yaml`  ",
        f"**Agenten:** DummyBrain / DummyMuscle (Baseline, kein RL)  ",
        f"**Simulationsdauer:** {len(ticks)} Ticks (≈ 1 Jahr)  ",
        f"**Datum:** {date.today().isoformat()}  ",
        f"**Datenbank:** `palaestrai.db` · Environment-ID: {ENV_ID}  ",
        f"",
        f"---",
        f"",
    ]

    # Systemübersicht
    lines += [
        f"## Systemübersicht",
        f"",
        f"| Kennzahl | Wert |",
        f"|---|---|",
        f"| Entities | {len(entities)} |",
        f"| Events (gesamt) | {len(events)} |",
        f"| davon PDL-Events | {len(pdl_events)} |",
        f"| davon GTA-Events (KG-Enrichment) | {len(gta_events)} |",
        f"| Mittlere System-Health | **{system_health:.3f}** |",
        f"| Entities unter Schwelle 0.5 | **{n_critical} von {len(entities)}** |",
        f"",
    ]

    # Plots
    lines += [
        f"## Visualisierungen",
        f"",
        f"### Health-Zeitreihen (alle Entities)",
        f"",
        f"![Health-Zeitreihen](enriched_health_timeseries.png)",
        f"",
        f"### Systemübersicht: Mittlere Health & Event-Aktivierungen",
        f"",
        f"![Systemübersicht](enriched_overview.png)",
        f"",
        f"### Top-5 kritischste Entities (Supply / Price / Health)",
        f"",
        f"![Worst Entities](enriched_worst_entities.png)",
        f"",
    ]

    # Entity-Tabelle
    lines += [
        f"## Entity-Health-Ranking",
        f"",
        f"| # | Entity | Mittl. Health | Min. Health | Mittl. Supply | Mittl. Price | Status |",
        f"|---|---|---|---|---|---|---|",
    ]
    for i, (eid, h) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | `{eid}` | {h:.3f} | {min_health[eid]:.3f} "
            f"| {mean_supply[eid]:.3f} | {mean_price[eid]:.3f} "
            f"| {health_emoji(h)} {health_status(h)} |"
        )
    lines += [""]

    # Kritische Entities Detail
    lines += [
        f"## Kritische Entities (Health < 0.5)",
        f"",
    ]
    critical = [(e, h) for e, h in ranked if h < 0.5]
    if critical:
        for eid, h in critical:
            lines += [
                f"### `{eid}`",
                f"",
                f"- Mittlere Health: **{h:.3f}** — {health_status(h)}",
                f"- Minimum Health: {min_health[eid]:.3f}",
                f"- Mittlere Supply: {mean_supply[eid]:.3f}",
                f"- Mittlere Price: {mean_price[eid]:.3f}",
                f"",
            ]
    else:
        lines += ["_Keine Entity durchgehend unter 0.5._", ""]

    # Event-Analyse
    active_events = sorted(
        [(e, f) for e, f in event_freq.items() if f > 0],
        key=lambda x: -x[1],
    )

    lines += [
        f"## Event-Analyse",
        f"",
        f"### PDL-Events (Szenario-definiert)",
        f"",
        f"| Event | Aktivierungsrate |",
        f"|---|---|",
    ]
    for e, f in active_events:
        if not e.startswith("gta_"):
            bar = "█" * int(f * 20)
            lines.append(f"| `{e}` | {f*100:.1f}% `{bar}` |")

    lines += [
        f"",
        f"### GTA-Events (KG-Enrichment, BACI/GTA-Handelsdaten)",
        f"",
        f"| Event | Aktivierungsrate |",
        f"|---|---|",
    ]
    for e, f in active_events:
        if e.startswith("gta_"):
            bar = "█" * int(f * 20)
            lines.append(f"| `{e}` | {f*100:.1f}% `{bar}` |")

    lines += [""]

    # Interpretation
    lines += [
        f"## Interpretation",
        f"",
        f"### Baseline-Verhalten (DummyBrain)",
        f"",
        f"Der DummyBrain agiert zufällig ohne Strategie. Die Ergebnisse zeigen das",
        f"**ungesteuerte Systemverhalten** unter kontinuierlicher zufälliger Störung.",
        f"",
        f"### Beobachtungen",
        f"",
        f"- **{n_critical} von {len(entities)} Entities** liegen im Durchschnitt unter der",
        f"  kritischen Schwelle von 0.5, darunter `feed_mills` (H={mean_health.get('feed_mills', 0):.3f})",
        f"  als am stärksten betroffene Entity.",
        f"- **GTA-Events** ({len(gta_events)} Stück aus dem KG-Enrichment) zeigen",
        f"  Aktivierungsraten >95% — sie werden als Cascade-Events modelliert und",
        f"  bleiben nach dem ersten Trigger dauerhaft aktiv.",
        f"- **Robuste Entities** wie `us_farms`, `strategic_feed_reserves` und `us_gulf_ports`",
        f"  (Health >{best5[-1][1]:.2f}) werden durch das Szenario kaum beeinträchtigt,",
        f"  da sie upstream liegen oder als Puffer fungieren.",
        f"- Die **mittlere System-Health von {system_health:.3f}** zeigt eine moderate",
        f"  Belastung des Gesamtsystems.",
        f"",
        f"### Nächste Schritte",
        f"",
        f"- PPO-Training starten: `python experiments/train_ppo.py 50`",
        f"- Vergleich Dummy vs. PPO-Defender",
        f"- GTA-Event-Kaskadierung im PDL-Modell prüfen (Dauerhaft-Aktivierung plausibel?)",
        f"",
        f"---",
        f"",
        f"_Generiert von `experiments/report_enriched.py` · PROVIDER-Projekt_",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    print("Lade Daten ...")
    ticks, series = load_timeseries(ENV_ID)
    print(f"  {len(ticks)} Ticks, {len(series)} Sensoren geladen")

    report = build_report(ticks, series)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Bericht gespeichert: {REPORT_PATH}")
