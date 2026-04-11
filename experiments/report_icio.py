"""HTML-Report-Generator für PROVIDER-Simulationsexperimente.

Erzeugt einen selbst-enthaltenen HTML-Report mit eingebetteten Plots,
automatischer Interpretation und ZIP-Archiv.

Verwendung:
    python experiments/report_icio.py experiments/s1-soja_icio.pdl.yaml
    python experiments/report_icio.py experiments/s1-soja_icio.pdl.yaml \\
        --title "Soja-Krise 2026" --db palaestrai.db --out-dir analysis/reports/
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import re
import sqlite3
import types
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from provider_sim.pdl.parser import load_pdl

# ── Schwellenwerte ───────────────────────────────────────────────────────────
_HEALTH_CRITICAL = 0.4
_HEALTH_WARNING  = 0.5
_HEALTH_REDUCED  = 0.7
_DELTA_SEVERE    = -0.3
_DELTA_MODERATE  = -0.1

PLOT_TITLES = [
    "Health aller Entitäten — Überblick (nach Gruppe eingefärbt)",
    "Health nach Lieferkettengruppen",
    "Health-Heatmap (alle Entitäten × alle Ticks)",
    "Kaskadeneffekt entlang der Hauptlieferkette",
    "Health-Vergleich: Start (T1) vs. Ende (T365)",
]

CASCADE_CHAIN = [
    "brazil_farms", "santos_port", "rotterdam_port",
    "eu_oil_mills", "feed_mills", "poultry_farms", "consumers",
]


# ── Daten laden ──────────────────────────────────────────────────────────────

def load_analysis_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_icio_run",
        Path(__file__).parent / "analyze_icio_run.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_plot_files(plots_dir: Path) -> List[Path]:
    paths = sorted(plots_dir.glob("icio_0[1-5]_*.png"))
    return paths


def png_to_base64(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def extract_db_metadata(db_path: Path) -> Dict:
    conn = sqlite3.connect(db_path)
    ticks = conn.execute("SELECT COUNT(*) FROM world_states").fetchone()[0]
    brain = "DummyBrain"
    try:
        row = conn.execute("SELECT brain_module FROM agents LIMIT 1").fetchone()
        if row:
            brain = row[0].split(".")[-1]
    except Exception:
        pass
    conn.close()
    return {"brain": brain, "ticks": ticks}


def detect_data_sources(pdl_doc) -> List[str]:
    sources = set()
    for e in pdl_doc.entities:
        extra = getattr(e, "extra", {}) or {}
        if any(k.startswith("baci_") for k in extra):
            sources.add("BACI Trade Data (UN Comtrade)")
        if any(k.startswith("icio_") or k == "icio_model" for k in extra):
            sources.add("OECD ICIO 2025")
        if any(k.startswith("exiobase_") for k in extra):
            sources.add("EXIOBASE 3")
        if any(k.startswith("wpi_") for k in extra):
            sources.add("WPI (World Port Index / NGA)")
    return sorted(sources)


# ── Analyse ──────────────────────────────────────────────────────────────────

def compute_health_table(
    series: Dict[str, List[float]],
    entity_labels: Dict[str, str],
    entity_group_fn,
) -> List[Dict]:
    rows = []
    for eid, vals in series.items():
        t1   = vals[0]
        t90  = vals[89]  if len(vals) > 89  else vals[-1]
        t180 = vals[179] if len(vals) > 179 else vals[-1]
        t365 = vals[-1]
        rows.append({
            "entity_id": eid,
            "label":     entity_labels.get(eid, eid),
            "group":     entity_group_fn(eid),
            "t1":   t1,
            "t90":  t90,
            "t180": t180,
            "t365": t365,
            "delta": t365 - t1,
        })
    rows.sort(key=lambda r: r["t365"])
    return rows


def compute_system_stats(series: Dict[str, List[float]]) -> Dict:
    finals = {eid: vals[-1] for eid, vals in series.items()}
    mean_final = float(np.mean(list(finals.values())))
    n_critical = sum(1 for v in finals.values() if v < _HEALTH_WARNING)
    worst = min(finals, key=finals.__getitem__)
    best  = max(finals, key=finals.__getitem__)
    return {
        "mean_health_final": mean_final,
        "n_critical": n_critical,
        "n_total": len(finals),
        "worst_entity": worst,
        "best_entity": best,
    }


def generate_interpretation(
    series: Dict[str, List[float]],
    entity_labels: Dict[str, str],
    groups: Dict[str, List[str]],
) -> Dict:
    # Top-3 Verlierer
    deltas = {eid: vals[-1] - vals[0] for eid, vals in series.items()}
    sorted_losers = sorted(deltas.items(), key=lambda x: x[1])
    top3 = []
    for eid, delta in sorted_losers[:3]:
        if delta < _DELTA_SEVERE:
            severity = "stark betroffen"
            color = "#e74c3c"
        elif delta < _DELTA_MODERATE:
            severity = "moderat betroffen"
            color = "#e67e22"
        else:
            severity = "leicht betroffen"
            color = "#f39c12"
        top3.append({
            "label": entity_labels.get(eid, eid),
            "delta_pct": delta * 100,
            "t365": series[eid][-1],
            "severity": severity,
            "color": color,
        })

    # Kaskadeneffekt — ersten Tick unter 0.7 ermitteln
    cascade_rows = []
    for eid in CASCADE_CHAIN:
        if eid not in series:
            continue
        vals = series[eid]
        first_drop = next((i for i, v in enumerate(vals) if v < 0.7), None)
        cascade_rows.append({
            "label": entity_labels.get(eid, eid),
            "t1":   vals[0],
            "t365": vals[-1],
            "first_drop_day": first_drop + 1 if first_drop is not None else None,
        })

    # Puffer-Performance
    buffer_ids = groups.get("Puffer/Input", [])
    buffer_rows = []
    for eid in buffer_ids:
        if eid not in series:
            continue
        t365 = series[eid][-1]
        if t365 >= _HEALTH_REDUCED:
            status = "widerstandsfähig"
            color = "#27ae60"
        elif t365 >= _HEALTH_WARNING:
            status = "eingeschränkt"
            color = "#f39c12"
        else:
            status = "kritisch"
            color = "#e74c3c"
        buffer_rows.append({
            "label": entity_labels.get(eid, eid),
            "t365": t365,
            "status": status,
            "color": color,
        })

    return {
        "top3_losers": top3,
        "cascade": cascade_rows,
        "buffers": buffer_rows,
    }


# ── HTML-Bausteine ───────────────────────────────────────────────────────────

def health_color(val: float) -> str:
    if val >= _HEALTH_REDUCED:
        # grün-Abstufung
        intensity = int(255 * (1 - (val - _HEALTH_REDUCED) / (1 - _HEALTH_REDUCED)) * 0.4)
        return f"#{'%02x' % (100 + intensity)}c878"
    if val >= _HEALTH_WARNING:
        return "#f0c040"
    if val >= _HEALTH_CRITICAL:
        return "#f0924a"
    return "#e05555"


def delta_color(val: float) -> str:
    return "#27ae60" if val >= 0 else "#e74c3c"


def fmt_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}%"


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px; line-height: 1.6; color: #2c3e50; background: #f8f9fa;
}
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: white; padding: 40px 24px; margin-bottom: 32px;
}
header .subtitle { opacity: 0.75; font-size: 13px; margin-top: 6px; }
header h1 { font-size: 26px; font-weight: 700; }
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
  margin-left: 10px; vertical-align: middle;
}
.badge-high { background: #e74c3c; color: white; }
.badge-medium { background: #f39c12; color: white; }
.badge-low { background: #27ae60; color: white; }
section {
  background: white; border-radius: 8px; padding: 28px;
  margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
section h2 {
  font-size: 18px; font-weight: 600; margin-bottom: 18px;
  padding-bottom: 10px; border-bottom: 2px solid #e8ecef; color: #1a1a2e;
}
section h3 { font-size: 15px; font-weight: 600; margin: 18px 0 10px; color: #2c3e50; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
  background: #f1f3f5; text-align: left; padding: 9px 12px;
  font-weight: 600; border-bottom: 2px solid #dee2e6; white-space: nowrap;
}
td { padding: 7px 12px; border-bottom: 1px solid #f1f3f5; }
tr:hover td { background: #fafbfc; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.meta-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;
}
.meta-card {
  background: #f8f9fa; border-radius: 6px; padding: 14px 16px;
  border-left: 3px solid #3498db;
}
.meta-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #7f8c8d; }
.meta-card .value { font-size: 20px; font-weight: 700; color: #2c3e50; margin-top: 2px; }
.plot-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(540px, 1fr)); gap: 20px;
}
figure { margin: 0; }
figure img { width: 100%; border-radius: 6px; border: 1px solid #e8ecef; }
figcaption {
  font-size: 12px; color: #7f8c8d; margin-top: 6px; text-align: center; font-style: italic;
}
.card-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;
}
.card {
  border-radius: 8px; padding: 18px; border-left: 4px solid;
}
.card .card-title { font-weight: 600; font-size: 14px; margin-bottom: 6px; }
.card .card-value { font-size: 22px; font-weight: 700; }
.card .card-sub { font-size: 12px; color: #7f8c8d; margin-top: 4px; }
.cascade-table td.drop { font-size: 12px; color: #7f8c8d; }
ol.phases { padding-left: 20px; }
ol.phases li { margin-bottom: 6px; }
footer {
  background: #2c3e50; color: #bdc3c7; text-align: center;
  padding: 20px; font-size: 12px; border-radius: 8px;
}
footer a { color: #7fb3d3; text-decoration: none; }
@media print {
  body { background: white; }
  section { box-shadow: none; border: 1px solid #ddd; }
  .plot-grid { grid-template-columns: 1fr; }
}
"""


def _badge(criticality: str) -> str:
    cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(
        criticality, "badge-medium"
    )
    return f'<span class="badge {cls}">{criticality}</span>'


def _section_setup(pdl_doc, db_meta: Dict, sources: List[str]) -> str:
    sc = pdl_doc.scenario
    n_entities = len(pdl_doc.entities)
    n_events   = len(pdl_doc.events)
    n_cascades = len(getattr(pdl_doc, "cascades", []) or [])
    rows = [
        ("Szenario-ID", sc.id),
        ("Sektor", sc.sector),
        ("Kritikalität", sc.criticality),
        ("Entities", str(n_entities)),
        ("Events", str(n_events)),
        ("Kaskaden", str(n_cascades)),
        ("Simulations-Ticks", str(db_meta["ticks"])),
        ("Brain-Typ", db_meta["brain"]),
        ("Datenquellen", ", ".join(sources) if sources else "—"),
    ]
    tr = "\n".join(
        f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in rows
    )
    return f"""
<section id="setup">
  <h2>Experiment-Setup</h2>
  <p style="margin-bottom:16px;color:#555">{sc.description}</p>
  <table style="max-width:600px">{tr}</table>
</section>"""


def _section_methodik() -> str:
    return """
<section id="methodik">
  <h2>Methodik</h2>
  <h3>5-Phasen-Simulationsschritt (pro Tick)</h3>
  <ol class="phases">
    <li><strong>Agent-Actions:</strong> Attacker reduziert Entity-Supply (×vulnerability), Defender addiert.</li>
    <li><strong>Event-Evaluierung:</strong> Root-Events probabilistisch, Condition-Events per Condition-AST.</li>
    <li><strong>Impact-Stack:</strong> Modifier-basiert, kein Per-Tick-Compounding. effective_supply = Π(1 + modifier_i).</li>
    <li><strong>Flow-Propagation:</strong> Topologische Sortierung (Kahn), supply = min(intrinsic, mean(incoming)).</li>
    <li><strong>Health-Berechnung:</strong> <code>health = clip(0.5·supply + 0.3·(1/price) + 0.2·min(demand,1), 0, 1)</code></li>
  </ol>
  <h3>ARL-Setup (diese Simulation)</h3>
  <p>Baseline-Lauf ohne RL-Training: Beide Agenten (Attacker, Defender) verwenden
  <strong>DummyBrain / DummyMuscle</strong> — keine Aktionen, reine Umgebungsdynamik.
  Natürliche Erholung: +2 %/Tick Richtung 1.0 wenn keine aktiven Events.</p>
  <h3>Zero-Sum-Reward</h3>
  <p><code>reward_attacker = mean(1 − health)</code> &nbsp;|&nbsp;
     <code>reward_defender = mean(health)</code> &nbsp;→ Summe immer 1.0</p>
</section>"""


def _section_ergebnisse(
    plot_b64: List[str],
    health_table: List[Dict],
    system_stats: Dict,
    entity_labels: Dict[str, str],
) -> str:
    # Kennzahlen-Karten
    cards_html = f"""
<div class="meta-grid" style="margin-bottom:24px">
  <div class="meta-card"><div class="label">Mittlere Health (T365)</div>
    <div class="value">{system_stats['mean_health_final']:.3f}</div></div>
  <div class="meta-card" style="border-color:#e74c3c"><div class="label">Kritische Entities (&lt;0.5)</div>
    <div class="value">{system_stats['n_critical']} / {system_stats['n_total']}</div></div>
  <div class="meta-card" style="border-color:#e74c3c"><div class="label">Schwächste Entity</div>
    <div class="value" style="font-size:14px">{entity_labels.get(system_stats['worst_entity'], system_stats['worst_entity'])}</div></div>
  <div class="meta-card" style="border-color:#27ae60"><div class="label">Stärkste Entity</div>
    <div class="value" style="font-size:14px">{entity_labels.get(system_stats['best_entity'], system_stats['best_entity'])}</div></div>
</div>"""

    # Plot-Grid
    figures = ""
    for b64, title in zip(plot_b64, PLOT_TITLES):
        figures += f"""
  <figure>
    <img src="{b64}" alt="{title}" loading="lazy">
    <figcaption>{title}</figcaption>
  </figure>"""

    # Health-Tabelle
    thead = "<tr><th>Entity</th><th>Gruppe</th><th class='num'>T1</th><th class='num'>T90</th><th class='num'>T180</th><th class='num'>T365</th><th class='num'>Δ</th></tr>"
    tbody = ""
    for r in health_table:
        def cell(v: float) -> str:
            bg = health_color(v)
            return f"<td class='num' style='background:{bg};color:#fff'>{v:.3f}</td>"
        dc = delta_color(r["delta"])
        tbody += (
            f"<tr><td>{r['label']}</td><td style='font-size:12px;color:#7f8c8d'>{r['group']}</td>"
            + cell(r["t1"]) + cell(r["t90"]) + cell(r["t180"]) + cell(r["t365"])
            + f"<td class='num' style='color:{dc};font-weight:600'>{fmt_pct(r['delta'])}</td></tr>"
        )

    return f"""
<section id="ergebnisse">
  <h2>Ergebnisse</h2>
  {cards_html}
  <div class="plot-grid">{figures}</div>
  <h3 style="margin-top:28px">Health-Übersicht (alle Entities)</h3>
  <table><thead>{thead}</thead><tbody>{tbody}</tbody></table>
  <p style="font-size:11px;color:#aaa;margin-top:8px">
    Farbskala: <span style="background:#64c878;color:#fff;padding:1px 6px;border-radius:3px">≥0.7 gut</span>
    <span style="background:#f0c040;color:#fff;padding:1px 6px;border-radius:3px">0.5–0.7 eingeschränkt</span>
    <span style="background:#f0924a;color:#fff;padding:1px 6px;border-radius:3px">0.4–0.5 gefährdet</span>
    <span style="background:#e05555;color:#fff;padding:1px 6px;border-radius:3px">&lt;0.4 kritisch</span>
  </p>
</section>"""


def _section_interpretation(interp: Dict) -> str:
    # Top-3 Verlierer
    cards = ""
    for r in interp["top3_losers"]:
        cards += f"""
  <div class="card" style="border-color:{r['color']};background:{r['color']}14">
    <div class="card-title">{r['label']}</div>
    <div class="card-value" style="color:{r['color']}">{r['delta_pct']:+.1f}%</div>
    <div class="card-sub">Health T365: {r['t365']:.3f} — {r['severity']}</div>
  </div>"""

    # Kaskaden-Tabelle
    cascade_rows = ""
    for r in interp["cascade"]:
        drop = f"Tag {r['first_drop_day']}" if r["first_drop_day"] else "nie"
        cascade_rows += (
            f"<tr><td>{r['label']}</td>"
            f"<td class='num'>{r['t1']:.3f}</td>"
            f"<td class='num'>{r['t365']:.3f}</td>"
            f"<td class='drop'>{drop}</td></tr>"
        )

    # Puffer
    buf_rows = ""
    for r in interp["buffers"]:
        buf_rows += (
            f"<tr><td>{r['label']}</td>"
            f"<td class='num'>{r['t365']:.3f}</td>"
            f"<td style='color:{r['color']};font-weight:600'>{r['status']}</td></tr>"
        )

    return f"""
<section id="interpretation">
  <h2>Interpretation</h2>

  <h3>Top-3 Verlierer (größter Health-Rückgang)</h3>
  <div class="card-grid">{cards}</div>

  <h3 style="margin-top:24px">Kaskadeneffekt entlang der Hauptlieferkette</h3>
  <p style="margin-bottom:12px;color:#555">
    Der Schock in der brasilianischen Soja-Produktion pflanzt sich stufenweise
    durch die gesamte Versorgungskette fort. Die Spalte „Erstmals &lt;0.7" zeigt
    die Verzögerung des Kaskadeneffekts pro Kettenglied.
  </p>
  <table class="cascade-table" style="max-width:550px">
    <thead><tr><th>Entity</th><th class="num">T1</th><th class="num">T365</th><th>Erstmals &lt;0.7</th></tr></thead>
    <tbody>{cascade_rows}</tbody>
  </table>

  <h3 style="margin-top:24px">Puffer- und Alternativ-Ressourcen</h3>
  <table style="max-width:480px">
    <thead><tr><th>Puffer</th><th class="num">Health T365</th><th>Status</th></tr></thead>
    <tbody>{buf_rows}</tbody>
  </table>
  <p style="margin-top:12px;color:#555;font-size:13px">
    <strong>Fazit (DummyBrain-Baseline):</strong>
    Ohne RL-gestützte Intervention kollabiert die Futtermittel-Wertschöpfungskette
    innerhalb von ~90 Tagen auf kritisches Niveau. Strategische Lagerbestände und
    alternative Proteinquellen puffern, reichen aber nicht aus, um den Kaskadeneffekt
    zu stoppen. Ein trainierter Defender-Agent könnte durch frühzeitige
    Umlenkungs-Aktionen (z.B. Hochfahren US-Lieferkette) gegensteuern.
  </p>
</section>"""


def build_html(
    pdl_doc,
    series: Dict[str, List[float]],
    health_table: List[Dict],
    system_stats: Dict,
    interpretation: Dict,
    plot_b64: List[str],
    db_meta: Dict,
    sources: List[str],
    entity_labels: Dict[str, str],
    report_title: str,
    generated_at: str,
) -> str:
    sc = pdl_doc.scenario
    title_display = report_title or sc.name
    crit_badge = _badge(sc.criticality)

    body = (
        _section_setup(pdl_doc, db_meta, sources)
        + _section_methodik()
        + _section_ergebnisse(plot_b64, health_table, system_stats, entity_labels)
        + _section_interpretation(interpretation)
    )

    footer_sources = " · ".join(sources) if sources else "keine externen Quellen"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_display}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <div class="container">
    <h1>{title_display} {crit_badge}</h1>
    <div class="subtitle">
      PROVIDER-Simulation · Szenario: {sc.id} · Generiert: {generated_at}
    </div>
  </div>
</header>
<div class="container">
  {body}
</div>
<footer>
  <strong>PROVIDER-Simulation</strong> · Szenario-ID: {sc.id} ·
  Generiert: {generated_at}<br>
  Datenquellen: {footer_sources}<br>
  <small>Erstellt mit report_icio.py · palaestrAI / provider_sim</small>
</footer>
</body>
</html>"""


def build_metadata_json(
    pdl_doc,
    series: Dict[str, List[float]],
    health_table: List[Dict],
    system_stats: Dict,
    db_meta: Dict,
    pdl_path: Path,
    db_path: Path,
    sources: List[str],
    generated_at: str,
) -> str:
    sc = pdl_doc.scenario
    def _str(v) -> str:
        return v.value if hasattr(v, "value") else str(v)

    return json.dumps(
        {
            "generated_at": generated_at,
            "scenario": {
                "id": sc.id,
                "name": sc.name,
                "sector": _str(sc.sector) if sc.sector else None,
                "criticality": _str(sc.criticality) if sc.criticality else None,
                "description": sc.description,
            },
            "experiment": {
                "brain": db_meta["brain"],
                "ticks": db_meta["ticks"],
                "n_entities": len(pdl_doc.entities),
                "n_events": len(pdl_doc.events),
            },
            "system_health_final": round(system_stats["mean_health_final"], 4),
            "n_critical_entities": system_stats["n_critical"],
            "entity_results": [
                {
                    "id": r["entity_id"],
                    "label": r["label"],
                    "t1": round(r["t1"], 4),
                    "t90": round(r["t90"], 4),
                    "t180": round(r["t180"], 4),
                    "t365": round(r["t365"], 4),
                    "delta": round(r["delta"], 4),
                }
                for r in health_table
            ],
            "sources": {
                "pdl": str(pdl_path),
                "db": str(db_path),
                "data_sources": sources,
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def write_zip(
    out_dir: Path,
    slug: str,
    html_content: str,
    metadata_json: str,
    plot_paths: List[Path],
    date_str: str,
) -> Path:
    zip_path = out_dir / f"{date_str}_{slug}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.html", html_content.encode("utf-8"))
        zf.writestr("metadata.json", metadata_json.encode("utf-8"))
        for p in plot_paths:
            zf.write(p, p.name)
    return zip_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTML-Report für PROVIDER-Simulationsexperiment")
    p.add_argument("pdl_path", type=Path, help="Pfad zur PDL-YAML-Datei")
    p.add_argument("--title", default="", help="Optionaler Berichtstitel")
    p.add_argument("--db", type=Path, default=Path(__file__).parent / "data" / "palaestrai.db")
    p.add_argument("--plots-dir", type=Path, default=base / "analysis")
    p.add_argument("--out-dir", type=Path, default=base / "analysis" / "reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generated_at = datetime.now().isoformat(timespec="seconds")
    date_str = generated_at[:10]

    print("Lade PDL-Szenario …")
    pdl_doc = load_pdl(args.pdl_path)

    print("Lade Analyse-Modul …")
    ana = load_analysis_module()

    print("Lade Zeitreihen aus DB …")
    series = ana.load_health_timeseries(args.db)
    print(f"  {len(series)} Entities, {len(next(iter(series.values())))} Ticks")

    db_meta  = extract_db_metadata(args.db)
    sources  = detect_data_sources(pdl_doc)
    plots    = load_plot_files(args.plots_dir)
    plot_b64 = [png_to_base64(p) for p in plots]
    print(f"  {len(plots)} Plots eingebettet")

    health_table  = compute_health_table(series, ana.ENTITY_LABELS, ana.entity_group)
    system_stats  = compute_system_stats(series)
    interpretation = generate_interpretation(series, ana.ENTITY_LABELS, ana.GROUPS)

    html = build_html(
        pdl_doc, series, health_table, system_stats, interpretation,
        plot_b64, db_meta, sources, ana.ENTITY_LABELS,
        args.title, generated_at,
    )
    metadata = build_metadata_json(
        pdl_doc, series, health_table, system_stats, db_meta,
        args.pdl_path, args.db, sources, generated_at,
    )

    # Szenario-Slug aus PDL-Dateiname
    stem = args.pdl_path.stem
    if stem.endswith(".pdl"):
        stem = stem[:-4]
    slug = re.sub(r"[^a-z0-9\-_]", "-", stem.lower())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = write_zip(args.out_dir, slug, html, metadata, plots, date_str)
    print(f"\nReport erstellt: {zip_path}")
    print(f"  ZIP-Inhalt: report.html, metadata.json, {len(plots)} PNGs")


if __name__ == "__main__":
    main()
