#!/usr/bin/env python3
"""
enrich_pdl_from_kg.py — Reichert ein PDL-Szenario mit BACI/GTA-Daten aus dem CoyPu-KG an.

Usage:
    python experiments/enrich_pdl_from_kg.py \\
        /path/to/scenario.pdl.yaml \\
        /path/to/baci_gta_kg.csv \\
        --output /path/to/scenario_enriched.pdl.yaml

Was es tut:
  1. Entity.extra  — BACI-Exportvolumina (t/Jahr, kUSD, t/Tag) je kartiertem Land
  2. Substitution.coverage — BACI-basierte Volumenquotienten (realistischere Werte)
  3. GTA-Events  — Neue regulatorische Events mit empirischen Eintrittswahrscheinlichkeiten
                   aus dem GTA-Interventionsverlauf (Red-Bewertungen)

Keine Abhängigkeit von provider_sim — nur PyYAML + csv aus der Standardbibliothek.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Mapping: PDL-Entity-ID → (ISO-3-Ländercode, relevante HS-4-Codes)
# Kann für andere Szenarien erweitert / überschrieben werden.
# ---------------------------------------------------------------------------
DEFAULT_ENTITY_MAP: Dict[str, Tuple[str, List[str]]] = {
    "brazil_farms":    ("BRA", ["1201", "2304"]),
    "argentina_farms": ("ARG", ["1201", "1202", "1507", "2304"]),
    "us_farms":        ("USA", ["1201", "2304"]),
}

# Lieferantensubstitution: sub_id → (Quell-Entity, Ziel-Entity)
DEFAULT_SUBSTITUTION_MAP: Dict[str, Tuple[str, str]] = {
    "sub_supplier_argentina": ("brazil_farms", "argentina_farms"),
    "sub_supplier_usa":       ("brazil_farms", "us_farms"),
}

# Hauptprodukt je Entity für die Substitutionsberechnung
DEFAULT_PRIMARY_HS4: Dict[str, str] = {
    "brazil_farms":    "1201",
    "argentina_farms": "1201",
    "us_farms":        "1201",
}

# GTA-Impact je Interventionstyp (prozentualer Supply-Schock)
GTA_IMPACT_MAP: Dict[str, str] = {
    "ExportBan":                    "-30%",
    "ExportQuota":                  "-15%",
    "ExportTax":                    "-10%",
    "ExportLicensingRequirement":   "-8%",
    "ImportTariff":                 "+5%",   # Importzölle erhöhen Exportnachfrage leicht
    "ImportBan":                    "-20%",
    "ImportLicensingRequirement":   "-5%",
    "ImportQuota":                  "-10%",
}

HS4_NAMES: Dict[str, str] = {
    "1201": "Soybeans",
    "1202": "Peanuts",
    "1204": "Linseed",
    "1205": "Rape/Colza seeds",
    "1206": "Sunflower seeds",
    "1207": "Other oil seeds",
    "1507": "Soybean oil",
    "1512": "Sunflower oil",
    "2304": "Soy meal (Sojaschrot)",
    "2306": "Oil-cake residues",
    "1001": "Wheat",
    "1005": "Maize",
}


# ---------------------------------------------------------------------------
# CSV laden
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# BACI-Aggregation: (country, hs4) → {value_kusd, qty_tons}
# ---------------------------------------------------------------------------

def aggregate_baci(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Summiert BACI-Volumen je (country, hs4). Jede Zeile kann mehrfach vorkommen
    (Long-Format: pro GTA-Intervention eine Zeile), daher deduplizieren."""
    seen: set = set()
    agg: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {"value_kusd": 0.0, "qty_tons": 0.0})
    for r in rows:
        key = (r["country"], r["hs4"])
        dedup_key = (r["country"], r["baci_year"], r["hs4"])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        agg[key]["value_kusd"] += float(r["export_kusd"] or 0)
        agg[key]["qty_tons"]   += float(r["export_tons"] or 0)
    return dict(agg)


# ---------------------------------------------------------------------------
# GTA-Aggregation: Eintrittswahrscheinlichkeiten aus Red-Interventionen
# ---------------------------------------------------------------------------

def aggregate_gta(
    rows: List[Dict[str, str]],
) -> Dict[Tuple[str, str, str], float]:
    """Gibt P(Red-Intervention) = Anzahl Jahre mit >=1 Red / Gesamtjahre im Datensatz
    je (country, hs4, intervention_type) zurück."""
    all_years = sorted({r["gta_year"] for r in rows if r["gta_year"]})
    n_years = len(all_years) if all_years else 1

    # Jahre mit Red-Bewertung je (country, hs4, type)
    red_years: Dict[Tuple[str, str, str], set] = defaultdict(set)
    for r in rows:
        if r["gta_evaluation"] == "Red" and r["gta_year"]:
            key = (r["country"], r["hs4"], r["gta_type"])
            red_years[key].add(r["gta_year"])

    return {k: len(yrs) / n_years for k, yrs in red_years.items()}


# ---------------------------------------------------------------------------
# PDL-YAML laden / speichern (raw, ohne provider_sim)
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)


# ---------------------------------------------------------------------------
# Schritt 1: Entity.extra mit BACI-Volumina anreichern
# ---------------------------------------------------------------------------

def enrich_entities(
    pdl: Dict,
    baci: Dict[Tuple[str, str], Dict[str, float]],
    entity_map: Dict[str, Tuple[str, List[str]]],
) -> List[str]:
    log = []
    for entity in pdl.get("entities", []):
        eid = entity.get("id", "")
        if eid not in entity_map:
            continue
        country, hs4_list = entity_map[eid]

        # Summe über alle relevanten HS-4-Codes
        total_value = sum(baci.get((country, h), {}).get("value_kusd", 0.0) for h in hs4_list)
        total_tons  = sum(baci.get((country, h), {}).get("qty_tons",   0.0) for h in hs4_list)

        if total_tons == 0:
            continue

        extra = entity.setdefault("extra", {})
        extra["baci_export_value_kusd"]    = round(total_value, 0)
        extra["baci_export_volume_t_year"] = round(total_tons, 0)
        extra["baci_capacity_t_day"]       = round(total_tons / 365, 1)
        extra["baci_year"]                 = "2021"
        extra["baci_country"]              = country
        extra["baci_hs4_codes"]            = hs4_list

        log.append(
            f"  Entity '{eid}' ({country}): "
            f"{total_tons:,.0f} t/Jahr | {total_value:,.0f} kUSD"
        )
    return log


# ---------------------------------------------------------------------------
# Schritt 2: Substitution.coverage aus BACI-Quotienten aktualisieren
# ---------------------------------------------------------------------------

def enrich_substitutions(
    pdl: Dict,
    baci: Dict[Tuple[str, str], Dict[str, float]],
    substitution_map: Dict[str, Tuple[str, str]],
    primary_hs4: Dict[str, str],
    entity_map: Dict[str, Tuple[str, List[str]]],
) -> List[str]:
    log = []
    for sub in pdl.get("substitutions", []):
        sid = sub.get("id", "")
        if sid not in substitution_map:
            continue
        src_entity, dst_entity = substitution_map[sid]

        src_country = entity_map.get(src_entity, (None,))[0]
        dst_country = entity_map.get(dst_entity, (None,))[0]
        src_hs4 = primary_hs4.get(src_entity)
        dst_hs4 = primary_hs4.get(dst_entity)

        if not all([src_country, dst_country, src_hs4, dst_hs4]):
            continue

        src_tons = baci.get((src_country, src_hs4), {}).get("qty_tons", 0.0)
        dst_tons = baci.get((dst_country, dst_hs4), {}).get("qty_tons", 0.0)

        if src_tons == 0:
            continue

        raw_coverage = dst_tons / src_tons
        # Begrenzen auf [0, 1], PDL-Wert ist ein Anteil
        new_coverage = round(min(raw_coverage, 1.0), 3)
        old_coverage = sub.get("coverage")

        sub["coverage"] = new_coverage
        sub.setdefault("_kg_source", {})["baci_coverage_formula"] = (
            f"{dst_country}/{dst_hs4} / {src_country}/{src_hs4} "
            f"= {dst_tons:,.0f} / {src_tons:,.0f} = {raw_coverage:.3f} → capped {new_coverage}"
        )

        log.append(
            f"  Substitution '{sid}': coverage {old_coverage} → {new_coverage} "
            f"({dst_country} {dst_tons:,.0f} t / {src_country} {src_tons:,.0f} t)"
        )
    return log


# ---------------------------------------------------------------------------
# Schritt 3: GTA-Events als regulatorische Events anhängen
# ---------------------------------------------------------------------------

def build_gta_events(
    gta_probs: Dict[Tuple[str, str, str], float],
    entity_map: Dict[str, Tuple[str, List[str]]],
    existing_event_ids: set,
    min_probability: float = 0.15,
) -> Tuple[List[Dict], List[str]]:
    """Erzeugt neue PDL-Events aus GTA-Interventionswahrscheinlichkeiten."""
    # Invertiertes Mapping: country → entity_id
    country_to_entity: Dict[str, str] = {v[0]: k for k, v in entity_map.items()}

    new_events: List[Dict] = []
    log: List[str] = []

    for (country, hs4, itype), prob in sorted(gta_probs.items()):
        if prob < min_probability:
            continue

        entity_id = country_to_entity.get(country)
        if not entity_id:
            continue  # Kein passendes Entity für dieses Land

        hs4_name  = HS4_NAMES.get(hs4, f"HS-{hs4}")
        impact_pct = GTA_IMPACT_MAP.get(itype, "-10%")
        direction = "export" if itype.startswith("Export") else "import"

        event_id = f"gta_{country.lower()}_{itype.lower()}_{hs4}"
        if event_id in existing_event_ids:
            continue

        event: Dict[str, Any] = {
            "id":   event_id,
            "name": f"GTA: {country} {itype} ({hs4_name})",
            "type": "regulatory",
            "trigger": {
                "target":      entity_id,
                "probability": round(prob, 2),
            },
            "impact": {
                "supply":   impact_pct,
                "duration": "180d",
            },
            "_kg_source": {
                "origin":            "CoyPu KG / GTA",
                "country":           country,
                "hs4":               hs4,
                "intervention_type": itype,
                "direction":         direction,
                "p_red_per_year":    round(prob, 2),
                "extracted":         str(date.today()),
            },
        }
        new_events.append(event)
        log.append(
            f"  Event '{event_id}': p={prob:.2f} | {itype} → {impact_pct} supply | target: {entity_id}"
        )

    return new_events, log


# ---------------------------------------------------------------------------
# Metadaten-Block im PDL
# ---------------------------------------------------------------------------

def add_metadata(pdl: Dict, csv_path: str) -> None:
    pdl.setdefault("_kg_enrichment", {}).update({
        "source_csv":  str(Path(csv_path).resolve()),
        "enriched_at": str(date.today()),
        "description": (
            "Entity.extra mit BACI-Handelsvolumina (2021) angereichert. "
            "Substitution.coverage aus BACI-Volumenquotienten berechnet. "
            "GTA-Events mit empirischen Eintrittswahrscheinlichkeiten (Red-Bewertungen) ergänzt."
        ),
    })


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def enrich(
    pdl_path: str,
    csv_path: str,
    output_path: Optional[str] = None,
    entity_map: Optional[Dict] = None,
    substitution_map: Optional[Dict] = None,
    primary_hs4: Optional[Dict] = None,
    min_gta_probability: float = 0.15,
) -> None:
    entity_map      = entity_map      or DEFAULT_ENTITY_MAP
    substitution_map = substitution_map or DEFAULT_SUBSTITUTION_MAP
    primary_hs4     = primary_hs4     or DEFAULT_PRIMARY_HS4

    print(f"Lade PDL:  {pdl_path}")
    print(f"Lade CSV:  {csv_path}")

    pdl  = load_yaml(pdl_path)
    rows = load_csv(csv_path)

    baci      = aggregate_baci(rows)
    gta_probs = aggregate_gta(rows)

    print(f"\nBACi-Einträge: {len(baci)} | GTA-Kombinationen mit Red: {len(gta_probs)}")

    # 1. Entities
    print("\n[1] Entity.extra — BACI-Volumina:")
    log = enrich_entities(pdl, baci, entity_map)
    print("\n".join(log) or "  (keine Matches)")

    # 2. Substitutionen
    print("\n[2] Substitution.coverage — BACI-Quotienten:")
    log = enrich_substitutions(pdl, baci, substitution_map, primary_hs4, entity_map)
    print("\n".join(log) or "  (keine Matches)")

    # 3. GTA-Events
    existing_ids = {e.get("id") for e in pdl.get("events", [])}
    new_events, log = build_gta_events(
        gta_probs, entity_map, existing_ids, min_probability=min_gta_probability
    )
    print(f"\n[3] GTA-Events — {len(new_events)} neue regulatorische Events (p ≥ {min_gta_probability}):")
    print("\n".join(log) or "  (keine Matches)")
    pdl.setdefault("events", []).extend(new_events)

    # Metadaten
    add_metadata(pdl, csv_path)

    # Ausgabe
    out = output_path or pdl_path.replace(".yaml", "_enriched.yaml").replace(".pdl.yaml", "_enriched.pdl.yaml")
    save_yaml(pdl, out)
    print(f"\nGespeichert: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDL-Szenario mit BACI/GTA-Daten aus dem CoyPu-KG anreichern"
    )
    parser.add_argument("pdl_path",  help="Pfad zum PDL-Szenario (.pdl.yaml)")
    parser.add_argument("csv_path",  help="Pfad zur baci_gta_kg.csv")
    parser.add_argument("--output", "-o", default=None,
                        help="Ausgabepfad (Standard: <pdl>_enriched.pdl.yaml)")
    parser.add_argument("--min-prob", type=float, default=0.15,
                        help="Mindest-Eintrittswahrscheinlichkeit für GTA-Events (Standard: 0.15)")

    args = parser.parse_args()
    enrich(
        pdl_path=args.pdl_path,
        csv_path=args.csv_path,
        output_path=args.output,
        min_gta_probability=args.min_prob,
    )


if __name__ == "__main__":
    main()
