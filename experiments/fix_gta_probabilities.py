#!/usr/bin/env python
"""Patcht GTA-Event-Wahrscheinlichkeiten in einem enriched PDL-Szenario.

Liest p_red_per_year aus _kg_source und schreibt die korrekte
per-Tick-Wahrscheinlichkeit in trigger.probability.

Usage:
    python experiments/fix_gta_probabilities.py scenarios/s1-soja_enriched.pdl.yaml
    python experiments/fix_gta_probabilities.py scenarios/s1-soja_enriched.pdl.yaml \
        --output scenarios/s1-soja_enriched_fixed.pdl.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def p_annual_to_per_tick(p_per_year: float) -> float:
    return round(1.0 - (1.0 - p_per_year) ** (1.0 / 365), 6)


def fix_probabilities(pdl_path: Path, output_path: Path) -> None:
    doc = yaml.safe_load(pdl_path.read_text(encoding="utf-8"))

    fixed = 0
    skipped = 0

    for event in doc.get("events", []):
        ev_id = event.get("id", "")
        if not ev_id.startswith("gta_"):
            continue

        kg = event.get("_kg_source", {})
        p_year = kg.get("p_red_per_year")
        if p_year is None:
            skipped += 1
            continue

        p_tick = p_annual_to_per_tick(p_year)
        old = event["trigger"].get("probability")
        event["trigger"]["probability"] = p_tick
        event["_kg_source"]["p_per_tick"] = p_tick
        print(f"  {ev_id}: {old} → {p_tick}  (p/Jahr={p_year})")
        fixed += 1

    output_path.write_text(
        yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\n{fixed} Events gepatcht, {skipped} übersprungen.")
    print(f"Gespeichert: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdl_path", help="Pfad zum enriched PDL-YAML")
    parser.add_argument("--output", "-o", default=None,
                        help="Ausgabepfad (Standard: überschreibt Input)")
    args = parser.parse_args()

    pdl_path = Path(args.pdl_path)
    output_path = Path(args.output) if args.output else pdl_path

    fix_probabilities(pdl_path, output_path)
