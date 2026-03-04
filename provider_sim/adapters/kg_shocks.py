"""Adapter: KG-Schocks (JSON dict) → PDL-Events für palestrai_simulation."""
from __future__ import annotations

from provider_sim.pdl.model import (
    Duration,
    Event,
    EventType,
    Impact,
    Percentage,
    Trigger,
)

# Mapping: coypu-kg-analyser target_id → provider_sim entity_id (S1 Soja)
S1_KG_TO_PDL: dict[str, str] = {
    "bra_soy_farm":     "brazil_farms",
    "arg_soy_farm":     "argentina_farms",
    "usa_soy_farm":     "us_farms",
    "santos_port":      "santos_port",
    "paranagua_port":   "paranagua_port",
    "rotterdam_port":   "rotterdam_port",
    "hamburg_port":     "hamburg_port",
    "us_gulf_ports":    "us_gulf_ports",
    "eu_oil_mills":     "eu_oil_mills",
    "feed_mills":       "feed_mills",
    "fertilizer_input": "fertilizer_supply",
    "energy_input":     "gas_supply",
    # deu_soy_farm: kein PDL-Äquivalent
    # rosario_port:  kein PDL-Äquivalent
}


def kg_shocks_to_events(
    shocks: list[dict],
    id_mapping: dict[str, str] | None = None,
    duration_days: int = 365,
    reference: str = "CoyPu KG Live-Abfrage",
) -> list[Event]:
    """Wandelt KG-Schocks in synthetische PDL-Events um.

    Regeln:
    - target_ids ohne Mapping werden übersprungen.
    - capacity-Schocks mit magnitude == 1.0 werden übersprungen (kein Schock).
    - price-Schocks mit magnitude == 1.0 werden übersprungen.
    - Alle generierten Events haben trigger.probability = 1.0 (immer aktiv).
    """
    mapping = id_mapping if id_mapping is not None else S1_KG_TO_PDL
    duration = Duration(raw=f"{duration_days}d", days=float(duration_days))
    events: list[Event] = []

    for shock in shocks:
        target_id = shock.get("target_id", "")
        shock_type = shock.get("shock_type", "")
        magnitude: float = shock.get("magnitude", 1.0)

        pdl_id = mapping.get(target_id)
        if pdl_id is None:
            continue

        if magnitude == 1.0:
            continue

        delta = magnitude - 1.0  # 0.6 → -0.4; 1.5 → +0.5
        sign = "+" if delta >= 0 else ""
        pct_raw = f"{sign}{int(round(delta * 100))}%"
        pct = Percentage(raw=pct_raw, decimal=delta)

        if shock_type == "capacity":
            impact = Impact(supply=pct, duration=duration)
        elif shock_type == "price":
            impact = Impact(price=pct, duration=duration)
        else:
            continue

        events.append(Event(
            id=f"kg_shock_{pdl_id}",
            name=f"KG-Schock: {pdl_id}",
            type=EventType.MARKET_SHOCK,
            trigger=Trigger(target=pdl_id, probability=1.0),
            impact=impact,
            reference=reference,
        ))

    return events
