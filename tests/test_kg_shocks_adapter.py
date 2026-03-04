"""Tests für KG-Schocks → PDL-Events Adapter."""
from __future__ import annotations

import pytest

from provider_sim.adapters.kg_shocks import kg_shocks_to_events, S1_KG_TO_PDL
from provider_sim.pdl.model import Event, EventType


def test_capacity_shock_maps_to_supply_event():
    """capacity-Schock mit magnitude=0.6 → Event mit supply=-40%."""
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.6}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert len(events) == 1
    ev = events[0]
    assert ev.id == "kg_shock_brazil_farms"
    assert ev.trigger.target == "brazil_farms"
    assert ev.trigger.probability == 1.0
    assert ev.trigger.condition is None
    assert ev.impact.supply is not None
    assert abs(ev.impact.supply.decimal - (-0.4)) < 1e-9
    assert ev.impact.price is None


def test_price_shock_maps_to_price_event():
    """price-Schock mit magnitude=1.5 → Event mit price=+50%."""
    shocks = [{"target_id": "fertilizer_input", "shock_type": "price", "magnitude": 1.5}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert len(events) == 1
    ev = events[0]
    assert ev.id == "kg_shock_fertilizer_supply"
    assert ev.trigger.target == "fertilizer_supply"
    assert ev.impact.price is not None
    assert abs(ev.impact.price.decimal - 0.5) < 1e-9
    assert ev.impact.supply is None


def test_magnitude_one_skipped_for_capacity():
    """capacity-Schock mit magnitude=1.0 → kein Event (kein Schock)."""
    shocks = [{"target_id": "santos_port", "shock_type": "capacity", "magnitude": 1.0}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert events == []


def test_unmapped_target_skipped():
    """target_id ohne PDL-Äquivalent → kein Event."""
    shocks = [{"target_id": "deu_soy_farm", "shock_type": "capacity", "magnitude": 0.6}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert events == []


def test_multiple_shocks_all_mapped():
    """Vollständige S1-Schockliste: 8 resultierende Events."""
    shocks = [
        {"target_id": "bra_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},
        {"target_id": "arg_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},
        {"target_id": "usa_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},
        {"target_id": "deu_soy_farm",     "shock_type": "capacity", "magnitude": 0.6},   # skip: kein Mapping
        {"target_id": "fertilizer_input", "shock_type": "price",    "magnitude": 1.5},
        {"target_id": "energy_input",     "shock_type": "price",    "magnitude": 1.5},
        {"target_id": "santos_port",      "shock_type": "capacity", "magnitude": 1.0},   # skip: magnitude=1.0
        {"target_id": "rosario_port",     "shock_type": "capacity", "magnitude": 0.85},  # skip: kein Mapping
        {"target_id": "paranagua_port",   "shock_type": "capacity", "magnitude": 0.85},
        {"target_id": "rotterdam_port",   "shock_type": "capacity", "magnitude": 1.0},   # skip: magnitude=1.0
        {"target_id": "hamburg_port",     "shock_type": "capacity", "magnitude": 1.0},   # skip: magnitude=1.0
        {"target_id": "us_gulf_ports",    "shock_type": "capacity", "magnitude": 1.0},   # skip: magnitude=1.0
        {"target_id": "eu_oil_mills",     "shock_type": "capacity", "magnitude": 0.8},
        {"target_id": "feed_mills",       "shock_type": "capacity", "magnitude": 0.95},
    ]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    assert len(events) == 8
    ids = {ev.id for ev in events}
    assert "kg_shock_brazil_farms" in ids
    assert "kg_shock_fertilizer_supply" in ids
    assert "kg_shock_paranagua_port" in ids


def test_event_has_correct_type_and_duration():
    """Events sind vom Typ MARKET_SHOCK und haben duration=365d."""
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.7}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL)
    ev = events[0]
    assert ev.type == EventType.MARKET_SHOCK
    assert ev.impact.duration is not None
    assert ev.impact.duration.days == 365


def test_custom_duration():
    """duration_days-Parameter wirkt sich auf Event-Duration aus."""
    shocks = [{"target_id": "bra_soy_farm", "shock_type": "capacity", "magnitude": 0.7}]
    events = kg_shocks_to_events(shocks, id_mapping=S1_KG_TO_PDL, duration_days=90)
    assert events[0].impact.duration.days == 90
