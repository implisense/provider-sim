"""Integration tests: load PDL → simulate 365 ticks → no crash, all 9 scenarios."""

from __future__ import annotations

from provider_sim.pdl.parser import load_pdl
from provider_sim.sim.engine import SimulationEngine


def test_full_simulation_365_ticks(any_scenario_path):
    """Load any PDL scenario and run 365 ticks without errors."""
    doc = load_pdl(any_scenario_path)
    engine = SimulationEngine(doc, seed=42)

    for _ in range(365):
        engine.step()

    assert engine.state.tick == 365

    # All health values remain bounded
    for eid in engine.state.entity_ids:
        es = engine.state.entities[eid]
        assert 0.0 <= es.health <= 1.0, f"{eid} health={es.health} out of bounds"
        assert es.supply >= 0.0, f"{eid} supply={es.supply} negative"
        assert es.price >= 0.0, f"{eid} price={es.price} negative"
