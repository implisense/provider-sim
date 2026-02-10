"""Tests for the simulation engine."""

from __future__ import annotations

import pytest

from provider_sim.sim.engine import SimulationEngine


class TestEngine:
    def test_tick_increment(self, soja_engine):
        assert soja_engine.state.tick == 0
        soja_engine.step()
        assert soja_engine.state.tick == 1

    def test_reset(self, soja_engine):
        for _ in range(10):
            soja_engine.step()
        assert soja_engine.state.tick == 10
        soja_engine.reset()
        assert soja_engine.state.tick == 0

    def test_health_bounded(self, soja_engine):
        for _ in range(50):
            soja_engine.step()
        for es in soja_engine.state.entities.values():
            assert 0.0 <= es.health <= 1.0

    def test_attacker_reduces_supply(self, soja_doc):
        eng = SimulationEngine(soja_doc, seed=0)
        eng.step(attacker_actions={"brazil_farms": 1.0})
        es = eng.state.entities["brazil_farms"]
        assert es.supply < 1.0

    def test_defender_restores_supply(self, soja_doc):
        eng = SimulationEngine(soja_doc, seed=0)
        # First reduce supply
        eng.step(attacker_actions={"brazil_farms": 1.0})
        supply_after_attack = eng.state.entities["brazil_farms"].supply
        # Then defend
        eng.step(defender_actions={"brazil_farms": 1.0})
        supply_after_defend = eng.state.entities["brazil_farms"].supply
        assert supply_after_defend >= supply_after_attack

    def test_cascade_propagation(self, soja_doc):
        """Force a root event and check if downstream effects appear."""
        eng = SimulationEngine(soja_doc, seed=0)

        # Force brazil_drought by setting it active directly
        eng.state.events["brazil_drought"].active = True
        eng.state.events["brazil_drought"].active_since_tick = 0
        eng.state.events["brazil_drought"].remaining_ticks = 90

        # Run enough ticks for cascading effects
        for _ in range(30):
            eng.step()

        # soy_export_reduction should have been triggered (condition: brazil_drought.active)
        assert eng.state.events["soy_export_reduction"].active or \
            eng.state.events["soy_export_reduction"].active_since_tick is not None or \
            eng.state.events["soy_export_reduction"].remaining_ticks < \
            eng.state.event_duration_ticks.get("soy_export_reduction", 120)

    def test_no_crash_365_ticks(self, soja_engine):
        for _ in range(365):
            soja_engine.step()
        assert soja_engine.state.tick == 365
