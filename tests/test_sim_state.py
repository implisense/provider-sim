"""Tests for the simulation state initialisation."""

from __future__ import annotations

from provider_sim.sim.state import build_state_from_pdl


class TestStateInit:
    def test_entity_count(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        assert len(state.entity_ids) == 20
        assert len(state.entities) == 20

    def test_event_count(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        assert len(state.event_ids) == 55
        assert len(state.events) == 55

    def test_initial_supply(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        for es in state.entities.values():
            assert es.supply == 1.0

    def test_downstream_edges(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        assert "santos_port" in state.downstream["brazil_farms"]
        assert "rotterdam_port" in state.downstream["santos_port"]

    def test_upstream_edges(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        assert "brazil_farms" in state.upstream["santos_port"]

    def test_dependencies(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        assert "gas_supply" in state.depends_on["eu_oil_mills"]

    def test_vulnerability(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        assert state.vulnerability["santos_port"] == 0.7

    def test_condition_state_protocol(self, soja_doc):
        state = build_state_from_pdl(soja_doc)
        assert state.is_event_active("brazil_drought") is False
        assert state.event_active_duration_days("brazil_drought") == 0.0
