from __future__ import annotations
import pathlib
import pytest
from provider_sim.sim.state import build_state_from_pdl
from provider_sim.pdl.parser import load_pdl


@pytest.fixture
def soja_icio_path():
    base = pathlib.Path(__file__).parent.parent
    p = base / "experiments" / "configs" / "s1-soja_icio.pdl.yaml"
    if not p.exists():
        pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
    return str(p)


class TestAblationState:
    def test_baci_capacity_extracted(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        state = build_state_from_pdl(doc)
        assert "brazil_farms" in state.baci_capacity
        assert max(state.baci_capacity.values()) == pytest.approx(1.0)
        assert state.baci_capacity["brazil_farms"] > state.baci_capacity["argentina_farms"]

    def test_icio_weight_extracted(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        state = build_state_from_pdl(doc)
        # brazil_farms hat baci_export_volume_t_year (105M t/year → max → 1.0)
        assert "brazil_farms" in state.icio_weight
        assert state.icio_weight["brazil_farms"] == pytest.approx(1.0)
        # argentina_farms hat weniger Volumen → kleiner
        assert state.icio_weight["argentina_farms"] < state.icio_weight["brazil_farms"]

    def test_entities_without_baci_not_in_dict(self, soja_icio_path):
        doc = load_pdl(soja_icio_path)
        state = build_state_from_pdl(doc)
        assert "consumers" not in state.baci_capacity
        # rotterdam_port hat kein baci_export_volume_t_year
        assert "rotterdam_port" not in state.icio_weight

    def test_base_scenario_has_empty_dicts(self):
        base = pathlib.Path(__file__).parent.parent
        # Suche ein Basis-Szenario ohne BACI/ICIO-Anreicherung
        candidates = []
        for scenario_dir in [
            base / "scenarios",
            base.parent.parent / "06_Szenarien" / "scenarios",
        ]:
            if scenario_dir.exists():
                candidates += list(scenario_dir.glob("s1-soja.pdl.yaml"))
        if not candidates:
            pytest.skip("Kein Basis-Szenario ohne BACI gefunden")
        doc = load_pdl(str(candidates[0]))
        state = build_state_from_pdl(doc)
        if state.baci_capacity:
            pytest.skip("Dieses Szenario hat auch BACI-Daten")
        assert state.baci_capacity == {}
        assert state.icio_weight == {}
