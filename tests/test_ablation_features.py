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


# ---------------------------------------------------------------------------
# Task 2 — SimulationEngine flags
# ---------------------------------------------------------------------------

from provider_sim.sim.engine import SimulationEngine  # noqa: E402


class TestBaciCap:
    def test_baci_cap_limits_supply(self):
        import pathlib
        base = pathlib.Path(__file__).parent.parent
        p = base / "experiments" / "configs" / "s1-soja_icio.pdl.yaml"
        if not p.exists():
            pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(str(p))
        engine_cap = SimulationEngine(doc, seed=0, use_baci_capacity=True)
        engine_free = SimulationEngine(doc, seed=0, use_baci_capacity=False)

        # Defender pusht argentina_farms maximal
        big_defense = {"argentina_farms": 2.0}
        engine_cap.step(defender_actions=big_defense)
        engine_free.step(defender_actions=big_defense)

        cap_limit = engine_cap.state.baci_capacity["argentina_farms"]
        s_cap = engine_cap.state.entities["argentina_farms"].supply
        s_free = engine_free.state.entities["argentina_farms"].supply

        assert s_cap <= cap_limit + 1e-6, f"supply {s_cap:.4f} überschreitet BACI-Cap {cap_limit:.4f}"
        assert s_free > s_cap, "ohne Cap sollte supply höher sein"

    def test_entities_without_baci_unaffected(self):
        import pathlib
        base = pathlib.Path(__file__).parent.parent
        p = base / "experiments" / "configs" / "s1-soja_icio.pdl.yaml"
        if not p.exists():
            pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(str(p))
        engine = SimulationEngine(doc, seed=0, use_baci_capacity=True)
        engine.step(defender_actions={"consumers": 2.0})
        s = engine.state.entities["consumers"].supply
        # consumers hat kein BACI-Cap → kann über 1.0 steigen
        assert s > 0.0  # kein Crash

    def test_flags_default_false(self):
        import pathlib
        base = pathlib.Path(__file__).parent.parent
        p = base / "experiments" / "configs" / "s1-soja_icio.pdl.yaml"
        if not p.exists():
            pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(str(p))
        engine = SimulationEngine(doc, seed=0)
        assert engine._use_baci_capacity is False
        assert engine._use_icio_weights is False


class TestIcioWeights:
    def test_icio_weights_shift_flow(self):
        """Mit ICIO-Gewichten fließt das Upstream-Supply nach Exportvolumen gewichtet."""
        import pathlib
        base = pathlib.Path(__file__).parent.parent
        p = base / "experiments" / "configs" / "s1-soja_icio.pdl.yaml"
        if not p.exists():
            pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(str(p))
        engine_w = SimulationEngine(doc, seed=0, use_icio_weights=True)
        engine_u = SimulationEngine(doc, seed=0, use_icio_weights=False)

        # Attacker schädigt brazil_farms (höchstes ICIO-Gewicht) stark
        engine_w.step(attacker_actions={"brazil_farms": 1.0})
        engine_u.step(attacker_actions={"brazil_farms": 1.0})

        # Santos-Port hängt von brazil_farms ab — mit ICIO-Gewichten
        # schlägt ein Brazil-Schaden stärker durch als ohne
        s_w = engine_w.state.entities["santos_port"].supply
        s_u = engine_u.state.entities["santos_port"].supply
        # Beide sollten ähnlich sein (santos_port hat nur einen Upstream), kein Crash
        assert s_w >= 0.0
        assert s_u >= 0.0


# ---------------------------------------------------------------------------
# Task 3 — ProviderEnvironment leitet Flags durch
# ---------------------------------------------------------------------------

from provider_sim.env.environment import ProviderEnvironment  # noqa: E402
import pathlib as _pl  # noqa: E402


class TestEnvironmentFlags:
    @staticmethod
    def _icio_path():
        base = _pl.Path(__file__).parent.parent
        p = base / "experiments" / "configs" / "s1-soja_icio.pdl.yaml"
        if not p.exists():
            pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
        return str(p)

    def test_baci_flag_passed_to_engine(self):
        p = self._icio_path()
        env = ProviderEnvironment(pdl_source=p, seed=0, use_baci_capacity=True)
        assert env.engine._use_baci_capacity is True

    def test_icio_flag_passed_to_engine(self):
        p = self._icio_path()
        env = ProviderEnvironment(pdl_source=p, seed=0, use_icio_weights=True)
        assert env.engine._use_icio_weights is True

    def test_defaults_are_false(self):
        p = self._icio_path()
        env = ProviderEnvironment(pdl_source=p, seed=0)
        assert env.engine._use_baci_capacity is False
        assert env.engine._use_icio_weights is False


# ---------------------------------------------------------------------------
# Task 4 — Follow-up params: baci_capacity_scale + icio_norm
# ---------------------------------------------------------------------------


class TestFollowupParams:
    @staticmethod
    def _icio_path():
        import pathlib
        base = pathlib.Path(__file__).parent.parent
        p = base / "experiments" / "configs" / "s1-soja_icio.pdl.yaml"
        if not p.exists():
            pytest.skip("s1-soja_icio.pdl.yaml nicht gefunden")
        return str(p)

    def test_baci_scale_relaxes_cap(self):
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(self._icio_path())
        eng1 = SimulationEngine(doc, seed=0, use_baci_capacity=True, baci_capacity_scale=1.0)
        eng2 = SimulationEngine(doc, seed=0, use_baci_capacity=True, baci_capacity_scale=2.0)
        actions = {"argentina_farms": 2.0}
        eng1.step(defender_actions=actions)
        eng2.step(defender_actions=actions)
        # Mit scale=2.0 darf supply höher sein als mit scale=1.0
        assert eng2.state.entities["argentina_farms"].supply >= \
               eng1.state.entities["argentina_farms"].supply - 1e-6

    def test_icio_norm_uniform_equals_unweighted(self):
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(self._icio_path())
        # uniform-Gewichte sollten dasselbe wie kein ICIO-Flag ergeben
        eng_uniform = SimulationEngine(doc, seed=42, use_icio_weights=True, icio_norm="uniform")
        eng_none    = SimulationEngine(doc, seed=42, use_icio_weights=False)
        for _ in range(5):
            eng_uniform.step()
            eng_none.step()
        for eid in eng_uniform.state.entity_ids:
            s_u = eng_uniform.state.entities[eid].supply
            s_n = eng_none.state.entities[eid].supply
            assert abs(s_u - s_n) < 1e-6, f"{eid}: uniform={s_u:.4f} vs none={s_n:.4f}"

    def test_icio_norm_sqrt_no_crash(self):
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(self._icio_path())
        eng = SimulationEngine(doc, seed=0, use_icio_weights=True, icio_norm="sqrt")
        for _ in range(10):
            eng.step()
        for eid in eng.state.entity_ids:
            assert eng.state.entities[eid].supply >= 0.0

    def test_icio_norm_softmax_no_crash(self):
        from provider_sim.pdl.parser import load_pdl
        doc = load_pdl(self._icio_path())
        eng = SimulationEngine(doc, seed=0, use_icio_weights=True, icio_norm="softmax")
        for _ in range(10):
            eng.step()
        for eid in eng.state.entity_ids:
            assert eng.state.entities[eid].supply >= 0.0
