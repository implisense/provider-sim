"""Shared fixtures for provider_sim tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from provider_sim.pdl.parser import load_pdl
from provider_sim.sim.engine import SimulationEngine

_REPO_ROOT = Path(__file__).parent.parent
# Prefer local scenarios/ in the repo; fall back to parent PROVIDER tree
_LOCAL_SCENARIOS = _REPO_ROOT / "scenarios"
_PROVIDER_SCENARIOS = _REPO_ROOT.parent.parent.parent / "06_Szenarien" / "scenarios"
SCENARIOS_DIR = _LOCAL_SCENARIOS if _LOCAL_SCENARIOS.exists() else _PROVIDER_SCENARIOS

SOJA_PATH = SCENARIOS_DIR / "s1-soja.pdl.yaml"


@pytest.fixture
def soja_path() -> Path:
    return SOJA_PATH


@pytest.fixture
def soja_doc():
    return load_pdl(SOJA_PATH)


@pytest.fixture
def soja_engine(soja_doc):
    return SimulationEngine(soja_doc, seed=42)


@pytest.fixture(
    params=sorted(SCENARIOS_DIR.glob("*.pdl.yaml")),
    ids=lambda p: p.stem,
)
def any_scenario_path(request) -> Path:
    return request.param
