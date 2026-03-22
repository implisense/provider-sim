"""Shared fixtures for provider_sim tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from provider_sim.pdl.parser import load_pdl
from provider_sim.sim.engine import SimulationEngine

SCENARIOS_DIR = (
    Path(__file__).parent.parent.parent.parent / "06_Szenarien" / "scenarios"
)

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
