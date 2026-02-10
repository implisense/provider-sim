"""Supply-chain simulation state: entity states, event states, graph topology."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from provider_sim.pdl.model import PdlDocument


@dataclass
class EntityState:
    supply: float = 1.0
    demand: float = 1.0
    price: float = 1.0
    health: float = 1.0
    supply_modifiers: Dict[str, float] = field(default_factory=dict)


@dataclass
class EventState:
    active: bool = False
    active_since_tick: Optional[int] = None
    remaining_ticks: int = 0


class SupplyChainState:
    """Holds the full simulation state for one PDL scenario."""

    def __init__(self) -> None:
        self.tick: int = 0
        self.max_ticks: int = 365
        self.entity_ids: List[str] = []
        self.event_ids: List[str] = []
        self.entities: Dict[str, EntityState] = {}
        self.events: Dict[str, EventState] = {}

        # Graph adjacency
        self.downstream: Dict[str, Set[str]] = defaultdict(set)
        self.upstream: Dict[str, Set[str]] = defaultdict(set)
        self.depends_on: Dict[str, Set[str]] = defaultdict(set)

        # Entity metadata from PDL
        self.vulnerability: Dict[str, float] = {}

        # Event metadata from PDL (duration in ticks, probability, trigger target)
        self.event_duration_ticks: Dict[str, int] = {}
        self.event_probability: Dict[str, float] = {}
        self.event_target: Dict[str, str] = {}

        self.rng: np.random.Generator = np.random.default_rng()

    # ---- ConditionState protocol ----

    def is_event_active(self, event_id: str) -> bool:
        es = self.events.get(event_id)
        return es.active if es else False

    def event_active_duration_days(self, event_id: str) -> float:
        es = self.events.get(event_id)
        if es is None or not es.active or es.active_since_tick is None:
            return 0.0
        return float(self.tick - es.active_since_tick)


def build_state_from_pdl(
    doc: PdlDocument,
    seed: int | None = None,
    max_ticks: int = 365,
) -> SupplyChainState:
    """Initialise a SupplyChainState from a parsed PdlDocument."""
    state = SupplyChainState()
    state.max_ticks = max_ticks
    state.rng = np.random.default_rng(seed)

    # Entities
    for ent in doc.entities:
        state.entity_ids.append(ent.id)
        state.entities[ent.id] = EntityState()
        state.vulnerability[ent.id] = ent.vulnerability

    # Events
    for ev in doc.events:
        state.event_ids.append(ev.id)
        state.events[ev.id] = EventState()
        dur = ev.impact.duration
        state.event_duration_ticks[ev.id] = int(dur.days) if dur else 30
        state.event_probability[ev.id] = ev.trigger.probability or 0.0
        state.event_target[ev.id] = ev.trigger.target

    # Graph from supply-chain stages
    for sc in doc.supply_chains:
        for stage in sc.stages:
            if len(stage) == 2:
                src, dst = stage[0], stage[1]
                state.downstream[src].add(dst)
                state.upstream[dst].add(src)

        for dep in sc.dependencies:
            state.depends_on[dep.from_entity].add(dep.to_entity)

    return state
