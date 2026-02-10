"""Five-phase simulation engine for supply-chain disruption scenarios."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np

from provider_sim.pdl.condition import ConditionNode, parse_condition
from provider_sim.pdl.model import Event, PdlDocument
from provider_sim.sim.state import SupplyChainState, build_state_from_pdl

# Recovery rate per tick towards 1.0 when no events are active
_RECOVERY_RATE = 0.02


class SimulationEngine:
    """Runs a supply-chain disruption simulation driven by a PDL document."""

    def __init__(
        self,
        doc: PdlDocument,
        seed: int | None = None,
        max_ticks: int = 365,
    ) -> None:
        self.doc = doc
        self._seed = seed
        self._max_ticks = max_ticks
        self.state = build_state_from_pdl(doc, seed=seed, max_ticks=max_ticks)

        # Pre-parse condition ASTs
        self._conditions: Dict[str, ConditionNode] = {}
        self._root_events: List[str] = []
        self._condition_events: List[str] = []

        # Impact caches: event_id → (attribute, decimal_modifier)
        self._supply_impact: Dict[str, float] = {}
        self._demand_impact: Dict[str, float] = {}
        self._price_impact: Dict[str, float] = {}

        for ev in doc.events:
            if ev.is_root_event:
                self._root_events.append(ev.id)
            elif ev.trigger.condition:
                self._conditions[ev.id] = parse_condition(ev.trigger.condition)
                self._condition_events.append(ev.id)

            imp = ev.impact
            if imp.supply is not None:
                self._supply_impact[ev.id] = imp.supply.decimal
            if imp.demand is not None:
                self._demand_impact[ev.id] = imp.demand.decimal
            if imp.price is not None:
                self._price_impact[ev.id] = imp.price.decimal

        # Topological order for flow propagation
        self._topo_order = self._compute_topo_order()

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def _compute_topo_order(self) -> List[str]:
        """Kahn's algorithm over downstream edges."""
        in_degree: Dict[str, int] = {eid: 0 for eid in self.state.entity_ids}
        for src, dsts in self.state.downstream.items():
            for dst in dsts:
                if dst in in_degree:
                    in_degree[dst] += 1

        queue = deque(eid for eid, deg in in_degree.items() if deg == 0)
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for dst in self.state.downstream.get(node, set()):
                if dst in in_degree:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        queue.append(dst)

        # Entities not reached (cycles) are appended at the end
        remaining = [eid for eid in self.state.entity_ids if eid not in set(order)]
        order.extend(remaining)
        return order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        attacker_actions: Optional[Dict[str, float]] = None,
        defender_actions: Optional[Dict[str, float]] = None,
    ) -> None:
        """Execute one simulation tick (5 phases)."""
        attacker_actions = attacker_actions or {}
        defender_actions = defender_actions or {}

        self._phase1_apply_actions(attacker_actions, defender_actions)
        self._phase2_evaluate_events(attacker_actions)
        self._phase3_apply_impacts()
        self._phase4_propagate_flow()
        self._phase5_compute_health()

        self.state.tick += 1

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = build_state_from_pdl(
            self.doc, seed=self._seed, max_ticks=self._max_ticks
        )

    # ------------------------------------------------------------------
    # Phase 1: Agent actions
    # ------------------------------------------------------------------

    def _phase1_apply_actions(
        self,
        attacker: Dict[str, float],
        defender: Dict[str, float],
    ) -> None:
        for eid, strength in attacker.items():
            es = self.state.entities.get(eid)
            if es is None:
                continue
            vuln = self.state.vulnerability.get(eid, 0.5)
            es.supply = max(0.0, es.supply - strength * vuln)

        for eid, strength in defender.items():
            es = self.state.entities.get(eid)
            if es is None:
                continue
            es.supply = min(2.0, es.supply + strength * 0.5)

    # ------------------------------------------------------------------
    # Phase 2: Event evaluation
    # ------------------------------------------------------------------

    def _phase2_evaluate_events(self, attacker_actions: Dict[str, float]) -> None:
        s = self.state

        for ev_id in self._root_events:
            evs = s.events[ev_id]
            if evs.active:
                self._tick_event(ev_id, evs)
                continue
            prob = s.event_probability[ev_id]
            # Attacker can force root events by targeting the event's entity
            target = s.event_target[ev_id]
            if target in attacker_actions:
                prob = min(1.0, prob + attacker_actions[target])
            if s.rng.random() < prob:
                self._activate_event(ev_id)

        for ev_id in self._condition_events:
            evs = s.events[ev_id]
            if evs.active:
                self._tick_event(ev_id, evs)
                continue
            cond = self._conditions[ev_id]
            if cond.evaluate(s):
                self._activate_event(ev_id)

    def _activate_event(self, ev_id: str) -> None:
        evs = self.state.events[ev_id]
        evs.active = True
        evs.active_since_tick = self.state.tick
        evs.remaining_ticks = self.state.event_duration_ticks[ev_id]

    def _tick_event(self, ev_id: str, evs) -> None:
        evs.remaining_ticks -= 1
        if evs.remaining_ticks <= 0:
            evs.active = False
            evs.active_since_tick = None

    # ------------------------------------------------------------------
    # Phase 3: Impact stack (modifier-based, no compounding)
    # ------------------------------------------------------------------

    def _phase3_apply_impacts(self) -> None:
        s = self.state

        # Clear all modifiers first, then re-apply from active events
        for eid in s.entity_ids:
            s.entities[eid].supply_modifiers.clear()

        for ev in self.doc.events:
            evs = s.events[ev.id]
            if not evs.active:
                continue
            target = ev.trigger.target
            es = s.entities.get(target)
            if es is None:
                continue

            # Supply modifier
            if ev.id in self._supply_impact:
                es.supply_modifiers[ev.id] = self._supply_impact[ev.id]

            # Demand impact (directly applied)
            if ev.id in self._demand_impact:
                es.demand = max(0.0, 1.0 + self._demand_impact[ev.id])

            # Price impact (directly applied)
            if ev.id in self._price_impact:
                es.price = max(0.01, 1.0 + self._price_impact[ev.id])

        # Compute effective supply from modifier stack
        for eid in s.entity_ids:
            es = s.entities[eid]
            if es.supply_modifiers:
                product = 1.0
                for mod in es.supply_modifiers.values():
                    product *= (1.0 + mod)
                es.supply = max(0.0, product)
            else:
                # Natural recovery towards 1.0
                if es.supply < 1.0:
                    es.supply = min(1.0, es.supply + _RECOVERY_RATE)

    # ------------------------------------------------------------------
    # Phase 4: Flow propagation (topological)
    # ------------------------------------------------------------------

    def _phase4_propagate_flow(self) -> None:
        s = self.state

        for eid in self._topo_order:
            es = s.entities[eid]
            ups = s.upstream.get(eid, set())

            if ups:
                incoming = [s.entities[u].supply for u in ups if u in s.entities]
                if incoming:
                    mean_incoming = sum(incoming) / len(incoming)
                    es.supply = min(es.supply, mean_incoming)

            # Dependency penalty
            deps = s.depends_on.get(eid, set())
            for dep_id in deps:
                dep_es = s.entities.get(dep_id)
                if dep_es and dep_es.supply < 0.5:
                    penalty = 0.3 * (1.0 - dep_es.supply)
                    es.supply = max(0.0, es.supply - penalty)

    # ------------------------------------------------------------------
    # Phase 5: Health
    # ------------------------------------------------------------------

    def _phase5_compute_health(self) -> None:
        for eid in self.state.entity_ids:
            es = self.state.entities[eid]
            inv_price = min(1.0, 1.0 / max(es.price, 0.01))
            demand_term = min(es.demand, 1.0)
            es.health = np.clip(
                0.5 * es.supply + 0.3 * inv_price + 0.2 * demand_term,
                0.0,
                1.0,
            )
