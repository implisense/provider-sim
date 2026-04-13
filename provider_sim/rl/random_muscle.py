"""RandomMuscle: Budget-normierte Zufallsaktionen fuer Curriculum-Training.

Gegenstueck zu DummyMuscle (Null-Aktionen): RandomMuscle waehlt zufaellige
Aktionen die auf das Budget normiert sind. Dient als "schwacher Gegner" in
Phase 1 des Curriculum-Trainings, bevor Self-Play startet.
"""
from __future__ import annotations

import numpy as np

try:
    from palaestrai.agent.muscle import Muscle
except ImportError:
    class Muscle:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


class RandomMuscle(Muscle):
    """Erzeugt budget-normierte Zufallsaktionen ohne jedes Lernen.

    Params (YAML):
        budget: float -- Aktionsbudget (Summe aller Aktionen <= budget)
        n_act:  int   -- Anzahl der Aktuatoren (default 20)
        seed:   int   -- RNG-Seed (default 0, fuer Reproduzierbarkeit)
    """

    def __init__(
        self, *args,
        budget: float = 0.8,
        n_act: int = 20,
        seed: int = 0,
        **kwargs,
    ) -> None:
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            pass
        self._budget = float(budget)
        self._n_act = int(n_act)
        self._rng = np.random.default_rng(seed)

    def setup(self) -> None:
        pass

    def propose_actions(self, sensors, actuators_available):
        raw = self._rng.random(self._n_act).astype(np.float32)
        s = float(raw.sum())
        if s > 0:
            raw = raw * (self._budget / s)
        raw = np.clip(raw, 0.0, 1.0)

        for actuator, val in zip(actuators_available, raw):
            actuator(np.array([float(val)], dtype=np.float32))

        return (actuators_available, None)

    def update(self, update) -> None:
        pass  # kein Lernen

    def prepare_model(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"RandomMuscle(budget={self._budget}, n_act={self._n_act})"
