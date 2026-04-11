"""PassiveMuscle: setzt alle Aktuatoren auf 0 (kein Eingriff)."""
from __future__ import annotations

import numpy as np

try:
    from palaestrai.agent.muscle import Muscle

    _HAS_PALAESTRAI = True
except ImportError:
    _HAS_PALAESTRAI = False

    class Muscle:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


class PassiveMuscle(Muscle):
    """Muscle that always proposes zero actions — no disruption, no defence."""

    def __init__(self, broker_uri, brain_uri, uid, brain_id, path):
        super().__init__(broker_uri, brain_uri, uid, brain_id, path)

    def setup(self):
        pass

    def propose_actions(self, sensors, actuators_available, is_terminal=False):
        for actuator in actuators_available:
            actuator(np.array([0.0], dtype=np.float32))
        return (
            actuators_available,
            actuators_available,
            [1 for _ in actuators_available],
            {},
        )

    def update(self, update):
        pass

    @property
    def parameters(self) -> dict:
        return {
            "uid": self.uid,
            "brain_uri": self._brain_uri,
            "broker_uri": self._broker_uri,
        }

    def __repr__(self):
        return f"PassiveMuscle(uid={self.uid!r})"

    def prepare_model(self):
        pass
