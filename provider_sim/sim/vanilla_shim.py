"""Shim for palaestrAI 3.5.8 SimulationController naming convention.

palaestrAI 3.5.8 experiment_run.py appends "SimulationController" to the
simulation class name when it does not already end with that suffix.
VanillaSimController ends with "SimController" (not "SimulationController"),
so palaestrAI would look for the non-existent VanillaSimControllerSimulationController.

This shim re-exports VanillaSimController under the name
VanillaSimulationController, which ends with "SimulationController" and
therefore passes the endswith-check without triggering the append.

YAML usage:
  simulation:
    name: provider_sim.sim.vanilla_shim:VanillaSimulationController
"""
from __future__ import annotations

from palaestrai.simulation.vanilla_sim_controller import VanillaSimController

VanillaSimulationController = VanillaSimController
