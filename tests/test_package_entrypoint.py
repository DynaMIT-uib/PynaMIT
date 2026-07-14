"""Package entry-point behavior."""

import runpy

import pynamit
import pynamit.simulation as simulation_api
from pynamit.simulation.workflows import standard as standard_workflow


def test_simulation_package_has_an_explicit_public_api():
    """The simulation package exports only its stable entry points."""
    assert simulation_api.__all__ == ["Simulation", "SimulationConfig"]
    assert simulation_api.Simulation is pynamit.Simulation
    assert simulation_api.SimulationConfig is pynamit.SimulationConfig
    assert not hasattr(simulation_api, "InputPipeline")
    assert not hasattr(simulation_api, "SimulationRunner")


def test_main_module_import_is_inert(monkeypatch):
    """Importing ``pynamit.__main__`` must not launch a simulation."""
    calls = []
    monkeypatch.setattr(standard_workflow, "run_pynamit", lambda: calls.append(None))

    runpy.run_module("pynamit.__main__", run_name="pynamit.__main_import_test__")

    assert calls == []


def test_main_module_executes_as_script(monkeypatch):
    """Executing ``pynamit.__main__`` preserves script behavior."""
    calls = []
    monkeypatch.setattr(standard_workflow, "run_pynamit", lambda: calls.append(None))

    runpy.run_module("pynamit.__main__", run_name="__main__")

    assert calls == [None]
