"""Package entry-point behavior."""

import importlib.util
import runpy

import pynamit
import pynamit.simulation as simulation_api
import pynamit.workflows as workflows
from pynamit.workflows import example as example_workflow
from pynamit.workflows import prepared_inputs as prepared_input_workflow


def test_simulation_package_has_an_explicit_public_api():
    """The simulation package exports only its stable entry points."""
    assert simulation_api.__all__ == ["InputPreparation", "Simulation", "SimulationConfig"]
    assert simulation_api.InputPreparation is pynamit.InputPreparation
    assert simulation_api.Simulation is pynamit.Simulation
    assert simulation_api.SimulationConfig is pynamit.SimulationConfig
    assert hasattr(pynamit, "SimulationResults")
    assert not hasattr(pynamit, "RunResults")
    assert not hasattr(simulation_api, "RunData")
    assert not hasattr(simulation_api, "_InputProjector")
    assert not hasattr(simulation_api, "_TimeEvolution")
    assert not hasattr(simulation_api.Simulation, "set_jr")
    assert not hasattr(simulation_api.Simulation, "set_u")


def test_simulation_workflow_names_are_short_and_explicit():
    """Common prepared-input workflows avoid repeated package names."""
    assert workflows.__all__ == ["prepare_example_inputs", "run_example", "run_from_inputs"]
    assert workflows.prepare_example_inputs.__module__ == "pynamit.workflows.example_inputs"
    assert workflows.run_example is example_workflow.run_example
    assert workflows.run_from_inputs is prepared_input_workflow.run_from_inputs
    assert prepared_input_workflow.__all__ == [
        "SIMULATION_MANIFEST_FILENAME",
        "load_prepared_inputs_into_simulation",
        "run_from_inputs",
    ]
    assert not hasattr(workflows, "prepare_inputs")
    assert not hasattr(workflows, "run_pynamit")
    assert not hasattr(prepared_input_workflow, "prepare_pynamit_inputs")
    assert not hasattr(prepared_input_workflow, "run_pynamit_from_inputs")
    assert not hasattr(prepared_input_workflow, "INPUT_MANIFEST_FILENAME")
    assert not hasattr(prepared_input_workflow, "write_input_manifest")
    assert not hasattr(pynamit, "BasisEvaluator")
    assert importlib.util.find_spec("pynamit.workflows.standard") is None
    assert importlib.util.find_spec("pynamit.simulation.run_data") is None


def test_main_module_import_is_inert(monkeypatch):
    """Importing ``pynamit.__main__`` must not launch a simulation."""
    calls = []
    monkeypatch.setattr(example_workflow, "run_example", lambda: calls.append(None))

    runpy.run_module("pynamit.__main__", run_name="pynamit.__main_import_test__")

    assert calls == []


def test_main_module_executes_as_script(monkeypatch):
    """Executing ``pynamit.__main__`` preserves script behavior."""
    calls = []
    monkeypatch.setattr(example_workflow, "run_example", lambda: calls.append(None))

    runpy.run_module("pynamit.__main__", run_name="__main__")

    assert calls == [None]
