"""Package entry-point behavior."""

import runpy

import pynamit
import pynamit.workflows as workflows
from pynamit.workflows import example as example_workflow
from pynamit.workflows import prepared_inputs as prepared_input_workflow


def test_package_exports_the_researcher_workflow():
    """The top-level API directly exposes the workflow classes."""
    from pynamit.results.simulation_results import SimulationResults
    from pynamit.simulation.config import SimulationConfig
    from pynamit.simulation.geometry import SimulationGeometry
    from pynamit.simulation.input_preparation import InputPreparation
    from pynamit.simulation.simulation import Simulation

    for cls in (
        InputPreparation,
        Simulation,
        SimulationConfig,
        SimulationGeometry,
        SimulationResults,
    ):
        assert getattr(pynamit, cls.__name__) is cls


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


def test_main_module_import_is_inert(monkeypatch):
    """Importing ``pynamit.__main__`` must not launch a simulation."""
    calls = []
    monkeypatch.setattr(example_workflow, "run_example", lambda: calls.append(None))

    runpy.run_module("pynamit.__main__", run_name="pynamit.__main_import_test__")

    assert calls == []


def test_main_module_executes_as_script(monkeypatch):
    """Executing ``pynamit.__main__`` preserves script behavior."""
    calls = []
    monkeypatch.setattr(example_workflow, "run_example", lambda **kwargs: calls.append(kwargs))

    runpy.run_module("pynamit.__main__", run_name="__main__")

    assert calls[0]["event_time"].isoformat() == "2001-05-12T21:45:00"
    assert calls[0]["kp"] == 5
    assert calls[0]["imf_Bz_nT"] == -4.0
    assert calls[0]["hwm_ap"] == (-1, 35)
