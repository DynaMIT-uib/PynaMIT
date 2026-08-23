"""Tests for the runnable MAGE workflow scripts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scripts.simulation.mage_prepare import CASE_DIRECTORY as MAGE_PREPARE_CASE
from scripts.simulation.mage_prepare import DEFAULT_GAMERA_DIRECTORY, DEFAULT_OUTPUT_PATH
from scripts.simulation.mage_project import CASE_DIRECTORY as MAGE_PROJECT_CASE
from scripts.simulation.mage_project import (
    DEFAULT_FORCING_PATH,
    DEFAULT_RESOLUTIONS_DIRECTORY,
    ProjectionSettings,
)
from scripts.simulation.mage_project import SETTINGS as MAGE_PROJECT_SETTINGS
from scripts.simulation.mage_project import main as project_mage_inputs
from scripts.simulation.mage_run import CASE_DIRECTORY as MAGE_SIMULATION_CASE
from scripts.simulation.mage_run import (
    DEFAULT_RESOLUTIONS_DIRECTORY as SIMULATION_RESOLUTIONS_DIRECTORY,
)
from scripts.simulation.mage_run import SETTINGS as MAGE_SIMULATION_SETTINGS
from scripts.simulation.mage_run import SimulationSweep, _last_projected_input_time
from scripts.simulation.mage_run import main as run_mage_simulations

from pynamit.simulation.config import SimulationConfig
from pynamit.storage import ArtifactStore
from pynamit.workflows.mage.projection import MAGE_MAIN_FIELD_KIND


def test_default_mage_case_paths_are_consistent():
    """Every MAGE stage uses the same event-local artifact tree."""
    assert MAGE_PREPARE_CASE == MAGE_PROJECT_CASE == MAGE_SIMULATION_CASE
    assert MAGE_PREPARE_CASE.name == "2011-10-24"
    assert DEFAULT_OUTPUT_PATH == DEFAULT_FORCING_PATH == MAGE_PREPARE_CASE / "forcing.h5"


def test_default_gamera_directory_is_cluster_path():
    """Preparation defaults to the intended MAGE machine data path."""
    assert DEFAULT_GAMERA_DIRECTORY == Path("/disk/Gamera_Dong")


def test_projected_input_default_matches_run_input_directory():
    """Projection and run scripts should agree on the input package."""
    assert MAGE_PROJECT_SETTINGS.resolutions_directory == DEFAULT_RESOLUTIONS_DIRECTORY
    assert MAGE_PROJECT_SETTINGS.resolutions == (20, 40, 60, 80)
    assert MAGE_PROJECT_CASE == MAGE_SIMULATION_CASE
    assert MAGE_PROJECT_CASE.name == "2011-10-24"
    assert SIMULATION_RESOLUTIONS_DIRECTORY == DEFAULT_RESOLUTIONS_DIRECTORY
    assert MAGE_SIMULATION_SETTINGS.resolutions_directory == DEFAULT_RESOLUTIONS_DIRECTORY
    assert MAGE_SIMULATION_SETTINGS.resolutions == MAGE_PROJECT_SETTINGS.resolutions
    assert MAGE_SIMULATION_SETTINGS.projection_name == MAGE_PROJECT_SETTINGS.projection_name
    assert MAGE_PROJECT_SETTINGS.cache_operators is True
    assert MAGE_PROJECT_SETTINGS.write_diagnostics is True
    assert MAGE_SIMULATION_SETTINGS.cache_operators is True


def test_mage_projection_sweeps_configured_resolutions(monkeypatch, tmp_path):
    """Every configured resolution uses the same projection path."""
    calls = []
    monkeypatch.setattr(
        "scripts.simulation.mage_project.prepare_inputs", lambda **kwargs: calls.append(kwargs)
    )
    settings = ProjectionSettings(
        forcing_path=tmp_path / "forcing.h5",
        resolutions_directory=tmp_path / "resolutions",
        resolutions=(20, 40, 60, 80),
        write_diagnostics=False,
    )

    project_mage_inputs(settings)

    assert [(call["nmax"], call["mmax"], call["ncs"]) for call in calls] == [
        (20, 20, 20),
        (40, 40, 40),
        (60, 60, 60),
        (80, 80, 80),
    ]
    assert [call["input_directory"] for call in calls] == [
        tmp_path / "resolutions" / "N20_M20_Ncs20" / "projections" / "default",
        tmp_path / "resolutions" / "N40_M40_Ncs40" / "projections" / "default",
        tmp_path / "resolutions" / "N60_M60_Ncs60" / "projections" / "default",
        tmp_path / "resolutions" / "N80_M80_Ncs80" / "projections" / "default",
    ]
    assert [call["operator_cache_directory"] for call in calls] == [
        tmp_path / "resolutions" / "N20_M20_Ncs20" / "operator_cache",
        tmp_path / "resolutions" / "N40_M40_Ncs40" / "operator_cache",
        tmp_path / "resolutions" / "N60_M60_Ncs60" / "operator_cache",
        tmp_path / "resolutions" / "N80_M80_Ncs80" / "operator_cache",
    ]


def test_mage_projection_writes_configured_diagnostics(monkeypatch, tmp_path):
    """The projection sweep also produces inspectable diagnostics."""
    input_directory = (
        tmp_path / "resolutions" / "N20_M20_Ncs20" / "projections" / "regularization-1"
    )
    monkeypatch.setattr(
        "scripts.simulation.mage_project.prepare_inputs",
        lambda **kwargs: kwargs["input_directory"],
    )
    calls = []
    monkeypatch.setattr(
        "pynamit.workflows.mage.write_input_projection_diagnostics",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    settings = ProjectionSettings(
        forcing_path=tmp_path / "forcing.h5",
        resolutions_directory=tmp_path / "resolutions",
        resolutions=(20,),
        projection_name="regularization-1",
        diagnostic_steps=(0, -1),
        diagnostic_fields=("etaP", "SigmaP"),
    )

    project_mage_inputs(settings)

    assert calls == [
        (
            (settings.forcing_path, input_directory),
            {
                "timesteps": (0, -1),
                "fields": ("etaP", "SigmaP"),
                "operator_cache_directory": (
                    tmp_path / "resolutions" / "N20_M20_Ncs20" / "operator_cache"
                ),
            },
        )
    ]


@pytest.mark.parametrize("resolutions", [(), (20, 0), (20, True), (20, 20)])
def test_mage_projection_validates_sweep_before_projecting(monkeypatch, tmp_path, resolutions):
    """Reject an invalid sweep before replacing projected packages."""
    calls = []
    monkeypatch.setattr(
        "scripts.simulation.mage_project.prepare_inputs", lambda **kwargs: calls.append(kwargs)
    )
    settings = ProjectionSettings(
        resolutions_directory=tmp_path / "resolutions",
        resolutions=resolutions,
        write_diagnostics=False,
    )

    with pytest.raises(ValueError, match="resolutions"):
        project_mage_inputs(settings)

    assert not calls


def test_mage_run_defaults_to_initialize_from_equilibrium_and_output():
    """MAGE starts from and records instantaneous equilibrium."""
    assert MAGE_SIMULATION_SETTINGS.initialize_from_equilibrium is True
    assert MAGE_SIMULATION_SETTINGS.run_equilibrium is True
    assert MAGE_SIMULATION_SETTINGS.magnetic_boundary_shielding is False
    assert MAGE_SIMULATION_SETTINGS.final_time is None


def test_mage_run_infers_final_time_from_projected_boundary_input():
    """An unedited run must stop at its last projected forcing."""
    store = SimpleNamespace(
        load_dataset=lambda key: SimpleNamespace(
            time=SimpleNamespace(values=np.array([0.0, 10.25, 20.5]))
        )
    )

    assert _last_projected_input_time(store) == 20.5


def test_mage_run_resolves_every_projected_resolution(monkeypatch, tmp_path):
    """A run sweep maps each projection to an independent named run."""
    resolutions_directory = tmp_path / "resolutions"
    for resolution in (20, 40):
        directory = (
            resolutions_directory
            / f"N{resolution}_M{resolution}_Ncs{resolution}"
            / "projections"
            / "comparison"
        )
        store = ArtifactStore(directory, preferred_dataset_storage="netcdf")
        config = SimulationConfig(
            Nmax=resolution,
            Mmax=resolution,
            Ncs=resolution,
            RM=7.0e6,
            main_field_kind=MAGE_MAIN_FIELD_KIND,
        )
        store.save_dataset(config.to_dataset(), "settings")
        store.save_dataset(xr.Dataset(coords={"time": [0.0, 10.0]}), "boundary_Br")

    settings = SimulationSweep(
        resolutions_directory=resolutions_directory,
        resolutions=(20, 40),
        projection_name="comparison",
        simulation_name="exponential",
        artifact_storage="netcdf",
    )
    calls = []
    monkeypatch.setattr(
        "scripts.simulation.mage_run.run_from_inputs",
        lambda input_directory, **kwargs: calls.append((input_directory, kwargs)),
    )

    run_mage_simulations(settings)

    assert [input_directory for input_directory, _ in calls] == [
        resolutions_directory / "N20_M20_Ncs20" / "projections" / "comparison",
        resolutions_directory / "N40_M40_Ncs40" / "projections" / "comparison",
    ]
    assert [kwargs["final_time"] for _, kwargs in calls] == [10.0, 10.0]
    assert [kwargs["simulation_directory"] for _, kwargs in calls] == [
        resolutions_directory / "N20_M20_Ncs20" / "simulations" / "exponential",
        resolutions_directory / "N40_M40_Ncs40" / "simulations" / "exponential",
    ]


def test_mage_run_validates_full_sweep_before_starting(monkeypatch, tmp_path):
    """A missing later projection cannot partially execute a sweep."""
    resolutions_directory = tmp_path / "resolutions"
    first_projection = resolutions_directory / "N20_M20_Ncs20" / "projections" / "comparison"
    store = ArtifactStore(first_projection, preferred_dataset_storage="netcdf")
    config = SimulationConfig(
        Nmax=20, Mmax=20, Ncs=20, RM=7.0e6, main_field_kind=MAGE_MAIN_FIELD_KIND
    )
    store.save_dataset(config.to_dataset(), "settings")
    store.save_dataset(xr.Dataset(coords={"time": [0.0, 10.0]}), "boundary_Br")

    settings = SimulationSweep(
        resolutions_directory=resolutions_directory,
        resolutions=(20, 40),
        projection_name="comparison",
        artifact_storage="netcdf",
    )
    calls = []
    monkeypatch.setattr(
        "scripts.simulation.mage_run.run_from_inputs",
        lambda input_directory, **kwargs: calls.append((input_directory, kwargs)),
    )

    with pytest.raises(ValueError, match="missing required artifact"):
        run_mage_simulations(settings)

    assert not calls
