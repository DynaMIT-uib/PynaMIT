"""Tests for persisted simulation data plumbing."""

import numpy as np
import pytest
import xarray as xr

import pynamit
from pynamit.simulation.api import Simulation
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.run_data import RunData
from pynamit.storage import ArtifactStore


def _settings(**attrs):
    defaults = {"Nmax": 3, "Mmax": 2, "Ncs": 4}
    defaults.update(attrs)
    return xr.Dataset(attrs=defaults)


def _output_payload(n_magnetic, n_surface):
    return {
        "induced_Br": np.zeros(n_magnetic),
        "boundary_jr": np.zeros(n_surface),
        "Phi": np.zeros(n_surface),
        "W": np.zeros(n_surface),
    }


def test_public_simulation_name_is_canonical():
    """The public facade has one descriptive class name."""
    assert pynamit.Simulation is Simulation
    assert "Dynamics" not in pynamit.__all__
    assert not hasattr(pynamit, "Dynamics")


def test_run_data_owns_schema_artifacts_and_field_series(tmp_path):
    """RunData creates and reloads the persisted run context."""
    run_dir = tmp_path / "run"
    settings = _settings(horizontal_basis_kind="CS", area_weighted_least_squares=1)
    data = RunData.open(settings, run_directory=run_dir, artifact_storage="netcdf")

    assert data.run_directory == str(run_dir.resolve())
    assert data.settings_saved is False
    assert data.gap_Br_response is None
    assert data.config.horizontal_basis_kind == "CS"
    assert data.config.boundary_jr_projection_basis == "CS"
    data.save_settings_if_missing()
    output_spaces = data.schema.output_field_spaces["dynamic"]
    n_magnetic = output_spaces["induced_Br"].coefficient_length
    n_surface = output_spaces["boundary_jr"].coefficient_length
    data.save_gap_Br_response_if_missing(np.zeros((n_magnetic, n_surface)))
    data.input_series.add_entry(
        "boundary_jr",
        {
            "boundary_jr": np.arange(
                data.schema.input_field_spaces["boundary_jr"].coefficient_length
            )
        },
        time=0.0,
    )
    data.save_input_dataset("boundary_jr")
    data.add_output_entry("dynamic", _output_payload(n_magnetic, n_surface), time=0.0)
    data.save_output_dataset("dynamic")

    reloaded = RunData.open(settings, run_directory=run_dir, artifact_storage="netcdf")

    assert reloaded.settings_saved is True
    assert reloaded.gap_Br_response is not None
    assert reloaded.gap_Br_response.dims == ("poloidal_i", "surface_i")
    assert "boundary_jr" in reloaded.input_series.datasets
    assert "dynamic" in reloaded.output_series.datasets
    np.testing.assert_allclose(reloaded.gap_Br_response.values, np.zeros((n_magnetic, n_surface)))
    np.testing.assert_allclose(
        reloaded.output_series.get_entry("dynamic", 0.0)["boundary_jr"], np.zeros(n_surface)
    )


def test_run_data_reuses_validated_config(tmp_path):
    """The runtime shares one immutable validated configuration."""
    config = SimulationConfig(Nmax=2, Mmax=1, Ncs=8, enable_pfac_coupling=False)

    data = RunData.open(config, run_directory=tmp_path / "run", artifact_storage="netcdf")

    assert data.config is config


def test_run_data_rejects_saved_settings_mismatch(tmp_path):
    """Saved settings guard against restarting with new args."""
    run_dir = tmp_path / "run"
    settings = _settings(Nmax=3)
    data = RunData.open(settings, run_directory=run_dir, artifact_storage="netcdf")
    data.save_settings_if_missing()

    with pytest.raises(ValueError, match="Mismatch"):
        RunData.open(_settings(Nmax=4), run_directory=run_dir, artifact_storage="netcdf")


def test_run_data_rejects_legacy_magnetic_schema(tmp_path):
    """A legacy run cannot be misread as physical magnetic variables."""
    run_dir = tmp_path / "run"
    legacy_settings = _settings()
    ArtifactStore(run_dir, preferred_dataset_storage="netcdf").save_dataset(
        legacy_settings, "settings"
    )

    with pytest.raises(ValueError, match="simulation schema"):
        RunData.open(legacy_settings, run_directory=run_dir, artifact_storage="netcdf")


def test_simulation_has_one_persistence_and_spatial_ownership_path(tmp_path):
    """Run data and geometry have one canonical owner each."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=True,
        artifact_storage="netcdf",
    )

    assert simulation.config is simulation.run_data.config
    assert simulation.geometry.horizontal_basis is simulation.run_data.schema.horizontal_basis
    assert simulation.response.geometry is simulation.geometry
    for redundant_name in (
        "settings",
        "io",
        "run_directory",
        "schema",
        "cs_basis",
        "horizontal_basis",
        "solid_harmonics",
        "input_field_spaces",
        "output_field_spaces",
        "input_series",
        "output_series",
        "main_field",
        "backend",
    ):
        assert not hasattr(simulation, redundant_name)
    assert simulation.run_data.gap_Br_response is None


@pytest.mark.parametrize(
    ("enable_pfac_coupling", "expected_persisted"), [(True, True), (False, False)]
)
def test_simulation_persists_only_active_gap_Br_response(
    tmp_path, enable_pfac_coupling, expected_persisted
):
    """Defer active gap-field work until model output is requested."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=enable_pfac_coupling,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "conductance"
    ].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(resistance_shape),
        log_ratio_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )
    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)

    assert simulation.run_data.gap_Br_response is None
    simulation.impose_equilibrium(time=0.0, quiet=True)
    assert (simulation.run_data.gap_Br_response is not None) is expected_persisted


def test_simulation_from_directory_uses_saved_configuration(tmp_path):
    """Saved settings seed restart construction."""
    run_dir = tmp_path / "run"
    original = Simulation(
        run_directory=str(run_dir),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        horizontal_basis_kind="CS",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    reloaded = Simulation.from_directory(
        str(run_dir), horizontal_basis_kind=None, artifact_storage="netcdf"
    )

    assert reloaded.config.Nmax == original.config.Nmax
    assert reloaded.config.Mmax == original.config.Mmax
    assert reloaded.config.horizontal_basis_kind == "CS"
    assert (
        reloaded.run_data.schema.horizontal_basis is not original.run_data.schema.horizontal_basis
    )
    assert reloaded.run_data.schema.horizontal_basis.index_length == (
        original.run_data.schema.horizontal_basis.index_length
    )


def test_simulation_from_directory_rejects_conflicting_override(tmp_path):
    """Explicit restart overrides must match saved settings."""
    run_dir = tmp_path / "run"
    Simulation(
        run_directory=str(run_dir),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    with pytest.raises(ValueError, match="Mismatch"):
        Simulation.from_directory(str(run_dir), Nmax=3, artifact_storage="netcdf")
