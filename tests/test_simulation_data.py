"""Tests for persisted simulation data plumbing."""

import numpy as np
import pytest
import xarray as xr

from pynamit.simulation.data import SimulationData
from pynamit.simulation.dynamics import Dynamics


def _settings(**attrs):
    defaults = {
        "Nmax": 3,
        "Mmax": 2,
        "Ncs": 4,
        "vector_jr": 1,
        "vector_Br": 1,
        "vector_conductance": 1,
        "vector_u": 1,
    }
    defaults.update(attrs)
    return xr.Dataset(attrs=defaults)


def _state_payload(n_coeffs):
    return {
        "m_ind": np.zeros(n_coeffs),
        "m_imp": np.zeros(n_coeffs),
        "Phi": np.zeros(n_coeffs),
        "W": np.zeros(n_coeffs),
    }


def test_simulation_data_owns_schema_io_and_timeseries(tmp_path):
    """SimulationData creates and reloads the persisted run context."""
    run_dir = tmp_path / "run"
    settings = _settings()
    data = SimulationData.create(
        settings,
        run_directory=run_dir,
        artifact_storage="netcdf",
        horizontal_basis_kind="CS",
        area_weighted_least_squares=True,
    )

    assert data.run_directory == str(run_dir.resolve())
    assert not data.settings_from_file
    assert not data.pfac_matrix_from_file
    assert data.input_timeseries.area_weighted_least_squares
    assert data.output_timeseries.area_weighted_least_squares

    data.save_settings_if_missing()
    n_state = data.schema.output_field_spaces["state"].coefficient_length
    data.save_pfac_matrix_if_missing(
        xr.DataArray(np.eye(n_state), dims=("row", "col"), name="PFAC_matrix")
    )
    data.input_timeseries.add_entry(
        "jr",
        {"jr": np.arange(data.schema.input_field_spaces["jr"].coefficient_length)},
        time=0.0,
    )
    data.save_input_dataset("jr")
    data.add_output_entry("state", _state_payload(n_state), time=0.0)
    data.save_output_dataset("state")

    reloaded = SimulationData.create(
        settings,
        run_directory=run_dir,
        artifact_storage="netcdf",
        horizontal_basis_kind="CS",
        area_weighted_least_squares=True,
    )

    assert reloaded.settings_from_file
    assert reloaded.pfac_matrix_from_file
    assert "jr" in reloaded.input_timeseries.datasets
    assert "state" in reloaded.output_timeseries.datasets
    np.testing.assert_allclose(reloaded.pfac_matrix.values, np.eye(n_state))
    np.testing.assert_allclose(
        reloaded.output_timeseries.get_entry("state", 0.0)["m_imp"],
        np.zeros(n_state),
    )


def test_simulation_data_rejects_saved_settings_mismatch(tmp_path):
    """Saved settings guard against restarting with new args."""
    run_dir = tmp_path / "run"
    settings = _settings(Nmax=3)
    data = SimulationData.create(
        settings,
        run_directory=run_dir,
        artifact_storage="netcdf",
        horizontal_basis_kind="SH",
    )
    data.save_settings_if_missing()

    with pytest.raises(ValueError, match="Mismatch"):
        SimulationData.create(
            _settings(Nmax=4),
            run_directory=run_dir,
            artifact_storage="netcdf",
            horizontal_basis_kind="SH",
        )


def test_dynamics_delegates_persistence_to_simulation_data(tmp_path):
    """Dynamics aliases are backed by SimulationData."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        artifact_storage="netcdf",
    )

    assert dynamics.data.io is dynamics.io
    assert dynamics.data.schema is dynamics.schema
    assert dynamics.data.input_timeseries is dynamics.input_timeseries
    assert dynamics.data.output_timeseries is dynamics.output_timeseries
    assert dynamics.data.pfac_matrix is not None
