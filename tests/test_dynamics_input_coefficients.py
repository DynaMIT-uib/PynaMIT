"""Tests for direct input-basis coefficient setters."""

import numpy as np

from pynamit.math.constants import RE
from pynamit.primitives.field_coefficients import FieldCoefficients
from pynamit.simulation.dynamics import Dynamics


def _small_dynamics(tmp_path, **kwargs):
    return Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        artifact_storage="netcdf",
        **kwargs,
    )


def test_set_jr_accepts_input_basis_coefficients(tmp_path):
    """Radial current coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["jr"].index_length
    jr_coeffs = np.arange(n_coeffs, dtype=float) + 0.25

    dynamics.set_jr(jr_coeffs, time=4.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["jr"]
    np.testing.assert_allclose(dataset["SH_jr"].isel(time=0).values, jr_coeffs)
    np.testing.assert_allclose(dataset.time.values, [4.0])


def test_set_Br_accepts_input_basis_coefficients(tmp_path):
    """Magnetospheric Br coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path, RM=4 * RE)
    n_coeffs = dynamics.input_field_spaces["Br"].index_length
    br_coeffs = np.linspace(-1.0, 1.0, n_coeffs)

    dynamics.set_Br(br_coeffs, time=2.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["Br"]
    np.testing.assert_allclose(dataset["SH_Br"].isel(time=0).values, br_coeffs)
    np.testing.assert_allclose(dataset.time.values, [2.0])


def test_set_neutral_wind_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Wind Helmholtz coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    dynamics.set_neutral_wind(cf_coeffs, df_coeffs, time=3.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values,
        np.concatenate([cf_coeffs, df_coeffs]),
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_set_u_uses_neutral_wind_api_without_set_wind(tmp_path):
    """Historical set_u delegates to set_neutral_wind."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    assert not hasattr(dynamics, "set_wind")
    dynamics.set_u(cf_coeffs, df_coeffs, time=3.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values,
        np.concatenate([cf_coeffs, df_coeffs]),
    )


def test_state_update_uses_field_coefficients_for_wind(tmp_path):
    """State coefficient storage does not need grid expansion."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    dynamics.set_neutral_wind(cf_coeffs, df_coeffs, time=3.0, coefficients=True)
    dynamics.state.update(dynamics.input_timeseries, time=3.0)

    assert isinstance(dynamics.state.u, FieldCoefficients)
    np.testing.assert_allclose(
        dynamics.state.u.coeffs,
        np.vstack([cf_coeffs, df_coeffs]),
    )


def test_set_Q_eff_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Q_eff Helmholtz coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    dynamics.set_Q_eff(cf_coeffs, df_coeffs, time=3.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["Q_eff"]
    np.testing.assert_allclose(
        dataset["SH_Q_eff"].isel(time=0).values,
        np.concatenate([cf_coeffs, df_coeffs]),
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_state_update_uses_field_coefficients_for_Q_eff(tmp_path):
    """Q_eff state storage keeps canonical coefficient shape."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    dynamics.set_Q_eff(cf_coeffs, df_coeffs, time=3.0, coefficients=True)
    dynamics.state.update(dynamics.input_timeseries, time=3.0)

    assert isinstance(dynamics.state.Q_eff, FieldCoefficients)
    np.testing.assert_allclose(
        dynamics.state.Q_eff.coeffs,
        np.vstack([cf_coeffs, df_coeffs]),
    )


def test_set_resistance_accepts_input_basis_coefficients(tmp_path):
    """Pedersen and Hall resistance coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["conductance"].index_length
    etaP_coeffs = np.arange(n_coeffs, dtype=float) + 1.0
    etaH_coeffs = np.arange(n_coeffs, dtype=float) - 2.0

    dynamics.set_resistance(etaP_coeffs, etaH_coeffs, time=5.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["conductance"]
    np.testing.assert_allclose(dataset["SH_etaP"].isel(time=0).values, etaP_coeffs)
    np.testing.assert_allclose(dataset["SH_etaH"].isel(time=0).values, etaH_coeffs)
    np.testing.assert_allclose(dataset.time.values, [5.0])


def test_set_resistance_can_store_native_grid_values_without_projection(tmp_path):
    """No-projection conductance stores CS grid values."""
    dynamics = _small_dynamics(tmp_path, project_conductance=False)
    grid = dynamics.state.geometry.grid
    etaP = np.linspace(0.1, 0.3, grid.size)
    etaH = np.linspace(-0.2, 0.2, grid.size)

    dynamics.set_resistance(etaP, etaH, lat=grid.lat, lon=grid.lon, time=6.0)

    dataset = dynamics.input_timeseries.datasets["conductance"]
    np.testing.assert_allclose(dataset["CS_etaP"].isel(time=0).values, etaP)
    np.testing.assert_allclose(dataset["CS_etaH"].isel(time=0).values, etaH)
    np.testing.assert_allclose(dataset.time.values, [6.0])

    dynamics.state.update(dynamics.input_timeseries, time=6.0)
    np.testing.assert_allclose(dynamics.state.etaP.coeffs, etaP)
    np.testing.assert_allclose(dynamics.state.etaH.coeffs, etaH)
    np.testing.assert_allclose(
        dynamics.state._conductance_synthesis_matrix(),
        np.eye(grid.size),
        atol=1e-12,
    )


def test_set_resistance_without_projection_requires_matching_grid(tmp_path):
    """Direct grid conductance rejects non-model grids."""
    dynamics = _small_dynamics(tmp_path, project_conductance=False)
    grid = dynamics.state.geometry.grid
    etaP = np.ones(grid.size)
    etaH = np.zeros(grid.size)

    with np.testing.assert_raises_regex(ValueError, "input grid to match"):
        dynamics.set_resistance(
            etaP,
            etaH,
            lat=grid.lat + np.linspace(0.0, 1e-3, grid.size),
            lon=grid.lon,
        )


def test_set_resistance_without_projection_rejects_projection_options(tmp_path):
    """Direct grid conductance rejects projection controls."""
    dynamics = _small_dynamics(tmp_path, project_conductance=False)
    grid = dynamics.state.geometry.grid
    etaP = np.ones(grid.size)
    etaH = np.zeros(grid.size)

    with np.testing.assert_raises_regex(ValueError, "reg_lambda"):
        dynamics.set_resistance(
            etaP,
            etaH,
            lat=grid.lat,
            lon=grid.lon,
            reg_lambda=1e-3,
        )


def test_set_conductance_delegates_resistance_conversion(tmp_path, monkeypatch):
    """Conductance inputs are converted once before delegation."""
    dynamics = _small_dynamics(tmp_path)
    hall = np.array([[3.0, 4.0]])
    pedersen = np.array([[4.0, 3.0]])
    recorded = {}

    def record_set_resistance(Pedersen, Hall, **kwargs):
        recorded["Pedersen"] = Pedersen
        recorded["Hall"] = Hall
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(dynamics, "set_resistance", record_set_resistance)

    dynamics.set_conductance(
        hall,
        pedersen,
        lat=np.array([60.0, 61.0]),
        lon=np.array([10.0, 11.0]),
        time=7.0,
        sqrt_weights=np.ones(2),
        reg_lambda=1e-3,
        pinv_rtol=1e-10,
    )

    denominator = hall**2 + pedersen**2
    np.testing.assert_allclose(recorded["Pedersen"], pedersen / denominator)
    np.testing.assert_allclose(recorded["Hall"], hall / denominator)
    assert recorded["kwargs"]["time"] == 7.0
    assert recorded["kwargs"]["reg_lambda"] == 1e-3
    assert recorded["kwargs"]["pinv_rtol"] == 1e-10
