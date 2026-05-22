"""Tests for direct input-basis coefficient setters."""

import numpy as np

from pynamit.math.constants import RE
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
    n_coeffs = dynamics.input_storage_bases["jr"].index_length
    jr_coeffs = np.arange(n_coeffs, dtype=float) + 0.25

    dynamics.set_jr(jr_coeffs, time=4.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["jr"]
    np.testing.assert_allclose(dataset["SH_jr"].isel(time=0).values, jr_coeffs)
    np.testing.assert_allclose(dataset.time.values, [4.0])


def test_set_Br_accepts_input_basis_coefficients(tmp_path):
    """Magnetospheric Br coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path, RM=4 * RE)
    n_coeffs = dynamics.input_storage_bases["Br"].index_length
    br_coeffs = np.linspace(-1.0, 1.0, n_coeffs)

    dynamics.set_Br(br_coeffs, time=2.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["Br"]
    np.testing.assert_allclose(dataset["SH_Br"].isel(time=0).values, br_coeffs)
    np.testing.assert_allclose(dataset.time.values, [2.0])


def test_set_wind_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Wind Helmholtz coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_storage_bases["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    dynamics.set_wind(cf_coeffs, df_coeffs, time=3.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values,
        np.concatenate([cf_coeffs, df_coeffs]),
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_set_resistance_accepts_input_basis_coefficients(tmp_path):
    """Pedersen and Hall resistance coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_storage_bases["conductance"].index_length
    etaP_coeffs = np.arange(n_coeffs, dtype=float) + 1.0
    etaH_coeffs = np.arange(n_coeffs, dtype=float) - 2.0

    dynamics.set_resistance(etaP_coeffs, etaH_coeffs, time=5.0, coefficients=True)

    dataset = dynamics.input_timeseries.datasets["conductance"]
    np.testing.assert_allclose(dataset["SH_etaP"].isel(time=0).values, etaP_coeffs)
    np.testing.assert_allclose(dataset["SH_etaH"].isel(time=0).values, etaH_coeffs)
    np.testing.assert_allclose(dataset.time.values, [5.0])


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
