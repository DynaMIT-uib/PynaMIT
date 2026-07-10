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


def test_dynamics_reuses_input_transforms_for_shared_representations(tmp_path):
    """Input transforms are shared by representation and grid."""
    dynamics = _small_dynamics(tmp_path)

    assert dynamics.input_transforms["jr"] is dynamics.input_transforms["Br"]
    assert dynamics.input_transforms["jr"] is dynamics.input_transforms["u"]
    assert dynamics.input_transforms["jr"] is dynamics.input_transforms["Q_eff"]
    assert dynamics.input_transforms["conductance"] is not dynamics.input_transforms["jr"]


def test_set_jr_accepts_input_basis_coefficients(tmp_path):
    """Radial current coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["jr"].index_length
    jr_coeffs = np.arange(n_coeffs, dtype=float) + 0.25

    dynamics.set_jr(jr_coefficients=jr_coeffs, time=4.0)

    dataset = dynamics.input_timeseries.datasets["jr"]
    np.testing.assert_allclose(dataset["SH_jr"].isel(time=0).values, jr_coeffs)
    np.testing.assert_allclose(dataset.time.values, [4.0])


def test_set_Br_accepts_input_basis_coefficients(tmp_path):
    """Magnetospheric Br coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path, RM=4 * RE)
    n_coeffs = dynamics.input_field_spaces["Br"].index_length
    br_coeffs = np.linspace(-1.0, 1.0, n_coeffs)

    dynamics.set_Br(Br_coefficients=br_coeffs, time=2.0)

    dataset = dynamics.input_timeseries.datasets["Br"]
    np.testing.assert_allclose(dataset["SH_Br"].isel(time=0).values, br_coeffs)
    np.testing.assert_allclose(dataset.time.values, [2.0])


def test_set_neutral_wind_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Wind Helmholtz coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    dynamics.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=3.0)

    dataset = dynamics.input_timeseries.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_set_u_uses_neutral_wind_api(tmp_path):
    """Historical set_u delegates to set_neutral_wind."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    dynamics.set_u(u_cf=cf_coeffs, u_df=df_coeffs, time=3.0)

    dataset = dynamics.input_timeseries.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )


def test_state_update_uses_field_coefficients_for_wind(tmp_path):
    """State coefficient storage does not need grid expansion."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    dynamics.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=3.0)
    dynamics.state.update(dynamics.input_timeseries, time=3.0)

    assert isinstance(dynamics.state.u, FieldCoefficients)
    np.testing.assert_allclose(dynamics.state.u.array, np.vstack([cf_coeffs, df_coeffs]))


def test_set_Q_eff_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Q_eff Helmholtz coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    dynamics.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=3.0)

    dataset = dynamics.input_timeseries.datasets["Q_eff"]
    np.testing.assert_allclose(
        dataset["SH_Q_eff"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_set_neutral_wind_rejects_existing_Q_eff_input(tmp_path):
    """Direct wind and Q_eff are mutually exclusive."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    dynamics.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        dynamics.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=1.0)


def test_set_Q_eff_rejects_existing_neutral_wind_input(tmp_path):
    """Q_eff cannot be added after direct wind input."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    dynamics.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        dynamics.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=1.0)


def test_state_calculation_rejects_simultaneous_wind_and_Q_eff(tmp_path):
    """State calculation should not double-count wind forcing."""
    dynamics = _small_dynamics(tmp_path)
    dynamics.state.u = object()
    dynamics.state.Q_eff = object()

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        dynamics.state.calculate_noind_coeffs()


def test_state_update_uses_field_coefficients_for_Q_eff(tmp_path):
    """Q_eff state storage keeps canonical coefficient shape."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    dynamics.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=3.0)
    dynamics.state.update(dynamics.input_timeseries, time=3.0)

    assert isinstance(dynamics.state.Q_eff, FieldCoefficients)
    np.testing.assert_allclose(dynamics.state.Q_eff.array, np.vstack([cf_coeffs, df_coeffs]))


def test_set_resistance_accepts_input_basis_coefficients(tmp_path):
    """EtaP and etaH resistance coefficients are stored directly."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["conductance"].index_length
    etaP_coeffs = np.arange(n_coeffs, dtype=float) + 1.0
    etaH_coeffs = np.arange(n_coeffs, dtype=float) - 2.0

    dynamics.set_resistance(etaP_coefficients=etaP_coeffs, etaH_coefficients=etaH_coeffs, time=5.0)

    dataset = dynamics.input_timeseries.datasets["conductance"]
    np.testing.assert_allclose(dataset["SH_etaP"].isel(time=0).values, etaP_coeffs)
    np.testing.assert_allclose(dataset["SH_etaH"].isel(time=0).values, etaH_coeffs)
    np.testing.assert_allclose(dataset.time.values, [5.0])


def test_coefficient_inputs_reject_projection_coordinates(tmp_path):
    """Direct coefficients should not specify sample geometry."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["jr"].index_length
    jr_coeffs = np.arange(n_coeffs, dtype=float)

    with np.testing.assert_raises_regex(ValueError, "lat"):
        dynamics.set_jr(jr_coefficients=jr_coeffs, lat=np.zeros(n_coeffs), time=0.0)


def test_tangential_coefficient_inputs_must_be_complete(tmp_path):
    """Helmholtz coefficients require both cf and df components."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)

    with np.testing.assert_raises_regex(ValueError, "u_df"):
        dynamics.set_neutral_wind(u_cf=cf_coeffs, time=0.0)


def test_tangential_inputs_reject_mixed_samples_and_coefficients(tmp_path):
    """Tangential setters should not mix samples with coefficients."""
    dynamics = _small_dynamics(tmp_path)
    n_coeffs = dynamics.input_field_spaces["Q_eff"].index_length
    values = np.zeros(dynamics.state.geometry.grid.size)
    coeffs = np.zeros(n_coeffs)

    with np.testing.assert_raises_regex(ValueError, "sample values"):
        dynamics.set_Q_eff(
            Q_eff_theta=values, Q_eff_phi=values, Q_eff_cf=coeffs, Q_eff_df=coeffs, time=0.0
        )


def test_set_resistance_can_store_native_cs_grid_values(tmp_path):
    """CS conductance basis stores native grid values."""
    dynamics = _small_dynamics(tmp_path, conductance_projection_basis="CS")
    grid = dynamics.state.geometry.grid
    etaP = np.linspace(0.1, 0.3, grid.size)
    etaH = np.linspace(-0.2, 0.2, grid.size)

    dynamics.set_resistance(etaP, etaH, lat=grid.lat, lon=grid.lon, time=6.0)

    dataset = dynamics.input_timeseries.datasets["conductance"]
    np.testing.assert_allclose(dataset["CS_etaP"].isel(time=0).values, etaP)
    np.testing.assert_allclose(dataset["CS_etaH"].isel(time=0).values, etaH)
    np.testing.assert_allclose(dataset.time.values, [6.0])

    dynamics.state.update(dynamics.input_timeseries, time=6.0)
    np.testing.assert_allclose(dynamics.state.etaP.array, etaP)
    np.testing.assert_allclose(dynamics.state.etaH.array, etaH)
    np.testing.assert_allclose(
        dynamics.state._resistance_synthesis_operator().to_matrix(backend="numpy"),
        np.eye(grid.size),
        atol=1e-12,
    )


def test_set_resistance_cs_basis_remaps_non_model_grid(tmp_path):
    """CS conductance basis can remap values from another grid."""
    dynamics = _small_dynamics(tmp_path, conductance_projection_basis="CS")
    grid = dynamics.state.geometry.grid
    etaP = np.ones(grid.size)
    etaH = np.zeros(grid.size)

    dynamics.set_resistance(
        etaP, etaH, lat=grid.lat + np.linspace(0.0, 1e-3, grid.size), lon=grid.lon, time=6.0
    )

    dataset = dynamics.input_timeseries.datasets["conductance"]
    assert "CS_etaP" in dataset
    assert "CS_etaH" in dataset
    assert np.all(np.isfinite(dataset["CS_etaP"].isel(time=0).values))
    assert np.all(np.isfinite(dataset["CS_etaH"].isel(time=0).values))


def test_set_resistance_cs_basis_rejects_least_squares_options(tmp_path):
    """CS conductance storage rejects least-squares controls."""
    dynamics = _small_dynamics(tmp_path, conductance_projection_basis="CS")
    grid = dynamics.state.geometry.grid
    etaP = np.ones(grid.size)
    etaH = np.zeros(grid.size)

    with np.testing.assert_raises_regex(ValueError, "reg_lambda"):
        dynamics.set_resistance(etaP, etaH, lat=grid.lat, lon=grid.lon, reg_lambda=1e-3)


def test_set_conductance_delegates_resistance_conversion(tmp_path, monkeypatch):
    """Conductance inputs are converted once before delegation."""
    dynamics = _small_dynamics(tmp_path)
    hall = np.array([[3, 4]])
    pedersen = np.array([[4, 3]])
    recorded = {}

    def record_set_resistance(etaP, etaH, **kwargs):
        recorded["etaP"] = etaP
        recorded["etaH"] = etaH
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
    np.testing.assert_allclose(recorded["etaP"], pedersen / denominator)
    np.testing.assert_allclose(recorded["etaH"], hall / denominator)
    assert recorded["etaP"].dtype.kind == "f"
    assert recorded["etaH"].dtype.kind == "f"
    assert recorded["kwargs"]["time"] == 7.0
    assert recorded["kwargs"]["reg_lambda"] == 1e-3
    assert recorded["kwargs"]["pinv_rtol"] == 1e-10
