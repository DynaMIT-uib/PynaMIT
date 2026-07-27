"""Tests for direct input-basis coefficient setters."""

import numpy as np

from pynamit.fields import FieldCoefficients
from pynamit.math.constants import RE
from pynamit.simulation.api import Simulation


def _small_simulation(tmp_path, **kwargs):
    return Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
        **kwargs,
    )


def test_simulation_reuses_input_transforms_for_shared_representations(tmp_path):
    """Input transforms are shared by representation and grid."""
    simulation = _small_simulation(tmp_path)
    pipeline = simulation._input_pipeline

    assert pipeline.projection_transforms == {}
    transforms = {
        key: pipeline.projection_transform_for(key)
        for key in ("jr", "Br", "u", "Q_eff", "E_neutral_wind", "resistance")
    }

    assert transforms["jr"] is transforms["Br"]
    assert transforms["jr"] is transforms["u"]
    assert transforms["jr"] is transforms["Q_eff"]
    assert transforms["jr"] is transforms["E_neutral_wind"]
    assert transforms["resistance"] is not transforms["jr"]
    assert transforms["jr"].grid is simulation.geometry.model_grid


def test_set_jr_accepts_input_basis_coefficients(tmp_path):
    """Radial current coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["jr"].index_length
    jr_coeffs = np.arange(n_coeffs, dtype=float) + 0.25

    simulation.set_jr(jr_coefficients=jr_coeffs, time=4.0)

    dataset = simulation.run_data.input_series.datasets["jr"]
    np.testing.assert_allclose(dataset["SH_jr"].isel(time=0).values, jr_coeffs)
    np.testing.assert_allclose(dataset.time.values, [4.0])
    assert simulation._input_pipeline.projection_transforms == {}


def test_set_Br_accepts_input_basis_coefficients(tmp_path):
    """Magnetospheric Br coefficients are stored directly."""
    simulation = _small_simulation(tmp_path, RM=4 * RE)
    n_coeffs = simulation.run_data.schema.input_field_spaces["Br"].index_length
    br_coeffs = np.linspace(-1.0, 1.0, n_coeffs)

    simulation.set_Br(Br_coefficients=br_coeffs, time=2.0)

    dataset = simulation.run_data.input_series.datasets["Br"]
    np.testing.assert_allclose(dataset["SH_Br"].isel(time=0).values, br_coeffs)
    np.testing.assert_allclose(dataset.time.values, [2.0])


def test_set_neutral_wind_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Wind Helmholtz coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    simulation.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=3.0)

    dataset = simulation.run_data.input_series.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_set_u_uses_neutral_wind_api(tmp_path):
    """Historical set_u delegates to set_neutral_wind."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    simulation.set_u(u_cf=cf_coeffs, u_df=df_coeffs, time=3.0)

    dataset = simulation.run_data.input_series.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )


def test_state_update_uses_field_coefficients_for_wind(tmp_path):
    """Response input storage does not need grid expansion."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    simulation.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=3.0)
    simulation.response.activate_inputs_at_time(simulation.run_data.input_series, time=3.0)

    assert isinstance(simulation.response.u, FieldCoefficients)
    np.testing.assert_allclose(simulation.response.u.array, np.vstack([cf_coeffs, df_coeffs]))


def test_nonwind_response_keeps_wind_operator_lazy(tmp_path):
    """A zero wind contribution should not build the wind operator."""
    simulation = _small_simulation(tmp_path)
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "resistance"
    ].coefficient_shape
    current_shape = simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape
    simulation.set_resistance(
        etaP_coefficients=np.ones(resistance_shape),
        etaH_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )
    simulation.set_jr(jr_coefficients=np.zeros(current_shape), time=0.0)
    simulation.response.activate_inputs_at_time(simulation.run_data.input_series, time=0.0)

    assert simulation.response._u_coeffs_to_E_coeffs_cache is None
    simulation.response.calculate_noninductive_response()
    assert simulation.response._u_coeffs_to_E_coeffs_cache is None


def test_set_Q_eff_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Q_eff Helmholtz coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    simulation.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=3.0)

    dataset = simulation.run_data.input_series.datasets["Q_eff"]
    np.testing.assert_allclose(
        dataset["SH_Q_eff"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_calculate_Q_eff_uses_canonical_input_series_owner(tmp_path):
    """Wind-equivalent Q_eff reads resistance through RunData."""
    simulation = _small_simulation(tmp_path)
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "resistance"
    ].coefficient_shape
    simulation.set_resistance(
        etaP_coefficients=np.ones(resistance_shape),
        etaH_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )
    grid = simulation.geometry.model_grid
    zeros = np.zeros(grid.size)

    q_theta, q_phi, q_lat, q_lon = simulation.calculate_Q_eff_from_neutral_wind(
        zeros, zeros, lat=grid.lat, lon=grid.lon, time=0.0
    )

    assert q_theta.shape == q_phi.shape == (1, grid.size)
    np.testing.assert_allclose(q_theta, 0.0, atol=1e-18)
    np.testing.assert_allclose(q_phi, 0.0, atol=1e-18)
    np.testing.assert_array_equal(q_lat, grid.lat)
    np.testing.assert_array_equal(q_lon, grid.lon)


def test_set_neutral_wind_rejects_existing_Q_eff_input(tmp_path):
    """Direct wind and Q_eff are mutually exclusive."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    simulation.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=1.0)


def test_set_Q_eff_rejects_existing_neutral_wind_input(tmp_path):
    """Q_eff cannot be added after direct wind input."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    simulation.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=1.0)


def test_state_calculation_rejects_simultaneous_wind_and_Q_eff(tmp_path):
    """Response calculation should not double-count wind forcing."""
    simulation = _small_simulation(tmp_path)
    simulation.response.u = object()
    simulation.response.Q_eff = object()

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.response.calculate_noninductive_response()


def test_E_neutral_wind_rejects_existing_neutral_wind_input(tmp_path):
    """Equivalent neutral-wind E cannot double-count direct wind."""
    simulation = _small_simulation(tmp_path)
    vector_length = simulation.run_data.schema.input_field_spaces["u"].index_length
    wind_cf = np.linspace(0.0, 1.0, vector_length)
    wind_df = np.linspace(1.0, 0.0, vector_length)
    simulation.set_neutral_wind(u_cf=wind_cf, u_df=wind_df, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.set_E_neutral_wind(
            E_neutral_wind_cf=-wind_cf, E_neutral_wind_df=-wind_df, time=1.0
        )


def test_state_calculation_rejects_simultaneous_wind_and_neutral_wind_E(tmp_path):
    """Response should reject alternate wind representations."""
    simulation = _small_simulation(tmp_path)
    simulation.response.u = object()
    simulation.response.E_neutral_wind = object()

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.response.calculate_noninductive_response()


def test_state_update_uses_field_coefficients_for_Q_eff(tmp_path):
    """Q_eff state storage keeps canonical coefficient shape."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    simulation.set_Q_eff(Q_eff_cf=cf_coeffs, Q_eff_df=df_coeffs, time=3.0)
    simulation.response.activate_inputs_at_time(simulation.run_data.input_series, time=3.0)

    assert isinstance(simulation.response.Q_eff, FieldCoefficients)
    np.testing.assert_allclose(simulation.response.Q_eff.array, np.vstack([cf_coeffs, df_coeffs]))


def test_set_resistance_accepts_input_basis_coefficients(tmp_path):
    """EtaP and etaH resistance coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["resistance"].index_length
    etaP_coeffs = np.arange(n_coeffs, dtype=float) + 1.0
    etaH_coeffs = np.arange(n_coeffs, dtype=float) - 2.0

    simulation.set_resistance(
        etaP_coefficients=etaP_coeffs, etaH_coefficients=etaH_coeffs, time=5.0
    )

    dataset = simulation.run_data.input_series.datasets["resistance"]
    np.testing.assert_allclose(dataset["SH_etaP"].isel(time=0).values, etaP_coeffs)
    np.testing.assert_allclose(dataset["SH_etaH"].isel(time=0).values, etaH_coeffs)
    np.testing.assert_allclose(dataset.time.values, [5.0])


def test_coefficient_inputs_reject_projection_coordinates(tmp_path):
    """Direct coefficients should not specify sample geometry."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["jr"].index_length
    jr_coeffs = np.arange(n_coeffs, dtype=float)

    with np.testing.assert_raises_regex(ValueError, "lat"):
        simulation.set_jr(jr_coefficients=jr_coeffs, lat=np.zeros(n_coeffs), time=0.0)


def test_tangential_coefficient_inputs_must_be_complete(tmp_path):
    """Helmholtz coefficients require both cf and df components."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)

    with np.testing.assert_raises_regex(ValueError, "u_df"):
        simulation.set_neutral_wind(u_cf=cf_coeffs, time=0.0)


def test_tangential_inputs_reject_mixed_samples_and_coefficients(tmp_path):
    """Tangential setters should not mix samples with coefficients."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.run_data.schema.input_field_spaces["Q_eff"].index_length
    values = np.zeros(simulation.geometry.model_grid.size)
    coeffs = np.zeros(n_coeffs)

    with np.testing.assert_raises_regex(ValueError, "sample values"):
        simulation.set_Q_eff(
            Q_eff_theta=values, Q_eff_phi=values, Q_eff_cf=coeffs, Q_eff_df=coeffs, time=0.0
        )


def test_set_resistance_can_store_native_cs_grid_values(tmp_path):
    """CS resistance basis stores native grid values."""
    simulation = _small_simulation(tmp_path, resistance_projection_basis="CS")
    grid = simulation.geometry.model_grid
    etaP = np.linspace(0.1, 0.3, grid.size)
    etaH = np.linspace(-0.2, 0.2, grid.size)

    simulation.set_resistance(etaP, etaH, lat=grid.lat, lon=grid.lon, time=6.0)

    dataset = simulation.run_data.input_series.datasets["resistance"]
    np.testing.assert_allclose(dataset["CS_etaP"].isel(time=0).values, etaP)
    np.testing.assert_allclose(dataset["CS_etaH"].isel(time=0).values, etaH)
    np.testing.assert_allclose(dataset.time.values, [6.0])

    simulation.response.activate_inputs_at_time(simulation.run_data.input_series, time=6.0)
    np.testing.assert_allclose(simulation.response.etaP.array, etaP)
    np.testing.assert_allclose(simulation.response.etaH.array, etaH)
    np.testing.assert_allclose(
        simulation.response._resistance_synthesis_operator().to_matrix(backend="numpy"),
        np.eye(grid.size),
        atol=1e-12,
    )


def test_identical_resistance_history_retains_closure_caches(tmp_path):
    """Repeated coefficient values do not rebuild the same closure."""
    simulation = _small_simulation(tmp_path)
    field_space = simulation.run_data.schema.input_field_spaces["resistance"]
    eta_p = np.ones((2, *field_space.coefficient_shape))
    eta_h = np.zeros_like(eta_p)
    simulation.set_resistance(etaP_coefficients=eta_p, etaH_coefficients=eta_h, time=[0.0, 1.0])

    response = simulation.response
    response.activate_inputs_at_time(simulation.run_data.input_series, time=0.0)
    sentinel = object()
    response._m_ind_feedback_matrix = sentinel
    first_fingerprint = response.resistance_fingerprint
    response.activate_inputs_at_time(simulation.run_data.input_series, time=1.0)

    assert response.resistance_fingerprint == first_fingerprint
    assert response._m_ind_feedback_matrix is sentinel


def test_set_resistance_cs_basis_remaps_non_model_grid(tmp_path):
    """CS resistance basis can remap values from another grid."""
    simulation = _small_simulation(tmp_path, resistance_projection_basis="CS")
    grid = simulation.geometry.model_grid
    etaP = np.ones(grid.size)
    etaH = np.zeros(grid.size)

    simulation.set_resistance(
        etaP, etaH, lat=grid.lat + np.linspace(0.0, 1e-3, grid.size), lon=grid.lon, time=6.0
    )

    dataset = simulation.run_data.input_series.datasets["resistance"]
    assert "CS_etaP" in dataset
    assert "CS_etaH" in dataset
    assert np.all(np.isfinite(dataset["CS_etaP"].isel(time=0).values))
    assert np.all(np.isfinite(dataset["CS_etaH"].isel(time=0).values))


def test_set_resistance_cs_basis_rejects_least_squares_options(tmp_path):
    """CS resistance storage rejects least-squares controls."""
    simulation = _small_simulation(tmp_path, resistance_projection_basis="CS")
    grid = simulation.geometry.model_grid
    etaP = np.ones(grid.size)
    etaH = np.zeros(grid.size)

    with np.testing.assert_raises_regex(ValueError, "reg_lambda"):
        simulation.set_resistance(etaP, etaH, lat=grid.lat, lon=grid.lon, reg_lambda=1e-3)


def test_set_conductance_delegates_resistance_conversion(tmp_path, monkeypatch):
    """Conductance inputs are converted once before delegation."""
    simulation = _small_simulation(tmp_path)
    hall = np.array([[3, 4]])
    pedersen = np.array([[4, 3]])
    recorded = {}

    def record_set_resistance(etaP, etaH, **kwargs):
        recorded["etaP"] = etaP
        recorded["etaH"] = etaH
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(simulation, "set_resistance", record_set_resistance)

    simulation.set_conductance(
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
