"""Tests for direct input-basis coefficient setters."""

import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.fields import FieldCoefficients
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_to_log_coordinates,
    resistance_to_log_conductance_coordinates,
)
from pynamit.simulation.simulation import Simulation


def _small_simulation(tmp_path, **kwargs):
    return Simulation(
        simulation_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
        **kwargs,
    )


def test_simulation_reuses_projection_transforms_for_shared_representations(tmp_path):
    """Input transforms are shared by representation and grid."""
    simulation = _small_simulation(tmp_path)
    projector = simulation._input_projector

    assert projector._projection_transforms == {}
    transforms = {
        key: projector.projection_transform(key)
        for key in ("boundary_jr", "boundary_Br", "u", "Q_eff", "E_neutral_wind", "conductance")
    }

    assert transforms["boundary_jr"] is transforms["boundary_Br"]
    assert transforms["boundary_jr"] is transforms["u"]
    assert transforms["boundary_jr"] is transforms["Q_eff"]
    assert transforms["boundary_jr"] is transforms["E_neutral_wind"]
    assert transforms["conductance"] is not transforms["boundary_jr"]
    assert transforms["boundary_jr"].grid is simulation.geometry.model_grid


def test_set_boundary_jr_accepts_input_basis_coefficients(tmp_path):
    """Radial current coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["boundary_jr"].index_length
    boundary_jr_coeffs = np.arange(n_coeffs, dtype=float) + 0.25

    simulation.set_boundary_jr(boundary_jr_coefficients=boundary_jr_coeffs, time=4.0)

    dataset = simulation.data.input_series.datasets["boundary_jr"]
    np.testing.assert_allclose(dataset["SH_boundary_jr"].isel(time=0).values, boundary_jr_coeffs)
    np.testing.assert_allclose(dataset.time.values, [4.0])
    assert simulation._input_projector._projection_transforms == {}


def test_set_boundary_Br_accepts_input_basis_coefficients(tmp_path):
    """Magnetospheric Br coefficients are stored directly."""
    simulation = _small_simulation(tmp_path, RM=4 * EARTH_RADIUS_M)
    n_coeffs = simulation.data.schema.input_field_spaces["boundary_Br"].index_length
    br_coeffs = np.linspace(-1.0, 1.0, n_coeffs)

    simulation.set_boundary_Br(boundary_Br_coefficients=br_coeffs, time=2.0)

    dataset = simulation.data.input_series.datasets["boundary_Br"]
    np.testing.assert_allclose(dataset["SH_boundary_Br"].isel(time=0).values, br_coeffs)
    np.testing.assert_allclose(dataset.time.values, [2.0])


def test_set_neutral_wind_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Wind Helmholtz coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    u_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.set_neutral_wind(u_coefficients=u_coefficients, time=3.0)

    dataset = simulation.data.input_series.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_set_neutral_wind_accepts_time_dependent_coefficient_arrays(tmp_path):
    """The leading coefficient-array axis corresponds to time."""
    simulation = _small_simulation(tmp_path)
    coefficient_shape = simulation.data.schema.input_field_spaces["u"].coefficient_shape
    u_coefficients = np.arange(2 * np.prod(coefficient_shape), dtype=float).reshape(
        (2, *coefficient_shape)
    )

    simulation.set_neutral_wind(u_coefficients=u_coefficients, time=[3.0, 4.0])

    dataset = simulation.data.input_series.datasets["u"]
    np.testing.assert_allclose(dataset["SH_u"].values, u_coefficients.reshape(2, -1))
    np.testing.assert_allclose(dataset.time.values, [3.0, 4.0])


def test_input_activation_uses_field_coefficients_for_wind(tmp_path):
    """Response input storage does not need grid expansion."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    u_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.set_neutral_wind(u_coefficients=u_coefficients, time=3.0)
    simulation.response.activate_inputs_at_time(simulation.data.input_series, time=3.0)

    assert isinstance(simulation.response.u, FieldCoefficients)
    np.testing.assert_allclose(simulation.response.u.array, np.vstack([cf_coeffs, df_coeffs]))


def test_nonwind_response_keeps_wind_operator_lazy(tmp_path):
    """A zero wind contribution should not build the wind operator."""
    simulation = _small_simulation(tmp_path)
    conductance_shape = simulation.data.schema.input_field_spaces["conductance"].coefficient_shape
    current_shape = simulation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(conductance_shape),
        log_ratio_coefficients=np.zeros(conductance_shape),
        time=0.0,
    )
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(current_shape), time=0.0)
    simulation.response.activate_inputs_at_time(simulation.data.input_series, time=0.0)

    assert simulation.response._u_coeffs_to_E_coeffs_operator_cache is None
    simulation.response.solve_noninductive_response()
    assert simulation.response._u_coeffs_to_E_coeffs_operator_cache is None


def test_set_Q_eff_accepts_helmholtz_input_basis_coefficients(tmp_path):
    """Q_eff Helmholtz coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    Q_eff_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.set_Q_eff(Q_eff_coefficients=Q_eff_coefficients, time=3.0)

    dataset = simulation.data.input_series.datasets["Q_eff"]
    np.testing.assert_allclose(
        dataset["SH_Q_eff"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_evaluate_Q_eff_uses_canonical_input_series_owner(tmp_path):
    """Q_eff reads conductance through SimulationData."""
    simulation = _small_simulation(tmp_path)
    conductance_shape = simulation.data.schema.input_field_spaces["conductance"].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(conductance_shape),
        log_ratio_coefficients=np.zeros(conductance_shape),
        time=0.0,
    )
    grid = simulation.geometry.model_grid
    zeros = np.zeros(grid.size)

    q_theta, q_phi, q_lat, q_lon = simulation.evaluate_Q_eff_from_neutral_wind(
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
    n_coeffs = simulation.data.schema.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    Q_eff_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.set_Q_eff(Q_eff_coefficients=Q_eff_coefficients, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.set_neutral_wind(u_coefficients=np.stack((cf_coeffs, df_coeffs)), time=1.0)


def test_set_Q_eff_rejects_existing_neutral_wind_input(tmp_path):
    """Q_eff cannot be added after direct wind input."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    u_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.set_neutral_wind(u_coefficients=u_coefficients, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.set_Q_eff(Q_eff_coefficients=np.stack((cf_coeffs, df_coeffs)), time=1.0)


def test_reopening_rejects_conflicting_stored_wind_forcing(tmp_path):
    """Stored input validation catches conflicts before evolution."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["Q_eff"].index_length
    zeros = np.zeros((2, n_coeffs))
    simulation.set_Q_eff(Q_eff_coefficients=zeros, time=0.0)

    simulation.data.input_series.add_entry("u", {"u": np.zeros((2, n_coeffs))}, time=0.0)
    simulation.data.input_series.save("u", simulation.data.artifact_store)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        _small_simulation(tmp_path)


def test_E_neutral_wind_rejects_existing_neutral_wind_input(tmp_path):
    """Equivalent neutral-wind E cannot double-count direct wind."""
    simulation = _small_simulation(tmp_path)
    vector_length = simulation.data.schema.input_field_spaces["u"].index_length
    wind_coefficients = np.stack(
        (np.linspace(0.0, 1.0, vector_length), np.linspace(1.0, 0.0, vector_length))
    )
    simulation.set_neutral_wind(u_coefficients=wind_coefficients, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.set_E_neutral_wind(E_neutral_wind_coefficients=-wind_coefficients, time=1.0)


def test_input_activation_uses_field_coefficients_for_Q_eff(tmp_path):
    """Active Q_eff keeps its canonical coefficient shape."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["Q_eff"].index_length
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    Q_eff_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.set_Q_eff(Q_eff_coefficients=Q_eff_coefficients, time=3.0)
    simulation.response.activate_inputs_at_time(simulation.data.input_series, time=3.0)

    assert isinstance(simulation.response.Q_eff, FieldCoefficients)
    np.testing.assert_allclose(simulation.response.Q_eff.array, np.vstack([cf_coeffs, df_coeffs]))


def test_set_conductance_accepts_canonical_input_basis_coefficients(tmp_path):
    """Store dimensionless magnitude/ratio coefficients directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["conductance"].index_length
    log_magnitude_coeffs = np.arange(n_coeffs, dtype=float) + 1.0
    log_ratio_coeffs = np.arange(n_coeffs, dtype=float) - 2.0

    simulation.set_conductance(
        log_magnitude_coefficients=log_magnitude_coeffs,
        log_ratio_coefficients=log_ratio_coeffs,
        time=5.0,
    )

    dataset = simulation.data.input_series.datasets["conductance"]
    np.testing.assert_allclose(
        dataset["SH_log_conductance_magnitude"].isel(time=0).values, log_magnitude_coeffs
    )
    np.testing.assert_allclose(
        dataset["SH_log_hall_to_pedersen_ratio"].isel(time=0).values, log_ratio_coeffs
    )
    np.testing.assert_allclose(dataset.time.values, [5.0])


def test_coefficient_inputs_reject_projection_coordinates(tmp_path):
    """Direct coefficients should not specify sample geometry."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["boundary_jr"].index_length
    boundary_jr_coeffs = np.arange(n_coeffs, dtype=float)

    with np.testing.assert_raises_regex(ValueError, "lat"):
        simulation.set_boundary_jr(
            boundary_jr_coefficients=boundary_jr_coeffs, lat=np.zeros(n_coeffs), time=0.0
        )


def test_tangential_coefficient_inputs_require_canonical_component_shape(tmp_path):
    """Helmholtz coefficients keep their component axis explicit."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["u"].index_length
    curl_free_coefficients = np.arange(n_coeffs, dtype=float)

    with np.testing.assert_raises_regex(ValueError, "must have shape"):
        simulation.set_neutral_wind(u_coefficients=curl_free_coefficients, time=0.0)


def test_tangential_inputs_reject_mixed_samples_and_coefficients(tmp_path):
    """Tangential setters should not mix samples with coefficients."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.data.schema.input_field_spaces["Q_eff"].index_length
    values = np.zeros(simulation.geometry.model_grid.size)
    coefficients = np.zeros((2, n_coeffs))

    with np.testing.assert_raises_regex(ValueError, "sample values"):
        simulation.set_Q_eff(
            Q_eff_theta=values, Q_eff_phi=values, Q_eff_coefficients=coefficients, time=0.0
        )


def test_set_conductance_can_store_native_cs_grid_values(tmp_path):
    """CS conductance basis stores native log-coordinate values."""
    simulation = _small_simulation(tmp_path, conductance_projection_basis="CS")
    grid = simulation.geometry.model_grid
    pedersen = np.linspace(1.0, 3.0, grid.size)
    hall = np.linspace(0.5, 2.0, grid.size)
    log_magnitude, log_ratio = conductance_to_log_coordinates(pedersen, hall)

    simulation.set_conductance(pedersen=pedersen, hall=hall, lat=grid.lat, lon=grid.lon, time=6.0)

    dataset = simulation.data.input_series.datasets["conductance"]
    np.testing.assert_allclose(
        dataset["CS_log_conductance_magnitude"].isel(time=0).values, log_magnitude
    )
    np.testing.assert_allclose(
        dataset["CS_log_hall_to_pedersen_ratio"].isel(time=0).values, log_ratio
    )
    np.testing.assert_allclose(dataset.time.values, [6.0])

    simulation.response.activate_inputs_at_time(simulation.data.input_series, time=6.0)
    np.testing.assert_allclose(simulation.response.log_conductance_magnitude.array, log_magnitude)
    np.testing.assert_allclose(simulation.response.log_hall_to_pedersen_ratio.array, log_ratio)
    conductance_basis = simulation.response.log_conductance_magnitude.field_space.basis
    np.testing.assert_allclose(
        conductance_basis.scalar_evaluation_operator(grid).to_matrix(backend="numpy"),
        np.eye(grid.size),
        atol=1e-12,
    )


def test_set_conductance_rejects_mixed_samples_and_coefficients(tmp_path):
    """Do not silently ignore samples when coefficients are supplied."""
    simulation = _small_simulation(tmp_path)
    field_space = simulation.data.schema.input_field_spaces["conductance"]
    coefficients = np.zeros(field_space.coefficient_shape)

    with np.testing.assert_raises_regex(ValueError, "cannot be combined with sample values"):
        simulation.set_conductance(
            pedersen=np.ones(simulation.geometry.model_grid.size),
            hall=np.ones(simulation.geometry.model_grid.size),
            log_magnitude_coefficients=coefficients,
            log_ratio_coefficients=coefficients,
        )


def test_identical_conductance_history_retains_closure_caches(tmp_path):
    """Repeated coefficient values do not rebuild the same closure."""
    simulation = _small_simulation(tmp_path)
    field_space = simulation.data.schema.input_field_spaces["conductance"]
    log_magnitude = np.zeros((2, *field_space.coefficient_shape))
    log_ratio = np.zeros_like(log_magnitude)
    simulation.set_conductance(
        log_magnitude_coefficients=log_magnitude, log_ratio_coefficients=log_ratio, time=[0.0, 1.0]
    )

    response = simulation.response
    response.activate_inputs_at_time(simulation.data.input_series, time=0.0)
    sentinel = object()
    response._induced_poloidal_potential_feedback_operator = sentinel
    first_fingerprint = response.conductance_fingerprint
    response.activate_inputs_at_time(simulation.data.input_series, time=1.0)

    assert response.conductance_fingerprint == first_fingerprint
    assert response._induced_poloidal_potential_feedback_operator is sentinel


def test_conductance_activation_fingerprints_stored_coefficients(tmp_path, monkeypatch):
    """Fingerprint stored conductance before backend transfer."""
    simulation = _small_simulation(tmp_path)
    field_space = simulation.data.schema.input_field_spaces["conductance"]
    coefficients = np.zeros(field_space.coefficient_shape)
    simulation.set_conductance(
        log_magnitude_coefficients=coefficients, log_ratio_coefficients=coefficients, time=0.0
    )

    def fail_on_backend_copy(_array):
        raise AssertionError("conductance activation copied an active backend array")

    monkeypatch.setattr("pynamit.simulation.response.to_numpy", fail_on_backend_copy)
    simulation.response.activate_inputs_at_time(simulation.data.input_series, time=0.0)

    assert simulation.response.conductance_fingerprint


def test_set_conductance_cs_basis_remaps_non_model_grid(tmp_path):
    """CS conductance basis can remap values from another grid."""
    simulation = _small_simulation(tmp_path, conductance_projection_basis="CS")
    grid = simulation.geometry.model_grid
    pedersen = np.ones(grid.size)
    hall = np.full(grid.size, 0.5)

    simulation.set_conductance(
        pedersen=pedersen,
        hall=hall,
        lat=grid.lat + np.linspace(0.0, 1e-3, grid.size),
        lon=grid.lon,
        time=6.0,
    )

    dataset = simulation.data.input_series.datasets["conductance"]
    assert "CS_log_conductance_magnitude" in dataset
    assert "CS_log_hall_to_pedersen_ratio" in dataset
    assert np.all(np.isfinite(dataset["CS_log_conductance_magnitude"].isel(time=0).values))
    assert np.all(np.isfinite(dataset["CS_log_hall_to_pedersen_ratio"].isel(time=0).values))


def test_set_conductance_cs_basis_rejects_least_squares_options(tmp_path):
    """CS conductance storage rejects least-squares controls."""
    simulation = _small_simulation(tmp_path, conductance_projection_basis="CS")
    grid = simulation.geometry.model_grid
    pedersen = np.ones(grid.size)
    hall = np.full(grid.size, 0.5)

    with np.testing.assert_raises_regex(ValueError, "reg_lambda"):
        simulation.set_conductance(
            pedersen=pedersen, hall=hall, lat=grid.lat, lon=grid.lon, reg_lambda=1e-3
        )


def test_set_conductance_projects_dimensionless_log_coordinates(tmp_path, monkeypatch):
    """Store conductance samples in canonical coordinates."""
    simulation = _small_simulation(tmp_path)
    hall = np.array([[3, 4]])
    pedersen = np.array([[4, 3]])
    recorded = {}

    def record_set_scalar_input(key, **kwargs):
        recorded["key"] = key
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(simulation._input_projector, "set_scalar_input", record_set_scalar_input)

    simulation.set_conductance(
        pedersen=pedersen,
        hall=hall,
        lat=np.array([60.0, 61.0]),
        lon=np.array([10.0, 11.0]),
        time=7.0,
        sqrt_weights=np.ones(2),
        reg_lambda=1e-3,
        pinv_rtol=1e-10,
    )

    expected_magnitude, expected_ratio = conductance_to_log_coordinates(pedersen, hall)
    assert recorded["key"] == "conductance"
    np.testing.assert_allclose(
        recorded["kwargs"]["samples"]["log_conductance_magnitude"], expected_magnitude
    )
    np.testing.assert_allclose(
        recorded["kwargs"]["samples"]["log_hall_to_pedersen_ratio"], expected_ratio
    )
    assert recorded["kwargs"]["time"] == 7.0
    assert recorded["kwargs"]["reg_lambda"] == 1e-3
    assert recorded["kwargs"]["pinv_rtol"] == 1e-10


def test_set_resistance_projects_direct_log_conductance_coordinates(tmp_path, monkeypatch):
    """Map resistance samples directly onto canonical coordinates."""
    simulation = _small_simulation(tmp_path)
    eta_p = np.array([[0.4, 0.2]])
    eta_h = np.array([[0.3, 0.1]])
    recorded = {}

    def record_set_scalar_input(key, **kwargs):
        recorded["key"] = key
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(simulation._input_projector, "set_scalar_input", record_set_scalar_input)

    simulation.set_resistance(
        etaP=eta_p,
        etaH=eta_h,
        lat=np.array([60.0, 61.0]),
        lon=np.array([10.0, 11.0]),
        time=7.0,
        sqrt_weights=np.ones(2),
        reg_lambda=1e-3,
        pinv_rtol=1e-10,
    )

    expected_magnitude, expected_ratio = resistance_to_log_conductance_coordinates(eta_p, eta_h)
    assert recorded["key"] == "conductance"
    np.testing.assert_allclose(
        recorded["kwargs"]["samples"]["log_conductance_magnitude"], expected_magnitude
    )
    np.testing.assert_allclose(
        recorded["kwargs"]["samples"]["log_hall_to_pedersen_ratio"], expected_ratio
    )
    assert recorded["kwargs"]["sample_label"] == "resistance samples"
    assert recorded["kwargs"]["time"] == 7.0
    assert recorded["kwargs"]["reg_lambda"] == 1e-3
    assert recorded["kwargs"]["pinv_rtol"] == 1e-10
