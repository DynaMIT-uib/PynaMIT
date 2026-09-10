"""Tests for direct input-basis coefficient setters."""

import numpy as np
import pytest
from kompe import SphericalGrid
from kompe.constants import EARTH_RADIUS_M
from kompe.math import LeastSquaresSolver

from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_to_log_coordinates,
    resistance_to_log_conductance_coordinates,
)
from pynamit.simulation.response import ElectrodynamicResponse
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
    projector = simulation.inputs._input_projector

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
    assert transforms["boundary_jr"] is simulation.geometry.horizontal_transform


def test_boundary_jr_coefficients_are_stored_directly(tmp_path):
    """Radial current coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["boundary_jr"].coefficient_count
    boundary_jr_coeffs = np.arange(n_coeffs, dtype=float) + 0.25

    simulation.inputs.set_coefficients("boundary_jr", boundary_jr_coeffs, time=4.0)

    dataset = simulation.results.input_series.datasets["boundary_jr"]
    np.testing.assert_allclose(dataset["SH_boundary_jr"].isel(time=0).values, boundary_jr_coeffs)
    np.testing.assert_allclose(dataset.time.values, [4.0])
    assert "surface_to_poloidal_operator" not in simulation.geometry.__dict__


def test_area_weighted_model_grid_inputs_reuse_the_known_measure(tmp_path):
    """The supplied model grid retains its known cell areas."""
    simulation = _small_simulation(tmp_path, area_weighted_least_squares=True)
    grid = simulation.geometry.model_grid
    transform = simulation.geometry.horizontal_transform
    values = np.random.default_rng(92).normal(size=grid.size) * 1e-6
    expected = transform.analyze_scalar(values)

    simulation.inputs.set_boundary_jr(values, time=0.0, grid=grid)

    stored = simulation.results.input_series.datasets["boundary_jr"]["SH_boundary_jr"]
    np.testing.assert_allclose(stored.isel(time=0), expected, atol=1e-14)
    assert not transform._analysis_transforms


def test_area_weighted_arbitrary_input_points_require_explicit_weights(tmp_path):
    """Unknown sampling measures fail before storing fitted inputs."""
    simulation = _small_simulation(tmp_path, area_weighted_least_squares=True)
    grid = simulation.geometry.model_grid
    theta, phi = grid.theta + 0.01, grid.phi
    values = np.cos(np.deg2rad(theta)) * 1e-6

    with pytest.raises(ValueError, match="explicit sqrt_weights"):
        simulation.inputs.set_boundary_jr(
            values, time=0.0, grid=SphericalGrid(theta=theta, phi=phi)
        )
    assert "boundary_jr" not in simulation.results.input_series.datasets
    simulation.inputs.set_boundary_jr(
        values, time=0.0, sqrt_weights=np.ones(grid.size), grid=SphericalGrid(theta=theta, phi=phi)
    )
    assert "boundary_jr" in simulation.results.input_series.datasets


def test_FAC_samples_follow_the_main_field_in_both_hemispheres(tmp_path, monkeypatch):
    """Positive FAC is inward in the north and outward in the south."""
    simulation = _small_simulation(tmp_path, main_field_kind="dipole")
    theta = np.array([30.0, 150.0])
    FAC = [[1e-6, 1e-6], [-1e-6, 2e-6]]
    recorded = {}

    def record_radial_current(values, **kwargs):
        recorded.update(values=values, **kwargs)

    monkeypatch.setattr(simulation.inputs, "set_boundary_jr", record_radial_current)
    simulation.inputs.set_FAC(
        FAC, time=[0.0, 2.0], grid=SphericalGrid(theta=theta, phi=[0.0, 0.0])
    )

    cos_theta = np.cos(np.deg2rad(theta))
    unit_br = -2 * cos_theta / np.sqrt(1 + 3 * cos_theta**2)
    np.testing.assert_allclose(recorded["values"], np.asarray(FAC) * unit_br)
    np.testing.assert_array_equal(recorded["time"], [0.0, 2.0])


def test_boundary_Br_coefficients_are_stored_directly(tmp_path):
    """Magnetospheric Br coefficients are stored directly."""
    simulation = _small_simulation(tmp_path, RM=4 * EARTH_RADIUS_M)
    n_coeffs = simulation.results.schema.input_field_spaces["boundary_Br"].coefficient_count
    br_coeffs = np.linspace(-1.0, 1.0, n_coeffs)

    simulation.inputs.set_coefficients("boundary_Br", br_coeffs, time=2.0)

    dataset = simulation.results.input_series.datasets["boundary_Br"]
    np.testing.assert_allclose(dataset["SH_boundary_Br"].isel(time=0).values, br_coeffs)
    np.testing.assert_allclose(dataset.time.values, [2.0])


def test_neutral_wind_coefficients_retain_helmholtz_components(tmp_path):
    """Wind Helmholtz coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["u"].coefficient_count
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    u_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.inputs.set_coefficients("u", u_coefficients, time=3.0)

    dataset = simulation.results.input_series.datasets["u"]
    np.testing.assert_allclose(
        dataset["SH_u"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_neutral_wind_coefficients_retain_time_axis(tmp_path):
    """The leading coefficient-array axis corresponds to time."""
    simulation = _small_simulation(tmp_path)
    coefficient_shape = simulation.results.schema.input_field_spaces["u"].shape
    u_coefficients = np.arange(2 * np.prod(coefficient_shape), dtype=float).reshape(
        (2, *coefficient_shape)
    )

    simulation.inputs.set_coefficients("u", u_coefficients, time=[3.0, 4.0])

    dataset = simulation.results.input_series.datasets["u"]
    np.testing.assert_allclose(dataset["SH_u"].values, u_coefficients.reshape(2, -1))
    np.testing.assert_allclose(dataset.time.values, [3.0, 4.0])


def test_wind_selection_retains_coefficient_shape(tmp_path):
    """Input selection does not need grid expansion."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["u"].coefficient_count
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 1.0

    u_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.inputs.set_coefficients("u", u_coefficients, time=3.0)
    series = simulation.results.input_series
    selected = series.get_entry("u", 3.0)["u"]
    np.testing.assert_allclose(selected, np.vstack([cf_coeffs, df_coeffs]))


def test_nonwind_response_keeps_wind_operator_lazy(tmp_path):
    """A zero wind contribution should not build the wind operator."""
    simulation = _small_simulation(tmp_path)
    conductance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    current_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(conductance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(conductance_shape),
        },
        time=0.0,
    )
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(current_shape), time=0.0)
    assert "u_coeffs_to_E_coeffs_operator" not in simulation.geometry.__dict__
    simulation.response.solve_noninductive_response()
    assert "u_coeffs_to_E_coeffs_operator" not in simulation.geometry.__dict__


def test_Q_eff_coefficients_retain_helmholtz_components(tmp_path):
    """Q_eff Helmholtz coefficients are stored directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["Q_eff"].coefficient_count
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    Q_eff_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.inputs.set_coefficients("Q_eff", Q_eff_coefficients, time=3.0)

    dataset = simulation.results.input_series.datasets["Q_eff"]
    np.testing.assert_allclose(
        dataset["SH_Q_eff"].isel(time=0).values, np.concatenate([cf_coeffs, df_coeffs])
    )
    np.testing.assert_allclose(dataset.time.values, [3.0])


def test_evaluate_Q_eff_uses_canonical_input_series_owner(tmp_path):
    """Q_eff reads the preparation's owned conductance series."""
    simulation = _small_simulation(tmp_path)
    conductance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(conductance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(conductance_shape),
        },
        time=0.0,
    )
    grid = simulation.geometry.model_grid
    zeros = np.zeros(grid.size)

    q_theta, q_phi, q_lat, q_lon = simulation.inputs.evaluate_Q_eff_from_neutral_wind(
        zeros, zeros, time=0.0, grid=grid
    )

    assert q_theta.shape == q_phi.shape == (1, grid.size)
    np.testing.assert_allclose(q_theta, 0.0, atol=1e-18)
    np.testing.assert_allclose(q_phi, 0.0, atol=1e-18)
    np.testing.assert_array_equal(q_lat, grid.lat)
    np.testing.assert_array_equal(q_lon, grid.lon)


@pytest.mark.parametrize("operation", ["evaluate", "fit"])
def test_wind_Q_eff_preparation_preserves_live_response(tmp_path, operation):
    """Preview and fitting leave the live closure state intact."""
    simulation = _small_simulation(tmp_path)
    shape = simulation.results.schema.input_field_spaces["conductance"].shape
    for time, value in ((0.0, 0.0), (10.0, 1.0)):
        simulation.inputs.set_coefficients(
            "conductance",
            {
                "log_conductance_magnitude": np.full(shape, value),
                "log_hall_to_pedersen_ratio": np.zeros(shape),
            },
            time=time,
        )
    response = simulation.response
    coefficients = response.log_conductance_magnitude
    resistance = response.resistance_tensor_on_grid
    wind_shape = simulation.results.schema.input_field_spaces["u"].shape
    wind = np.zeros((1,) + wind_shape)
    projector = simulation.inputs._input_projector
    if operation == "evaluate":
        projector.evaluate_Q_eff_from_neutral_wind([10.0], wind)
    else:
        projector.fit_Q_eff_from_neutral_wind([10.0], wind)
    assert simulation.current_time == 0.0
    assert response.log_conductance_magnitude is coefficients
    assert response.resistance_tensor_on_grid is resistance


def test_set_neutral_wind_rejects_existing_Q_eff_input(tmp_path):
    """Direct wind and Q_eff are mutually exclusive."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["Q_eff"].coefficient_count
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    Q_eff_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.inputs.set_coefficients("Q_eff", Q_eff_coefficients, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.inputs.set_coefficients("u", np.stack((cf_coeffs, df_coeffs)), time=1.0)


def test_set_Q_eff_rejects_existing_neutral_wind_input(tmp_path):
    """Q_eff cannot be added after direct wind input."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["u"].coefficient_count
    cf_coeffs = np.arange(n_coeffs, dtype=float)
    df_coeffs = -np.arange(n_coeffs, dtype=float)
    u_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.inputs.set_coefficients("u", u_coefficients, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.inputs.set_coefficients("Q_eff", np.stack((cf_coeffs, df_coeffs)), time=1.0)


def test_reopening_rejects_conflicting_stored_wind_forcing(tmp_path):
    """Stored input validation catches conflicts before evolution."""
    from pynamit.results import SimulationResults

    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["Q_eff"].coefficient_count
    zeros = np.zeros((2, n_coeffs))
    simulation.inputs.set_coefficients("Q_eff", zeros, time=0.0)

    simulation.results.input_series.add_entry("u", {"u": np.zeros((2, n_coeffs))}, time=0.0)
    simulation.results.input_series.save("u", simulation.results.artifact_store)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        _small_simulation(tmp_path)
    results = SimulationResults.from_directory(simulation.simulation_directory)
    for _ in range(2):
        with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
            results.load_input_series()


def test_E_neutral_wind_rejects_existing_neutral_wind_input(tmp_path):
    """Equivalent neutral-wind E cannot double-count direct wind."""
    simulation = _small_simulation(tmp_path)
    vector_length = simulation.results.schema.input_field_spaces["u"].coefficient_count
    wind_coefficients = np.stack(
        (np.linspace(0.0, 1.0, vector_length), np.linspace(1.0, 0.0, vector_length))
    )
    simulation.inputs.set_coefficients("u", wind_coefficients, time=0.0)

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        simulation.inputs.set_coefficients("E_neutral_wind", -wind_coefficients, time=1.0)


def test_Q_eff_selection_retains_coefficient_shape(tmp_path):
    """Selected Q_eff keeps its canonical coefficient shape."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["Q_eff"].coefficient_count
    cf_coeffs = np.arange(n_coeffs, dtype=float) + 2.0
    df_coeffs = -np.arange(n_coeffs, dtype=float) - 3.0

    Q_eff_coefficients = np.stack((cf_coeffs, df_coeffs))
    simulation.inputs.set_coefficients("Q_eff", Q_eff_coefficients, time=3.0)
    series = simulation.results.input_series
    selected = series.get_entry("Q_eff", 3.0)["Q_eff"]
    np.testing.assert_allclose(selected, np.vstack([cf_coeffs, df_coeffs]))


def test_conductance_coefficients_retain_log_coordinates(tmp_path):
    """Store dimensionless magnitude/ratio coefficients directly."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["conductance"].coefficient_count
    log_magnitude_coeffs = np.arange(n_coeffs, dtype=float) + 1.0
    log_ratio_coeffs = np.arange(n_coeffs, dtype=float) - 2.0

    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": log_magnitude_coeffs,
            "log_hall_to_pedersen_ratio": log_ratio_coeffs,
        },
        time=5.0,
    )

    dataset = simulation.results.input_series.datasets["conductance"]
    np.testing.assert_allclose(
        dataset["SH_log_conductance_magnitude"].isel(time=0).values, log_magnitude_coeffs
    )
    np.testing.assert_allclose(
        dataset["SH_log_hall_to_pedersen_ratio"].isel(time=0).values, log_ratio_coeffs
    )
    np.testing.assert_allclose(dataset.time.values, [5.0])


def test_input_grid_retains_its_own_quadrature_weights(tmp_path):
    """A supplied grid carries the measure of the sampled field."""
    simulation = _small_simulation(tmp_path, area_weighted_least_squares=True)
    model_grid = simulation.model_grid
    weights = np.linspace(0.1, 2.0, model_grid.size)
    grid = SphericalGrid(theta=model_grid.theta, phi=model_grid.phi, area_weights=weights)
    values = np.sin(np.arange(grid.size)) * 1e-6
    simulation.inputs.set_boundary_jr(np.stack((values, 2 * values)), grid=grid, time=[0.0, 1.0])
    basis = simulation.results.schema.input_field_spaces["boundary_jr"].basis
    A = np.asarray(basis.scalar_evaluation_array(grid))
    expected = np.linalg.lstsq(
        np.sqrt(weights)[:, None] * A, np.sqrt(weights) * values, rcond=1e-15
    )[0]
    actual = simulation.results.input_series.get_entry("boundary_jr", 1.0)["boundary_jr"]
    np.testing.assert_allclose(actual, 2 * expected, rtol=1e-10, atol=1e-20)


def test_tangential_coefficient_inputs_require_canonical_component_shape(tmp_path):
    """Helmholtz coefficients keep their component axis explicit."""
    simulation = _small_simulation(tmp_path)
    n_coeffs = simulation.results.schema.input_field_spaces["u"].coefficient_count
    curl_free_coefficients = np.arange(n_coeffs, dtype=float)

    with np.testing.assert_raises_regex(ValueError, "requires shape"):
        simulation.inputs.set_coefficients("u", curl_free_coefficients, time=0.0)


def test_set_conductance_can_store_native_cs_grid_values(tmp_path):
    """CS conductance basis stores native log-coordinate values."""
    simulation = _small_simulation(tmp_path, conductance_basis="CS")
    grid = simulation.geometry.model_grid
    pedersen = np.linspace(1.0, 3.0, grid.size)
    hall = np.linspace(0.5, 2.0, grid.size)
    log_magnitude, log_ratio = conductance_to_log_coordinates(pedersen, hall)

    simulation.inputs.set_conductance(pedersen=pedersen, hall=hall, time=6.0, grid=grid)

    dataset = simulation.results.input_series.datasets["conductance"]
    np.testing.assert_allclose(
        dataset["CS_log_conductance_magnitude"].isel(time=0).values, log_magnitude
    )
    np.testing.assert_allclose(
        dataset["CS_log_hall_to_pedersen_ratio"].isel(time=0).values, log_ratio
    )
    np.testing.assert_allclose(dataset.time.values, [6.0])

    response = simulation.response_at_time(6.0)
    np.testing.assert_allclose(response.log_conductance_magnitude, log_magnitude)
    np.testing.assert_allclose(response.log_hall_to_pedersen_ratio, log_ratio)
    conductance_basis = response._conductance_space.basis
    np.testing.assert_allclose(
        conductance_basis.scalar_evaluation_operator(grid).to_matrix(backend="numpy"),
        np.eye(grid.size),
        atol=1e-12,
    )


def test_identical_conductance_history_retains_closure_caches(tmp_path):
    """Repeated coefficient values do not rebuild the same closure."""
    simulation = _small_simulation(tmp_path)
    field_space = simulation.results.schema.input_field_spaces["conductance"]
    log_magnitude = np.zeros((2, *field_space.shape))
    log_ratio = np.zeros_like(log_magnitude)
    simulation.inputs.set_coefficients(
        "conductance",
        {"log_conductance_magnitude": log_magnitude, "log_hall_to_pedersen_ratio": log_ratio},
        time=[0.0, 1.0],
    )

    response = simulation.response
    sentinel = object()
    response.induced_poloidal_potential_feedback_operator = sentinel
    first_fingerprint = response.conductance_fingerprint
    first_coefficients = response.log_conductance_magnitude
    assert simulation.response_at_time(1.0) is response

    assert response.conductance_fingerprint == first_fingerprint
    assert response.induced_poloidal_potential_feedback_operator is sentinel
    assert response.log_conductance_magnitude is first_coefficients


def test_response_detects_small_edits_to_live_input_datasets(tmp_path, monkeypatch):
    """Selection snapshots detect small edits to live input arrays."""
    simulation = _small_simulation(tmp_path)
    shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(shape),
            "log_hall_to_pedersen_ratio": np.zeros(shape),
        },
    )
    series = simulation.results.input_series
    selected = {}
    get_entry = series.get_entry

    def capture_entry(key, *args, **kwargs):
        selected[key] = get_entry(key, *args, **kwargs)
        return selected[key]

    monkeypatch.setattr(series, "get_entry", capture_entry)
    response = simulation.response_at_time(0.0)
    for name, values in selected["conductance"].items():
        np.testing.assert_array_equal(response._conductance_values[name], values)
    first = response.log_conductance_magnitude
    fingerprint = response.conductance_fingerprint

    simulation.inputs["conductance"]["SH_log_conductance_magnitude"].values[0, 0] = 1e-12
    updated = simulation.response_at_time(0.0)

    assert updated is not response
    assert updated.log_conductance_magnitude is not first
    assert updated.conductance_fingerprint != fingerprint
    np.testing.assert_array_equal(first, np.zeros(shape))
    assert updated.log_conductance_magnitude[0] == 1e-12


def test_responses_select_shared_inputs_independently(tmp_path):
    """One response must not consume another's input update."""
    simulation = _small_simulation(tmp_path)
    shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(shape),
            "log_hall_to_pedersen_ratio": np.zeros(shape),
        },
    )
    first = simulation.response
    series = simulation.results.input_series
    second = ElectrodynamicResponse.from_conductance(
        simulation.geometry,
        simulation.config,
        series.get_field_space("conductance"),
        series.get_entry("conductance", 0.0),
    )

    assert second.log_conductance_magnitude is not None
    assert second.conductance_fingerprint == first.conductance_fingerprint
    np.testing.assert_allclose(second.resistance_tensor_on_grid, first.resistance_tensor_on_grid)


def test_evolution_drops_inputs_unavailable_at_selected_time(tmp_path):
    """Backward selection must not retain a future forcing."""
    simulation = _small_simulation(tmp_path)
    conductance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(conductance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(conductance_shape),
        },
    )
    shape = simulation.results.schema.input_field_spaces["u"].shape
    simulation.inputs.set_coefficients("u", np.ones(shape), time=1.0)
    evolution = simulation._time_evolution
    series = simulation.results.input_series

    def select(time):
        return evolution._response_and_forcing(next(series.iter_intervals(time, time))[2])

    select(1.0)
    assert "u" in evolution._forcing_values

    select(0.0)
    assert "u" not in evolution._forcing_values
    select(1.0)
    assert "u" in evolution._forcing_values
    simulation.inputs.datasets.pop("u")
    select(1.0)
    assert "u" not in evolution._forcing_values


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("reuse_preconditioner", [False, True])
def test_changed_conductance_reuses_only_independent_response_state(
    tmp_path, coupled, reuse_preconditioner
):
    """A current-only toroidal fit is independent of conductance."""
    simulation = _small_simulation(
        tmp_path,
        enable_interhemispheric_coupling=coupled,
        reuse_preconditioner=reuse_preconditioner,
    )
    shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.stack([np.zeros(shape), np.full(shape, 0.01)]),
            "log_hall_to_pedersen_ratio": np.zeros((2, *shape)),
        },
        time=[0.0, 1.0],
    )
    response = simulation.response
    series = simulation.results.input_series
    problem = response._toroidal_potential_problem
    resistance = response.resistance_tensor_on_grid
    fingerprint = response.conductance_fingerprint
    # Test ownership without constructing an expensive preconditioner.
    preconditioner = object()
    response._toroidal_potential_preconditioner = preconditioner
    updated = simulation.response_at_time(1.0)

    assert response.conductance_fingerprint == fingerprint
    assert response.resistance_tensor_on_grid is resistance
    assert updated.conductance_fingerprint != fingerprint
    assert updated.resistance_tensor_on_grid is not resistance
    assert (updated._toroidal_potential_problem is problem) == (not coupled)
    assert (updated._toroidal_potential_preconditioner is preconditioner) == (
        not coupled or reuse_preconditioner
    )
    fresh = ElectrodynamicResponse.from_conductance(
        simulation.geometry,
        simulation.config,
        series.get_field_space("conductance"),
        series.get_entry("conductance", 1.0),
    )
    np.testing.assert_allclose(updated.resistance_tensor_on_grid, fresh.resistance_tensor_on_grid)


def test_changing_conductance_does_not_force_unused_solver_state(tmp_path):
    """Sharing a fixed geometry must not eagerly construct a solve."""
    simulation = _small_simulation(tmp_path)
    shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.stack([np.zeros(shape), np.full(shape, 0.01)]),
            "log_hall_to_pedersen_ratio": np.zeros((2, *shape)),
        },
        time=[0.0, 1.0],
    )
    first = simulation.response_at_time(0.0)
    second = simulation.response_at_time(1.0)
    assert first is not second
    for response in (first, second):
        assert "_toroidal_potential_problem" not in response.__dict__
        assert "_toroidal_potential_response_solver" not in response.__dict__
        assert "_toroidal_potential_preconditioner" not in response.__dict__


def test_conductance_selection_fingerprints_stored_coefficients(tmp_path):
    """Fingerprint stored conductance before backend transfer."""
    simulation = _small_simulation(tmp_path)
    field_space = simulation.results.schema.input_field_spaces["conductance"]
    coefficients = np.zeros(field_space.shape)
    simulation.inputs.set_coefficients(
        "conductance",
        {"log_conductance_magnitude": coefficients, "log_hall_to_pedersen_ratio": coefficients},
        time=0.0,
    )

    response = simulation.response
    assert all(isinstance(value, np.ndarray) for value in response._conductance_values.values())
    assert response.conductance_fingerprint


@pytest.mark.parametrize("coupled", [False, True])
def test_changed_conductance_response_matches_fresh_solve(tmp_path, coupled, monkeypatch):
    """Reused solvers match the response from a fresh model."""
    simulation = _small_simulation(tmp_path, enable_interhemispheric_coupling=coupled)
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=np.stack([np.full(grid.size, 2.0), np.full(grid.size, 3.0)]),
        hall=np.ones((2, grid.size)),
        time=[0.0, 1.0],
        grid=grid,
    )
    space = simulation.results.schema.input_field_spaces["boundary_jr"]
    simulation.inputs.set_coefficients("boundary_jr", np.linspace(-1e-6, 1e-6, space.size))
    response = simulation.response
    build_solver = LeastSquaresSolver.prepare
    builds = []

    def record_build(solver, problem, **kwargs):
        builds.append(problem)
        return build_solver(solver, problem, **kwargs)

    monkeypatch.setattr(LeastSquaresSolver, "prepare", record_build)
    series = simulation.results.input_series
    forcing = series.get_entry("boundary_jr", 0.0)
    response.solve_noninductive_response(**forcing)
    updated = simulation.response_at_time(1.0)
    actual = updated.solve_noninductive_response(**forcing)
    assert len(builds) == (2 if coupled else 1)

    fresh = ElectrodynamicResponse.from_conductance(
        simulation.geometry,
        simulation.config,
        series.get_field_space("conductance"),
        series.get_entry("conductance", 1.0),
    )
    for values, expected in zip(actual, fresh.solve_noninductive_response(**forcing), strict=True):
        np.testing.assert_allclose(values, expected, rtol=1e-11, atol=1e-13)


def test_set_conductance_cs_basis_remaps_non_model_grid(tmp_path):
    """CS conductance basis can remap values from another grid."""
    simulation = _small_simulation(tmp_path, conductance_basis="CS")
    grid = simulation.geometry.model_grid
    pedersen = np.ones(grid.size)
    hall = np.full(grid.size, 0.5)

    simulation.inputs.set_conductance(
        pedersen=pedersen,
        hall=hall,
        time=6.0,
        grid=SphericalGrid(lat=grid.lat + np.linspace(0.0, 1e-3, grid.size), lon=grid.lon),
    )

    dataset = simulation.results.input_series.datasets["conductance"]
    assert "CS_log_conductance_magnitude" in dataset
    assert "CS_log_hall_to_pedersen_ratio" in dataset
    assert np.all(np.isfinite(dataset["CS_log_conductance_magnitude"].isel(time=0).values))
    assert np.all(np.isfinite(dataset["CS_log_hall_to_pedersen_ratio"].isel(time=0).values))


def test_set_conductance_cs_basis_rejects_least_squares_options(tmp_path):
    """CS conductance storage rejects least-squares controls."""
    simulation = _small_simulation(tmp_path, conductance_basis="CS")
    grid = simulation.geometry.model_grid
    pedersen = np.ones(grid.size)
    hall = np.full(grid.size, 0.5)

    with np.testing.assert_raises_regex(ValueError, "fitting controls"):
        simulation.inputs.set_conductance(pedersen=pedersen, hall=hall, reg_lambda=1e-3, grid=grid)


def test_set_conductance_projects_dimensionless_log_coordinates(tmp_path, monkeypatch):
    """Store conductance samples in canonical coordinates."""
    simulation = _small_simulation(tmp_path)
    hall = np.array([[3, 4]])
    pedersen = np.array([[4, 3]])
    recorded = {}

    def record_projection(key, samples, **kwargs):
        recorded["samples"] = samples
        recorded["key"] = key
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(
        simulation.inputs._input_projector, "project_and_store_input", record_projection
    )

    simulation.inputs.set_conductance(
        pedersen=pedersen,
        hall=hall,
        time=7.0,
        sqrt_weights=np.ones(2),
        reg_lambda=1e-3,
        grid=SphericalGrid(lat=np.array([60.0, 61.0]), lon=np.array([10.0, 11.0])),
    )

    expected_magnitude, expected_ratio = conductance_to_log_coordinates(pedersen, hall)
    assert recorded["key"] == "conductance"
    np.testing.assert_allclose(
        recorded["samples"]["log_conductance_magnitude"], expected_magnitude
    )
    np.testing.assert_allclose(recorded["samples"]["log_hall_to_pedersen_ratio"], expected_ratio)
    assert recorded["kwargs"]["time"] == 7.0
    assert recorded["kwargs"]["reg_lambda"] == 1e-3


def test_set_resistance_projects_direct_log_conductance_coordinates(tmp_path, monkeypatch):
    """Map resistance samples directly onto canonical coordinates."""
    simulation = _small_simulation(tmp_path)
    etaP = np.array([[0.4, 0.2]])
    etaH = np.array([[0.3, 0.1]])
    recorded = {}

    def record_projection(key, samples, **kwargs):
        recorded["samples"] = samples
        recorded["key"] = key
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(
        simulation.inputs._input_projector, "project_and_store_input", record_projection
    )

    simulation.inputs.set_resistance(
        etaP=etaP,
        etaH=etaH,
        time=7.0,
        sqrt_weights=np.ones(2),
        reg_lambda=1e-3,
        grid=SphericalGrid(lat=np.array([60.0, 61.0]), lon=np.array([10.0, 11.0])),
    )

    expected_magnitude, expected_ratio = resistance_to_log_conductance_coordinates(etaP, etaH)
    assert recorded["key"] == "conductance"
    np.testing.assert_allclose(
        recorded["samples"]["log_conductance_magnitude"], expected_magnitude
    )
    np.testing.assert_allclose(recorded["samples"]["log_hall_to_pedersen_ratio"], expected_ratio)
    assert recorded["kwargs"]["time"] == 7.0
    assert recorded["kwargs"]["reg_lambda"] == 1e-3
