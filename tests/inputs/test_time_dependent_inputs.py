"""End-to-end tests for explicitly time-dependent prepared inputs."""

import numpy as np
import pytest
from kompe import SphericalGrid
from kompe.math import get_array_module

from pynamit import InputPreparation
from pynamit.simulation.input_projection import _scalar_sample_rows
from pynamit.workflows.prepared_inputs import run_from_inputs


@pytest.mark.parametrize("grid_shape", [(3, 4), (1, 4), (4, 1), ()])
@pytest.mark.parametrize("layout", ["single", "time_first", "time_last"])
def test_input_boundary_distinguishes_spatial_axes_from_time(grid_shape, layout):
    """Normalize provider layouts before Kompe analysis."""
    xp = get_array_module()
    theta = np.arange(np.prod(grid_shape), dtype=float).reshape(grid_shape) + 20.0
    grid = SphericalGrid(theta=theta, phi=np.zeros(grid_shape))
    count = 1 if layout == "single" else 5
    rows = xp.arange(count * grid.size, dtype=float).reshape(count, *grid_shape)
    values = rows[0] if layout == "single" else rows
    if layout == "time_last":
        values = xp.moveaxis(values, 0, -1)
    actual = _scalar_sample_rows(values, grid)
    assert isinstance(actual, xp.ndarray)
    np.testing.assert_array_equal(actual, rows.reshape(count, grid.size))


@pytest.mark.parametrize(
    "layout", ["single", "time_first", "time_last", "flat_time_first", "flat_time_last"]
)
def test_sample_inputs_retain_spatial_grids_and_time_axes(layout):
    """Project ordinary mesh-shaped fields without manual flattening."""
    preparation = InputPreparation(Nmax=2, Mmax=1, Ncs=4)
    theta = np.linspace(20.0, 160.0, 6)[:, None]
    phi = np.linspace(0.0, 300.0, 7)[None, :]
    grid = SphericalGrid(theta=theta, phi=phi)
    xp = get_array_module()
    amplitudes = np.array([1.0]) if layout == "single" else np.array([1.0, 1.4, 0.3])
    times = None if layout == "single" else [0.0, 3.0, 7.0]
    theta_radians = np.deg2rad(grid.theta.reshape(grid.shape))

    def samples(pattern):
        rows = xp.asarray(amplitudes[:, None, None] * pattern)
        if layout == "single":
            return rows[0]
        if layout.startswith("flat"):
            rows = rows.reshape(amplitudes.size, grid.size)
        return xp.moveaxis(rows, 0, -1) if layout.endswith("last") else rows

    # For a dipole, FAC * br = 1e-6 cos(theta). The radial current
    # and these two wind components are exactly representable at l=1.
    FAC = -0.5e-6 * np.sqrt(1.0 + 3.0 * np.cos(theta_radians) ** 2)
    preparation.set_FAC(samples(FAC), time=times, grid=SphericalGrid(theta=theta, phi=phi))
    preparation.set_neutral_wind(
        samples(25.0 * np.sin(theta_radians)),
        samples(80.0 * np.sin(theta_radians)),
        time=times,
        grid=SphericalGrid(theta=theta, phi=phi),
    )
    preparation.set_conductance(
        pedersen=samples(np.full(grid.shape, 2.0)),
        hall=samples(np.full(grid.shape, 3.0)),
        time=times,
        grid=SphericalGrid(theta=theta, phi=phi),
    )

    series = preparation.input_series
    current_map = series.get_field_space("boundary_jr").basis.scalar_evaluation_operator(grid)
    wind_map = series.get_field_space("u").basis.helmholtz_synthesis_operator(grid)
    conductance_map = series.get_field_space("conductance").basis.scalar_evaluation_operator(grid)
    for amplitude, time in zip(amplitudes, [0.0] if times is None else times, strict=True):
        current = current_map.matvec(series.get_entry("boundary_jr", time)["boundary_jr"])
        wind = wind_map.matvec(series.get_entry("u", time)["u"]).reshape(2, *grid.shape)
        conductance = series.get_entry("conductance", time)
        np.testing.assert_allclose(
            current.reshape(grid.shape), amplitude * 1e-6 * np.cos(theta_radians), atol=1e-19
        )
        np.testing.assert_allclose(wind[0], amplitude * 25.0 * np.sin(theta_radians), atol=1e-10)
        np.testing.assert_allclose(wind[1], amplitude * 80.0 * np.sin(theta_radians), atol=1e-10)
        np.testing.assert_allclose(
            conductance_map.matvec(conductance["log_conductance_magnitude"]),
            np.log(amplitude * np.sqrt(13.0)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            conductance_map.matvec(conductance["log_hall_to_pedersen_ratio"]),
            np.log(1.5),
            atol=1e-12,
        )
    for dataset in preparation.datasets.values():
        np.testing.assert_array_equal(dataset.time.values, [0.0] if times is None else times)


def test_projected_batches_preserve_times_variables_and_vector_components():
    """Recover scalar pairs and vector potentials in one input batch."""
    preparation = InputPreparation(Nmax=2, Mmax=1, Ncs=8, conductance_basis="CS")
    grid = preparation.model_grid
    times = [0.0, 1.0, 3.0]
    pedersen = np.array([2.0, 4.0, 3.0])[:, None] + np.linspace(0.0, 1.0, grid.size)
    hall = np.array([3.0, 1.0, 2.0])[:, None] + np.linspace(1.0, 0.0, grid.size)
    preparation.set_conductance(pedersen=pedersen, hall=hall, time=times, grid=grid)

    wind_basis = preparation.schema.input_field_spaces["u"].basis
    wind_coefficients = np.arange(6 * wind_basis.coefficient_count, dtype=float).reshape(3, 2, -1)
    wind_samples = wind_basis.helmholtz_synthesis_operator(grid)(
        np.moveaxis(wind_coefficients, 0, -1)
    )
    preparation.set_neutral_wind(
        u_theta=wind_samples[0].T, u_phi=wind_samples[1].T, time=times, grid=grid
    )

    for index, time in enumerate(times):
        conductance = preparation.input_series.get_entry("conductance", time)
        np.testing.assert_allclose(
            conductance["log_conductance_magnitude"],
            np.log(np.hypot(pedersen[index], hall[index])),
        )
        np.testing.assert_allclose(
            conductance["log_hall_to_pedersen_ratio"], np.log(hall[index] / pedersen[index])
        )
        wind = preparation.input_series.get_entry("u", time)
        np.testing.assert_allclose(wind["u"], wind_coefficients[index], atol=1e-10)


def test_prepared_coefficient_series_drives_a_simulation(tmp_path):
    """Exercise an explicit input history end to end."""
    input_directory = tmp_path / "inputs"
    preparation = InputPreparation(
        input_directory=input_directory,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        t0="2020-01-01 00:00:00",
        artifact_storage="netcdf",
    )

    time = np.array([0.0, 0.05, 0.1])
    conductance_shape = preparation.schema.input_field_spaces["conductance"].shape
    log_magnitude = np.zeros((time.size, *conductance_shape))
    log_ratio = np.zeros_like(log_magnitude)
    preparation.set_coefficients(
        "conductance",
        {"log_conductance_magnitude": log_magnitude, "log_hall_to_pedersen_ratio": log_ratio},
        time=time,
    )

    current_shape = preparation.schema.input_field_spaces["boundary_jr"].shape
    current_pattern = np.linspace(-1.0e-6, 1.0e-6, np.prod(current_shape)).reshape(current_shape)
    boundary_jr = np.stack((current_pattern, 1.5 * current_pattern, 2.0 * current_pattern))
    preparation.set_coefficients("boundary_jr", boundary_jr, time=time)
    preparation.write_manifest(source="test_time_dependent_inputs")

    simulation = run_from_inputs(
        input_directory,
        simulation_directory=tmp_path / "simulation",
        final_time=0.1,
        dt=0.05,
        output_interval=0.05,
        samples_per_write=1,
        initialize_from_equilibrium=False,
        sample_equilibrium=False,
        artifact_storage="netcdf",
    )

    interpolated = simulation.results.input_series.get_entry(
        "boundary_jr", 0.025, interpolation=True
    )
    np.testing.assert_allclose(interpolated["boundary_jr"], 1.25 * current_pattern)
    np.testing.assert_allclose(
        simulation.results.output_series.datasets["dynamic"].time.values, time
    )
    assert (
        np.linalg.norm(
            simulation.results.output_series.datasets["dynamic"]["SH_induced_Br"].values[-1]
        )
        > 0.0
    )
