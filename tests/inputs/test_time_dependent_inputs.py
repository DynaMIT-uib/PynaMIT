"""End-to-end tests for explicitly time-dependent prepared inputs."""

import numpy as np

from pynamit import InputPreparation
from pynamit.workflows.prepared_inputs import run_from_inputs


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
    conductance_shape = preparation.data.schema.input_field_spaces["conductance"].coefficient_shape
    log_magnitude = np.zeros((time.size, *conductance_shape))
    log_ratio = np.zeros_like(log_magnitude)
    preparation.set_conductance(
        log_magnitude_coefficients=log_magnitude, log_ratio_coefficients=log_ratio, time=time
    )

    current_shape = preparation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    current_pattern = np.linspace(-1.0e-6, 1.0e-6, np.prod(current_shape)).reshape(current_shape)
    boundary_jr = np.stack((current_pattern, 1.5 * current_pattern, 2.0 * current_pattern))
    preparation.set_boundary_jr(boundary_jr_coefficients=boundary_jr, time=time)
    preparation.write_manifest(source="test_time_dependent_inputs")

    simulation = run_from_inputs(
        input_directory,
        simulation_directory=tmp_path / "simulation",
        final_time=0.1,
        dt=0.05,
        samples_per_write=1,
        initialize_from_equilibrium=False,
        run_equilibrium=False,
        artifact_storage="netcdf",
    )

    interpolated = simulation.data.input_series.get_entry("boundary_jr", 0.025, interpolation=True)
    np.testing.assert_allclose(interpolated["boundary_jr"], 1.25 * current_pattern)
    np.testing.assert_allclose(simulation.data.output_series.datasets["dynamic"].time.values, time)
    assert (
        np.linalg.norm(
            simulation.data.output_series.datasets["dynamic"]["SH_induced_Br"].values[-1]
        )
        > 0.0
    )
