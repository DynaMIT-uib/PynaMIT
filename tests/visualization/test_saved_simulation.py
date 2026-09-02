"""Tests for saved simulation result access and evaluation."""

import numpy as np

import pynamit
from pynamit.results import SimulationResults, evaluate_simulation_output


def test_simulation_results_loads_core_visualization_objects(tmp_path):
    """SimulationResults owns saved settings, schema, and geometry."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    results = SimulationResults.from_directory(
        simulation.data.simulation_directory, build_geometry=True
    )
    input_series = results.load_input_series()
    output_series = results.load_output_series()

    assert results.config.Nmax == 2
    assert not hasattr(results, "run_directory")
    assert results.schema.horizontal_basis is results.schema.mean_free_sh_basis
    assert results.main_field.kind == simulation.geometry.main_field.kind
    assert results.boundary_jr_to_gap_Br_matrix is None
    assert results.geometry is not None
    assert input_series.field_spaces == results.schema.input_field_spaces
    assert output_series.field_spaces == results.schema.output_field_spaces


def test_simulation_results_is_the_core_saved_simulation_api(tmp_path):
    """Load saved datasets without live time evolution."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    shape = simulation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(shape), time=2.0)

    results = pynamit.SimulationResults.from_directory(tmp_path)

    assert set(results.inputs) == {"boundary_jr"}
    assert results.outputs == {}
    np.testing.assert_allclose(results.times, [2.0])
    assert results.simulation_directory == str(tmp_path.resolve())


def test_evaluate_simulation_output_matches_live_and_saved_sources(tmp_path):
    """Physical output evaluation needs no plotting wrapper."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    conductance_shape = simulation.data.schema.input_field_spaces["conductance"].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(conductance_shape),
        log_ratio_coefficients=np.zeros(conductance_shape),
        time=0.0,
    )
    current_shape = simulation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(current_shape), time=0.0)
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)

    live = evaluate_simulation_output(simulation, 0.0)
    saved = evaluate_simulation_output(SimulationResults.from_directory(tmp_path), 0.0)
    basic_results = SimulationResults.from_directory(tmp_path)
    basic = evaluate_simulation_output(basic_results, 0.0, include_derived=False)

    expected = {
        "induced_Br",
        "boundary_jr",
        "Phi",
        "W",
        "E_theta",
        "E_phi",
        "E_mag",
        "equivalent_current_function",
        "JS_theta",
        "JS_phi",
        "JS_mag",
        "joule_heating",
    }
    assert set(live) == expected
    assert set(saved) == expected
    assert set(basic) == {"induced_Br", "boundary_jr", "Phi", "W"}
    assert basic_results._input_series is None
    for name in expected:
        np.testing.assert_allclose(saved[name], live[name])


def test_simulation_results_loads_requested_datasets(tmp_path):
    """Required and optional dataset loading is explicit."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    results = SimulationResults.from_directory(
        simulation.data.simulation_directory,
        required_datasets=("settings",),
        optional_datasets=("missing_optional",),
    )

    assert set(results.datasets) == {"settings"}
