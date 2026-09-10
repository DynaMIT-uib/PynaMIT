"""Tests for saved simulation result access and evaluation."""

import numpy as np
from kompe import SphericalGrid
from kompe.math import get_array_module

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

    results = SimulationResults.from_directory(simulation.results.simulation_directory)
    input_series = results.load_input_series()
    output_series = results.load_output_series()

    assert results.config.Nmax == 2
    assert not hasattr(results, "run_directory")
    assert results.geometry.horizontal_basis is results.geometry.poloidal_basis
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
    shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(shape), time=2.0)

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
    conductance_shape = simulation.results.schema.input_field_spaces["conductance"].shape
    simulation.inputs.set_coefficients(
        "conductance",
        {
            "log_conductance_magnitude": np.zeros(conductance_shape),
            "log_hall_to_pedersen_ratio": np.zeros(conductance_shape),
        },
        time=0.0,
    )
    current_shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(current_shape), time=0.0)
    simulation.set_state(
        simulation.equilibrium_coefficients(time=0.0, interpolation=True)["induced_Br"], time=0.0
    )
    simulation.record_state(save=True)

    grid = SphericalGrid(lat=np.array([[30.0], [60.0]]), lon=np.array([[0.0, 90.0, 180.0]]))
    live = evaluate_simulation_output(simulation.results, 0.0, grid=grid)
    saved = evaluate_simulation_output(SimulationResults.from_directory(tmp_path), 0.0, grid=grid)
    basic_results = SimulationResults.from_directory(tmp_path)
    basic = evaluate_simulation_output(
        basic_results, 0.0, grid=grid, field_names={"induced_Br", "boundary_jr", "Phi", "W"}
    )

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
    assert set(basic_results.datasets) == {"settings", "dynamic"}
    for name in expected:
        assert live[name].shape == grid.shape == (2, 3)
        assert isinstance(live[name], get_array_module().ndarray)
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
        simulation.results.simulation_directory,
        required_datasets=("settings",),
        optional_datasets=("missing_optional",),
    )

    assert set(results.datasets) == {"settings"}


def test_saved_results_load_one_stream_and_keep_one_owner(tmp_path, monkeypatch):
    """Selecting one input does not load siblings or copy ownership."""
    simulation = pynamit.Simulation(
        tmp_path, Nmax=2, Mmax=1, Ncs=4, enable_pfac_coupling=False, artifact_storage="netcdf"
    )
    shape = simulation.results.schema.input_field_spaces["boundary_jr"].shape
    simulation.inputs.set_coefficients("boundary_jr", np.zeros(shape), time=0.0)
    results = SimulationResults.from_directory(tmp_path)
    load_dataset = results.artifact_store.load_dataset
    calls = []

    def load(key, **kwargs):
        calls.append(key)
        return load_dataset(key, **kwargs)

    monkeypatch.setattr(results.artifact_store, "load_dataset", load)
    series = results.load_input_series("boundary_jr")
    assert calls == ["boundary_jr"]
    assert set(results.datasets) == {"settings", "boundary_jr"}
    assert results._output_series.datasets == {}
    assert results.load_input_series("boundary_jr") is series
    assert results.data_var_name("boundary_jr", "boundary_jr") == "SH_boundary_jr"
    assert calls == ["boundary_jr"]

    replacement = series.datasets["boundary_jr"].assign_attrs(note="inspection")
    series.datasets["boundary_jr"] = replacement
    assert results.datasets["boundary_jr"] is replacement
    results.load_input_series("u")
    results.load_input_series("u")
    assert calls == ["boundary_jr", "u"]
