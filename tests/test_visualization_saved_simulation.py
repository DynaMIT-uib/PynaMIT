"""Tests for saved simulation visualization views."""

import numpy as np

import pynamit
from pynamit.plotting.legacy import PynamEye
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
    assert results.gap_Br_response is None
    assert results.geometry is not None
    assert input_series.field_spaces == results.schema.input_field_spaces
    assert output_series.field_spaces == results.schema.output_field_spaces


def test_simulation_results_is_the_core_saved_simulation_api(tmp_path):
    """The core results object loads datasets without a live runner."""
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


def test_pynameye_uses_simulation_results(tmp_path):
    """PynamEye is a frontend over SimulationResults."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.data.schema.input_field_spaces["conductance"].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(resistance_shape),
        log_ratio_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )

    wind_shape = simulation.data.schema.input_field_spaces["u"].coefficient_shape
    u_cf = np.linspace(0.0, 1.0, wind_shape[1])
    u_df = np.linspace(2.0, 3.0, wind_shape[1])
    simulation.set_neutral_wind(u_cf=u_cf, u_df=u_df, time=0.0)

    boundary_jr_shape = simulation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)

    eye = PynamEye(
        simulation.data.simulation_directory, Nlat=6, Nlon=8, NCS_plot=4, equilibrium=False
    )

    assert isinstance(eye.results, SimulationResults)
    assert not hasattr(eye, "run_view")
    assert eye.schema is eye.results.schema
    assert eye.geometry is eye.results.geometry
    assert eye.main_field is eye.results.main_field
    np.testing.assert_allclose(
        eye.induced_Br_to_Br_operator.to_matrix(backend="numpy"),
        np.eye(eye.geometry.poloidal_basis.index_length),
    )
    np.testing.assert_allclose(
        eye.boundary_jr_to_jr_operator.to_matrix(backend="numpy"),
        np.eye(eye.geometry.horizontal_basis.index_length),
    )
    expected_model_lat, expected_model_lon = eye.main_field.geo_to_model_coordinates(
        eye.lat, eye.lon, event_time=eye.t0
    )
    np.testing.assert_allclose(eye.global_grid.lat, expected_model_lat.reshape(-1))
    np.testing.assert_allclose(eye.global_grid.lon, expected_model_lon.reshape(-1))
    np.testing.assert_allclose(eye.polar_grid.lat, eye.mlat.reshape(-1))
    np.testing.assert_allclose(eye.polar_grid.lon, eye.mlon.reshape(-1))
    np.testing.assert_allclose(eye.m_Phi, eye.Phi_coeffs * eye.RI)
    np.testing.assert_allclose(eye.m_W, eye.W_coeffs * eye.RI)
    np.testing.assert_allclose(eye.m_u_cf, u_cf)
    np.testing.assert_allclose(eye.m_u_df, u_df)
    np.testing.assert_allclose(eye.u.array, np.stack([u_cf, u_df]))


def test_pynameye_supports_equilibrium_only_output(tmp_path):
    """PynamEye does not require a dynamic artifact for equilibrium."""
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
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)
    simulation.data.artifact_store.remove_artifact("dynamic")

    eye = PynamEye(tmp_path, Nlat=6, Nlon=8, NCS_plot=4, equilibrium=True)

    assert "dynamic" not in eye.datasets
    assert "equilibrium" in eye.datasets
    assert eye.induced_Br.shape == (simulation.geometry.poloidal_basis.index_length,)


def test_pynameye_reuses_earth_fixed_geographic_mapping(tmp_path):
    """PynamEye does not rebuild fixed GEO display geometry."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="kaiju_dipole",
        main_field_epoch=2011.8,
        t0="2011-10-24T18:00:10",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.data.schema.input_field_spaces["conductance"].coefficient_shape
    times = np.array([0.0, 3600.0])
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros((2, *resistance_shape)),
        log_ratio_coefficients=np.zeros((2, *resistance_shape)),
        time=times,
    )
    boundary_jr_shape = simulation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    simulation.set_boundary_jr(
        boundary_jr_coefficients=np.zeros((2, *boundary_jr_shape)), time=times
    )
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)
    simulation.impose_equilibrium(time=3600.0, save=True, quiet=True)
    eye = PynamEye(tmp_path, Nlat=6, Nlon=8, NCS_plot=4)
    initial_lon = eye.global_grid.lon.copy()
    initial_vector_lon = np.asarray(eye.global_vector_lon).copy()

    eye.set_time(3600.0)

    np.testing.assert_allclose(eye.global_grid.lon, initial_lon)
    np.testing.assert_allclose(eye.global_vector_lon, initial_vector_lon)
    expected_lat, expected_lon = eye.main_field.geo_to_model_coordinates(
        eye.lat, eye.lon, event_time=eye.time
    )
    np.testing.assert_allclose(eye.global_grid.lat, expected_lat.reshape(-1))
    np.testing.assert_allclose(eye.global_grid.lon, expected_lon.reshape(-1))


def test_pynameye_joule_uses_total_boundary_driven_current(tmp_path):
    """Legacy Joule plots include prescribed-boundary sheet current."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * 6381e3,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.data.schema.input_field_spaces["conductance"].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(resistance_shape),
        log_ratio_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )
    boundary_Br_shape = simulation.data.schema.input_field_spaces["boundary_Br"].coefficient_shape
    boundary_Br = np.zeros(boundary_Br_shape)
    boundary_Br[0] = 1.0e-9
    simulation.set_boundary_Br(boundary_Br_coefficients=boundary_Br, time=0.0)
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)

    eye = PynamEye(
        simulation.data.simulation_directory, Nlat=6, Nlon=8, NCS_plot=4, equilibrium=False
    )
    eye._plot_filled_contour = lambda values, _axis, _region, **_kwargs: values

    assert eye.sheet_current_maps == {}
    plotted = eye.plot_joule(object())
    assert set(eye.sheet_current_maps) == {"global"}
    current_maps = eye.sheet_current_maps["global"]
    expected = (
        np.asarray(current_maps["induced_Br_to_JS"]).dot(eye.induced_Br)
        + np.asarray(current_maps["boundary_jr_to_JS"]).dot(eye.boundary_jr)
        + np.asarray(current_maps["boundary_Br_to_JS"]).dot(eye.boundary_Br)
    ).reshape(2, -1)

    np.testing.assert_allclose(eye._JS, expected, atol=1e-15)
    np.testing.assert_allclose(plotted, eye._Q)
    assert np.nanmin(eye._Q) >= -1e-15


def test_pynameye_wind_plot_uses_wind_projection_basis(tmp_path):
    """PynamEye evaluates wind on the saved input coefficient basis."""
    simulation = pynamit.Simulation(
        simulation_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
        horizontal_basis_kind="SH",
        u_projection_basis="CS",
    )
    resistance_shape = simulation.data.schema.input_field_spaces["conductance"].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(resistance_shape),
        log_ratio_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )

    wind_shape = simulation.data.schema.input_field_spaces["u"].coefficient_shape
    u_cf = np.zeros(wind_shape[1])
    u_df = np.zeros(wind_shape[1])
    u_cf[0] = 1.0
    u_df[1] = 0.5
    simulation.set_neutral_wind(u_cf=u_cf, u_df=u_df, time=0.0)

    boundary_jr_shape = simulation.data.schema.input_field_spaces["boundary_jr"].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)

    eye = PynamEye(
        simulation.data.simulation_directory, Nlat=6, Nlon=8, NCS_plot=4, equilibrium=False
    )
    cs_wind_space = pynamit.FieldSpace(
        eye.schema.cs_basis, field_type="tangential", mean_free=True
    )
    cs_wind = np.zeros(cs_wind_space.coefficient_shape)
    cs_wind[0, 0] = 1.0
    cs_wind[1, 1] = 0.5
    eye.u = pynamit.FieldCoefficients(cs_wind_space, cs_wind)
    eye.m_u = eye.u.array

    captured = {}

    def capture_quiver(east, north, ax, region="global", **kwargs):
        del ax, kwargs
        captured["region"] = region
        captured["east"] = np.asarray(east)
        captured["north"] = np.asarray(north)
        return captured

    eye._quiver = capture_quiver

    assert eye.plot_wind(object()) is captured
    assert captured["region"] == "global"
    assert captured["east"].shape == (eye.global_vector_grid.size,)
    assert captured["north"].shape == (eye.global_vector_grid.size,)
