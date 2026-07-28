"""Tests for saved-run visualization views."""

import numpy as np

import pynamit
from pynamit.visualization.pynameye import PynamEye
from pynamit.visualization.saved_run import SavedRunView


def test_saved_run_view_loads_core_visualization_objects(tmp_path):
    """Saved-run view centralizes settings, schema, and geometry."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    run_view = SavedRunView.from_directory(simulation.run_data.run_directory, build_geometry=True)
    input_series = run_view.load_input_series()
    output_series = run_view.load_output_series()

    assert run_view.config.Nmax == 2
    assert run_view.schema.horizontal_basis is run_view.schema.mean_free_sh_basis
    assert run_view.main_field.kind == simulation.geometry.main_field.kind
    assert run_view.pfac_matrix is None
    assert run_view.geometry is not None
    assert input_series.field_spaces == run_view.schema.input_field_spaces
    assert output_series.field_spaces == run_view.schema.output_field_spaces


def test_saved_run_view_loads_requested_datasets(tmp_path):
    """Required and optional dataset loading is explicit."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    run_view = SavedRunView.from_directory(
        simulation.run_data.run_directory,
        required_datasets=("settings",),
        optional_datasets=("missing_optional",),
    )

    assert set(run_view.datasets) == {"settings"}


def test_pynameye_uses_saved_run_view(tmp_path):
    """PynamEye is a frontend over the saved-run view."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "conductance"
    ].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(resistance_shape),
        log_ratio_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )

    wind_shape = simulation.run_data.schema.input_field_spaces["u"].coefficient_shape
    u_cf = np.linspace(0.0, 1.0, wind_shape[1])
    u_df = np.linspace(2.0, 3.0, wind_shape[1])
    simulation.set_neutral_wind(u_cf=u_cf, u_df=u_df, time=0.0)

    jr_shape = simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape
    simulation.set_jr(jr_coefficients=np.zeros(jr_shape), time=0.0)
    simulation.impose_steady_state(time=0.0, save=True, quiet=True)

    eye = PynamEye(
        simulation.run_data.run_directory, Nlat=6, Nlon=8, NCS_plot=4, steady_state=False
    )

    assert isinstance(eye.run_view, SavedRunView)
    assert eye.schema is eye.run_view.schema
    assert eye.geometry is eye.run_view.geometry
    assert eye.main_field is eye.run_view.main_field
    assert eye.m_ind_to_Br_operator is eye.geometry.m_ind_to_Br_operator
    assert eye.m_imp_to_jr_operator is eye.geometry.m_imp_to_jr_operator
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


def test_pynameye_reuses_earth_fixed_geographic_mapping(tmp_path):
    """PynamEye does not rebuild fixed GEO display geometry."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="kaiju_dipole",
        main_field_epoch=2011.8,
        t0="2011-10-24T18:00:10",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "conductance"
    ].coefficient_shape
    times = np.array([0.0, 3600.0])
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros((2, *resistance_shape)),
        log_ratio_coefficients=np.zeros((2, *resistance_shape)),
        time=times,
    )
    jr_shape = simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape
    simulation.set_jr(jr_coefficients=np.zeros((2, *jr_shape)), time=times)
    simulation.impose_steady_state(time=0.0, save=True, quiet=True)
    simulation.impose_steady_state(time=3600.0, save=True, quiet=True)
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
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * 6381e3,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "conductance"
    ].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(resistance_shape),
        log_ratio_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )
    br_shape = simulation.run_data.schema.input_field_spaces["Br"].coefficient_shape
    Br = np.zeros(br_shape)
    Br[0] = 1.0e-9
    simulation.set_Br(Br_coefficients=Br, time=0.0)
    simulation.impose_steady_state(time=0.0, save=True, quiet=True)

    eye = PynamEye(
        simulation.run_data.run_directory, Nlat=6, Nlon=8, NCS_plot=4, steady_state=False
    )
    eye._plot_filled_contour = lambda values, _axis, _region, **_kwargs: values

    assert eye.sheet_current_maps == {}
    plotted = eye.plot_joule(object())
    assert set(eye.sheet_current_maps) == {"global"}
    current_maps = eye.sheet_current_maps["global"]
    expected = (
        np.asarray(current_maps["m_ind_to_JS"]).dot(eye.m_ind)
        + np.asarray(current_maps["m_imp_to_JS"]).dot(eye.m_imp)
        + np.asarray(current_maps["Br_to_JS"]).dot(eye.m_Br)
    ).reshape(2, -1)

    np.testing.assert_allclose(eye._JS, expected, atol=1e-15)
    np.testing.assert_allclose(plotted, eye._Q)
    assert np.nanmin(eye._Q) >= -1e-15


def test_pynameye_wind_plot_uses_wind_projection_basis(tmp_path):
    """PynamEye evaluates wind on the saved input coefficient basis."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
        horizontal_basis_kind="SH",
        u_projection_basis="CS",
    )
    resistance_shape = simulation.run_data.schema.input_field_spaces[
        "conductance"
    ].coefficient_shape
    simulation.set_conductance(
        log_magnitude_coefficients=np.zeros(resistance_shape),
        log_ratio_coefficients=np.zeros(resistance_shape),
        time=0.0,
    )

    wind_shape = simulation.run_data.schema.input_field_spaces["u"].coefficient_shape
    u_cf = np.zeros(wind_shape[1])
    u_df = np.zeros(wind_shape[1])
    u_cf[0] = 1.0
    u_df[1] = 0.5
    simulation.set_neutral_wind(u_cf=u_cf, u_df=u_df, time=0.0)

    jr_shape = simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape
    simulation.set_jr(jr_coefficients=np.zeros(jr_shape), time=0.0)
    simulation.impose_steady_state(time=0.0, save=True, quiet=True)

    eye = PynamEye(
        simulation.run_data.run_directory, Nlat=6, Nlon=8, NCS_plot=4, steady_state=False
    )
    cs_wind_space = pynamit.FieldSpace.from_representation(
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
