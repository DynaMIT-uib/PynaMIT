"""Tests for saved-run visualization views."""

import numpy as np

import pynamit
from pynamit.visualization.pynameye import PynamEye
from pynamit.visualization.saved_run import SavedRunView


def test_saved_run_view_loads_core_visualization_objects(tmp_path):
    """Saved-run view centralizes settings, schema, and geometry."""
    dynamics = pynamit.Dynamics(
        run_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, ignore_PFAC=True, artifact_storage="netcdf"
    )

    run_view = SavedRunView.from_directory(
        dynamics.run_directory, require_pfac_matrix=True, build_geometry=True
    )
    input_timeseries = run_view.load_input_timeseries()
    output_timeseries = run_view.load_output_timeseries()

    assert run_view.settings is run_view.datasets["settings"]
    assert run_view.config.Nmax == 2
    assert run_view.schema.horizontal_basis is run_view.schema.sh_basis_mean_free
    assert run_view.mainfield.kind == dynamics.mainfield.kind
    assert run_view.pfac_matrix is not None
    assert run_view.geometry is not None
    assert input_timeseries.field_spaces == run_view.schema.input_field_spaces
    assert output_timeseries.field_spaces == run_view.schema.output_field_spaces


def test_saved_run_view_loads_requested_datasets(tmp_path):
    """Required and optional dataset loading is explicit."""
    dynamics = pynamit.Dynamics(
        run_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, ignore_PFAC=True, artifact_storage="netcdf"
    )

    run_view = SavedRunView.from_directory(
        dynamics.run_directory,
        required_datasets=("settings",),
        optional_datasets=("missing_optional",),
    )

    assert set(run_view.datasets) == {"settings"}


def test_pynameye_uses_saved_run_view(tmp_path):
    """PynamEye is a frontend over the saved-run view."""
    dynamics = pynamit.Dynamics(
        run_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, ignore_PFAC=True, artifact_storage="netcdf"
    )
    conductance_shape = dynamics.input_field_spaces["conductance"].coefficient_shape
    etaP = np.zeros(conductance_shape)
    etaH = np.zeros(conductance_shape)
    etaP[0] = 1.0
    dynamics.set_resistance(etaP_coefficients=etaP, etaH_coefficients=etaH, time=0.0)

    wind_shape = dynamics.input_field_spaces["u"].coefficient_shape
    u_cf = np.linspace(0.0, 1.0, wind_shape[1])
    u_df = np.linspace(2.0, 3.0, wind_shape[1])
    dynamics.set_neutral_wind(u_cf=u_cf, u_df=u_df, time=0.0)

    jr_shape = dynamics.input_field_spaces["jr"].coefficient_shape
    dynamics.set_jr(jr_coefficients=np.zeros(jr_shape), time=0.0)
    dynamics.impose_steady_state(time=0.0, save=True, quiet=True)

    eye = PynamEye(dynamics.run_directory, Nlat=6, Nlon=8, NCS_plot=4, steady_state=False)

    assert isinstance(eye.run_view, SavedRunView)
    assert eye.schema is eye.run_view.schema
    assert eye.geometry is eye.run_view.geometry
    assert eye.mainfield is eye.run_view.mainfield
    assert eye.m_ind_to_Br_operator is eye.geometry.m_ind_to_Br_operator
    assert eye.m_imp_to_jr_operator is eye.geometry.m_imp_to_jr_operator
    np.testing.assert_allclose(eye.m_Phi, eye.Phi_coeffs * eye.RI)
    np.testing.assert_allclose(eye.m_W, eye.W_coeffs * eye.RI)
    np.testing.assert_allclose(eye.m_u_cf, u_cf)
    np.testing.assert_allclose(eye.m_u_df, u_df)
    np.testing.assert_allclose(eye.u.array, np.stack([u_cf, u_df]))


def test_pynameye_wind_plot_uses_wind_projection_basis(tmp_path):
    """PynamEye evaluates wind on the saved input coefficient basis."""
    dynamics = pynamit.Dynamics(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        artifact_storage="netcdf",
        horizontal_basis_kind="SH",
        u_projection_basis="CS",
    )
    conductance_shape = dynamics.input_field_spaces["conductance"].coefficient_shape
    etaP = np.zeros(conductance_shape)
    etaH = np.zeros(conductance_shape)
    etaP[0] = 1.0
    dynamics.set_resistance(etaP_coefficients=etaP, etaH_coefficients=etaH, time=0.0)

    wind_shape = dynamics.input_field_spaces["u"].coefficient_shape
    u_cf = np.zeros(wind_shape[1])
    u_df = np.zeros(wind_shape[1])
    u_cf[0] = 1.0
    u_df[1] = 0.5
    dynamics.set_neutral_wind(u_cf=u_cf, u_df=u_df, time=0.0)

    jr_shape = dynamics.input_field_spaces["jr"].coefficient_shape
    dynamics.set_jr(jr_coefficients=np.zeros(jr_shape), time=0.0)
    dynamics.impose_steady_state(time=0.0, save=True, quiet=True)

    eye = PynamEye(dynamics.run_directory, Nlat=6, Nlon=8, NCS_plot=4, steady_state=False)
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
