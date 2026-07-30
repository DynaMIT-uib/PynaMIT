"""Tests for visualization package exports."""

import importlib

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pynamit


def test_magnetic_boundary_operators_are_available():
    """JS operators are available from their owning modules."""
    grid_evaluation = importlib.import_module("pynamit.visualization.grid_evaluation")
    magnetic_boundary = importlib.import_module(
        "pynamit.simulation.electrodynamics.magnetic_boundary"
    )

    assert callable(magnetic_boundary.induced_Br_to_gridded_JS_operator)
    assert callable(magnetic_boundary.boundary_jr_to_gridded_JS_operator)
    assert callable(grid_evaluation.build_JS_operators)


def test_pynameye_is_available_from_visualization():
    """PynamEye is available from the canonical visualization path."""
    visualization = importlib.import_module("pynamit.visualization")
    visualization_pynameye = importlib.import_module("pynamit.visualization.pynameye")

    assert visualization.PynamEye is visualization_pynameye.PynamEye
    assert hasattr(visualization.PynamEye, "style_global_axis")


def test_input_projection_comparison_recipe_is_importable():
    """Input diagnostic recipes are available from their module."""
    diagnostics = importlib.import_module("pynamit.visualization.input_projection_comparison")

    assert callable(diagnostics.plot_input_projection_comparison)
    assert callable(diagnostics.write_input_projection_diagnostics)


def test_projected_input_inspector_is_visualization_api():
    """Projected-input inspection is exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    input_projection = importlib.import_module("pynamit.visualization.input_projection")

    assert visualization.evaluate_projected_input is input_projection.evaluate_projected_input


def test_projection_diagnostics_are_visualization_api():
    """Projection-comparison entry points are exported lazily."""
    visualization = importlib.import_module("pynamit.visualization")
    diagnostics = importlib.import_module("pynamit.visualization.input_projection_comparison")

    assert (
        visualization.plot_input_projection_comparison
        is diagnostics.plot_input_projection_comparison
    )
    assert (
        visualization.write_input_projection_diagnostics
        is diagnostics.write_input_projection_diagnostics
    )


def test_saved_run_view_is_visualization_api():
    """Saved-run view is exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    saved_run = importlib.import_module("pynamit.visualization.saved_run")

    assert visualization.SavedRunView is saved_run.SavedRunView


def test_panel_figure_specs_are_visualization_api():
    """Panel and publication-script primitives are exported lazily."""
    visualization = importlib.import_module("pynamit.visualization")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    figure_builder = importlib.import_module("pynamit.visualization.figure_builder")
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    assert visualization.PynamitFigureSpec is figure_specs.PynamitFigureSpec
    assert visualization.render_pynamit_figure is figure_builder.render_pynamit_figure
    assert visualization.save_pynamit_movie is figure_builder.save_pynamit_movie
    assert visualization.SavedCoefficientFieldView is run_fields.SavedCoefficientFieldView


def test_saved_field_view_loads_projected_input_package_without_output(tmp_path):
    """Projection packages should be inspectable before a run exists."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

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

    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)

    boundary_Br_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_Br"
    ].coefficient_shape
    simulation.set_boundary_Br(boundary_Br_coefficients=np.zeros(boundary_Br_shape), time=0.0)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)

    assert view.has_model_output is False
    assert view.n_time == 1
    assert "dynamic" not in view.run_view.datasets
    assert {"boundary_Br", "boundary_jr", "conductance"}.issubset(view.run_view.datasets)
    assert view.output_evaluator is None
    assert view.run_view.geometry is None
    assert view.output_evaluation_context is None
    assert view.sheet_current_maps is None
    assert view.input_evaluators["boundary_jr"] is view.input_evaluators["boundary_Br"]
    assert view.input_evaluators["u"] is None
    assert view.input_evaluators["Q_eff"] is None
    assert view.input_evaluators["E_neutral_wind"] is None


def test_saved_field_view_loads_without_boundary_br(tmp_path):
    """Ordinary runs without RM/Br artifacts can be inspected."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

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
    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    assert view.run_view.geometry is None
    assert view.output_evaluation_context is None
    assert view.sheet_current_maps is None
    with pytest.raises(ValueError, match="Unknown output fields"):
        view.solution_comparison_grid_fields(0, field_names={"not-a-field"})
    assert view.run_view.geometry is None

    fields = view.solution_comparison_grid_fields(0, field_names={"Br"})
    input_fields = view.input_grid_fields(0)

    assert view.has_model_output
    assert view.run_view.geometry is not None
    assert view.output_evaluation_context is not None
    assert view.sheet_current_maps is None
    assert "boundary_Br" not in view.run_view.datasets
    assert set(view.available_inputs) == {"boundary_jr", "conductance"}
    assert set(fields) == {"Br_dynamic", "Br_equilibrium"}
    assert fields["Br_dynamic"].shape == view.lat.shape
    assert np.all(np.isnan(input_fields["Br"]))


def test_saved_output_joule_uses_pedersen_dissipation():
    """Saved-output Joule heating follows the Pedersen closure."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    class IdentityEvaluator:
        scalar_coeffs_to_grid = np.eye(2)
        scalar_coeffs_to_grid_operator = np.eye(2)

        @staticmethod
        def synthesize_helmholtz(coeffs):
            return np.asarray(coeffs)[0], np.asarray(coeffs)[1]

    output = xr.Dataset(
        {
            "SH_induced_Br": (("time", "coefficient"), [[1.0, 2.0]]),
            "SH_boundary_jr": (("time", "coefficient"), [[3.0, 4.0]]),
            "SH_Phi": (("time", "coefficient"), [[5.0, 6.0]]),
            "SH_W": (("time", "coefficient"), [[7.0, 8.0]]),
        },
        coords={"time": [0.0]},
    )
    eta_p = np.array([2.0, 3.0])
    conductance = xr.Dataset(
        {
            "SH_log_conductance_magnitude": (
                ("time", "coefficient"),
                [np.log(1.0 / (np.sqrt(2.0) * eta_p))],
            ),
            "SH_log_hall_to_pedersen_ratio": (("time", "coefficient"), [np.zeros(2)]),
        },
        coords={"time": [0.0]},
    )
    zero_map = np.zeros((2, 2))
    output_evaluation_context = {
        "RI": 2.0,
        "induced_Br_to_Br": zero_map,
        "boundary_jr_to_jr": zero_map,
        "induced_Br_to_Jeq": zero_map,
        "pedersen_geometry": np.moveaxis(
            np.array([[[2.0, 1.0], [1.0, 3.0]], [[4.0, 0.0], [0.0, 5.0]]]), 0, -1
        ),
    }
    sheet_current_maps = {
        "induced_Br_to_JS": np.array([np.eye(2), np.zeros((2, 2))]),
        "boundary_jr_to_JS": np.array([np.zeros((2, 2)), np.eye(2)]),
        "boundary_Br_to_JS": None,
    }

    fields = run_fields.compute_solution_comparison_fields_at_index(
        0,
        {"dynamic": output, "conductance": conductance},
        IdentityEvaluator(),
        IdentityEvaluator(),
        output_evaluation_context,
        sheet_current_maps,
    )["dynamic"]

    np.testing.assert_allclose(fields["joule"], [70.0, 288.0])

    equilibrium_fields = run_fields.compute_solution_comparison_fields_at_index(
        0,
        {"equilibrium": output, "conductance": conductance},
        IdentityEvaluator(),
        IdentityEvaluator(),
        output_evaluation_context,
        sheet_current_maps,
    )["equilibrium"]

    np.testing.assert_allclose(equilibrium_fields["joule"], fields["joule"])


def test_saved_field_view_supports_equilibrium_only_output(tmp_path):
    """An equilibrium-only run remains visualizable."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

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
    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)
    simulation.run_data.artifact_store.remove_artifact("dynamic")

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    fields = view.solution_comparison_grid_fields(0)

    assert view.has_model_output
    assert "dynamic" not in view.run_view.datasets
    assert "equilibrium" in view.run_view.datasets
    assert view.run_view.geometry is not None
    assert "Br_equilibrium" in fields
    assert "Br_dynamic" not in fields


def test_saved_field_view_aligns_inputs_by_time_not_index(tmp_path):
    """Sparse outputs should use dense inputs at matching times."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * 6381e3,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    br_shape = simulation.run_data.schema.input_field_spaces["boundary_Br"].coefficient_shape
    br_coefficients = np.zeros((3, *br_shape))
    br_coefficients[0] = 1.0
    br_coefficients[1] = 2.0
    br_coefficients[2] = 3.0
    simulation.set_boundary_Br(
        boundary_Br_coefficients=br_coefficients, time=np.array([0.0, 10.0, 20.0])
    )
    xr.Dataset(coords={"time": np.array([0.0, 20.0])}).to_netcdf(tmp_path / "dynamic.ncdf")

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    fields = view.input_grid_fields(1)
    expected = (
        view.input_evaluators["boundary_Br"]
        .scalar_coeffs_to_grid.dot(br_coefficients[2])
        .reshape(view.lat.shape)
    )

    assert view.n_time == 2
    assert view.run_view.datasets["boundary_Br"].sizes["time"] == 3
    np.testing.assert_allclose(fields["Br"], expected)


def test_saved_field_view_inspects_neutral_wind_electric_field_input(tmp_path):
    """Projected neutral-wind E packages should be inspectable."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    coeff_length = simulation.run_data.schema.input_field_spaces["E_neutral_wind"].index_length
    cf_coeffs = np.zeros(coeff_length)
    df_coeffs = np.zeros(coeff_length)
    cf_coeffs[0] = 1.0e-3
    simulation.set_E_neutral_wind(
        E_neutral_wind_cf=cf_coeffs, E_neutral_wind_df=df_coeffs, time=0.0
    )

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)
    fields = view.input_grid_fields(0)

    assert view.available_inputs == ("E_neutral_wind",)
    assert fields["E_neutral_wind_theta"].shape == view.wind_lat.shape
    assert fields["E_neutral_wind_phi"].shape == view.wind_lat.shape
    assert np.any(np.isfinite(fields["E_neutral_wind_theta"]))


def test_saved_field_view_keeps_model_and_geographic_evaluation_grids_separate(tmp_path):
    """Geographic maps must not replace the magnetic hemisphere grid."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="dipole",
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
    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)
    simulation.impose_equilibrium(time=0.0, save=True, quiet=True)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path, nlat=6, nlon=8)
    geographic = view._get_geographic_evaluation()
    geographic_output_evaluator = view._geographic_output_evaluator(geographic)
    expected_lat, expected_lon = view.run_view.main_field.geo_to_model_coordinates(
        view.lat, view.lon, event_time=view._fallback_start_time()
    )

    np.testing.assert_allclose(view.output_evaluator.grid.lat, view.lat.reshape(-1))
    np.testing.assert_allclose(view.output_evaluator.grid.lon, view.lon.reshape(-1))
    np.testing.assert_allclose(geographic_output_evaluator.grid.lat, expected_lat.reshape(-1))
    np.testing.assert_allclose(geographic_output_evaluator.grid.lon, expected_lon.reshape(-1))
    assert geographic_output_evaluator.grid != view.output_evaluator.grid
    assert view._get_geographic_evaluation() is geographic
    assert view.geographic_map_context() == run_fields.MapCoordinateContext.geographic(
        pd.Timestamp(view._fallback_start_time()).to_pydatetime()
    )


def test_saved_field_view_reuses_earth_fixed_geographic_mapping(tmp_path):
    """Kaiju model and display geometry stay fixed in GEO."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")
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
    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(
        boundary_jr_coefficients=np.zeros((2, *boundary_jr_shape)), time=np.array([0.0, 3600.0])
    )

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path, nlat=6, nlon=8)
    first_time = view.timestamp_at_index(0)
    last_time = view.timestamp_at_index(1)
    first = view._get_geographic_evaluation(first_time)
    last = view._get_geographic_evaluation(last_time)

    assert first is last
    np.testing.assert_allclose(first.scalar_grid.lat, last.scalar_grid.lat)
    np.testing.assert_allclose(first.scalar_grid.lon, last.scalar_grid.lon)
    assert view._get_geographic_evaluation(last_time) is last
    assert (
        view.model_map_context(first_time).noon_longitude
        != view.model_map_context(last_time).noon_longitude
    )


def test_kaiju_hemisphere_plot_coordinates_are_magnetic(tmp_path):
    """Kaiju polar plots rotate GEO samples into MAG and use MLT."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")
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
    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path, nlat=6, nlon=8)
    magnetic_latitude, magnetic_longitude = view.magnetic_plot_coordinates()
    expected = view.run_view.main_field.geographic_to_magnetic_coordinates(view.lat, view.lon)
    timestamp = view.timestamp_at_index(0)
    context = view.magnetic_map_context(timestamp)

    np.testing.assert_allclose(magnetic_latitude, expected[0])
    np.testing.assert_allclose(magnetic_longitude, expected[1])
    assert not np.allclose(magnetic_latitude, view.lat)
    assert context.longitude_kind == "magnetic"
    assert context.local_time_kind == "magnetic"
    assert context.noon_longitude == pytest.approx(
        view.run_view.main_field.magnetic_noon_longitude(pd.Timestamp(timestamp).to_pydatetime())
    )


def test_geographic_input_vectors_are_rotated_to_geographic_components(tmp_path):
    """Global quivers use geographic tangent-vector components."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    wind_shape = simulation.run_data.schema.input_field_spaces["u"].coefficient_shape
    u_cf = np.linspace(0.0, 1.0, wind_shape[1])
    u_df = np.linspace(1.0, 0.0, wind_shape[1])
    simulation.set_neutral_wind(u_cf=u_cf, u_df=u_df, time=0.0)

    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path, wind_nlat=5, wind_nlon=7)
    fields = view.input_grid_fields(0, coordinate_system="geographic")
    evaluation = view._get_geographic_evaluation()
    coefficients = view.dataset_values("u", "u")[0]
    model_theta, model_phi = evaluation.input_evaluators["u"].synthesize_helmholtz(coefficients)
    _, _, expected_east, expected_north = view.run_view.main_field.model_to_geo_coordinates(
        evaluation.vector_grid.lat.reshape(view.wind_lat.shape),
        evaluation.vector_grid.lon.reshape(view.wind_lon.shape),
        model_phi.reshape(view.wind_lat.shape),
        -model_theta.reshape(view.wind_lat.shape),
    )

    np.testing.assert_allclose(fields["wind_phi"], expected_east.reshape(view.wind_lat.shape))
    np.testing.assert_allclose(fields["wind_theta"], -expected_north.reshape(view.wind_lat.shape))


def test_saved_field_view_rejects_unknown_display_coordinate_system(tmp_path):
    """Display-coordinate selection should fail explicitly on typos."""
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    simulation = pynamit.Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )
    boundary_jr_shape = simulation.run_data.schema.input_field_spaces[
        "boundary_jr"
    ].coefficient_shape
    simulation.set_boundary_jr(boundary_jr_coefficients=np.zeros(boundary_jr_shape), time=0.0)
    view = run_fields.SavedCoefficientFieldView.from_directory(tmp_path)

    with pytest.raises(ValueError, match="coordinate_system"):
        view.input_grid_fields(0, coordinate_system="geomagnetic-ish")


@pytest.mark.parametrize(
    ("plot_type", "expected_coordinate_system"),
    [("global", "geographic"), ("hemispheres", "model")],
)
def test_field_renderer_selects_coordinates_for_map_type(
    monkeypatch, plot_type, expected_coordinate_system
):
    """Global maps are geographic while hemispheres stay magnetic."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.visualization.field_comparison_figures")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    map_coordinates = importlib.import_module("pynamit.visualization.map_coordinates")

    class CapturingView:
        lat, lon = np.meshgrid(np.linspace(-80.0, 80.0, 4), np.linspace(-180.0, 180.0, 5))
        run_view = type("RunView", (), {"datasets": {"dynamic": object()}})()
        coordinate_system = None

        def solution_comparison_grid_fields(self, index, *, field_names, coordinate_system):
            del index, field_names
            self.coordinate_system = coordinate_system
            return {"Br_dynamic": np.zeros(self.lat.shape)}

        @staticmethod
        def timestamp_at_index(index):
            return pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=index)

        @staticmethod
        def geographic_map_context(reference_time=None):
            return map_coordinates.MapCoordinateContext.from_noon_longitude(
                30.0,
                longitude_kind="geographic",
                local_time_kind="solar",
                reference_time=reference_time,
            )

        @classmethod
        def magnetic_plot_coordinates(cls):
            return cls.lat, cls.lon

        magnetic_map_context = geographic_map_context
        model_map_context = geographic_map_context

    monkeypatch.setattr(
        figures, "_draw_field_comparison_artists", lambda *args, **kwargs: ([], None, None)
    )
    view = CapturingView()
    spec = figure_specs.PynamitFigureSpec(
        plot_type=plot_type, show_inductive=True, show_noninductive=False, show_difference=False
    )

    figure = figures.FieldComparisonRenderer(spec, view=view).render()
    try:
        assert view.coordinate_system == expected_coordinate_system
        if plot_type == "global":
            geo_axes = [axis for axis in figure.axes if hasattr(axis, "projection")]
            assert len(geo_axes) == 1
            assert geo_axes[0].projection.equals(ccrs.PlateCarree(central_longitude=30.0))
    finally:
        plt.close(figure)


def test_field_renderer_applies_manual_fill_and_line_levels(monkeypatch):
    """Manual controls replace selected main-field presets."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.visualization.field_comparison_figures")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    map_coordinates = importlib.import_module("pynamit.visualization.map_coordinates")

    class View:
        lat, lon = np.meshgrid(np.linspace(-80.0, 80.0, 4), np.linspace(-180.0, 180.0, 5))
        run_view = type("RunView", (), {"datasets": {"equilibrium": object()}})()

        @classmethod
        def solution_comparison_grid_fields(cls, index, *, field_names, coordinate_system):
            del index, field_names, coordinate_system
            return {
                "jr_equilibrium": np.zeros(cls.lat.shape),
                "Phi_equilibrium": np.zeros(cls.lat.shape),
            }

        @staticmethod
        def timestamp_at_index(index):
            return pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=index)

        @staticmethod
        def geographic_map_context(reference_time=None):
            return map_coordinates.MapCoordinateContext.geographic(reference_time)

    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return [], None, None

    monkeypatch.setattr(figures, "_draw_field_comparison_artists", capture)
    spec = figure_specs.PynamitFigureSpec(
        plot_type="global",
        fill="jr",
        lines="Phi",
        show_inductive=False,
        show_noninductive=True,
        show_difference=False,
        manual_color_min=-5e-7,
        manual_color_max=5e-7,
        line_first_abs_level=4.0,
        line_interval=4.0,
        line_levels_per_sign=3,
    )

    figure = figures.FieldComparisonRenderer(spec, view=View()).render()
    try:
        np.testing.assert_allclose(
            captured["plot_kwargs"]["jr"]["levels"], np.linspace(-5e-7, 5e-7, 18)
        )
        np.testing.assert_allclose(
            captured["plot_kwargs"]["Phi"]["levels"], [-12.0, -8.0, -4.0, 4.0, 8.0, 12.0]
        )
    finally:
        plt.close(figure)


def test_hemisphere_renderer_uses_cutoff_and_writes_coordinate_labels(monkeypatch):
    """Polar axes use the requested edge and show MLT orientation."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.visualization.field_comparison_figures")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    captured = []

    class FakePolar:
        def __init__(self, axis, min_abs_latitude):
            self.ax = axis
            self.min_abs_latitude = min_abs_latitude
            self.lat_labels = []
            self.lt_labels = 0

        def writeLATlabels(self, **kwargs):
            self.lat_labels.append(kwargs)

        def writeLTlabels(self):
            self.lt_labels += 1

    def fake_polar(axis, min_abs_latitude):
        polar = FakePolar(axis, min_abs_latitude)
        captured.append(polar)
        return polar

    monkeypatch.setattr(figures, "make_hemisphere_polarplot", fake_polar)
    renderer = object.__new__(figures.FieldComparisonRenderer)
    renderer.spec = figure_specs.PynamitFigureSpec(
        plot_type="hemispheres",
        show_north=True,
        show_south=False,
        hemisphere_min_abs_latitude=42.0,
    )

    figure, _, _ = renderer._create_hemisphere_axes([("dynamic", "Inductive")], 1)
    try:
        assert len(captured) == 1
        assert captured[0].min_abs_latitude == 42.0
        assert captured[0].lat_labels == [
            {"color": "black", "backgroundcolor": (0, 0, 0, 0), "north": True}
        ]
        assert captured[0].lt_labels == 1
    finally:
        plt.close(figure)


def test_line_legend_omits_unused_difference_interval():
    """A single-field plot should not advertise difference contours."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.visualization.field_comparison_figures")
    styles = importlib.import_module("pynamit.visualization.figure_styles")
    figure, axis = plt.subplots()
    try:
        figures.FieldComparisonRenderer._draw_map_line_legend(
            axis,
            ["Phi"],
            styles.FIELD_PLOT_KWARGS,
            styles.FIELD_DIFF_KWARGS,
            include_difference=False,
        )
        labels = [text.get_text() for text in axis.texts]
        assert len(labels) == 1
        assert "8 kV" in labels[0]
        assert "diff" not in labels[0]
    finally:
        plt.close(figure)


def test_input_summary_keeps_polar_jr_model_aligned(monkeypatch):
    """Mixed input figures request both model and geographic fields."""
    import matplotlib.pyplot as plt

    figures = importlib.import_module("pynamit.visualization.input_driver_figures")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    map_coordinates = importlib.import_module("pynamit.visualization.map_coordinates")

    class CapturingView:
        lat, lon = np.meshgrid(np.linspace(-80.0, 80.0, 4), np.linspace(-180.0, 180.0, 5))
        wind_lat, wind_lon = lat, lon

        def __init__(self):
            self.coordinate_systems = []

        def input_grid_fields(self, index, *, coordinate_system):
            del index
            self.coordinate_systems.append(coordinate_system)
            shape = self.lat.shape
            return {
                "jr": np.zeros(shape),
                "Br": np.zeros(shape),
                "sigmaP": np.ones(shape),
                "sigmaH": np.ones(shape),
                "wind_theta": np.zeros(shape),
                "wind_phi": np.zeros(shape),
                "Q_eff_theta": np.full(shape, np.nan),
                "Q_eff_phi": np.full(shape, np.nan),
                "E_neutral_wind_theta": np.full(shape, np.nan),
                "E_neutral_wind_phi": np.full(shape, np.nan),
            }

        @staticmethod
        def timestamp_at_index(index):
            return np.datetime64("2020-01-01") + np.timedelta64(index, "s")

        @staticmethod
        def geographic_map_context(reference_time=None):
            return map_coordinates.MapCoordinateContext.from_noon_longitude(
                30.0,
                longitude_kind="geographic",
                local_time_kind="solar",
                reference_time=reference_time,
            )

        @classmethod
        def magnetic_plot_coordinates(cls):
            return cls.lat, cls.lon

        magnetic_map_context = geographic_map_context
        model_map_context = geographic_map_context

    monkeypatch.setattr(figures.InputDriverRenderer, "_draw_jr_hemispheres", lambda *args: None)
    monkeypatch.setattr(
        figures.InputDriverRenderer, "_draw_global_scalars", lambda *args: (None, None)
    )
    monkeypatch.setattr(figures.InputDriverRenderer, "_draw_tangential_source", lambda *args: None)
    monkeypatch.setattr(figures.InputDriverRenderer, "_draw_colorbars", lambda *args: None)
    view = CapturingView()
    spec = figure_specs.PynamitFigureSpec(plot_type="input_summary")

    figure = figures.InputDriverRenderer(spec, view=view).render()
    try:
        assert view.coordinate_systems == ["model", "geographic"]
        geo_axes = [axis for axis in figure.axes if hasattr(axis, "projection")]
        assert len(geo_axes) == 4
        assert all(
            axis.projection.equals(ccrs.PlateCarree(central_longitude=30.0)) for axis in geo_axes
        )
    finally:
        plt.close(figure)


def test_publication_script_export_is_jupyter_friendly():
    """Figure specs can produce editable publication scripts."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    spec = figure_specs.PynamitFigureSpec(run_directory="run", plot_type="global")
    script = figure_specs.publication_script_for_spec(spec, output_path="figures/test.png")

    assert script.startswith("# %%")
    assert "render_pynamit_figure" in script
    assert '"run_directory": "run"' in script
    assert "fig.savefig" in script


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_index": -1},
        {"time_range": (0.5, 2)},
        {"dbdt_window_points": 1.5},
        {"color_scale_percentile": np.nan},
        {"manual_color_min": -1.0},
        {"manual_color_min": 1.0, "manual_color_max": -1.0},
        {"line_first_abs_level": 1.0},
        {"line_first_abs_level": 1.0, "line_interval": 0.0, "line_levels_per_sign": 2},
        {"sim_time_offset_seconds": np.inf},
    ],
)
def test_figure_spec_rejects_invalid_renderer_values(kwargs):
    """Figure specs reject values renderers cannot interpret."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    with pytest.raises(ValueError):
        figure_specs.PynamitFigureSpec(**kwargs)


def test_figure_spec_migrates_fixed_color_scale_to_manual():
    """Saved fixed-mode specs retain their preset semantics."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    spec = figure_specs.PynamitFigureSpec(color_scale_mode="fixed")

    assert spec.color_scale_mode == "manual"


def test_figure_spec_preserves_removed_options_as_extra_metadata():
    """Old configuration files retain a removed option as metadata."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    spec = figure_specs.PynamitFigureSpec.from_dict({"conductance_overlay": "hall"})

    assert spec.extra == {"conductance_overlay": "hall"}


def test_run_plot_defaults_are_applied(tmp_path):
    """Run directories can carry plotting defaults."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    (tmp_path / "pynamit_plot_defaults.json").write_text(
        """
        {
          "data_directory": "/tmp/mag_data",
          "plot_defaults": {
            "plot_type": "ground_timeseries",
            "ground_station": "OTT",
            "time_range": [4, 12],
            "extra_frontend_note": "kept"
          }
        }
        """,
        encoding="utf-8",
    )

    spec = figure_specs.figure_spec_from_run_defaults(tmp_path)

    assert spec.run_directory == str(tmp_path)
    assert spec.data_directory == "/tmp/mag_data"
    assert spec.plot_type == "ground_timeseries"
    assert spec.ground_station == "OTT"
    assert spec.time_range == (4, 12)
    assert spec.extra["extra_frontend_note"] == "kept"


def test_saved_field_view_cache_replaces_stale_run_version(tmp_path, monkeypatch):
    """Live-run updates replace rather than accumulate cached views."""
    figure_context = importlib.import_module("pynamit.visualization.figure_context")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    fingerprint = [("dynamic", 1)]
    views = iter((object(), object()))
    monkeypatch.setattr(
        figure_context, "_artifact_fingerprint", lambda _directory: tuple(fingerprint)
    )
    monkeypatch.setattr(
        figure_context.SavedCoefficientFieldView,
        "from_directory",
        staticmethod(lambda _directory: next(views)),
    )
    figure_context.clear_saved_field_view_cache()
    spec = figure_specs.PynamitFigureSpec(run_directory=str(tmp_path))

    first = figure_context.get_saved_field_view(spec)
    assert figure_context.get_saved_field_view(spec) is first
    fingerprint[0] = ("dynamic", 2)
    second = figure_context.get_saved_field_view(spec)

    assert second is not first
    assert len(figure_context._VIEW_CACHE) == 1


def test_saved_field_view_fingerprint_detects_nested_store_changes(tmp_path):
    """In-place Zarr chunk additions invalidate a saved field view."""
    figure_context = importlib.import_module("pynamit.visualization.figure_context")
    chunk_directory = tmp_path / "dynamic.zarr" / "SH_induced_Br"
    chunk_directory.mkdir(parents=True)
    (chunk_directory / "0").write_bytes(b"first")

    before = figure_context._artifact_fingerprint(tmp_path)
    (chunk_directory / "1").write_bytes(b"second")
    after = figure_context._artifact_fingerprint(tmp_path)

    assert after != before


def test_panel_app_module_imports_when_panel_is_installed():
    """Panel app construction API is importable when Panel exists."""
    pytest.importorskip("panel")
    visualization = importlib.import_module("pynamit.visualization")
    panel_app = importlib.import_module("pynamit.visualization.panel_app")

    assert visualization.PynamitPanelApp is panel_app.PynamitPanelApp
    assert callable(panel_app.build_pynamit_panel_app)
