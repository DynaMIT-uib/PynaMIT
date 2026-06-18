"""Tests for visualization package exports."""

import importlib

import pynamit
import pytest


def test_visualization_package_exports_building_blocks():
    """Public visualization exports are reusable building blocks."""
    visualization = importlib.import_module("pynamit.visualization")
    state_fields = importlib.import_module("pynamit.visualization.state_fields")

    assert visualization.evaluate_Br is state_fields.evaluate_Br
    assert visualization.evaluate_jr is state_fields.evaluate_jr
    assert visualization.evaluate_JS is state_fields.evaluate_JS


def test_pynameye_is_available_from_visualization():
    """PynamEye is available from the canonical visualization path."""
    visualization_pynameye = importlib.import_module("pynamit.visualization.pynameye")

    assert pynamit.PynamEye is visualization_pynameye.PynamEye
    assert hasattr(pynamit.PynamEye, "style_global_axis")


def test_input_projection_comparison_recipe_is_importable():
    """Input diagnostic recipes are available from their module."""
    diagnostics = importlib.import_module("pynamit.visualization.input_projection_comparison")

    assert callable(diagnostics.plot_input_projection_comparison)


def test_map_coordinate_context_is_visualization_api():
    """Map coordinate context is exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    coordinates = importlib.import_module("pynamit.visualization.map_coordinates")

    assert visualization.MapCoordinateContext is coordinates.MapCoordinateContext


def test_projected_input_inspector_is_visualization_api():
    """Projected-input inspection is exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    input_projection = importlib.import_module("pynamit.visualization.input_projection")

    assert visualization.evaluate_projected_input is input_projection.evaluate_projected_input
    assert pynamit.evaluate_projected_input is input_projection.evaluate_projected_input


def test_saved_run_view_is_visualization_api():
    """Saved-run view is exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    saved_run = importlib.import_module("pynamit.visualization.saved_run")

    assert visualization.SavedRunView is saved_run.SavedRunView


def test_field_maps_are_visualization_api():
    """Shared field-map helpers are exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    field_maps = importlib.import_module("pynamit.visualization.field_maps")

    assert (
        visualization.evaluate_conductance_coefficients
        is field_maps.evaluate_conductance_coefficients
    )
    assert visualization.evaluate_wind_coefficients is field_maps.evaluate_wind_coefficients
    assert (
        visualization.evaluate_joule_from_coefficients
        is field_maps.evaluate_joule_from_coefficients
    )
    assert visualization.evaluate_JS_from_maps is field_maps.evaluate_JS_from_maps


def test_plot_setup_helpers_are_visualization_api():
    """Shared plot setup helpers are exported from visualization."""
    visualization = importlib.import_module("pynamit.visualization")
    plot_helpers = importlib.import_module("pynamit.visualization.plot_helpers")

    assert visualization.build_percentile_color_scale is plot_helpers.build_percentile_color_scale
    assert visualization.draw_line_contour_legend is plot_helpers.draw_line_contour_legend
    assert (
        visualization.suppress_empty_contour_warnings
        is plot_helpers.suppress_empty_contour_warnings
    )


def test_map_curve_and_hemisphere_helpers_are_visualization_api():
    """Notebook-independent map primitives are exported."""
    visualization = importlib.import_module("pynamit.visualization")
    hemisphere = importlib.import_module("pynamit.visualization.hemisphere")
    map_curves = importlib.import_module("pynamit.visualization.map_curves")
    map_panels = importlib.import_module("pynamit.visualization.map_panels")

    assert visualization.hemisphere_masks_for_latitude is hemisphere.hemisphere_masks_for_latitude
    assert visualization.build_even_global_sites is map_curves.build_even_global_sites
    assert visualization.build_timeseries_curve_layers is map_curves.build_timeseries_curve_layers
    assert visualization.split_wrapped_curve is map_curves.split_wrapped_curve
    assert visualization.draw_curve_scale_inset is map_curves.draw_curve_scale_inset
    assert visualization.draw_field_comparison_artists is map_panels.draw_field_comparison_artists


def test_time_series_helpers_are_visualization_api():
    """Notebook-independent time-series primitives are exported."""
    visualization = importlib.import_module("pynamit.visualization")
    time_series = importlib.import_module("pynamit.visualization.time_series")

    assert visualization.resample_series_to_times is time_series.resample_series_to_times
    assert (
        visualization.compute_centered_difference_series_at_times
        is time_series.compute_centered_difference_series_at_times
    )
    assert (
        visualization.compute_time_derivative_matrix is time_series.compute_time_derivative_matrix
    )
    assert visualization.prominent_peak_candidates is time_series.prominent_peak_candidates
    assert (
        visualization.first_event_peak_abs_value_and_time
        is time_series.first_event_peak_abs_value_and_time
    )
    assert (
        visualization.vector_magnitude_from_component_series
        is time_series.vector_magnitude_from_component_series
    )


def test_station_data_helpers_are_visualization_api():
    """Ground-station data primitives are exported."""
    visualization = importlib.import_module("pynamit.visualization")
    station_data = importlib.import_module("pynamit.visualization.station_data")

    assert visualization.normalize_station_metadata is station_data.normalize_station_metadata
    assert (
        visualization.load_iaga2002_magnetometer_data
        is station_data.load_iaga2002_magnetometer_data
    )
    assert (
        visualization.station_window_has_nonzero_measurements
        is station_data.station_window_has_nonzero_measurements
    )


def test_panel_figure_specs_are_visualization_api():
    """Panel and publication-script primitives are exported lazily."""
    visualization = importlib.import_module("pynamit.visualization")
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")
    figure_builder = importlib.import_module("pynamit.visualization.figure_builder")
    run_fields = importlib.import_module("pynamit.visualization.run_fields")

    assert visualization.PynamitFigureSpec is figure_specs.PynamitFigureSpec
    assert (
        visualization.figure_spec_from_run_defaults is figure_specs.figure_spec_from_run_defaults
    )
    assert visualization.find_run_plot_defaults is figure_specs.find_run_plot_defaults
    assert visualization.load_run_plot_defaults is figure_specs.load_run_plot_defaults
    assert visualization.publication_script_for_spec is figure_specs.publication_script_for_spec
    assert visualization.render_pynamit_figure is figure_builder.render_pynamit_figure
    assert visualization.SavedCoefficientFieldView is run_fields.SavedCoefficientFieldView


def test_publication_script_export_is_jupyter_friendly():
    """Figure specs can produce editable publication scripts."""
    figure_specs = importlib.import_module("pynamit.visualization.figure_specs")

    spec = figure_specs.PynamitFigureSpec(run_directory="run", plot_type="global")
    script = figure_specs.publication_script_for_spec(spec, output_path="figures/test.png")

    assert script.startswith("# %%")
    assert "render_pynamit_figure" in script
    assert '"run_directory": "run"' in script
    assert "fig.savefig" in script


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


def test_panel_app_module_imports_when_panel_is_installed():
    """Panel app construction API is importable when Panel exists."""
    pytest.importorskip("panel")
    visualization = importlib.import_module("pynamit.visualization")
    panel_app = importlib.import_module("pynamit.visualization.panel_app")

    assert visualization.PynamitPanelApp is panel_app.PynamitPanelApp
    assert callable(panel_app.build_pynamit_panel_app)
