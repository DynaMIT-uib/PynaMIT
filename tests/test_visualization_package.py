"""Tests for visualization package exports."""

import importlib

import pynamit


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
    assert (
        visualization.suppress_empty_contour_warnings
        is plot_helpers.suppress_empty_contour_warnings
    )


def test_map_curve_and_hemisphere_helpers_are_visualization_api():
    """Notebook-independent map primitives are exported."""
    visualization = importlib.import_module("pynamit.visualization")
    hemisphere = importlib.import_module("pynamit.visualization.hemisphere")
    map_curves = importlib.import_module("pynamit.visualization.map_curves")

    assert visualization.hemisphere_masks_for_latitude is hemisphere.hemisphere_masks_for_latitude
    assert visualization.build_even_global_sites is map_curves.build_even_global_sites
    assert visualization.split_wrapped_curve is map_curves.split_wrapped_curve


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
    assert (
        visualization.vector_magnitude_from_component_series
        is time_series.vector_magnitude_from_component_series
    )
