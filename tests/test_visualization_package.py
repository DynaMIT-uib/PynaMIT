"""Tests for visualization package exports."""

import importlib

import pynamit


def test_visualization_package_exports_building_blocks():
    """Public visualization exports are reusable building blocks."""
    visualization = importlib.import_module("pynamit.visualization")
    state_fields = importlib.import_module("pynamit.visualization.state_fields")

    assert visualization.evaluate_Br is state_fields.evaluate_Br
    assert visualization.evaluate_jr is state_fields.evaluate_jr
    assert visualization.evaluate_sheet_current is state_fields.evaluate_sheet_current
    for name in [
        "debugplot",
        "globalplot",
        "plot_global_polar_map",
        "plot_state_diagnostics",
        "plot_input_vs_interpolated",
    ]:
        assert not hasattr(visualization, name)
        assert not hasattr(pynamit, name)


def test_pynameye_is_available_from_visualization():
    """PynamEye is available from the canonical visualization path."""
    visualization_pynameye = importlib.import_module("pynamit.visualization.pynameye")

    assert pynamit.PynamEye is visualization_pynameye.PynamEye


def test_input_vs_interpolated_is_module_recipe_not_package_api():
    """Input diagnostic recipes stay out of the package-level API."""
    diagnostics = importlib.import_module("pynamit.visualization.input_vs_interpolated")

    assert callable(diagnostics.plot_input_vs_interpolated)
    assert not hasattr(pynamit, "plot_input_vs_interpolated")


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
