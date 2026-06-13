"""Visualization helpers for PynaMIT simulation results."""

_LAZY_EXPORTS = {
    "add_panel_label": ("pynamit.visualization.plot_helpers", "add_panel_label"),
    "apply_local_time_grid_labels": (
        "pynamit.visualization.local_time",
        "apply_local_time_grid_labels",
    ),
    "artifact_path": ("pynamit.visualization.artifacts", "artifact_path"),
    "build_evaluator": ("pynamit.visualization.grid_evaluation", "build_evaluator"),
    "build_plot_grid": ("pynamit.visualization.grid_evaluation", "build_plot_grid"),
    "build_sheet_current_operators": (
        "pynamit.visualization.grid_evaluation",
        "build_sheet_current_operators",
    ),
    "compute_conversion_factors": (
        "pynamit.visualization.grid_evaluation",
        "compute_conversion_factors",
    ),
    "contour_kwargs_for_display": (
        "pynamit.visualization.plot_helpers",
        "contour_kwargs_for_display",
    ),
    "datetime_to_utc_hours": (
        "pynamit.visualization.local_time",
        "datetime_to_utc_hours",
    ),
    "evaluate_Br": ("pynamit.visualization.state_fields", "evaluate_Br"),
    "evaluate_Br_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_Br_coefficients",
    ),
    "evaluate_Phi": ("pynamit.visualization.state_fields", "evaluate_Phi"),
    "evaluate_Phi_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_Phi_coefficients",
    ),
    "evaluate_W": ("pynamit.visualization.state_fields", "evaluate_W"),
    "evaluate_W_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_W_coefficients",
    ),
    "evaluate_equivalent_current_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_equivalent_current_coefficients",
    ),
    "evaluate_equivalent_current_function": (
        "pynamit.visualization.state_fields",
        "evaluate_equivalent_current_function",
    ),
    "evaluate_jr": ("pynamit.visualization.state_fields", "evaluate_jr"),
    "evaluate_jr_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_jr_coefficients",
    ),
    "evaluate_sheet_current": (
        "pynamit.visualization.state_fields",
        "evaluate_sheet_current",
    ),
    "evaluate_sheet_current_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_sheet_current_coefficients",
    ),
    "format_contour_interval": (
        "pynamit.visualization.plot_helpers",
        "format_contour_interval",
    ),
    "format_local_time_longitude_label": (
        "pynamit.visualization.local_time",
        "format_local_time_longitude_label",
    ),
    "get_ticks_from_levels": (
        "pynamit.visualization.plot_helpers",
        "get_ticks_from_levels",
    ),
    "load_dataarray_artifact": (
        "pynamit.visualization.artifacts",
        "load_dataarray_artifact",
    ),
    "load_dataset_artifact": (
        "pynamit.visualization.artifacts",
        "load_dataset_artifact",
    ),
    "load_settings_and_basis": (
        "pynamit.visualization.grid_evaluation",
        "load_settings_and_basis",
    ),
    "local_noon_longitude": (
        "pynamit.visualization.local_time",
        "local_noon_longitude",
    ),
    "local_time_grid_longitudes": (
        "pynamit.visualization.local_time",
        "local_time_grid_longitudes",
    ),
    "local_time_hours_to_longitude": (
        "pynamit.visualization.local_time",
        "local_time_hours_to_longitude",
    ),
    "local_time_longitude_to_geographic": (
        "pynamit.visualization.local_time",
        "local_time_longitude_to_geographic",
    ),
    "longitude_to_local_time_from_noon_longitude": (
        "pynamit.visualization.local_time",
        "longitude_to_local_time_from_noon_longitude",
    ),
    "longitude_to_local_time_hours": (
        "pynamit.visualization.local_time",
        "longitude_to_local_time_hours",
    ),
    "make_local_time_longitude_formatter": (
        "pynamit.visualization.local_time",
        "make_local_time_longitude_formatter",
    ),
    "MapCoordinateContext": (
        "pynamit.visualization.map_coordinates",
        "MapCoordinateContext",
    ),
    "evaluate_projected_input": (
        "pynamit.visualization.input_projection",
        "evaluate_projected_input",
    ),
    "PynamEye": ("pynamit.visualization.pynameye", "PynamEye"),
    "remove_artists": ("pynamit.visualization.plot_helpers", "remove_artists"),
    "resistance_to_conductance": (
        "pynamit.visualization.grid_evaluation",
        "resistance_to_conductance",
    ),
    "resolve_xarray_artifact_path": (
        "pynamit.visualization.artifacts",
        "resolve_xarray_artifact_path",
    ),
    "set_contour_edges_to_face": (
        "pynamit.visualization.plot_helpers",
        "set_contour_edges_to_face",
    ),
    "stabilize_polarplot": (
        "pynamit.visualization.plot_helpers",
        "stabilize_polarplot",
    ),
    "style_global_axis": (
        "pynamit.visualization.plot_helpers",
        "style_global_axis",
    ),
    "style_global_comparison_axis": (
        "pynamit.visualization.plot_helpers",
        "style_global_comparison_axis",
    ),
    "style_global_input_axis": (
        "pynamit.visualization.plot_helpers",
        "style_global_input_axis",
    ),
    "symmetric_contour_levels_without_zero": (
        "pynamit.visualization.plot_helpers",
        "symmetric_contour_levels_without_zero",
    ),
    "wrap_longitude_180": (
        "pynamit.visualization.local_time",
        "wrap_longitude_180",
    ),
    "xarray_artifact_exists": (
        "pynamit.visualization.artifacts",
        "xarray_artifact_exists",
    ),
}


def __getattr__(name):
    """Load heavier visualization helpers only when requested."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return public visualization attributes including lazy exports."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "evaluate_Br",
    "evaluate_Br_coefficients",
    "evaluate_Phi",
    "evaluate_Phi_coefficients",
    "evaluate_W",
    "evaluate_W_coefficients",
    "evaluate_equivalent_current_coefficients",
    "evaluate_equivalent_current_function",
    "evaluate_jr",
    "evaluate_jr_coefficients",
    "evaluate_sheet_current",
    "evaluate_sheet_current_coefficients",
    "evaluate_projected_input",
    "add_panel_label",
    "apply_local_time_grid_labels",
    "artifact_path",
    "build_evaluator",
    "build_plot_grid",
    "build_sheet_current_operators",
    "compute_conversion_factors",
    "contour_kwargs_for_display",
    "datetime_to_utc_hours",
    "format_contour_interval",
    "format_local_time_longitude_label",
    "get_ticks_from_levels",
    "load_dataarray_artifact",
    "load_dataset_artifact",
    "load_settings_and_basis",
    "local_noon_longitude",
    "local_time_grid_longitudes",
    "local_time_hours_to_longitude",
    "local_time_longitude_to_geographic",
    "longitude_to_local_time_from_noon_longitude",
    "longitude_to_local_time_hours",
    "make_local_time_longitude_formatter",
    "MapCoordinateContext",
    "PynamEye",
    "remove_artists",
    "resistance_to_conductance",
    "resolve_xarray_artifact_path",
    "set_contour_edges_to_face",
    "stabilize_polarplot",
    "style_global_axis",
    "style_global_comparison_axis",
    "style_global_input_axis",
    "symmetric_contour_levels_without_zero",
    "wrap_longitude_180",
    "xarray_artifact_exists",
]
