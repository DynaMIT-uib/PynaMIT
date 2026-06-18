"""Visualization helpers for PynaMIT simulation results."""

_LAZY_EXPORTS = {
    "add_panel_label": ("pynamit.visualization.plot_helpers", "add_panel_label"),
    "apply_local_time_grid_labels": (
        "pynamit.visualization.local_time",
        "apply_local_time_grid_labels",
    ),
    "artifact_path": ("pynamit.visualization.artifacts", "artifact_path"),
    "build_even_global_sites": ("pynamit.visualization.map_curves", "build_even_global_sites"),
    "build_evaluator": ("pynamit.visualization.grid_evaluation", "build_evaluator"),
    "build_plot_grid": ("pynamit.visualization.grid_evaluation", "build_plot_grid"),
    "build_JS_operators": ("pynamit.visualization.grid_evaluation", "build_JS_operators"),
    "build_percentile_color_scale": (
        "pynamit.visualization.plot_helpers",
        "build_percentile_color_scale",
    ),
    "build_timeseries_curve_layers": (
        "pynamit.visualization.map_curves",
        "build_timeseries_curve_layers",
    ),
    "compute_conversion_factors": (
        "pynamit.visualization.grid_evaluation",
        "compute_conversion_factors",
    ),
    "compute_centered_difference_matrix_at_times": (
        "pynamit.visualization.time_series",
        "compute_centered_difference_matrix_at_times",
    ),
    "compute_centered_difference_series_at_times": (
        "pynamit.visualization.time_series",
        "compute_centered_difference_series_at_times",
    ),
    "compute_time_derivative_matrix": (
        "pynamit.visualization.time_series",
        "compute_time_derivative_matrix",
    ),
    "coerce_hemisphere_min_abs_latitude": (
        "pynamit.visualization.hemisphere",
        "coerce_hemisphere_min_abs_latitude",
    ),
    "contour_kwargs_for_display": (
        "pynamit.visualization.plot_helpers",
        "contour_kwargs_for_display",
    ),
    "curve_layer_zoffset": ("pynamit.visualization.map_curves", "curve_layer_zoffset"),
    "curve_site_group_zorders": ("pynamit.visualization.map_curves", "curve_site_group_zorders"),
    "datetime_to_utc_hours": ("pynamit.visualization.local_time", "datetime_to_utc_hours"),
    "datetime_index_to_epoch_ns": (
        "pynamit.visualization.time_series",
        "datetime_index_to_epoch_ns",
    ),
    "draw_timeseries_curve_map": ("pynamit.visualization.map_curves", "draw_timeseries_curve_map"),
    "draw_field_comparison_artists": (
        "pynamit.visualization.map_panels",
        "draw_field_comparison_artists",
    ),
    "draw_line_contour_legend": ("pynamit.visualization.plot_helpers", "draw_line_contour_legend"),
    "download_and_load_iaga2002_station_data": (
        "pynamit.visualization.station_data",
        "download_and_load_iaga2002_station_data",
    ),
    "evaluate_Br": ("pynamit.visualization.state_fields", "evaluate_Br"),
    "evaluate_Br_coefficients": ("pynamit.visualization.state_fields", "evaluate_Br_coefficients"),
    "evaluate_Phi": ("pynamit.visualization.state_fields", "evaluate_Phi"),
    "evaluate_Phi_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_Phi_coefficients",
    ),
    "evaluate_W": ("pynamit.visualization.state_fields", "evaluate_W"),
    "evaluate_W_coefficients": ("pynamit.visualization.state_fields", "evaluate_W_coefficients"),
    "evaluate_equivalent_current_coefficients": (
        "pynamit.visualization.state_fields",
        "evaluate_equivalent_current_coefficients",
    ),
    "evaluate_equivalent_current_function": (
        "pynamit.visualization.state_fields",
        "evaluate_equivalent_current_function",
    ),
    "evaluate_conductance_coefficients": (
        "pynamit.visualization.field_maps",
        "evaluate_conductance_coefficients",
    ),
    "evaluate_conductance_values": (
        "pynamit.visualization.field_maps",
        "evaluate_conductance_values",
    ),
    "evaluate_electric_field_coefficients": (
        "pynamit.visualization.field_maps",
        "evaluate_electric_field_coefficients",
    ),
    "evaluate_joule_from_coefficients": (
        "pynamit.visualization.field_maps",
        "evaluate_joule_from_coefficients",
    ),
    "evaluate_joule_from_fields": (
        "pynamit.visualization.field_maps",
        "evaluate_joule_from_fields",
    ),
    "evaluate_jr": ("pynamit.visualization.state_fields", "evaluate_jr"),
    "evaluate_jr_coefficients": ("pynamit.visualization.state_fields", "evaluate_jr_coefficients"),
    "evaluate_JS": ("pynamit.visualization.state_fields", "evaluate_JS"),
    "evaluate_JS_coefficients": ("pynamit.visualization.state_fields", "evaluate_JS_coefficients"),
    "evaluate_JS_from_maps": ("pynamit.visualization.field_maps", "evaluate_JS_from_maps"),
    "evaluate_tangential_coefficients": (
        "pynamit.visualization.field_maps",
        "evaluate_tangential_coefficients",
    ),
    "evaluate_wind_coefficients": (
        "pynamit.visualization.field_maps",
        "evaluate_wind_coefficients",
    ),
    "format_contour_interval": ("pynamit.visualization.plot_helpers", "format_contour_interval"),
    "format_local_time_longitude_label": (
        "pynamit.visualization.local_time",
        "format_local_time_longitude_label",
    ),
    "first_event_peak_abs_value_and_time": (
        "pynamit.visualization.time_series",
        "first_event_peak_abs_value_and_time",
    ),
    "get_ticks_from_levels": ("pynamit.visualization.plot_helpers", "get_ticks_from_levels"),
    "get_time_index_median_cadence_seconds": (
        "pynamit.visualization.time_series",
        "get_time_index_median_cadence_seconds",
    ),
    "geographic_local_time_mask": (
        "pynamit.visualization.map_curves",
        "geographic_local_time_mask",
    ),
    "hemisphere_masks_for_latitude": (
        "pynamit.visualization.hemisphere",
        "hemisphere_masks_for_latitude",
    ),
    "interpolate_curve_value_at_normalized_position": (
        "pynamit.visualization.map_curves",
        "interpolate_curve_value_at_normalized_position",
    ),
    "load_dataarray_artifact": ("pynamit.visualization.artifacts", "load_dataarray_artifact"),
    "load_dataset_artifact": ("pynamit.visualization.artifacts", "load_dataset_artifact"),
    "load_settings_and_basis": (
        "pynamit.visualization.grid_evaluation",
        "load_settings_and_basis",
    ),
    "load_iaga2002_magnetometer_data": (
        "pynamit.visualization.station_data",
        "load_iaga2002_magnetometer_data",
    ),
    "local_peak_abs_value_and_time": (
        "pynamit.visualization.time_series",
        "local_peak_abs_value_and_time",
    ),
    "local_noon_longitude": ("pynamit.visualization.local_time", "local_noon_longitude"),
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
    "local_time_window_extent": ("pynamit.visualization.map_curves", "local_time_window_extent"),
    "local_time_window_is_full": ("pynamit.visualization.map_curves", "local_time_window_is_full"),
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
    "make_hemisphere_polarplot": ("pynamit.visualization.hemisphere", "make_hemisphere_polarplot"),
    "MapCoordinateContext": ("pynamit.visualization.map_coordinates", "MapCoordinateContext"),
    "most_prominent_peak_abs_value_and_time": (
        "pynamit.visualization.time_series",
        "most_prominent_peak_abs_value_and_time",
    ),
    "normalize_station_metadata": (
        "pynamit.visualization.station_data",
        "normalize_station_metadata",
    ),
    "evaluate_projected_input": (
        "pynamit.visualization.input_projection",
        "evaluate_projected_input",
    ),
    "prominent_peak_candidates": (
        "pynamit.visualization.time_series",
        "prominent_peak_candidates",
    ),
    "PynamEye": ("pynamit.visualization.pynameye", "PynamEye"),
    "remove_artists": ("pynamit.visualization.plot_helpers", "remove_artists"),
    "resample_matrix_to_times": ("pynamit.visualization.time_series", "resample_matrix_to_times"),
    "resample_series_to_times": ("pynamit.visualization.time_series", "resample_series_to_times"),
    "resistance_to_conductance": (
        "pynamit.visualization.grid_evaluation",
        "resistance_to_conductance",
    ),
    "SavedRunView": ("pynamit.visualization.saved_run", "SavedRunView"),
    "resolve_xarray_artifact_path": (
        "pynamit.visualization.artifacts",
        "resolve_xarray_artifact_path",
    ),
    "set_contour_edges_to_face": (
        "pynamit.visualization.plot_helpers",
        "set_contour_edges_to_face",
    ),
    "shift_station_datetime_index": (
        "pynamit.visualization.station_data",
        "shift_station_datetime_index",
    ),
    "split_wrapped_curve": ("pynamit.visualization.map_curves", "split_wrapped_curve"),
    "stabilize_polarplot": ("pynamit.visualization.plot_helpers", "stabilize_polarplot"),
    "station_component_columns": (
        "pynamit.visualization.station_data",
        "station_component_columns",
    ),
    "station_has_complete_nonzero_components_at_times": (
        "pynamit.visualization.station_data",
        "station_has_complete_nonzero_components_at_times",
    ),
    "station_source_time_window": (
        "pynamit.visualization.station_data",
        "station_source_time_window",
    ),
    "station_window_has_nonzero_measurements": (
        "pynamit.visualization.station_data",
        "station_window_has_nonzero_measurements",
    ),
    "style_global_axis": ("pynamit.visualization.plot_helpers", "style_global_axis"),
    "style_global_comparison_axis": (
        "pynamit.visualization.plot_helpers",
        "style_global_comparison_axis",
    ),
    "style_global_input_axis": ("pynamit.visualization.plot_helpers", "style_global_input_axis"),
    "suppress_empty_contour_warnings": (
        "pynamit.visualization.plot_helpers",
        "suppress_empty_contour_warnings",
    ),
    "symmetric_contour_levels_without_zero": (
        "pynamit.visualization.plot_helpers",
        "symmetric_contour_levels_without_zero",
    ),
    "vector_magnitude_from_component_series": (
        "pynamit.visualization.time_series",
        "vector_magnitude_from_component_series",
    ),
    "vector_magnitude_preserve_shape": (
        "pynamit.visualization.time_series",
        "vector_magnitude_preserve_shape",
    ),
    "wrap_longitude_180": ("pynamit.visualization.local_time", "wrap_longitude_180"),
    "wrap_longitudes": ("pynamit.visualization.map_curves", "wrap_longitudes"),
    "xarray_artifact_exists": ("pynamit.visualization.artifacts", "xarray_artifact_exists"),
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


__all__ = sorted(_LAZY_EXPORTS)
