"""Bind Panel widgets to serializable PynaMIT figure specs."""

from __future__ import annotations

from pynamit.visualization.figure_specs import PynamitFigureSpec


def set_widget_value(widget, value):
    """Set a widget value only when it changed."""
    if value != widget.value:
        widget.value = value


def current_figure_spec(app) -> PynamitFigureSpec:
    """Return the spec described by the current Panel controls."""
    return PynamitFigureSpec(
        run_directory=app.run_directory.value,
        data_directory=app.spec.data_directory,
        plot_type=app.plot_type.value,
        time_index=int(app.time_index.value),
        time_range=tuple(int(value) for value in app.time_range.value),
        fill=app.fill.value,
        lines=app.lines.value,
        show_north=bool(app.show_north.value),
        show_south=bool(app.show_south.value),
        hemisphere_min_abs_latitude=float(app.min_abs_lat.value),
        ground_station=str(app.station.value).upper(),
        ground_component=app.ground_component.value,
        ground_quantity=app.ground_quantity.value,
        include_station_data=bool(app.include_station_data.value),
        show_inductive=bool(app.show_inductive.value),
        show_noninductive=bool(app.show_noninductive.value),
        show_difference=bool(app.show_difference.value),
        show_reference_line=bool(app.show_reference_line.value),
        reference_time_of_day_utc=str(app.reference_time.value),
        show_station_labels=bool(app.spec.show_station_labels),
        sim_time_offset_seconds=float(app.sim_time_offset.value),
        data_time_offset_seconds=float(app.data_time_offset.value),
        dbdt_window_points=max(1, int(app.dbdt_window_points.value)),
        ground_model_lt_count=max(1, int(app.ground_model_lt_count.value)),
        ground_model_lat_count=max(1, int(app.ground_model_lat_count.value)),
        ground_model_visual_even=bool(app.ground_model_visual_even.value),
        show_pedersen_conductance_overlay=bool(app.show_pedersen_conductance_overlay.value),
        show_hall_conductance_overlay=bool(app.show_hall_conductance_overlay.value),
        min_abs_dip_latitude=float(app.low_lat_cutoff.value),
        low_latitude_scale=float(app.low_lat_scale.value),
        show_dip_equator_curve=bool(app.show_dip_equator_curve.value),
        show_low_latitude_curve=bool(app.show_low_lat_curve.value),
        curve_scale_mode=app.curve_scale_mode.value,
        curve_scale_value=float(app.curve_scale.value),
        curve_time_scale=float(app.time_scale.value),
        color_scale_mode=app.color_scale_mode.value,
        color_scale_percentile=float(app.color_scale_percentile.value),
        geo_lat_min=float(app.geo_lat_min.value),
        geo_lat_max=float(app.geo_lat_max.value),
        local_time_min=float(app.local_time_min.value),
        local_time_max=float(app.local_time_max.value),
        zoom_window=bool(app.zoom_window.value),
        movie_filename=str(app.movie_filename.value),
        movie_fps=float(app.movie_fps.value),
        movie_dpi=int(app.spec.movie_dpi),
        extra=dict(app.spec.extra),
    )


def apply_figure_spec_to_widgets(app, spec: PynamitFigureSpec) -> None:
    """Apply one figure spec to the Panel controls."""
    plot_values = set(app.plot_type.options.values())
    fill_values = set(app.fill.options.values())
    line_values = set(app.lines.options.values())
    component_values = set(app.ground_component.options.values())
    quantity_values = set(app.ground_quantity.options.values())
    max_time = int(app.time_index.end)
    time_start, time_end = [int(value) for value in spec.time_range]
    time_start = max(0, min(time_start, max_time))
    time_end = max(time_start, min(time_end, max_time))
    if time_start == 0 and time_end == 0 and max_time > 0:
        time_end = min(max_time, 60)

    set_widget_value(app.run_directory, spec.run_directory)
    set_widget_value(
        app.plot_type, spec.plot_type if spec.plot_type in plot_values else "ground_curve_map"
    )
    set_widget_value(app.time_index, max(0, min(int(spec.time_index), max_time)))
    set_widget_value(app.time_range, (time_start, time_end))
    set_widget_value(app.fill, spec.fill if spec.fill in fill_values else "Br")
    set_widget_value(app.lines, spec.lines if spec.lines in line_values else "none")
    set_widget_value(app.show_north, bool(spec.show_north))
    set_widget_value(app.show_south, bool(spec.show_south))
    set_widget_value(app.min_abs_lat, float(spec.hemisphere_min_abs_latitude))
    set_widget_value(app.station, str(spec.ground_station).upper())
    set_widget_value(
        app.ground_component,
        spec.ground_component if spec.ground_component in component_values else "Magnitude",
    )
    set_widget_value(
        app.ground_quantity,
        spec.ground_quantity if spec.ground_quantity in quantity_values else "dbdt",
    )
    set_widget_value(app.include_station_data, bool(spec.include_station_data))
    set_widget_value(app.show_inductive, bool(spec.show_inductive))
    set_widget_value(app.show_noninductive, bool(spec.show_noninductive))
    set_widget_value(app.show_difference, bool(spec.show_difference))
    set_widget_value(app.sim_time_offset, float(spec.sim_time_offset_seconds))
    set_widget_value(app.data_time_offset, float(spec.data_time_offset_seconds))
    set_widget_value(app.dbdt_window_points, max(1, int(spec.dbdt_window_points)))
    set_widget_value(app.ground_model_lt_count, max(1, int(spec.ground_model_lt_count)))
    set_widget_value(app.ground_model_lat_count, max(1, int(spec.ground_model_lat_count)))
    set_widget_value(app.ground_model_visual_even, bool(spec.ground_model_visual_even))
    set_widget_value(
        app.show_pedersen_conductance_overlay, bool(spec.show_pedersen_conductance_overlay)
    )
    set_widget_value(app.show_hall_conductance_overlay, bool(spec.show_hall_conductance_overlay))
    set_widget_value(app.show_reference_line, bool(spec.show_reference_line))
    set_widget_value(app.reference_time, str(spec.reference_time_of_day_utc))
    set_widget_value(
        app.curve_scale_mode,
        spec.curve_scale_mode if spec.curve_scale_mode in {"manual", "auto"} else "manual",
    )
    set_widget_value(app.curve_scale, float(spec.curve_scale_value))
    set_widget_value(app.time_scale, float(spec.curve_time_scale))
    set_widget_value(app.low_lat_cutoff, float(spec.min_abs_dip_latitude))
    set_widget_value(app.low_lat_scale, float(spec.low_latitude_scale))
    set_widget_value(app.show_dip_equator_curve, bool(spec.show_dip_equator_curve))
    set_widget_value(app.show_low_lat_curve, bool(spec.show_low_latitude_curve))
    set_widget_value(
        app.color_scale_mode,
        spec.color_scale_mode if spec.color_scale_mode in {"fixed", "percentile"} else "fixed",
    )
    set_widget_value(app.color_scale_percentile, float(spec.color_scale_percentile))
    set_widget_value(app.geo_lat_min, float(spec.geo_lat_min))
    set_widget_value(app.geo_lat_max, float(spec.geo_lat_max))
    set_widget_value(app.local_time_min, float(spec.local_time_min))
    set_widget_value(app.local_time_max, float(spec.local_time_max))
    set_widget_value(app.zoom_window, bool(spec.zoom_window))
    set_widget_value(app.movie_filename, str(spec.movie_filename))
    set_widget_value(app.movie_fps, float(spec.movie_fps))


__all__ = ["apply_figure_spec_to_widgets", "current_figure_spec", "set_widget_value"]
