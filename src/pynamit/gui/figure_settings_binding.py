"""Bind Panel widgets to serializable PynaMIT figure settings."""

from __future__ import annotations

from pynamit.plotting.figure_settings import FigureSettings
from pynamit.plotting.figure_styles import (
    manual_color_control_units,
    manual_color_display_value,
    manual_color_limits,
    manual_line_parameters,
    map_line_keys,
)


def set_widget_value(widget, value):
    """Set a widget value only when it changed."""
    if value != widget.value:
        widget.value = value


def current_figure_settings(app) -> FigureSettings:
    """Return the settings described by the current Panel controls."""
    fill_key = app.fill.value if app.fill.value != "none" else "Br"
    _, color_display_scale = manual_color_control_units(fill_key)

    return FigureSettings(
        simulation_directory=app.simulation_directory.value,
        data_directory=app.figure_settings.data_directory,
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
        show_station_labels=bool(app.figure_settings.show_station_labels),
        sim_time_offset_seconds=float(app.sim_time_offset.value),
        data_time_offset_seconds=float(app.data_time_offset.value),
        dbdt_window_points=int(app.dbdt_window_points.value),
        ground_model_lt_count=int(app.ground_model_lt_count.value),
        ground_model_lat_count=int(app.ground_model_lat_count.value),
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
        manual_color_min=float(app.manual_color_min.value) / color_display_scale,
        manual_color_max=float(app.manual_color_max.value) / color_display_scale,
        line_first_abs_level=float(app.line_first_abs_level.value),
        line_interval=float(app.line_interval.value),
        line_levels_per_sign=int(app.line_levels_per_sign.value),
        geo_lat_min=float(app.geo_lat_min.value),
        geo_lat_max=float(app.geo_lat_max.value),
        local_time_min=float(app.local_time_min.value),
        local_time_max=float(app.local_time_max.value),
        zoom_window=bool(app.zoom_window.value),
        movie_filename=str(app.movie_filename.value),
        movie_fps=float(app.movie_fps.value),
        movie_dpi=int(app.figure_settings.movie_dpi),
    )


def apply_figure_settings_to_widgets(app, settings: FigureSettings) -> None:
    """Apply figure settings to the Panel controls."""
    max_time = int(app.time_index.end)
    time_start, time_end = [int(value) for value in settings.time_range]
    time_start = max(0, min(time_start, max_time))
    time_end = max(time_start, min(time_end, max_time))
    if time_start == 0 and time_end == 0 and max_time > 0:
        time_end = min(max_time, 60)

    set_widget_value(app.simulation_directory, settings.simulation_directory)
    set_widget_value(app.plot_type, settings.plot_type)
    set_widget_value(app.time_index, max(0, min(int(settings.time_index), max_time)))
    set_widget_value(app.time_range, (time_start, time_end))
    set_widget_value(app.fill, settings.fill)
    set_widget_value(app.lines, settings.lines)
    set_widget_value(app.show_north, bool(settings.show_north))
    set_widget_value(app.show_south, bool(settings.show_south))
    set_widget_value(app.min_abs_lat, float(settings.hemisphere_min_abs_latitude))
    set_widget_value(app.station, str(settings.ground_station).upper())
    set_widget_value(app.ground_component, settings.ground_component)
    set_widget_value(app.ground_quantity, settings.ground_quantity)
    set_widget_value(app.include_station_data, bool(settings.include_station_data))
    set_widget_value(app.show_inductive, bool(settings.show_inductive))
    set_widget_value(app.show_noninductive, bool(settings.show_noninductive))
    set_widget_value(app.show_difference, bool(settings.show_difference))
    set_widget_value(app.sim_time_offset, float(settings.sim_time_offset_seconds))
    set_widget_value(app.data_time_offset, float(settings.data_time_offset_seconds))
    set_widget_value(app.dbdt_window_points, int(settings.dbdt_window_points))
    set_widget_value(app.ground_model_lt_count, int(settings.ground_model_lt_count))
    set_widget_value(app.ground_model_lat_count, int(settings.ground_model_lat_count))
    set_widget_value(app.ground_model_visual_even, bool(settings.ground_model_visual_even))
    set_widget_value(
        app.show_pedersen_conductance_overlay, bool(settings.show_pedersen_conductance_overlay)
    )
    set_widget_value(
        app.show_hall_conductance_overlay, bool(settings.show_hall_conductance_overlay)
    )
    set_widget_value(app.show_reference_line, bool(settings.show_reference_line))
    set_widget_value(app.reference_time, str(settings.reference_time_of_day_utc))
    set_widget_value(app.curve_scale_mode, settings.curve_scale_mode)
    set_widget_value(app.curve_scale, float(settings.curve_scale_value))
    set_widget_value(app.time_scale, float(settings.curve_time_scale))
    set_widget_value(app.low_lat_cutoff, float(settings.min_abs_dip_latitude))
    set_widget_value(app.low_lat_scale, float(settings.low_latitude_scale))
    set_widget_value(app.show_dip_equator_curve, bool(settings.show_dip_equator_curve))
    set_widget_value(app.show_low_lat_curve, bool(settings.show_low_latitude_curve))
    set_widget_value(app.color_scale_mode, settings.color_scale_mode)
    set_widget_value(app.color_scale_percentile, float(settings.color_scale_percentile))
    fill_key = settings.fill if settings.fill != "none" else "Br"
    if settings.manual_color_min is None:
        color_min, color_max = manual_color_limits(fill_key)
    else:
        color_min, color_max = settings.manual_color_min, settings.manual_color_max
    set_widget_value(app.manual_color_min, manual_color_display_value(fill_key, color_min))
    set_widget_value(app.manual_color_max, manual_color_display_value(fill_key, color_max))
    if settings.line_first_abs_level is None:
        line_keys = map_line_keys(settings.lines)
        line_start, line_interval, line_count = manual_line_parameters(
            line_keys[0] if line_keys else "Phi"
        )
    else:
        line_start = settings.line_first_abs_level
        line_interval = settings.line_interval
        line_count = settings.line_levels_per_sign
    set_widget_value(app.line_first_abs_level, float(line_start))
    set_widget_value(app.line_interval, float(line_interval))
    set_widget_value(app.line_levels_per_sign, int(line_count))
    app._sync_style_control_labels()
    set_widget_value(app.geo_lat_min, float(settings.geo_lat_min))
    set_widget_value(app.geo_lat_max, float(settings.geo_lat_max))
    set_widget_value(app.local_time_min, float(settings.local_time_min))
    set_widget_value(app.local_time_max, float(settings.local_time_max))
    set_widget_value(app.zoom_window, bool(settings.zoom_window))
    set_widget_value(app.movie_filename, str(settings.movie_filename))
    set_widget_value(app.movie_fps, float(settings.movie_fps))


__all__ = ["apply_figure_settings_to_widgets", "current_figure_settings", "set_widget_value"]
