"""Matplotlib figure builders for saved PynaMIT runs."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynamit.math.constants import RE
from pynamit.simulation.config import setting_value
from pynamit.sphere import Grid, SolidHarmonics
from pynamit.visualization.figure_specs import PynamitFigureSpec
from pynamit.visualization.hemisphere import (
    hemisphere_masks_for_latitude,
    make_hemisphere_polarplot,
)
from pynamit.visualization.grid_evaluation import build_evaluator
from pynamit.visualization.map_curves import (
    build_even_global_sites,
    draw_curve_scale_inset,
    draw_timeseries_curve_map,
)
from pynamit.visualization.map_panels import draw_field_comparison_artists
from pynamit.visualization.plot_helpers import (
    add_panel_label,
    contour_kwargs_for_display,
    draw_line_contour_legend,
    get_ticks_from_levels,
    set_contour_edges_to_face,
    style_global_comparison_axis,
    style_global_input_axis,
    symmetric_contour_levels_without_zero,
)
from pynamit.visualization.run_fields import SavedCoefficientFieldView
from pynamit.visualization.station_data import (
    download_and_load_iaga2002_station_data,
    normalize_station_metadata,
    shift_station_datetime_index,
)
from pynamit.visualization.time_series import (
    compute_centered_difference_matrix_at_times,
    compute_centered_difference_series_at_times,
    get_time_index_median_cadence_seconds,
    resample_matrix_to_times,
    vector_magnitude_preserve_shape,
)


FIELD_PLOT_KWARGS = {
    "Br": {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-8.5, 8.5, 18) * 1e-8,
        "extend": "both",
        "symbol": "$B_r$",
        "units": "T",
    },
    "jr": {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-8.5, 8.5, 18) * 1e-7,
        "extend": "both",
        "symbol": "$j_r$",
        "units": "A/m$^2$",
    },
    "joule": {
        "cmap": plt.cm.bwr,
        "levels": np.linspace(-8.5, 8.5, 18) * 1e-3,
        "extend": "max",
        "symbol": "Joule heat",
        "units": "W/m$^2$",
    },
    "Jeq": {
        "colors": "black",
        "levels": np.linspace(-4, 4, 50) * 1e5,
        "symbol": "$J_{eq}$",
        "units": "A",
    },
    "Phi": {
        "colors": "black",
        "levels": symmetric_contour_levels_without_zero(170.0, 4.0),
        "symbol": "$\\Phi$",
        "units": "kV",
    },
    "W": {
        "colors": "green",
        "levels": symmetric_contour_levels_without_zero(40.0, 4.0),
        "symbol": "$W$",
        "units": "kV",
    },
}

FIELD_DIFF_KWARGS = {
    "Br": {
        **FIELD_PLOT_KWARGS["Br"],
        "levels": FIELD_PLOT_KWARGS["Br"]["levels"] * 0.5,
        "symbol": "$\\Delta B_r$",
    },
    "jr": {
        **FIELD_PLOT_KWARGS["jr"],
        "levels": FIELD_PLOT_KWARGS["jr"]["levels"] * 0.5,
        "symbol": "$\\Delta j_r$",
    },
    "joule": {
        **FIELD_PLOT_KWARGS["joule"],
        "levels": FIELD_PLOT_KWARGS["joule"]["levels"] * 0.5,
        "symbol": "$\\Delta$ Joule heat",
        "extend": "both",
    },
    "Jeq": {
        **FIELD_PLOT_KWARGS["Jeq"],
        "levels": FIELD_PLOT_KWARGS["Jeq"]["levels"] * 0.5,
        "symbol": "$\\Delta J_{eq}$",
    },
    "Phi": {
        **FIELD_PLOT_KWARGS["Phi"],
        "levels": symmetric_contour_levels_without_zero(36.0, 8.0),
        "symbol": "$\\Delta \\Phi$",
    },
    "W": {
        **FIELD_PLOT_KWARGS["W"],
        "levels": symmetric_contour_levels_without_zero(40.0, 4.0),
        "symbol": "$\\Delta W$",
    },
}

INPUT_SUMMARY_KWARGS = {
    "jr": {
        "cmap": plt.cm.bwr,
        "levels": symmetric_contour_levels_without_zero(0.9, 0.1),
        "extend": "both",
        "symbol": r"$j_r$",
        "units": r"$\mu$A/m$^2$",
        "scale": 1e6,
    },
    "Br": {
        "cmap": plt.cm.bwr,
        "levels": symmetric_contour_levels_without_zero(16.0, 2.0),
        "extend": "both",
        "symbol": r"$B_r(r=R_M)$",
        "units": "nT",
        "scale": 1e9,
    },
    "conductance": {
        "cmap": plt.cm.viridis,
        "levels": np.linspace(0.0, 40.0, 21),
        "extend": "max",
        "symbol": r"$\Sigma$",
        "units": "S",
        "scale": 1.0,
    },
    "wind": {
        "cmap": plt.cm.coolwarm,
        "levels": np.linspace(-500.0, 500.0, 21),
        "extend": "both",
        "symbol": r"$u$",
        "units": "m/s",
        "scale": 1.0,
    },
}

_VIEW_CACHE: dict[tuple[str, int, int], SavedCoefficientFieldView] = {}
_GROUND_FIELD_CACHE = {}


def _as_spec(spec):
    if isinstance(spec, PynamitFigureSpec):
        return spec
    return PynamitFigureSpec.from_dict(spec)


def clear_saved_field_view_cache():
    """Clear cached saved-run field views."""
    _VIEW_CACHE.clear()


def get_saved_field_view(spec):
    """Return a cached coefficient-field view for a figure spec."""
    spec = _as_spec(spec)
    key = (str(spec.run_directory), 60, 100)
    view = _VIEW_CACHE.get(key)
    if view is None:
        view = SavedCoefficientFieldView.from_directory(spec.run_directory)
        _VIEW_CACHE[key] = view
    return view


def map_line_keys(value):
    """Return contour-line field keys for one UI value."""
    value = str(value)
    if value == "none":
        return []
    if value == "Phi_W":
        return ["Phi", "W"]
    return [value]


def _figure_time_string(timestamp):
    try:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        if isinstance(timestamp, (int, float, np.floating)):
            return str(dt.timedelta(seconds=float(timestamp)))
        return str(timestamp)


def _create_comparison_axes(spec):
    if spec.plot_type == "hemispheres":
        rows = []
        if spec.show_north:
            rows.append("north")
        if spec.show_south:
            rows.append("south")
        if not rows:
            rows = ["north"]
        fig = plt.figure(figsize=(13, 4.2 * len(rows)), constrained_layout=True)
        grid = gridspec.GridSpec(
            len(rows),
            6,
            figure=fig,
            width_ratios=[1, 1, 1, 0.05, 0.05, 0.08],
            wspace=0.2,
            hspace=0.15,
        )
        axes_groups = []
        for row_index, hemisphere in enumerate(rows):
            axes = [
                make_hemisphere_polarplot(fig.add_subplot(grid[row_index, col]))
                for col in range(3)
            ]
            if row_index == 0:
                for axis, title in zip(axes, ["Inductive", "Non-inductive", "Difference"]):
                    axis.ax.set_title(title, fontsize=14)
            axes[0].ax.text(
                -0.4,
                0.5,
                hemisphere.upper(),
                transform=axes[0].ax.transAxes,
                ha="center",
                va="center",
                rotation=90,
                fontsize=14,
            )
            axes_groups.append({"hemisphere": hemisphere, "axes": axes})
        colorbar_axes = [
            fig.add_subplot(grid[:, 3]),
            fig.add_subplot(grid[:, 4]),
            fig.add_subplot(grid[:, 5]),
        ]
        return fig, axes_groups, colorbar_axes

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    grid = gridspec.GridSpec(
        1, 6, figure=fig, width_ratios=[1, 1, 1, 0.05, 0.05, 0.08], wspace=0.1
    )
    axes = [fig.add_subplot(grid[0, col], projection=ccrs.PlateCarree()) for col in range(3)]
    for axis, title in zip(axes, ["Inductive", "Non-inductive", "Difference"]):
        axis.set_title(title, fontsize=14)
    for index, axis in enumerate(axes):
        style_global_comparison_axis(axis, left_labels=(index == 0), bottom_labels=True)
    colorbar_axes = [
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[0, 4]),
        fig.add_subplot(grid[0, 5]),
    ]
    return fig, [{"hemisphere": "global", "axes": axes}], colorbar_axes


def _draw_line_legend(axis, overlay_keys, kwargs_source, title):
    draw_line_contour_legend(axis, overlay_keys, kwargs_source, title=title)


def _draw_map_line_legend(axis, overlay_keys):
    axis.cla()
    axis.axis("off")
    if not overlay_keys:
        return
    y_positions = np.linspace(0.78, 0.22, len(overlay_keys)) if len(overlay_keys) > 1 else [0.5]
    for y_pos, key in zip(y_positions, overlay_keys):
        kwargs = FIELD_PLOT_KWARGS[key]
        diff_kwargs = FIELD_DIFF_KWARGS[key]
        state_interval = kwargs["levels"][1] - kwargs["levels"][0]
        diff_interval = diff_kwargs["levels"][1] - diff_kwargs["levels"][0]
        units = kwargs.get("units", "")
        label = (
            f"{kwargs.get('symbol', key)} lines: {state_interval:.3g}; "
            f"diff {diff_interval:.3g} {units}"
        )
        axis.text(
            0.5,
            y_pos,
            label,
            ha="center",
            va="center",
            rotation="vertical",
            color=kwargs.get("colors", "black"),
            fontsize=10,
        )
    axis.set_title("Lines", fontsize=9, pad=6)


def _draw_comparison_colorbars(fig, colorbar_axes, main_mappable, diff_mappable, fill, lines):
    overlay_keys = map_line_keys(lines)
    filled_key = None if str(fill) == "none" else str(fill)
    cax_state, cax_diff, cax_lines = colorbar_axes

    if main_mappable is not None and filled_key is not None:
        kwargs = FIELD_PLOT_KWARGS[filled_key]
        colorbar = fig.colorbar(main_mappable, cax=cax_state, ticks=get_ticks_from_levels(kwargs))
        colorbar.set_label(f"{kwargs.get('symbol', filled_key)} ({kwargs.get('units', '')})")
    else:
        _draw_line_legend(cax_state, overlay_keys, FIELD_PLOT_KWARGS, "State lines")

    if diff_mappable is not None and filled_key is not None:
        kwargs = FIELD_DIFF_KWARGS[filled_key]
        colorbar = fig.colorbar(diff_mappable, cax=cax_diff, ticks=get_ticks_from_levels(kwargs))
        colorbar.set_label(f"{kwargs.get('symbol', filled_key)} ({kwargs.get('units', '')})")
    else:
        _draw_line_legend(cax_diff, overlay_keys, FIELD_DIFF_KWARGS, "Difference lines")

    if filled_key is not None:
        _draw_map_line_legend(cax_lines, overlay_keys)
    else:
        cax_lines.cla()
        cax_lines.axis("off")


def render_field_comparison_figure(spec, view=None):
    """Render inductive/non-inductive map panels."""
    spec = _as_spec(spec)
    view = get_saved_field_view(spec) if view is None else view
    index = int(max(0, min(int(spec.time_index), view.n_time - 1)))
    fields = view.state_comparison_grid_fields(index)
    timestamp = view.timestamp_at_index(index)

    fig, axes_groups, colorbar_axes = _create_comparison_axes(spec)
    _, main_mappable, diff_mappable = draw_field_comparison_artists(
        axes_groups,
        spec.fill,
        map_line_keys(spec.lines),
        fields,
        view.lat,
        view.lon,
        timestamp,
        plot_kwargs=FIELD_PLOT_KWARGS,
        diff_kwargs=FIELD_DIFF_KWARGS,
        dipole_obj=None,
        hemisphere_min_abs_latitude=spec.hemisphere_min_abs_latitude,
    )
    _draw_comparison_colorbars(
        fig, colorbar_axes, main_mappable, diff_mappable, spec.fill, spec.lines
    )

    fill_label = "no fill" if spec.fill == "none" else FIELD_PLOT_KWARGS[spec.fill]["symbol"]
    line_label = ", ".join(map_line_keys(spec.lines)) if map_line_keys(spec.lines) else "none"
    fig.suptitle(
        f"Time: {_figure_time_string(timestamp)} | filled: {fill_label}; lines: {line_label}",
        fontsize=15,
    )
    return fig


def _draw_input_scalar(axis, lon, lat, values, kwargs, title):
    display = np.asarray(values, dtype=float) * float(kwargs.get("scale", 1.0))
    contour = axis.contourf(
        lon,
        lat,
        display,
        levels=kwargs["levels"],
        cmap=kwargs.get("cmap"),
        extend=kwargs.get("extend", "neither"),
        transform=ccrs.PlateCarree(),
    )
    set_contour_edges_to_face(contour)
    style_global_input_axis(axis, draw_labels=True)
    axis.set_title(title, fontsize=10)
    return contour


def render_input_summary_figure(spec, view=None):
    """Render projected input drivers."""
    spec = _as_spec(spec)
    view = get_saved_field_view(spec) if view is None else view
    index = int(max(0, min(int(spec.time_index), view.n_time - 1)))
    fields = view.input_grid_fields(index)
    timestamp = view.timestamp_at_index(index)

    fig = plt.figure(figsize=(14, 7.875))
    figure_aspect = 16.0 / 9.0
    top_center_y = 0.70
    bottom_center_y = 0.255
    bottom_map_width = 0.285
    bottom_map_height = 0.5 * figure_aspect * bottom_map_width
    polar_width = 0.190
    polar_height = figure_aspect * polar_width
    br_map_height = polar_height
    br_map_width = 2.0 * br_map_height / figure_aspect
    bottom_x = {"wind": 0.035, "sigmaP": 0.350, "sigmaH": 0.665}
    top_x = {
        "jr_n": bottom_x["wind"] + 0.5 * (bottom_map_width - polar_width),
        "jr_s": bottom_x["sigmaP"] + 0.5 * (bottom_map_width - polar_width),
        "Br": 0.985 - br_map_width,
    }
    layout = {
        "jr_n": [top_x["jr_n"], top_center_y - 0.5 * polar_height, polar_width, polar_height],
        "jr_s": [top_x["jr_s"], top_center_y - 0.5 * polar_height, polar_width, polar_height],
        "Br": [top_x["Br"], top_center_y - 0.5 * br_map_height, br_map_width, br_map_height],
        "wind": [
            bottom_x["wind"],
            bottom_center_y - 0.5 * bottom_map_height,
            bottom_map_width,
            bottom_map_height,
        ],
        "sigmaP": [
            bottom_x["sigmaP"],
            bottom_center_y - 0.5 * bottom_map_height,
            bottom_map_width,
            bottom_map_height,
        ],
        "sigmaH": [
            bottom_x["sigmaH"],
            bottom_center_y - 0.5 * bottom_map_height,
            bottom_map_width,
            bottom_map_height,
        ],
    }
    layout["jr_cbar"] = [
        layout["jr_n"][0] + 0.030,
        layout["jr_n"][1] - 0.035,
        layout["jr_s"][0] + layout["jr_s"][2] - layout["jr_n"][0] - 0.060,
        0.026,
    ]
    layout["Br_cbar"] = [layout["Br"][0], layout["Br"][1] - 0.050, layout["Br"][2], 0.026]
    layout["conductance_cbar"] = [
        layout["sigmaH"][0] + layout["sigmaH"][2] + 0.014,
        layout["sigmaH"][1],
        0.015,
        layout["sigmaH"][3],
    ]

    pax_jr_n = make_hemisphere_polarplot(
        fig.add_axes(layout["jr_n"]), min_abs_latitude=spec.hemisphere_min_abs_latitude
    )
    pax_jr_s = make_hemisphere_polarplot(
        fig.add_axes(layout["jr_s"]), min_abs_latitude=spec.hemisphere_min_abs_latitude
    )
    global_projection = ccrs.PlateCarree()
    ax_br = fig.add_axes(layout["Br"], projection=global_projection)
    ax_wind = fig.add_axes(layout["wind"], projection=global_projection)
    ax_sigma_p = fig.add_axes(layout["sigmaP"], projection=global_projection)
    ax_sigma_h = fig.add_axes(layout["sigmaH"], projection=global_projection)

    mlt = (view.lon + 180.0) % 360.0 / 15.0
    north_mask, south_mask = hemisphere_masks_for_latitude(
        view.lat, spec.hemisphere_min_abs_latitude
    )
    jr_kwargs = INPUT_SUMMARY_KWARGS["jr"]
    jr_display = fields["jr"] * jr_kwargs.get("scale", 1.0)
    jr_plot_kwargs = contour_kwargs_for_display(jr_kwargs)
    jr_n = pax_jr_n.contourf(
        view.lat[north_mask], mlt[north_mask], jr_display[north_mask], **jr_plot_kwargs
    )
    jr_s = pax_jr_s.contourf(
        view.lat[south_mask], mlt[south_mask], jr_display[south_mask], **jr_plot_kwargs
    )
    set_contour_edges_to_face(jr_n)
    set_contour_edges_to_face(jr_s)
    pax_jr_n.ax.set_title(r"Input $j_r$ north", fontsize=11)
    pax_jr_s.ax.set_title(r"Input $j_r$ south", fontsize=11)
    try:
        pax_jr_n.writeLATlabels(color="black", backgroundcolor=(0, 0, 0, 0), north=True)
        pax_jr_n.writeLTlabels()
        pax_jr_s.writeLATlabels(color="black", backgroundcolor=(0, 0, 0, 0), north=False)
        pax_jr_s.writeLTlabels()
    except Exception:
        pass

    br_mappable = None
    conductance_mappable = None
    for axis, title, field_key, kwargs_key, left_labels, bottom_labels in [
        (ax_br, r"Input $B_r$ at $R_M$", "Br", "Br", False, True),
        (ax_sigma_p, "Pedersen conductance", "sigmaP", "conductance", False, True),
        (ax_sigma_h, "Hall conductance", "sigmaH", "conductance", False, True),
    ]:
        style_global_input_axis(axis, left_labels=left_labels, bottom_labels=bottom_labels)
        plot_kwargs = INPUT_SUMMARY_KWARGS[kwargs_key]
        contour = axis.contourf(
            view.lon,
            view.lat,
            fields[field_key] * plot_kwargs.get("scale", 1.0),
            transform=ccrs.PlateCarree(),
            **contour_kwargs_for_display(plot_kwargs),
        )
        set_contour_edges_to_face(contour)
        axis.set_title(title, fontsize=11)
        if field_key == "Br":
            br_mappable = contour
        else:
            conductance_mappable = contour

    style_global_input_axis(ax_wind, left_labels=True, bottom_labels=True)
    u_north = -fields["wind_theta"]
    u_east = fields["wind_phi"]
    if np.any(np.isfinite(u_north) & np.isfinite(u_east)):
        wind_quiver = ax_wind.quiver(
            view.wind_lon,
            view.wind_lat,
            u_east,
            u_north,
            transform=ccrs.PlateCarree(),
            color="0.08",
            scale=1800,
            width=0.0022,
            headwidth=3.4,
            headaxislength=3.4,
            minlength=0.02,
            zorder=4,
        )
        ax_wind.quiverkey(
            wind_quiver,
            0.08,
            0.08,
            200,
            "200 m/s",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": 8},
        )
    else:
        ax_wind.text(
            0.5,
            0.5,
            "Ordinary wind u not stored for this run",
            transform=ax_wind.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )
    ax_wind.set_title("Input horizontal wind", fontsize=11)

    for label, axis in zip(
        ["a", "b", "c", "d", "e", "f"],
        [pax_jr_n.ax, pax_jr_s.ax, ax_br, ax_wind, ax_sigma_p, ax_sigma_h],
    ):
        add_panel_label(axis, label)

    jr_cbar = fig.colorbar(jr_n, cax=fig.add_axes(layout["jr_cbar"]), orientation="horizontal")
    jr_cbar.set_label(f"{jr_kwargs['symbol']} ({jr_kwargs['units']})", size=10)
    jr_cbar.ax.tick_params(labelsize=8)
    br_kwargs = INPUT_SUMMARY_KWARGS["Br"]
    br_cbar = fig.colorbar(
        br_mappable, cax=fig.add_axes(layout["Br_cbar"]), orientation="horizontal"
    )
    br_cbar.set_label(f"{br_kwargs['symbol']} ({br_kwargs['units']})", size=10)
    br_cbar.ax.tick_params(labelsize=8)
    conductance_kwargs = INPUT_SUMMARY_KWARGS["conductance"]
    conductance_cbar = fig.colorbar(
        conductance_mappable, cax=fig.add_axes(layout["conductance_cbar"]), orientation="vertical"
    )
    conductance_cbar.set_label(
        f"{conductance_kwargs['symbol']} ({conductance_kwargs['units']})", size=10
    )
    conductance_cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Input drivers at {_figure_time_string(timestamp)}", fontsize=15, y=0.975)
    return fig


def _ground_time_index(view):
    return view.time_index


def _ground_field_matrices(view, site_lat, site_lon):
    lat_arr = np.asarray(site_lat, dtype=float).reshape(-1)
    lon_arr = np.asarray(site_lon, dtype=float).reshape(-1)
    if lat_arr.size != lon_arr.size:
        raise ValueError("site_lat and site_lon must have the same length.")

    key = (
        str(view.run_directory),
        tuple(np.round(lat_arr, 8).tolist()),
        tuple(np.round(lon_arr, 8).tolist()),
    )
    cached = _GROUND_FIELD_CACHE.get(key)
    if cached is not None:
        return cached

    grid = Grid(lat=lat_arr, lon=lon_arr)
    evaluator = build_evaluator(view.sh_basis, grid)
    ri = float(setting_value(view.settings, "RI"))
    solid_harmonics = SolidHarmonics(view.sh_basis)
    ve_to_ground = solid_harmonics.regular_reference_shift(ri, RE)
    m_ind_to_br = -(ri**2) * view.sh_basis.laplacian(ri)
    m_ind_to_br_ground = ve_to_ground * m_ind_to_br * evaluator.G
    m_ind_to_bh_ground = (view.sh_basis.n + 1) * ve_to_ground * evaluator.G_grad

    m_ind = view.datasets["state"].SH_m_ind.values.T
    m_ind_steady = view.datasets["steady_state"].SH_m_ind.values.T
    cached = (
        m_ind_to_br_ground.dot(m_ind),
        m_ind_to_bh_ground.dot(m_ind),
        m_ind_to_br_ground.dot(m_ind_steady),
        m_ind_to_bh_ground.dot(m_ind_steady),
    )
    _GROUND_FIELD_CACHE[key] = cached
    return cached


def _ground_component_base(component):
    component = str(component)
    if component.startswith("Abs") and component[3:] in {"North", "East", "Down"}:
        return component[3:]
    return component


def _ground_component_uses_abs(component):
    component = str(component)
    return component.startswith("Abs") and component[3:] in {"North", "East", "Down"}


def _ground_component_matrix(component, br_values, bh_values):
    base = _ground_component_base(component)
    if base == "North":
        values = -np.asarray(bh_values[0], dtype=float) * 1e9
    elif base == "East":
        values = np.asarray(bh_values[1], dtype=float) * 1e9
    elif base == "Down":
        values = -np.asarray(br_values, dtype=float) * 1e9
    elif base == "Magnitude":
        values = vector_magnitude_preserve_shape(
            [
                _ground_component_matrix("North", br_values, bh_values),
                _ground_component_matrix("East", br_values, bh_values),
                _ground_component_matrix("Down", br_values, bh_values),
            ]
        )
    else:
        raise ValueError(f"Unsupported ground component: {component!r}")
    return np.abs(values) if _ground_component_uses_abs(component) else values


def _ground_matrix_at_times(
    component, br_values, bh_values, source_times, target_times, *, quantity="b"
):
    source_index = pd.DatetimeIndex(source_times)
    target_index = pd.DatetimeIndex(target_times)
    if str(quantity) != "dbdt":
        return resample_matrix_to_times(
            source_index, _ground_component_matrix(component, br_values, bh_values), target_index
        )
    base = _ground_component_base(component)
    cadence = get_time_index_median_cadence_seconds(source_index)
    if base == "Magnitude":
        return vector_magnitude_preserve_shape(
            [
                _ground_matrix_at_times(
                    sub_component,
                    br_values,
                    bh_values,
                    source_index,
                    target_index,
                    quantity="dbdt",
                )
                for sub_component in ("North", "East", "Down")
            ]
        )
    values = compute_centered_difference_matrix_at_times(
        source_index,
        _ground_component_matrix(base, br_values, bh_values),
        target_index,
        half_window_points=1,
        cadence_seconds=cadence,
    )
    return np.abs(values) if _ground_component_uses_abs(component) else values


def _ground_value_scale(layers, *, fallback=10.0):
    finite = []
    for layer in layers:
        values = np.asarray(layer["values"], dtype=float)
        valid = np.abs(values[np.isfinite(values)])
        if valid.size:
            finite.append(valid)
    if not finite:
        return max(0.5 * float(fallback), np.finfo(float).tiny), float(fallback)
    display = float(np.nanpercentile(np.concatenate(finite), 95.0))
    if not np.isfinite(display) or display <= 0.0:
        display = float(fallback)
    display = float(np.ceil(2.0 * display))
    return max(0.5 * display, np.finfo(float).tiny), display


def _duration_label(time_index):
    if len(time_index) < 2:
        return "0 s"
    total_seconds = int(round((time_index[-1] - time_index[0]).total_seconds()))
    minutes, seconds = divmod(max(total_seconds, 0), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        parts = [f"{hours} h"]
        if minutes:
            parts.append(f"{minutes} m")
        if seconds and not minutes:
            parts.append(f"{seconds} s")
        return " ".join(parts)
    if minutes:
        return f"{minutes} m" if seconds == 0 else f"{minutes} m {seconds} s"
    return f"{seconds} s"


def _scale_label(value, unit):
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))} {unit}"
    if abs(value) < 1.0:
        return f"{value:.2g} {unit}"
    return f"{value:.1f} {unit}"


def _ground_target_times(spec, view):
    start, end = [int(value) for value in spec.time_range]
    start = max(0, min(start, view.n_time - 1))
    end = max(start, min(end, view.n_time - 1))
    if end == start:
        end = min(view.n_time - 1, start + min(60, max(view.n_time - 1, 1)))
    return _ground_time_index(view)[start : end + 1]


def _ground_station_table(spec):
    run_dir = Path(spec.run_directory).expanduser()
    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    if spec.data_directory:
        candidates.append(Path(spec.data_directory).expanduser() / "stations_full_list.csv")
    candidates.extend(
        [
            run_dir / "mag_data" / "stations_full_list.csv",
            run_dir / "data" / "mag_data" / "stations_full_list.csv",
            Path("mag_data/stations_full_list.csv"),
            Path("notebooks/mag_data/stations_full_list.csv"),
            repo_root / "notebooks" / "mag_data" / "stations_full_list.csv",
        ]
    )
    for candidate in candidates:
        try:
            return normalize_station_metadata(pd.read_csv(candidate)), str(candidate)
        except FileNotFoundError:
            continue
    raise ValueError(
        "Could not find stations_full_list.csv. Set data_directory in "
        "pynamit_plot_defaults.json or place station data in mag_data/."
    )


def _ground_signal_label(quantity):
    """Return a reader-facing ground signal label."""
    return "dB/dt" if str(quantity) == "dbdt" else "B"


def _ground_reference_line(spec, target_times, display_scale):
    if not spec.show_reference_line or len(target_times) < 2:
        return None
    try:
        reference_time = pd.Timestamp(f"{target_times[0].date()} {spec.reference_time_of_day_utc}")
    except ValueError:
        return None
    total_seconds = (target_times[-1] - target_times[0]).total_seconds()
    if total_seconds <= 0.0:
        return None
    position = (reference_time - target_times[0]).total_seconds() / total_seconds
    if position < 0.0 or position > 1.0:
        return None
    return {
        "position": position,
        "time": reference_time,
        "label": reference_time.strftime("%H:%M:%S UTC"),
        "color": "#0072B2",
        "linewidth": 1.5,
        "linestyle": (0, (1, 1)),
        "value_span": (2.0 / 3.0) * float(display_scale),
    }


def render_ground_curve_map_figure(spec, view=None):
    """Render a ground magnetic time-curve map."""
    spec = _as_spec(spec)
    view = get_saved_field_view(spec) if view is None else view
    target_times = _ground_target_times(spec, view)
    normalized_time = np.linspace(0.0, 1.0, len(target_times))

    lon, lat = build_even_global_sites(
        min_lat=spec.geo_lat_min,
        max_lat=spec.geo_lat_max,
        lat_count=8,
        equatorial_count=24,
        reference_time=target_times[0],
    )
    br_ind, bh_ind, br_steady, bh_steady = _ground_field_matrices(view, lat, lon)
    source_times = _ground_time_index(view) + pd.to_timedelta(
        float(spec.sim_time_offset_seconds), unit="s"
    )
    layers = []
    if spec.show_inductive:
        layers.append(
            {
                "series_key": "inductive",
                "label": "Inductive",
                "values": _ground_matrix_at_times(
                    spec.ground_component,
                    br_ind,
                    bh_ind,
                    source_times,
                    target_times,
                    quantity=spec.ground_quantity,
                ),
                "color": "#D55E00",
                "linewidth": 1.25,
                "zorder": 7,
            }
        )
    if spec.show_noninductive:
        layers.append(
            {
                "series_key": "magnetostatic",
                "label": "Non-inductive",
                "values": _ground_matrix_at_times(
                    spec.ground_component,
                    br_steady,
                    bh_steady,
                    source_times,
                    target_times,
                    quantity=spec.ground_quantity,
                ),
                "color": "#009E73",
                "linewidth": 1.25,
                "linestyle": (0, (3, 1.5)),
                "zorder": 6,
            }
        )
    if not layers:
        raise ValueError("Enable at least one model series for the ground curve map.")

    curve_width_deg = 10.0 * float(spec.curve_time_scale)
    curve_height_deg = 4.0
    value_scale, display_scale = (
        (0.5 * float(spec.curve_scale_value), float(spec.curve_scale_value))
        if spec.curve_scale_mode == "manual"
        else _ground_value_scale(layers, fallback=spec.curve_scale_value)
    )
    signal_label = _ground_signal_label(spec.ground_quantity)
    unit = "nT/s" if spec.ground_quantity == "dbdt" else "nT"
    reference_line = _ground_reference_line(spec, target_times, display_scale)
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(color="0.5", linewidth=0.8, zorder=2)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.8,
        color="0.8",
        linestyle="--",
        zorder=1,
    )
    gl.top_labels = False
    gl.right_labels = False

    draw_timeseries_curve_map(
        ax,
        site_lon=lon,
        site_lat=lat,
        normalized_time=normalized_time,
        layers=layers,
        curve_width_deg=curve_width_deg,
        curve_height_deg=curve_height_deg,
        value_scale=value_scale,
        central_longitude=0.0,
        reference_line=reference_line,
        legend_kwargs={"loc": "lower right", "framealpha": 0.92, "fontsize": 8},
        reference_color="#0072B2",
        reference_linewidth=1.5,
        reference_linestyle=(0, (1, 1)),
    )
    draw_curve_scale_inset(
        ax,
        curve_width_deg=curve_width_deg,
        curve_height_deg=curve_height_deg,
        value_scale=value_scale,
        scale_display_value=display_scale,
        scale_annotation=_scale_label(display_scale, unit),
        duration_annotation=_duration_label(target_times),
    )
    title = (
        f"Ground {signal_label} Curve Map: {spec.ground_component}; "
        f"{target_times[0].strftime('%H:%M:%S')} to {target_times[-1].strftime('%H:%M:%S')}; "
        f"scale {display_scale:g} {unit}"
    )
    fig.suptitle(title, fontsize=14)
    return fig


def _station_measured_dataframe(spec, station_code, target_index):
    try:
        _, stations_path = _ground_station_table(spec)
    except ValueError:
        return None
    data_dir = str(pd.io.common.stringify_path(stations_path)).rsplit("/", 1)[0]
    measured = download_and_load_iaga2002_station_data(
        station_code, target_index[0], data_dir, logger=None
    )
    if measured is None:
        return None
    measured_index = shift_station_datetime_index(
        measured.index, data_time_offset_seconds=spec.data_time_offset_seconds
    )
    return pd.DataFrame(
        {
            "North": measured[f"{station_code}X"].to_numpy(dtype=float),
            "East": measured[f"{station_code}Y"].to_numpy(dtype=float),
            "Down": measured[f"{station_code}Z"].to_numpy(dtype=float),
        },
        index=measured_index,
    )


def render_ground_timeseries_figure(spec, view=None):
    """Render selected-station ground magnetic time series."""
    spec = _as_spec(spec)
    view = get_saved_field_view(spec) if view is None else view
    stations, _ = _ground_station_table(spec)
    station_code = str(spec.ground_station).upper()
    rows = stations[stations["IAGA"] == station_code]
    if rows.empty:
        raise ValueError(f"Unknown station {station_code!r}.")
    station = rows.iloc[0]
    br_ind, bh_ind, br_steady, bh_steady = _ground_field_matrices(
        view, [station["GEOLAT"]], [station["GEOLON"]]
    )
    source_times = _ground_time_index(view) + pd.to_timedelta(
        float(spec.sim_time_offset_seconds), unit="s"
    )
    target_times = _ground_target_times(spec, view)
    measured = _station_measured_dataframe(spec, station_code, target_times)

    components = ["North", "East", "Down"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    x_start = target_times[0]
    x_end = target_times[-1]
    for axis, component in zip(axes, components):
        if measured is not None and spec.include_station_data:
            values = measured.loc[target_times[0] : target_times[-1], component]
            if spec.ground_quantity == "dbdt":
                values = pd.Series(
                    compute_centered_difference_series_at_times(
                        measured.index,
                        measured[component].to_numpy(dtype=float),
                        values.index,
                        half_window_points=1,
                    ),
                    index=values.index,
                )
            axis.plot(values.index, values, color="black", label="Measured")
        for br_values, bh_values, label, color, linestyle, enabled in [
            (br_ind, bh_ind, "Inductive", "#D55E00", "-", spec.show_inductive),
            (br_steady, bh_steady, "Non-inductive", "#009E73", "--", spec.show_noninductive),
        ]:
            if not enabled:
                continue
            values = _ground_matrix_at_times(
                component,
                br_values,
                bh_values,
                source_times,
                target_times,
                quantity=spec.ground_quantity,
            )[0]
            axis.plot(target_times, values, color=color, linestyle=linestyle, label=label)
        if spec.show_reference_line:
            try:
                ref_time = pd.Timestamp(
                    f"{target_times[0].date()} {spec.reference_time_of_day_utc}"
                )
                if x_start <= ref_time <= x_end:
                    axis.axvline(ref_time, color="#0072B2", linestyle=(0, (1, 1)), zorder=20)
            except ValueError:
                pass
        axis.set_xlim(x_start, x_end)
        axis.set_ylabel("nT/s" if spec.ground_quantity == "dbdt" else "nT")
        axis.set_title(component)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend(loc="best")
    axes[-1].set_xlabel(f"Time on {target_times[0].strftime('%Y-%m-%d')}")
    fig.suptitle(
        f"Ground {_ground_signal_label(spec.ground_quantity)} at {station_code}", fontsize=14
    )
    return fig


def render_pynamit_figure(spec):
    """Render a Matplotlib figure from a serializable figure spec."""
    spec = _as_spec(spec)
    if spec.plot_type in {"global", "hemispheres"}:
        return render_field_comparison_figure(spec)
    if spec.plot_type == "input_summary":
        return render_input_summary_figure(spec)
    if spec.plot_type == "ground_curve_map":
        return render_ground_curve_map_figure(spec)
    if spec.plot_type == "ground_timeseries":
        return render_ground_timeseries_figure(spec)
    raise NotImplementedError(f"{spec.plot_type!r} is not implemented in the Panel renderer.")


__all__ = [
    "FIELD_DIFF_KWARGS",
    "FIELD_PLOT_KWARGS",
    "INPUT_SUMMARY_KWARGS",
    "clear_saved_field_view_cache",
    "get_saved_field_view",
    "map_line_keys",
    "render_field_comparison_figure",
    "render_ground_curve_map_figure",
    "render_ground_timeseries_figure",
    "render_input_summary_figure",
    "render_pynamit_figure",
]
