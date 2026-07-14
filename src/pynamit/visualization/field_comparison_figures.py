"""Figure renderers for inductive/non-inductive field comparisons."""

from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from pynamit.visualization.figure_context import (
    as_figure_spec,
    figure_time_string,
    get_saved_field_view,
)
from pynamit.visualization.figure_styles import FIELD_DIFF_KWARGS, FIELD_PLOT_KWARGS, map_line_keys
from pynamit.visualization.hemisphere import (
    hemisphere_masks_for_latitude,
    make_hemisphere_polarplot,
)
from pynamit.visualization.plot_helpers import (
    contour_kwargs_for_display,
    draw_line_contour_legend,
    format_contour_interval,
    get_ticks_from_levels,
    percentile_contour_levels,
    set_contour_edges_to_face,
    style_global_comparison_axis,
)


def _axes_from_group(group):
    return group.get("axes", []) if isinstance(group, dict) else group


def _hemisphere_from_group(group, index):
    if isinstance(group, dict):
        return group.get("hemisphere", "global")
    return "north" if index == 0 else "south"


def _is_polarplot_axis(axis):
    return axis.__class__.__name__ == "Polarplot"


def _polar_comparison_coordinates(lat, lon, current_time, dipole_obj, minimum_latitude):
    """Return polar coordinates and hemisphere masks."""
    if dipole_obj:
        polar_lat, magnetic_lon = dipole_obj.geo2mag(lat, lon)
        polar_time = dipole_obj.mlon2mlt(magnetic_lon, current_time)
    else:
        polar_lat = lat
        polar_time = (lon + 180.0) % 360.0 / 15.0
    north_mask, south_mask = hemisphere_masks_for_latitude(polar_lat, minimum_latitude)
    return polar_lat, polar_time, north_mask, south_mask


def _field_panel_spec(field_key, fields_dict, plot_kwargs, diff_kwargs, panel_keys):
    """Return one field's styles and values by panel."""
    styles, fields = {}, {}
    if any(key in {"state", "diff"} for key in panel_keys):
        styles["state"] = plot_kwargs[field_key]
        fields["state"] = fields_dict[f"{field_key}_state"]
    if any(key in {"steady", "diff"} for key in panel_keys):
        styles["steady"] = plot_kwargs[field_key]
        fields["steady"] = fields_dict[f"{field_key}_steady"]
    if "diff" in panel_keys:
        styles["diff"] = diff_kwargs[field_key]
        fields["diff"] = fields["state"] - fields["steady"]
    return styles, fields


def _contour_plot_arguments(is_polar, mask, polar_lat, polar_time, lon, lat, field):
    """Return plotting arguments for one contour."""
    if is_polar:
        return (polar_lat[mask], polar_time[mask], field[mask]), {}
    return (lon, lat, field), {"transform": ccrs.PlateCarree()}


def _draw_field_comparison_artists(
    axes_groups,
    filled_key,
    overlay_keys,
    fields_dict,
    lat,
    lon,
    current_time,
    *,
    plot_kwargs,
    diff_kwargs,
    dipole_obj=None,
    hemisphere_min_abs_latitude=50.0,
    panel_keys=("state", "steady", "diff"),
):
    """Draw state, steady-state, and difference fields."""
    new_artists, main_mappable, diff_mappable = [], None, None
    filled_key = None if str(filled_key) == "none" else str(filled_key)
    overlay_keys = list(overlay_keys)
    panel_keys = list(panel_keys)
    polar_x, polar_y, polar_north_mask, polar_south_mask = _polar_comparison_coordinates(
        lat, lon, current_time, dipole_obj, hemisphere_min_abs_latitude
    )

    if filled_key is not None:
        fill_kwargs, fill_fields = _field_panel_spec(
            filled_key, fields_dict, plot_kwargs, diff_kwargs, panel_keys
        )
    else:
        fill_kwargs, fill_fields = {}, {}

    overlay_specs = [
        _field_panel_spec(key, fields_dict, plot_kwargs, diff_kwargs, panel_keys)
        for key in overlay_keys
    ]

    for group_index, group in enumerate(axes_groups):
        axes = _axes_from_group(group)
        hemisphere = _hemisphere_from_group(group, group_index)
        is_polar = bool(axes) and _is_polarplot_axis(axes[0])
        if hemisphere == "north":
            mask = polar_north_mask
        elif hemisphere == "south":
            mask = polar_south_mask
        else:
            mask = None

        for panel_index, axis in enumerate(axes):
            panel_key = panel_keys[panel_index] if panel_index < len(panel_keys) else "empty"
            if panel_key not in {"state", "steady", "diff"}:
                continue

            if filled_key is not None:
                display_kwargs = contour_kwargs_for_display(fill_kwargs[panel_key])
                plot_args, transform_args = _contour_plot_arguments(
                    is_polar, mask, polar_x, polar_y, lon, lat, fill_fields[panel_key]
                )
                artist = axis.contourf(*plot_args, **transform_args, **display_kwargs)
                set_contour_edges_to_face(artist)
                new_artists.append(artist)
                if panel_key in {"state", "steady"}:
                    main_mappable = artist
                if panel_key == "diff":
                    diff_mappable = artist

            for overlay_kwargs, overlay_fields in overlay_specs:
                display_kwargs = contour_kwargs_for_display(overlay_kwargs[panel_key])
                plot_args, transform_args = _contour_plot_arguments(
                    is_polar, mask, polar_x, polar_y, lon, lat, overlay_fields[panel_key]
                )
                new_artists.append(axis.contour(*plot_args, **transform_args, **display_kwargs))

    return new_artists, main_mappable, diff_mappable


class FieldComparisonRenderer:
    """Render global or hemisphere comparisons from one saved run."""

    def __init__(self, spec, view=None):
        self.spec = as_figure_spec(spec)
        self.view = get_saved_field_view(self.spec) if view is None else view

    def render(self):
        """Render inductive/non-inductive map panels."""
        has_state = "state" in self.view.run_view.datasets
        has_steady = "steady_state" in self.view.run_view.datasets
        if self.spec.show_inductive and not has_state:
            raise ValueError(
                "This run has no inductive state output. Disable Inductive plots, "
                "or rerun with run_inductive=True."
            )
        if self.spec.show_noninductive and not has_steady:
            raise ValueError(
                "This run has no steady_state output. Disable Non-inductive plots, "
                "or rerun with run_steady_state=True."
            )
        if self.spec.show_difference and not (has_state and has_steady):
            raise ValueError("Difference plots require both state and steady_state outputs.")
        field_names = set(map_line_keys(self.spec.lines))
        if self.spec.fill != "none":
            field_names.add(self.spec.fill)
        fields = self.view.state_comparison_grid_fields(
            self.spec.time_index, field_names=field_names
        )
        timestamp = self.view.timestamp_at_index(self.spec.time_index)
        plot_kwargs = {key: dict(value) for key, value in FIELD_PLOT_KWARGS.items()}
        diff_kwargs = {key: dict(value) for key, value in FIELD_DIFF_KWARGS.items()}
        filled_key = None if str(self.spec.fill) == "none" else str(self.spec.fill)
        if filled_key is not None and self.spec.color_scale_mode == "percentile":
            percentile_fields = []
            if self.spec.show_inductive:
                percentile_fields.append(fields[f"{filled_key}_state"])
            if self.spec.show_noninductive:
                percentile_fields.append(fields[f"{filled_key}_steady"])
            plot_kwargs[filled_key]["levels"] = percentile_contour_levels(
                percentile_fields,
                FIELD_PLOT_KWARGS[filled_key]["levels"],
                percentile=self.spec.color_scale_percentile,
                strictly_positive=filled_key == "joule",
            )
            if self.spec.show_difference:
                diff_field = fields[f"{filled_key}_state"] - fields[f"{filled_key}_steady"]
                diff_kwargs[filled_key]["levels"] = percentile_contour_levels(
                    [diff_field],
                    FIELD_DIFF_KWARGS[filled_key]["levels"],
                    percentile=self.spec.color_scale_percentile,
                    strictly_positive=False,
                )

        panel_specs = self._panel_specs()
        fig, axes_groups, colorbar_axes = self._create_axes(panel_specs)
        _, main_mappable, diff_mappable = _draw_field_comparison_artists(
            axes_groups,
            self.spec.fill,
            map_line_keys(self.spec.lines),
            fields,
            self.view.lat,
            self.view.lon,
            timestamp,
            plot_kwargs=plot_kwargs,
            diff_kwargs=diff_kwargs,
            dipole_obj=None,
            hemisphere_min_abs_latitude=self.spec.hemisphere_min_abs_latitude,
            panel_keys=[key for key, _ in panel_specs],
        )
        self._draw_colorbars(
            fig, colorbar_axes, main_mappable, diff_mappable, plot_kwargs, diff_kwargs
        )

        fill_label = (
            "no fill" if self.spec.fill == "none" else FIELD_PLOT_KWARGS[self.spec.fill]["symbol"]
        )
        line_keys = map_line_keys(self.spec.lines)
        line_label = ", ".join(line_keys) if line_keys else "none"
        fig.suptitle(
            f"Time: {figure_time_string(timestamp)} | filled: {fill_label}; lines: {line_label}",
            fontsize=15,
        )
        return fig

    def _panel_specs(self):
        panels = []
        if self.spec.show_inductive:
            panels.append(("state", "Inductive"))
        if self.spec.show_noninductive:
            panels.append(("steady", "Non-inductive"))
        if self.spec.show_difference:
            panels.append(("diff", "Difference"))
        return panels or [("empty", "No data selected")]

    def _create_axes(self, panel_specs):
        n_panels = max(1, len(panel_specs))
        if self.spec.plot_type == "hemispheres":
            return self._create_hemisphere_axes(panel_specs, n_panels)
        return self._create_global_axes(panel_specs, n_panels)

    def _create_hemisphere_axes(self, panel_specs, n_panels):
        rows = []
        if self.spec.show_north:
            rows.append("north")
        if self.spec.show_south:
            rows.append("south")
        if not rows:
            rows = ["north"]
        fig = plt.figure(figsize=(13, 4.2 * len(rows)), constrained_layout=True)
        grid = gridspec.GridSpec(
            len(rows),
            n_panels + 3,
            figure=fig,
            width_ratios=[1] * n_panels + [0.05, 0.05, 0.08],
            wspace=0.2,
            hspace=0.15,
        )
        axes_groups = []
        for row_index, hemisphere in enumerate(rows):
            axes = [
                make_hemisphere_polarplot(fig.add_subplot(grid[row_index, col]))
                for col in range(n_panels)
            ]
            if row_index == 0:
                for axis, (_, title) in zip(axes, panel_specs):
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
            fig.add_subplot(grid[:, n_panels]),
            fig.add_subplot(grid[:, n_panels + 1]),
            fig.add_subplot(grid[:, n_panels + 2]),
        ]
        return fig, axes_groups, colorbar_axes

    def _create_global_axes(self, panel_specs, n_panels):
        fig = plt.figure(figsize=(13, 6), constrained_layout=True)
        grid = gridspec.GridSpec(
            1,
            n_panels + 3,
            figure=fig,
            width_ratios=[1] * n_panels + [0.05, 0.05, 0.08],
            wspace=0.1,
        )
        axes = [
            fig.add_subplot(grid[0, col], projection=ccrs.PlateCarree()) for col in range(n_panels)
        ]
        for axis, (_, title) in zip(axes, panel_specs):
            axis.set_title(title, fontsize=14)
        for index, axis in enumerate(axes):
            style_global_comparison_axis(axis, left_labels=(index == 0), bottom_labels=True)
        colorbar_axes = [
            fig.add_subplot(grid[0, n_panels]),
            fig.add_subplot(grid[0, n_panels + 1]),
            fig.add_subplot(grid[0, n_panels + 2]),
        ]
        return fig, [{"hemisphere": "global", "axes": axes}], colorbar_axes

    def _draw_colorbars(
        self, fig, colorbar_axes, main_mappable, diff_mappable, plot_kwargs, diff_kwargs
    ):
        overlay_keys = map_line_keys(self.spec.lines)
        filled_key = None if str(self.spec.fill) == "none" else str(self.spec.fill)
        cax_state, cax_diff, cax_lines = colorbar_axes

        if main_mappable is not None and filled_key is not None:
            kwargs = plot_kwargs[filled_key]
            colorbar = fig.colorbar(
                main_mappable, cax=cax_state, ticks=get_ticks_from_levels(kwargs)
            )
            colorbar.set_label(f"{kwargs.get('symbol', filled_key)} ({kwargs.get('units', '')})")
        else:
            self._draw_line_legend(cax_state, overlay_keys, plot_kwargs, "State lines")

        if diff_mappable is not None and filled_key is not None:
            kwargs = diff_kwargs[filled_key]
            colorbar = fig.colorbar(
                diff_mappable, cax=cax_diff, ticks=get_ticks_from_levels(kwargs)
            )
            colorbar.set_label(f"{kwargs.get('symbol', filled_key)} ({kwargs.get('units', '')})")
        else:
            self._draw_line_legend(cax_diff, overlay_keys, diff_kwargs, "Difference lines")

        if filled_key is not None:
            self._draw_map_line_legend(cax_lines, overlay_keys, plot_kwargs, diff_kwargs)
        else:
            cax_lines.cla()
            cax_lines.axis("off")

    @staticmethod
    def _draw_line_legend(axis, overlay_keys, kwargs_source, title):
        draw_line_contour_legend(axis, overlay_keys, kwargs_source, title=title)

    @staticmethod
    def _draw_map_line_legend(axis, overlay_keys, plot_kwargs, diff_kwargs):
        axis.cla()
        axis.axis("off")
        if not overlay_keys:
            return
        y_positions = (
            [0.5] if len(overlay_keys) == 1 else np.linspace(0.78, 0.22, len(overlay_keys))
        )
        for y_pos, key in zip(y_positions, overlay_keys):
            kwargs = plot_kwargs[key]
            field_diff_kwargs = diff_kwargs[key]
            state_interval = kwargs["levels"][1] - kwargs["levels"][0]
            diff_interval = field_diff_kwargs["levels"][1] - field_diff_kwargs["levels"][0]
            units = kwargs.get("units", "")
            label = (
                f"{kwargs.get('symbol', key)} lines: "
                f"{format_contour_interval(state_interval, units)}; "
                f"diff {format_contour_interval(diff_interval, units)}"
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


__all__ = ["FieldComparisonRenderer"]
