"""Figure renderers for dynamic/equilibrium field comparisons."""

from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from pynamit.plotting.figure_styles import (
    FIELD_DIFF_KWARGS,
    FIELD_PLOT_KWARGS,
    manual_color_levels,
    manual_line_levels,
    map_line_keys,
)
from pynamit.plotting.hemisphere import hemisphere_masks_for_latitude, make_hemisphere_polarplot
from pynamit.plotting.plot_data import _coerce_figure_settings, format_figure_time, get_plot_data
from pynamit.plotting.plot_helpers import (
    contour_kwargs_for_display,
    draw_line_contour_legend,
    format_contour_interval,
    get_ticks_from_levels,
    percentile_contour_levels,
    set_contour_edges_to_face,
    style_global_comparison_axis,
)


def _polar_comparison_coordinates(lat, lon, coordinate_context, minimum_latitude):
    """Return polar coordinates and hemisphere masks."""
    polar_time = coordinate_context.longitude_to_local_time(lon)
    north_mask, south_mask = hemisphere_masks_for_latitude(lat, minimum_latitude)
    return lat, polar_time, north_mask, south_mask


def _field_panel_spec(field_key, fields_dict, plot_kwargs, diff_kwargs, panel_keys):
    """Return one field's styles and values by panel."""
    styles, fields = {}, {}
    if any(key in {"dynamic", "diff"} for key in panel_keys):
        styles["dynamic"] = plot_kwargs[field_key]
        fields["dynamic"] = fields_dict[f"{field_key}_dynamic"]
    if any(key in {"equilibrium", "diff"} for key in panel_keys):
        styles["equilibrium"] = plot_kwargs[field_key]
        fields["equilibrium"] = fields_dict[f"{field_key}_equilibrium"]
    if "diff" in panel_keys:
        styles["diff"] = diff_kwargs[field_key]
        fields["diff"] = fields["dynamic"] - fields["equilibrium"]
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
    coordinate_context,
    *,
    plot_kwargs,
    diff_kwargs,
    hemisphere_min_abs_latitude=50.0,
    panel_keys=("dynamic", "equilibrium", "diff"),
):
    """Draw dynamic, equilibrium, and difference fields."""
    new_artists, main_mappable, diff_mappable = [], None, None
    filled_key = None if str(filled_key) == "none" else str(filled_key)
    overlay_keys = list(overlay_keys)
    panel_keys = list(panel_keys)
    polar_x, polar_y, polar_north_mask, polar_south_mask = _polar_comparison_coordinates(
        lat, lon, coordinate_context, hemisphere_min_abs_latitude
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

    for group in axes_groups:
        axes = group["axes"]
        hemisphere = group["hemisphere"]
        is_polar = hemisphere != "global"
        if hemisphere == "north":
            mask = polar_north_mask
        elif hemisphere == "south":
            mask = polar_south_mask
        else:
            mask = None

        for panel_index, axis in enumerate(axes):
            panel_key = panel_keys[panel_index] if panel_index < len(panel_keys) else "empty"
            if panel_key not in {"dynamic", "equilibrium", "diff"}:
                continue

            if filled_key is not None:
                display_kwargs = contour_kwargs_for_display(fill_kwargs[panel_key])
                plot_args, transform_args = _contour_plot_arguments(
                    is_polar, mask, polar_x, polar_y, lon, lat, fill_fields[panel_key]
                )
                artist = axis.contourf(*plot_args, **transform_args, **display_kwargs)
                set_contour_edges_to_face(artist)
                new_artists.append(artist)
                if panel_key in {"dynamic", "equilibrium"}:
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
    """Render map comparisons from one saved simulation."""

    def __init__(self, settings, plot_data=None):
        self.settings = _coerce_figure_settings(settings)
        self.plot_data = get_plot_data(self.settings) if plot_data is None else plot_data

    def render(self):
        """Render dynamic/equilibrium map panels."""
        has_dynamic = "dynamic" in self.plot_data.results.datasets
        has_equilibrium = "equilibrium" in self.plot_data.results.datasets
        if self.settings.show_dynamic and not has_dynamic:
            raise ValueError(
                "This simulation has no dynamic output. Disable Dynamic plots, "
                "or rerun with run_dynamic=True."
            )
        if self.settings.show_equilibrium and not has_equilibrium:
            raise ValueError(
                "This simulation has no equilibrium output. Disable Equilibrium plots, "
                "or rerun with run_equilibrium=True."
            )
        if self.settings.show_difference and not (has_dynamic and has_equilibrium):
            raise ValueError("Difference plots require both dynamic and equilibrium outputs.")
        field_names = set(map_line_keys(self.settings.lines))
        if self.settings.fill != "none":
            field_names.add(self.settings.fill)
        display_coordinate_system = (
            "geographic" if self.settings.plot_type == "global" else "model"
        )
        fields = self.plot_data.output_plot_data(
            self.settings.time_index,
            field_names=field_names,
            coordinate_system=display_coordinate_system,
        )
        timestamp = self.plot_data.timestamp_at_index(self.settings.time_index)
        if self.settings.plot_type == "global":
            display_latitude, display_longitude = self.plot_data.lat, self.plot_data.lon
            display_coordinate_context = self.plot_data.geographic_map_context(timestamp)
        else:
            display_latitude, display_longitude = self.plot_data.magnetic_plot_coordinates()
            display_coordinate_context = self.plot_data.magnetic_map_context(timestamp)
        plot_kwargs = {key: dict(value) for key, value in FIELD_PLOT_KWARGS.items()}
        diff_kwargs = {key: dict(value) for key, value in FIELD_DIFF_KWARGS.items()}
        filled_key = None if str(self.settings.fill) == "none" else str(self.settings.fill)
        if filled_key is not None:
            if self.settings.color_scale_mode == "percentile":
                percentile_fields = []
                if self.settings.show_dynamic:
                    percentile_fields.append(fields[f"{filled_key}_dynamic"])
                if self.settings.show_equilibrium:
                    percentile_fields.append(fields[f"{filled_key}_equilibrium"])
                plot_kwargs[filled_key]["levels"] = percentile_contour_levels(
                    percentile_fields,
                    FIELD_PLOT_KWARGS[filled_key]["levels"],
                    percentile=self.settings.color_scale_percentile,
                    strictly_positive=filled_key == "joule",
                )
                if self.settings.show_difference:
                    diff_field = (
                        fields[f"{filled_key}_dynamic"] - fields[f"{filled_key}_equilibrium"]
                    )
                    diff_kwargs[filled_key]["levels"] = percentile_contour_levels(
                        [diff_field],
                        FIELD_DIFF_KWARGS[filled_key]["levels"],
                        percentile=self.settings.color_scale_percentile,
                        strictly_positive=False,
                    )
            elif self.settings.manual_color_min is not None:
                plot_kwargs[filled_key]["levels"] = manual_color_levels(
                    filled_key, self.settings.manual_color_min, self.settings.manual_color_max
                )

        line_keys = map_line_keys(self.settings.lines)
        if self.settings.line_first_abs_level is not None:
            levels = manual_line_levels(
                self.settings.line_first_abs_level,
                self.settings.line_interval,
                self.settings.line_levels_per_sign,
            )
            for key in line_keys:
                plot_kwargs[key]["levels"] = levels

        panel_specs = self._panel_specs()
        fig, axes_groups, colorbar_axes = self._create_axes(panel_specs, timestamp)
        _, main_mappable, diff_mappable = _draw_field_comparison_artists(
            axes_groups,
            self.settings.fill,
            map_line_keys(self.settings.lines),
            fields,
            display_latitude,
            display_longitude,
            display_coordinate_context,
            plot_kwargs=plot_kwargs,
            diff_kwargs=diff_kwargs,
            hemisphere_min_abs_latitude=self.settings.hemisphere_min_abs_latitude,
            panel_keys=[key for key, _ in panel_specs],
        )
        self._draw_colorbars(
            fig, colorbar_axes, main_mappable, diff_mappable, plot_kwargs, diff_kwargs
        )

        fill_label = (
            "no fill"
            if self.settings.fill == "none"
            else FIELD_PLOT_KWARGS[self.settings.fill]["symbol"]
        )
        line_keys = map_line_keys(self.settings.lines)
        line_label = ", ".join(line_keys) if line_keys else "none"
        fig.suptitle(
            f"Time: {format_figure_time(timestamp)} | filled: {fill_label}; lines: {line_label}",
            fontsize=15,
        )
        return fig

    def _panel_specs(self):
        panels = []
        if self.settings.show_dynamic:
            panels.append(("dynamic", "Dynamic"))
        if self.settings.show_equilibrium:
            panels.append(("equilibrium", "Equilibrium"))
        if self.settings.show_difference:
            panels.append(("diff", "Difference"))
        return panels or [("empty", "No data selected")]

    def _create_axes(self, panel_specs, timestamp):
        n_panels = max(1, len(panel_specs))
        if self.settings.plot_type == "hemispheres":
            return self._create_hemisphere_axes(panel_specs, n_panels)
        return self._create_global_axes(panel_specs, n_panels, timestamp)

    def _create_hemisphere_axes(self, panel_specs, n_panels):
        rows = []
        if self.settings.show_north:
            rows.append("north")
        if self.settings.show_south:
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
                make_hemisphere_polarplot(
                    fig.add_subplot(grid[row_index, col]),
                    min_abs_latitude=self.settings.hemisphere_min_abs_latitude,
                )
                for col in range(n_panels)
            ]
            for axis in axes:
                axis.writeLATlabels(
                    color="black", backgroundcolor=(0, 0, 0, 0), north=hemisphere == "north"
                )
                axis.writeLTlabels()
            if row_index == 0:
                for axis, (_, title) in zip(axes, panel_specs, strict=True):
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

    def _create_global_axes(self, panel_specs, n_panels, timestamp):
        coordinate_context = self.plot_data.geographic_map_context(timestamp)
        fig = plt.figure(figsize=(13, 6), constrained_layout=True)
        grid = gridspec.GridSpec(
            1,
            n_panels + 3,
            figure=fig,
            width_ratios=[1] * n_panels + [0.05, 0.05, 0.08],
            wspace=0.1,
        )
        axes = [
            fig.add_subplot(grid[0, col], projection=coordinate_context.projection())
            for col in range(n_panels)
        ]
        for axis, (_, title) in zip(axes, panel_specs, strict=True):
            axis.set_title(title, fontsize=14)
        for index, axis in enumerate(axes):
            style_global_comparison_axis(
                axis,
                coordinate_context=coordinate_context,
                left_labels=(index == 0),
                bottom_labels=True,
            )
        colorbar_axes = [
            fig.add_subplot(grid[0, n_panels]),
            fig.add_subplot(grid[0, n_panels + 1]),
            fig.add_subplot(grid[0, n_panels + 2]),
        ]
        return fig, [{"hemisphere": "global", "axes": axes}], colorbar_axes

    def _draw_colorbars(
        self, fig, colorbar_axes, main_mappable, diff_mappable, plot_kwargs, diff_kwargs
    ):
        overlay_keys = map_line_keys(self.settings.lines)
        filled_key = None if str(self.settings.fill) == "none" else str(self.settings.fill)
        cax_dynamic, cax_diff, cax_lines = colorbar_axes

        if main_mappable is not None and filled_key is not None:
            kwargs = plot_kwargs[filled_key]
            ticks = get_ticks_from_levels(kwargs)
            if (
                self.settings.color_scale_mode == "manual"
                and self.settings.manual_color_min is not None
            ):
                ticks = np.linspace(
                    self.settings.manual_color_min, self.settings.manual_color_max, 5
                )
            colorbar = fig.colorbar(main_mappable, cax=cax_dynamic, ticks=ticks)
            colorbar.set_label(f"{kwargs.get('symbol', filled_key)} ({kwargs.get('units', '')})")
        else:
            self._draw_line_legend(cax_dynamic, overlay_keys, plot_kwargs, "Lines")

        if self.settings.show_difference and diff_mappable is not None and filled_key is not None:
            kwargs = diff_kwargs[filled_key]
            colorbar = fig.colorbar(
                diff_mappable, cax=cax_diff, ticks=get_ticks_from_levels(kwargs)
            )
            colorbar.set_label(f"{kwargs.get('symbol', filled_key)} ({kwargs.get('units', '')})")
        elif self.settings.show_difference:
            self._draw_line_legend(cax_diff, overlay_keys, diff_kwargs, "Difference lines")
        else:
            cax_diff.cla()
            cax_diff.axis("off")

        if filled_key is not None:
            self._draw_map_line_legend(
                cax_lines,
                overlay_keys,
                plot_kwargs,
                diff_kwargs,
                include_difference=self.settings.show_difference,
            )
        else:
            cax_lines.cla()
            cax_lines.axis("off")

    @staticmethod
    def _draw_line_legend(axis, overlay_keys, kwargs_source, title):
        draw_line_contour_legend(axis, overlay_keys, kwargs_source, title=title)

    @staticmethod
    def _draw_map_line_legend(
        axis, overlay_keys, plot_kwargs, diff_kwargs, *, include_difference=True
    ):
        axis.cla()
        axis.axis("off")
        if not overlay_keys:
            return
        y_positions = (
            [0.5] if len(overlay_keys) == 1 else np.linspace(0.78, 0.22, len(overlay_keys))
        )
        for y_pos, key in zip(y_positions, overlay_keys, strict=True):
            kwargs = plot_kwargs[key]
            dynamic_interval = kwargs["levels"][1] - kwargs["levels"][0]
            units = kwargs.get("units", "")
            label = (
                f"{kwargs.get('symbol', key)} lines: "
                f"{format_contour_interval(dynamic_interval, units)}"
            )
            if include_difference:
                field_diff_kwargs = diff_kwargs[key]
                diff_interval = field_diff_kwargs["levels"][1] - field_diff_kwargs["levels"][0]
                label += f"; diff {format_contour_interval(diff_interval, units)}"
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
