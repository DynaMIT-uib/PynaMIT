"""Figure renderers for inductive/non-inductive field comparisons."""

from __future__ import annotations

import datetime as dt

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from pynamit.visualization.figure_context import SavedRunFigureContext
from pynamit.visualization.figure_styles import (
    FIELD_DIFF_KWARGS,
    FIELD_PLOT_KWARGS,
    format_contour_interval,
    map_line_keys,
    percentile_contour_levels,
)
from pynamit.visualization.hemisphere import make_hemisphere_polarplot
from pynamit.visualization.map_panels import draw_field_comparison_artists
from pynamit.visualization.plot_helpers import (
    draw_line_contour_legend,
    get_ticks_from_levels,
    style_global_comparison_axis,
)


def figure_time_string(timestamp):
    """Return a compact title-friendly timestamp label."""
    try:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        if isinstance(timestamp, (int, float)):
            return str(dt.timedelta(seconds=float(timestamp)))
        return str(timestamp)


class FieldComparisonRenderer:
    """Render global or hemisphere comparisons from one saved run."""

    def __init__(self, spec, view=None):
        self.context = SavedRunFigureContext.from_spec(spec, view=view)
        self.spec = self.context.spec
        self.view = self.context.view

    def render(self):
        """Render inductive/non-inductive map panels."""
        if (self.spec.show_noninductive or self.spec.show_difference) and (
            "steady_state" not in self.view.datasets
        ):
            raise ValueError(
                "This run has no steady_state output. Disable Non-inductive/Difference "
                "plots, or rerun with save_steady_states=True."
            )
        fields = self.view.state_comparison_grid_fields(self.context.time_index)
        timestamp = self.context.timestamp
        plot_kwargs = {key: dict(value) for key, value in FIELD_PLOT_KWARGS.items()}
        diff_kwargs = {key: dict(value) for key, value in FIELD_DIFF_KWARGS.items()}
        filled_key = None if str(self.spec.fill) == "none" else str(self.spec.fill)
        if filled_key is not None and self.spec.color_scale_mode == "percentile":
            state_field = fields[f"{filled_key}_state"]
            percentile_fields = [state_field]
            if "steady_state" in self.view.datasets:
                steady_field = fields[f"{filled_key}_steady"]
                diff_field = state_field - steady_field
                percentile_fields.append(steady_field)
            else:
                diff_field = state_field
            plot_kwargs[filled_key]["levels"] = percentile_contour_levels(
                percentile_fields,
                FIELD_PLOT_KWARGS[filled_key]["levels"],
                percentile=self.spec.color_scale_percentile,
                strictly_positive=filled_key == "joule",
            )
            diff_kwargs[filled_key]["levels"] = percentile_contour_levels(
                [diff_field],
                FIELD_DIFF_KWARGS[filled_key]["levels"],
                percentile=self.spec.color_scale_percentile,
                strictly_positive=False,
            )

        panel_specs = self._panel_specs()
        fig, axes_groups, colorbar_axes = self._create_axes(panel_specs)
        _, main_mappable, diff_mappable = draw_field_comparison_artists(
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


def render_field_comparison_figure(spec, view=None):
    """Render inductive/non-inductive map panels."""
    return FieldComparisonRenderer(spec, view=view).render()


__all__ = ["FieldComparisonRenderer", "figure_time_string", "render_field_comparison_figure"]
