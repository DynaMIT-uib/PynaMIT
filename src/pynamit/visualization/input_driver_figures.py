"""Figure renderer for projected input-driver summaries."""

from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from pynamit.visualization.field_comparison_figures import figure_time_string
from pynamit.visualization.figure_context import SavedRunFigureContext
from pynamit.visualization.figure_styles import INPUT_SUMMARY_KWARGS, percentile_contour_levels
from pynamit.visualization.hemisphere import (
    hemisphere_masks_for_latitude,
    make_hemisphere_polarplot,
)
from pynamit.visualization.plot_helpers import (
    add_panel_label,
    contour_kwargs_for_display,
    set_contour_edges_to_face,
    style_global_input_axis,
)


class InputDriverRenderer:
    """Render projected input drivers on the saved-run grid."""

    def __init__(self, spec, view=None):
        self.context = SavedRunFigureContext.from_spec(spec, view=view)
        self.spec = self.context.spec
        self.view = self.context.view

    def render(self):
        """Render projected input drivers."""
        fields = self.view.input_grid_fields(self.context.time_index)
        timestamp = self.context.timestamp
        input_kwargs = self._plot_kwargs(fields)

        fig = plt.figure(figsize=(14, 7.875))
        layout = self._layout()

        pax_jr_n = make_hemisphere_polarplot(
            fig.add_axes(layout["jr_n"]), min_abs_latitude=self.spec.hemisphere_min_abs_latitude
        )
        pax_jr_s = make_hemisphere_polarplot(
            fig.add_axes(layout["jr_s"]), min_abs_latitude=self.spec.hemisphere_min_abs_latitude
        )
        global_projection = ccrs.PlateCarree()
        ax_br = fig.add_axes(layout["Br"], projection=global_projection)
        ax_wind = fig.add_axes(layout["wind"], projection=global_projection)
        ax_sigma_p = fig.add_axes(layout["sigmaP"], projection=global_projection)
        ax_sigma_h = fig.add_axes(layout["sigmaH"], projection=global_projection)

        jr_n = self._draw_jr_hemispheres(fields, input_kwargs["jr"], pax_jr_n, pax_jr_s)
        br_mappable, conductance_mappable = self._draw_global_scalars(
            fields, input_kwargs, ax_br, ax_sigma_p, ax_sigma_h
        )
        self._draw_wind(fields, ax_wind)

        for label, axis in zip(
            ["a", "b", "c", "d", "e", "f"],
            [pax_jr_n.ax, pax_jr_s.ax, ax_br, ax_wind, ax_sigma_p, ax_sigma_h],
        ):
            add_panel_label(axis, label)

        self._draw_colorbars(fig, layout, jr_n, br_mappable, conductance_mappable, input_kwargs)
        fig.suptitle(f"Input drivers at {figure_time_string(timestamp)}", fontsize=15, y=0.975)
        return fig

    def _plot_kwargs(self, fields):
        kwargs = {key: dict(value) for key, value in INPUT_SUMMARY_KWARGS.items()}
        if self.spec.color_scale_mode != "percentile":
            return kwargs
        kwargs["jr"]["levels"] = percentile_contour_levels(
            [fields["jr"] * kwargs["jr"].get("scale", 1.0)],
            INPUT_SUMMARY_KWARGS["jr"]["levels"],
            percentile=self.spec.color_scale_percentile,
            strictly_positive=False,
        )
        kwargs["Br"]["levels"] = percentile_contour_levels(
            [fields["Br"] * kwargs["Br"].get("scale", 1.0)],
            INPUT_SUMMARY_KWARGS["Br"]["levels"],
            percentile=self.spec.color_scale_percentile,
            strictly_positive=False,
        )
        kwargs["conductance"]["levels"] = percentile_contour_levels(
            [
                fields["sigmaP"] * kwargs["conductance"].get("scale", 1.0),
                fields["sigmaH"] * kwargs["conductance"].get("scale", 1.0),
            ],
            INPUT_SUMMARY_KWARGS["conductance"]["levels"],
            percentile=self.spec.color_scale_percentile,
            strictly_positive=True,
        )
        return kwargs

    @staticmethod
    def _layout():
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
        return layout

    def _draw_jr_hemispheres(self, fields, jr_kwargs, pax_jr_n, pax_jr_s):
        mlt = (self.view.lon + 180.0) % 360.0 / 15.0
        north_mask, south_mask = hemisphere_masks_for_latitude(
            self.view.lat, self.spec.hemisphere_min_abs_latitude
        )
        jr_display = fields["jr"] * jr_kwargs.get("scale", 1.0)
        jr_plot_kwargs = contour_kwargs_for_display(jr_kwargs)
        jr_n = pax_jr_n.contourf(
            self.view.lat[north_mask], mlt[north_mask], jr_display[north_mask], **jr_plot_kwargs
        )
        jr_s = pax_jr_s.contourf(
            self.view.lat[south_mask], mlt[south_mask], jr_display[south_mask], **jr_plot_kwargs
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
        return jr_n

    def _draw_global_scalars(self, fields, input_kwargs, ax_br, ax_sigma_p, ax_sigma_h):
        br_mappable = None
        conductance_mappable = None
        for axis, title, field_key, kwargs_key, left_labels, bottom_labels in [
            (ax_br, r"Input $B_r$ at $R_M$", "Br", "Br", False, True),
            (ax_sigma_p, "Pedersen conductance", "sigmaP", "conductance", False, True),
            (ax_sigma_h, "Hall conductance", "sigmaH", "conductance", False, True),
        ]:
            style_global_input_axis(axis, left_labels=left_labels, bottom_labels=bottom_labels)
            plot_kwargs = input_kwargs[kwargs_key]
            contour = axis.contourf(
                self.view.lon,
                self.view.lat,
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
        return br_mappable, conductance_mappable

    def _draw_wind(self, fields, axis):
        style_global_input_axis(axis, left_labels=True, bottom_labels=True)
        u_north = -fields["wind_theta"]
        u_east = fields["wind_phi"]
        if np.any(np.isfinite(u_north) & np.isfinite(u_east)):
            wind_quiver = axis.quiver(
                self.view.wind_lon,
                self.view.wind_lat,
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
            axis.quiverkey(
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
            axis.text(
                0.5,
                0.5,
                "Ordinary wind u not stored for this run",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="0.35",
            )
        axis.set_title("Input horizontal wind", fontsize=11)

    @staticmethod
    def _draw_colorbars(fig, layout, jr_n, br_mappable, conductance_mappable, input_kwargs):
        jr_kwargs = input_kwargs["jr"]
        jr_cbar = fig.colorbar(jr_n, cax=fig.add_axes(layout["jr_cbar"]), orientation="horizontal")
        jr_cbar.set_label(f"{jr_kwargs['symbol']} ({jr_kwargs['units']})", size=10)
        jr_cbar.ax.tick_params(labelsize=8)

        br_kwargs = input_kwargs["Br"]
        br_cbar = fig.colorbar(
            br_mappable, cax=fig.add_axes(layout["Br_cbar"]), orientation="horizontal"
        )
        br_cbar.set_label(f"{br_kwargs['symbol']} ({br_kwargs['units']})", size=10)
        br_cbar.ax.tick_params(labelsize=8)

        conductance_kwargs = input_kwargs["conductance"]
        conductance_cbar = fig.colorbar(
            conductance_mappable,
            cax=fig.add_axes(layout["conductance_cbar"]),
            orientation="vertical",
        )
        conductance_cbar.set_label(
            f"{conductance_kwargs['symbol']} ({conductance_kwargs['units']})", size=10
        )
        conductance_cbar.ax.tick_params(labelsize=8)


def render_input_summary_figure(spec, view=None):
    """Render projected input drivers."""
    return InputDriverRenderer(spec, view=view).render()


__all__ = ["InputDriverRenderer", "render_input_summary_figure"]
