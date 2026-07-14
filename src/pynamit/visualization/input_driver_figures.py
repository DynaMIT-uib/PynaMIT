"""Figure renderer for projected input-driver summaries."""

from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from pynamit.visualization.figure_context import (
    as_figure_spec,
    figure_time_string,
    get_saved_field_view,
)
from pynamit.visualization.figure_styles import INPUT_SUMMARY_KWARGS
from pynamit.visualization.hemisphere import (
    hemisphere_masks_for_latitude,
    make_hemisphere_polarplot,
)
from pynamit.visualization.plot_helpers import (
    add_panel_label,
    contour_kwargs_for_display,
    percentile_contour_levels,
    set_contour_edges_to_face,
    style_global_input_axis,
)


class InputDriverRenderer:
    """Render projected input drivers on the saved-run grid."""

    def __init__(self, spec, view=None):
        self.spec = as_figure_spec(spec)
        self.view = get_saved_field_view(self.spec) if view is None else view

    def render(self):
        """Render projected input drivers."""
        fields = self.view.input_grid_fields(self.spec.time_index)
        timestamp = self.view.timestamp_at_index(self.spec.time_index)
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
        ax_source = fig.add_axes(layout["source"], projection=global_projection)
        ax_sigma_p = fig.add_axes(layout["sigmaP"], projection=global_projection)
        ax_sigma_h = fig.add_axes(layout["sigmaH"], projection=global_projection)

        jr_n = self._draw_jr_hemispheres(fields, input_kwargs["jr"], pax_jr_n, pax_jr_s)
        br_mappable, conductance_mappable = self._draw_global_scalars(
            fields, input_kwargs, ax_br, ax_sigma_p, ax_sigma_h
        )
        self._draw_tangential_source(fields, ax_source)

        for label, axis in zip(
            ["a", "b", "c", "d", "e", "f"],
            [pax_jr_n.ax, pax_jr_s.ax, ax_br, ax_source, ax_sigma_p, ax_sigma_h],
            strict=True,
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
    def _has_finite(values):
        return np.any(np.isfinite(values))

    @staticmethod
    def _mark_missing(axis, message):
        axis.text(
            0.5,
            0.5,
            message,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )

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
        bottom_x = {"source": 0.035, "sigmaP": 0.350, "sigmaH": 0.665}
        top_x = {
            "jr_n": bottom_x["source"] + 0.5 * (bottom_map_width - polar_width),
            "jr_s": bottom_x["sigmaP"] + 0.5 * (bottom_map_width - polar_width),
            "Br": 0.985 - br_map_width,
        }
        layout = {
            "jr_n": [top_x["jr_n"], top_center_y - 0.5 * polar_height, polar_width, polar_height],
            "jr_s": [top_x["jr_s"], top_center_y - 0.5 * polar_height, polar_width, polar_height],
            "Br": [top_x["Br"], top_center_y - 0.5 * br_map_height, br_map_width, br_map_height],
            "source": [
                bottom_x["source"],
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
        if not self._has_finite(jr_display):
            for pax in (pax_jr_n, pax_jr_s):
                self._mark_missing(pax.ax, "Input jr not stored")
            pax_jr_n.ax.set_title(r"Input $j_r$ north", fontsize=11)
            pax_jr_s.ax.set_title(r"Input $j_r$ south", fontsize=11)
            return None
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
        for axis, title, missing_message, field_key, kwargs_key, left_labels, bottom_labels in [
            (ax_br, r"Input $B_r$ at $R_M$", r"Input $B_r$ not stored", "Br", "Br", False, True),
            (
                ax_sigma_p,
                "Pedersen conductance",
                "Conductance not stored",
                "sigmaP",
                "conductance",
                False,
                True,
            ),
            (
                ax_sigma_h,
                "Hall conductance",
                "Conductance not stored",
                "sigmaH",
                "conductance",
                False,
                True,
            ),
        ]:
            style_global_input_axis(axis, left_labels=left_labels, bottom_labels=bottom_labels)
            plot_kwargs = input_kwargs[kwargs_key]
            if not self._has_finite(fields[field_key]):
                self._mark_missing(axis, missing_message)
                axis.set_title(title, fontsize=11)
                continue
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

    def _draw_tangential_source(self, fields, axis):
        style_global_input_axis(axis, left_labels=True, bottom_labels=True)
        source_options = [
            {
                "theta": fields["E_source_theta"],
                "phi": fields["E_source_phi"],
                "scale_factor": 1.0e3,
                "title": r"Input direct $E_\mathrm{source}$",
                "key_value": 10.0,
                "key_label": "10 mV/m",
                "scale": 90.0,
            },
            {
                "theta": fields["Q_eff_theta"],
                "phi": fields["Q_eff_phi"],
                "scale_factor": 1.0e3,
                "title": r"Input $Q_\mathrm{eff}$",
                "key_value": 50.0,
                "key_label": "50 mA/m",
                "scale": 450.0,
            },
            {
                "theta": fields["wind_theta"],
                "phi": fields["wind_phi"],
                "scale_factor": 1.0,
                "title": "Input horizontal wind",
                "key_value": 200.0,
                "key_label": "200 m/s",
                "scale": 1800.0,
            },
        ]
        selected = None
        for option in source_options:
            north = -np.asarray(option["theta"], dtype=float) * option["scale_factor"]
            east = np.asarray(option["phi"], dtype=float) * option["scale_factor"]
            if np.any(np.isfinite(north) & np.isfinite(east)):
                selected = {**option, "north": north, "east": east}
                break

        if selected is not None:
            quiver = axis.quiver(
                self.view.wind_lon,
                self.view.wind_lat,
                selected["east"],
                selected["north"],
                transform=ccrs.PlateCarree(),
                color="0.08",
                scale=selected["scale"],
                width=0.0022,
                headwidth=3.4,
                headaxislength=3.4,
                minlength=0.02,
                zorder=4,
            )
            axis.quiverkey(
                quiver,
                0.08,
                0.08,
                selected["key_value"],
                selected["key_label"],
                labelpos="E",
                coordinates="axes",
                fontproperties={"size": 8},
            )
            axis.set_title(selected["title"], fontsize=11)
        else:
            axis.text(
                0.5,
                0.5,
                "No tangential wind/source input stored",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="0.35",
            )
            axis.set_title("Input tangential source", fontsize=11)

    @staticmethod
    def _draw_colorbars(fig, layout, jr_n, br_mappable, conductance_mappable, input_kwargs):
        jr_axis = fig.add_axes(layout["jr_cbar"])
        if jr_n is not None:
            jr_kwargs = input_kwargs["jr"]
            jr_cbar = fig.colorbar(jr_n, cax=jr_axis, orientation="horizontal")
            jr_cbar.set_label(f"{jr_kwargs['symbol']} ({jr_kwargs['units']})", size=10)
            jr_cbar.ax.tick_params(labelsize=8)
        else:
            jr_axis.axis("off")

        br_axis = fig.add_axes(layout["Br_cbar"])
        if br_mappable is not None:
            br_kwargs = input_kwargs["Br"]
            br_cbar = fig.colorbar(br_mappable, cax=br_axis, orientation="horizontal")
            br_cbar.set_label(f"{br_kwargs['symbol']} ({br_kwargs['units']})", size=10)
            br_cbar.ax.tick_params(labelsize=8)
        else:
            br_axis.axis("off")

        conductance_axis = fig.add_axes(layout["conductance_cbar"])
        if conductance_mappable is not None:
            conductance_kwargs = input_kwargs["conductance"]
            conductance_cbar = fig.colorbar(
                conductance_mappable, cax=conductance_axis, orientation="vertical"
            )
            conductance_cbar.set_label(
                f"{conductance_kwargs['symbol']} ({conductance_kwargs['units']})", size=10
            )
            conductance_cbar.ax.tick_params(labelsize=8)
        else:
            conductance_axis.axis("off")


__all__ = ["InputDriverRenderer"]
