"""Panel frontend for saved PynaMIT visualization runs."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import traceback

import matplotlib.pyplot as plt

from pynamit.visualization.figure_builder import (
    clear_saved_field_view_cache,
    get_saved_field_view,
    render_pynamit_figure,
)
from pynamit.visualization.figure_specs import (
    MAP_FILL_OPTIONS,
    MAP_LINE_OPTIONS,
    PynamitFigureSpec,
    figure_spec_from_run_defaults,
    publication_script_for_spec,
)


def _panel():
    try:
        import panel as pn
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The Panel plotting app requires panel. Install it in the active environment, "
            "for example with `conda install -c conda-forge panel` or `pip install panel`."
        ) from exc
    pn.extension(sizing_mode="stretch_width")
    return pn


def _default_run_directory():
    for candidate in (Path("."), Path("sim_dir"), Path("notebooks/sim_dir")):
        if (candidate / "settings.zarr").exists() or (candidate / "settings.ncdf").exists():
            return str(candidate)
    return "sim_dir"


class PynamitPanelApp:
    """Interactive Panel application for saved PynaMIT figures."""

    def __init__(self, run_directory=None):
        self.pn = _panel()
        self.spec = figure_spec_from_run_defaults(str(run_directory or _default_run_directory()))
        self.view = None
        self.figure = None
        self._busy = False
        self._loaded_run_directory = None

        pn = self.pn
        self.run_directory = pn.widgets.TextInput(
            name="Run directory", value=self.spec.run_directory, min_width=280
        )
        self.load_button = pn.widgets.Button(name="Load", button_type="primary", width=85)
        plot_type_options = {
            "Ground curve map": "ground_curve_map",
            "Ground time series": "ground_timeseries",
            "Global maps": "global",
            "Hemispheres": "hemispheres",
            "Input drivers": "input_summary",
        }
        self.plot_type = pn.widgets.Select(
            name="Plot",
            options=plot_type_options,
            value=(
                self.spec.plot_type
                if self.spec.plot_type in plot_type_options.values()
                else "ground_curve_map"
            ),
            width=180,
        )
        self.time_index = pn.widgets.IntSlider(
            name="Time", start=0, end=0, value=0, step=1, min_width=320
        )
        self.time_label = pn.pane.Markdown("", width=260)
        self.time_range = pn.widgets.IntRangeSlider(
            name="Time range", start=0, end=0, value=(0, 0), step=1, min_width=320
        )

        self.fill = pn.widgets.Select(
            name="Filled contours",
            options={label: key for key, label in MAP_FILL_OPTIONS.items()},
            value=self.spec.fill if self.spec.fill in MAP_FILL_OPTIONS else "Br",
            width=210,
        )
        self.lines = pn.widgets.Select(
            name="Contour lines",
            options={label: key for key, label in MAP_LINE_OPTIONS.items()},
            value=self.spec.lines if self.spec.lines in MAP_LINE_OPTIONS else "none",
            width=210,
        )
        self.show_north = pn.widgets.Checkbox(name="North", value=self.spec.show_north, width=90)
        self.show_south = pn.widgets.Checkbox(name="South", value=self.spec.show_south, width=90)
        self.min_abs_lat = pn.widgets.FloatInput(
            name="Min |lat|",
            value=self.spec.hemisphere_min_abs_latitude,
            start=0,
            end=89.9,
            width=130,
        )
        self.station = pn.widgets.TextInput(
            name="Station", value=self.spec.ground_station, width=120
        )
        self.ground_component = pn.widgets.Select(
            name="Component",
            options={
                "Magnitude": "Magnitude",
                "|North|": "AbsNorth",
                "|East|": "AbsEast",
                "|Down|": "AbsDown",
            },
            value=(
                self.spec.ground_component
                if self.spec.ground_component in {"Magnitude", "AbsNorth", "AbsEast", "AbsDown"}
                else "Magnitude"
            ),
            width=150,
        )
        self.ground_quantity = pn.widgets.Select(
            name="Signal",
            options={"dB/dt": "dbdt", "B": "b"},
            value=(
                self.spec.ground_quantity if self.spec.ground_quantity in {"dbdt", "b"} else "dbdt"
            ),
            width=110,
        )
        self.include_station_data = pn.widgets.Checkbox(
            name="Include data", value=self.spec.include_station_data, width=120
        )
        self.show_inductive = pn.widgets.Checkbox(
            name="Inductive", value=self.spec.show_inductive, width=100
        )
        self.show_noninductive = pn.widgets.Checkbox(
            name="Non-inductive", value=self.spec.show_noninductive, width=130
        )
        self.show_reference_line = pn.widgets.Checkbox(
            name="Reference line", value=self.spec.show_reference_line, width=130
        )
        self.reference_time = pn.widgets.TextInput(
            name="Ref. UTC", value=self.spec.reference_time_of_day_utc, width=130
        )
        self.curve_scale = pn.widgets.FloatInput(
            name="Scale", value=self.spec.curve_scale_value, start=0.01, width=110
        )
        self.time_scale = pn.widgets.FloatInput(
            name="Time x", value=self.spec.curve_time_scale, start=0.1, width=110
        )
        self.geo_lat_min = pn.widgets.FloatInput(
            name="Geo lat min", value=self.spec.geo_lat_min, width=130
        )
        self.geo_lat_max = pn.widgets.FloatInput(
            name="Geo lat max", value=self.spec.geo_lat_max, width=130
        )

        self.redraw_button = pn.widgets.Button(name="Redraw", button_type="primary", width=95)
        self.save_button = pn.widgets.Button(name="Save figure", button_type="warning", width=120)
        self.output_filename = pn.widgets.TextInput(
            name="Output", value="pynamit_figure.png", width=260
        )
        self.script_download = pn.widgets.FileDownload(
            label="Download .py",
            filename="pynamit_figure.py",
            callback=self._download_script,
            button_type="success",
            width=130,
        )
        self.spec_download = pn.widgets.FileDownload(
            label="Download spec",
            filename="pynamit_figure.json",
            callback=self._download_spec,
            button_type="success",
            width=135,
        )
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.plot_pane = pn.pane.Matplotlib(
            object=None,
            tight=True,
            format="png",
            dpi=120,
            sizing_mode="stretch_both",
            min_height=560,
        )

        self.load_button.on_click(self._load_run)
        self.redraw_button.on_click(self._redraw)
        self.save_button.on_click(self._save_figure)
        for widget in (
            self.plot_type,
            self.time_index,
            self.time_range,
            self.fill,
            self.lines,
            self.show_north,
            self.show_south,
            self.min_abs_lat,
            self.station,
            self.ground_component,
            self.ground_quantity,
            self.include_station_data,
            self.show_inductive,
            self.show_noninductive,
            self.show_reference_line,
            self.reference_time,
            self.curve_scale,
            self.time_scale,
            self.geo_lat_min,
            self.geo_lat_max,
        ):
            widget.param.watch(self._control_changed, "value")

        self._load_run()

    def _current_spec(self):
        return PynamitFigureSpec(
            run_directory=self.run_directory.value,
            data_directory=self.spec.data_directory,
            plot_type=self.plot_type.value,
            time_index=int(self.time_index.value),
            time_range=tuple(int(value) for value in self.time_range.value),
            fill=self.fill.value,
            lines=self.lines.value,
            show_north=bool(self.show_north.value),
            show_south=bool(self.show_south.value),
            hemisphere_min_abs_latitude=float(self.min_abs_lat.value),
            ground_station=str(self.station.value).upper(),
            ground_component=self.ground_component.value,
            ground_quantity=self.ground_quantity.value,
            include_station_data=bool(self.include_station_data.value),
            show_inductive=bool(self.show_inductive.value),
            show_noninductive=bool(self.show_noninductive.value),
            show_reference_line=bool(self.show_reference_line.value),
            reference_time_of_day_utc=str(self.reference_time.value),
            show_station_labels=bool(self.spec.show_station_labels),
            conductance_overlay=self.spec.conductance_overlay,
            sim_time_offset_seconds=float(self.spec.sim_time_offset_seconds),
            data_time_offset_seconds=float(self.spec.data_time_offset_seconds),
            min_abs_dip_latitude=float(self.spec.min_abs_dip_latitude),
            low_latitude_scale=float(self.spec.low_latitude_scale),
            curve_scale_mode=self.spec.curve_scale_mode,
            curve_scale_value=float(self.curve_scale.value),
            curve_time_scale=float(self.time_scale.value),
            geo_lat_min=float(self.geo_lat_min.value),
            geo_lat_max=float(self.geo_lat_max.value),
            local_time_min=float(self.spec.local_time_min),
            local_time_max=float(self.spec.local_time_max),
            zoom_window=bool(self.spec.zoom_window),
            extra=dict(self.spec.extra),
        )

    def _set_widget_value(self, widget, value):
        if value != widget.value:
            widget.value = value

    def _apply_spec_to_widgets(self, spec):
        plot_values = set(self.plot_type.options.values())
        fill_values = set(self.fill.options.values())
        line_values = set(self.lines.options.values())
        component_values = set(self.ground_component.options.values())
        quantity_values = set(self.ground_quantity.options.values())
        max_time = int(self.time_index.end)
        time_start, time_end = [int(value) for value in spec.time_range]
        time_start = max(0, min(time_start, max_time))
        time_end = max(time_start, min(time_end, max_time))
        if time_start == 0 and time_end == 0 and max_time > 0:
            time_end = min(max_time, 60)

        self._set_widget_value(self.run_directory, spec.run_directory)
        self._set_widget_value(
            self.plot_type, spec.plot_type if spec.plot_type in plot_values else "ground_curve_map"
        )
        self._set_widget_value(self.time_index, max(0, min(int(spec.time_index), max_time)))
        self._set_widget_value(self.time_range, (time_start, time_end))
        self._set_widget_value(self.fill, spec.fill if spec.fill in fill_values else "Br")
        self._set_widget_value(self.lines, spec.lines if spec.lines in line_values else "none")
        self._set_widget_value(self.show_north, bool(spec.show_north))
        self._set_widget_value(self.show_south, bool(spec.show_south))
        self._set_widget_value(self.min_abs_lat, float(spec.hemisphere_min_abs_latitude))
        self._set_widget_value(self.station, str(spec.ground_station).upper())
        self._set_widget_value(
            self.ground_component,
            spec.ground_component if spec.ground_component in component_values else "Magnitude",
        )
        self._set_widget_value(
            self.ground_quantity,
            spec.ground_quantity if spec.ground_quantity in quantity_values else "dbdt",
        )
        self._set_widget_value(self.include_station_data, bool(spec.include_station_data))
        self._set_widget_value(self.show_inductive, bool(spec.show_inductive))
        self._set_widget_value(self.show_noninductive, bool(spec.show_noninductive))
        self._set_widget_value(self.show_reference_line, bool(spec.show_reference_line))
        self._set_widget_value(self.reference_time, str(spec.reference_time_of_day_utc))
        self._set_widget_value(self.curve_scale, float(spec.curve_scale_value))
        self._set_widget_value(self.time_scale, float(spec.curve_time_scale))
        self._set_widget_value(self.geo_lat_min, float(spec.geo_lat_min))
        self._set_widget_value(self.geo_lat_max, float(spec.geo_lat_max))

    def _set_status(self, message, *, error=False):
        prefix = "**Error:** " if error else ""
        self.status.object = f"{prefix}{message}" if message else ""

    def _load_run(self, event=None):
        if self._busy:
            return
        self._busy = True
        should_redraw = False
        try:
            clear_saved_field_view_cache()
            run_directory = self.run_directory.value
            expanded_run_directory = str(Path(run_directory).expanduser())
            if expanded_run_directory != self._loaded_run_directory:
                self.spec = figure_spec_from_run_defaults(run_directory)
            else:
                self.spec = self._current_spec()
            self.view = get_saved_field_view(self.spec)
            self.time_index.end = max(0, self.view.n_time - 1)
            self.time_range.end = max(0, self.view.n_time - 1)
            if self.time_range.value == (0, 0):
                self.time_range.value = (0, min(int(self.time_range.end), 60))
            self._apply_spec_to_widgets(self.spec)
            self._loaded_run_directory = expanded_run_directory
            self._set_status(f"Loaded `{Path(self.spec.run_directory).expanduser()}`.")
            should_redraw = True
        except Exception:
            self._set_status(traceback.format_exc(limit=6), error=True)
        finally:
            self._busy = False
        if should_redraw:
            self._redraw()

    def _control_changed(self, event=None):
        if self._busy:
            return
        if self.view is None:
            return
        self._redraw()

    def _redraw(self, event=None):
        if self._busy:
            return
        self._busy = True
        try:
            self.spec = self._current_spec()
            self._sync_visibility()
            view = self.view if self.view is not None else get_saved_field_view(self.spec)
            index = min(max(0, int(self.spec.time_index)), view.n_time - 1)
            time_text = view.timestamp_at_index(index).strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.object = f"**{time_text}**"
            if self.figure is not None:
                plt.close(self.figure)
            self.figure = render_pynamit_figure(self.spec)
            self.plot_pane.object = self.figure
            self._set_status("")
        except Exception:
            self._set_status(traceback.format_exc(limit=8), error=True)
        finally:
            self._busy = False

    def _save_figure(self, event=None):
        if self.figure is None:
            self._redraw()
        if self.figure is None:
            return
        try:
            path = Path(self.output_filename.value).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
            self._set_status(f"Saved figure to [{path}]({path})")
        except Exception:
            self._set_status(traceback.format_exc(limit=6), error=True)

    def _download_script(self):
        spec = self._current_spec()
        text = publication_script_for_spec(spec, output_path=self.output_filename.value)
        return StringIO(text)

    def _download_spec(self):
        return StringIO(self._current_spec().to_json())

    def _download_png(self):
        if self.figure is None:
            self._redraw()
        buffer = BytesIO()
        self.figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        return buffer

    def panel(self):
        """Return the Panel layout."""
        pn = self.pn
        mode_controls = pn.Card(
            self._control_row(self.run_directory, self.load_button, self.plot_type),
            self._control_row(self.time_index, self.time_label),
            title="Run",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        map_controls = pn.Card(
            self._control_row(
                self.fill, self.lines, self.show_north, self.show_south, self.min_abs_lat
            ),
            title="Map",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        ground_controls = pn.Card(
            self._control_row(
                self.station,
                self.ground_component,
                self.ground_quantity,
                self.include_station_data,
                self.show_inductive,
                self.show_noninductive,
            ),
            self._control_row(
                self.show_reference_line,
                self.reference_time,
                self.curve_scale,
                self.time_scale,
                self.geo_lat_min,
                self.geo_lat_max,
            ),
            self.time_range,
            title="Ground",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        output_controls = pn.Card(
            self._control_row(
                self.output_filename,
                self.save_button,
                self.script_download,
                self.spec_download,
                self.redraw_button,
            ),
            title="Output",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        self.map_controls = map_controls
        self.ground_controls = ground_controls
        self._sync_visibility()
        controls = pn.Column(
            mode_controls,
            map_controls,
            ground_controls,
            output_controls,
            self.status,
            min_width=360,
            max_width=720,
            sizing_mode="stretch_width",
            styles={"flex": "1 1 560px"},
        )
        plot_area = pn.Column(
            self.plot_pane, min_width=320, sizing_mode="stretch_both", styles={"flex": "4 1 760px"}
        )
        return pn.FlexBox(
            controls,
            plot_area,
            flex_direction="row",
            flex_wrap="wrap",
            align_items="flex-start",
            gap="14px",
            sizing_mode="stretch_both",
        )

    def _control_row(self, *objects):
        return self.pn.FlexBox(
            *objects,
            flex_direction="row",
            flex_wrap="wrap",
            align_items="flex-end",
            gap="8px 12px",
            sizing_mode="stretch_width",
        )

    def _sync_visibility(self):
        plot_type = self.plot_type.value
        if hasattr(self, "map_controls"):
            self.map_controls.visible = plot_type in {"global", "hemispheres"}
        if hasattr(self, "ground_controls"):
            self.ground_controls.visible = plot_type in {"ground_curve_map", "ground_timeseries"}
        self.time_index.visible = plot_type in {"global", "hemispheres", "input_summary"}
        self.time_range.visible = plot_type in {"ground_curve_map", "ground_timeseries"}
        self.ground_component.visible = plot_type == "ground_curve_map"
        self.include_station_data.visible = plot_type == "ground_timeseries"
        self.station.visible = plot_type == "ground_timeseries"


def build_pynamit_panel_app(run_directory=None):
    """Build and return the Panel layout for saved-run plotting."""
    return PynamitPanelApp(run_directory=run_directory).panel()


def servable(run_directory=None, title="PynaMIT Plot"):
    """Create a servable Panel app."""
    app = build_pynamit_panel_app(run_directory=run_directory)
    return app.servable(title=title)


__all__ = ["PynamitPanelApp", "build_pynamit_panel_app", "servable"]
