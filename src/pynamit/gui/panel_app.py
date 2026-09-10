"""Interactive Panel application for PynaMIT."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
from collections import deque
from dataclasses import replace
from functools import partial
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
from kompe.math import get_backend

from pynamit.gui.figure_settings_binding import (
    apply_figure_settings_to_widgets,
    current_figure_settings,
    manual_color_values,
    manual_line_values,
    set_widget_value,
)
from pynamit.plotting.figure_builder import render_figure
from pynamit.plotting.figure_settings import (
    FIGURE_DEFAULTS_FILENAME,
    MAP_FILL_OPTIONS,
    MAP_LINE_OPTIONS,
    PLOT_TYPE_OPTIONS,
    FigureSettings,
    publication_script,
)
from pynamit.plotting.figure_styles import (
    manual_color_control_units,
    manual_color_display_value,
    manual_color_limits,
    manual_line_parameters,
    map_line_keys,
)
from pynamit.plotting.plot_data import get_plot_data
from pynamit.simulation.config import INTEGRATORS
from pynamit.simulation.evolution import DEFAULT_ATOL, DEFAULT_DT_SECONDS, DEFAULT_RTOL

PANEL_PLOT_TYPE_OPTIONS = {label: key for key, label in PLOT_TYPE_OPTIONS.items()}
MAP_PLOT_TYPES = {"global", "hemispheres"}
GROUND_PLOT_TYPES = {"ground_curve_map", "ground_timeseries"}
MOVIE_PLOT_TYPES = MAP_PLOT_TYPES | {"input_summary"}
PANEL_LINE_UNITS = {"Phi": "kV", "W": "kV", "Jeq": "A"}
logger = logging.getLogger(__name__)


def _absolute_path(value):
    """Resolve a user-entered path without accepting an empty field."""
    text = str(value).strip()
    if not text:
        raise ValueError("Path cannot be empty.")
    return Path(text).expanduser().resolve()


def _panel():
    try:
        import panel as pn
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The Panel plotting app requires panel. Install it in the active environment, "
            "for example with `conda install -c conda-forge panel` or `pip install panel`."
        ) from exc
    pn.extension("modal", sizing_mode="stretch_width")
    return pn


def _has_pynamit_settings(directory):
    directory = Path(directory)
    return (directory / "settings.zarr").exists() or (directory / "settings.ncdf").exists()


def _default_simulation_directory():
    """Use an explicit preference or the current artifact directory."""
    configured = os.environ.get("PYNAMIT_SIMULATION_DIR")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(Path.cwd()) if _has_pynamit_settings(Path.cwd()) else ""


class PynamitGUI:
    """Prepare inputs, run simulations, and inspect their figures."""

    def __init__(self, simulation_directory=None):
        self.pn = _panel()
        directory = (
            _default_simulation_directory()
            if simulation_directory is None
            else str(Path(simulation_directory).expanduser().resolve())
        )
        self.figure_settings = FigureSettings(simulation_directory=directory)
        self.plot_data = None
        self.figure = None
        self._rendered_settings = None
        self._busy = False
        self._syncing_style_controls = False
        self._loaded_simulation_directory = None
        self._loaded_input_directory = None
        self._workflow_process = None
        self._workflow_cancelled = False

        self._build_mode_widgets()
        default_input_directory = str(Path.cwd() / "prepared_inputs")
        self._build_input_preparation_widgets(default_input_directory)
        self._build_simulation_widgets(default_input_directory)
        self._build_data_widgets()
        self._build_visualization_widgets()
        self._build_output_widgets()
        self._layout = self._build_layout()
        self._bind_callbacks()
        if self.pn.state.curdoc is not None:
            self.pn.state.on_session_destroyed(self._session_destroyed)
        if directory:
            self._load_simulation()
        else:
            self._set_status(
                "Choose a simulation or projected-input directory, then press **Load**."
            )

    def _build_mode_widgets(self):
        pn = self.pn
        self.app_mode = pn.widgets.Select(
            name="Mode",
            options={
                "Visualize simulation": "visualize",
                "Prepare example inputs": "prepare_example_inputs",
                "Run simulation": "run_simulation",
            },
            value="visualize",
            width=180,
        )
        self.simulation_directory = pn.widgets.TextInput(
            name="Simulation directory",
            value=self.figure_settings.simulation_directory,
            placeholder="Path to saved simulation or projected inputs",
            min_width=280,
        )
        self.load_button = pn.widgets.Button(name="Load", button_type="primary", width=85)
        self.plot_type = pn.widgets.Select(
            name="Plot",
            options=PANEL_PLOT_TYPE_OPTIONS,
            value=self.figure_settings.plot_type,
            width=180,
        )
        self.time_index = pn.widgets.IntSlider(
            name="Sample", start=0, end=0, value=0, step=1, min_width=280
        )
        self.time_label = pn.pane.Markdown("", width=260)
        self.time_range = pn.widgets.IntRangeSlider(
            name="Sample range (curves / movie)",
            start=0,
            end=0,
            value=(0, 0),
            step=1,
            min_width=280,
        )

    def _build_input_preparation_widgets(self, default_input_directory):
        pn = self.pn
        self.prepared_input_directory = pn.widgets.TextInput(
            name="Input package", value=default_input_directory, min_width=280
        )
        self.prepare_Nmax = pn.widgets.IntInput(name="Nmax", value=20, start=1, width=90)
        self.prepare_Mmax = pn.widgets.IntInput(name="Mmax", value=20, start=0, width=90)
        self.prepare_Ncs = pn.widgets.IntInput(name="Ncs", value=30, start=4, step=2, width=90)
        self.prepare_horizontal_basis = pn.widgets.Select(
            name="Basis",
            options={"Spherical harmonics": "SH", "Cubed sphere": "CS"},
            value="SH",
            width=170,
        )
        self.prepare_use_boundary_jr = pn.widgets.Checkbox(
            name="Boundary jr", value=True, width=110
        )
        self.prepare_use_wind = pn.widgets.Checkbox(name="u", value=False, width=70)
        self.prepare_use_q_eff = pn.widgets.Checkbox(name="Q_eff from u", value=False, width=120)
        self.prepare_button = pn.widgets.Button(
            name="Prepare example inputs", button_type="primary", width=220
        )
        self.example_parameters = {
            "event_time": pn.widgets.DatetimeInput(
                name="Event time (UTC)", value=datetime.datetime(2001, 5, 12, 21, 45), width=220
            ),
            "kp": pn.widgets.FloatInput(name="Kp", value=5, start=0, end=9, width=100),
            "starlight_conductance_S": pn.widgets.FloatInput(
                name="Starlight conductance (S)", value=1, width=200
            ),
            "solar_wind_speed_km_s": pn.widgets.FloatInput(
                name="Solar wind speed (km/s)", value=300, width=200
            ),
            "imf_By_nT": pn.widgets.FloatInput(name="IMF By (nT)", value=0, width=130),
            "imf_Bz_nT": pn.widgets.FloatInput(name="IMF Bz (nT)", value=-4, width=130),
            "dipole_tilt_deg": pn.widgets.FloatInput(name="Dipole tilt (°)", value=20, width=130),
            "f107_sfu": pn.widgets.FloatInput(name="F10.7 (sfu)", value=100, width=130),
            "amps_min_latitude_deg": pn.widgets.FloatInput(
                name="AMPS minimum latitude (°)", value=50, width=200
            ),
            "hwm_ap": pn.widgets.LiteralInput(name="HWM Ap inputs", value=(-1, 35), type=tuple),
        }

    def _build_simulation_widgets(self, default_input_directory):
        pn = self.pn
        self.simulation_input_directory = pn.widgets.TextInput(
            name="Input package", value=default_input_directory, min_width=280
        )
        self.new_simulation_directory = pn.widgets.TextInput(
            name="Simulation output", value=str(Path.cwd() / "simulation"), min_width=280
        )
        self.sim_final_time = pn.widgets.FloatInput(
            name="Final time (s)", value=100.0, start=0.0, width=120
        )
        self.sim_dt = pn.widgets.FloatInput(
            name="dt (s)", value=DEFAULT_DT_SECONDS, start=1e-12, width=110
        )
        self.sim_samples_per_write = pn.widgets.IntInput(
            name="Samples per save", value=200, start=1, width=150
        )
        self.sim_output_interval = pn.widgets.FloatInput(
            name="Output interval (s)", value=1.0, start=1e-12, width=150
        )
        self.sim_integrator = pn.widgets.Select(
            name="Integrator", options=list(INTEGRATORS.values()), value="euler", width=140
        )
        self.sim_rtol = pn.widgets.FloatInput(name="ODE rtol", value=DEFAULT_RTOL, width=130)
        self.sim_atol = pn.widgets.FloatInput(name="ODE atol (T)", value=DEFAULT_ATOL, width=150)
        self.sim_run_dynamic = pn.widgets.Checkbox(name="Dynamic", value=True, width=110)

        def update_integrator_controls(_event=None):
            method = self.sim_integrator.value
            dynamic = self.sim_run_dynamic.value
            self.sim_integrator.disabled = not dynamic
            self.sim_dt.disabled = not dynamic or method != "euler"
            self.sim_rtol.disabled = self.sim_atol.disabled = not dynamic or method in (
                "euler",
                "exponential",
            )

        self.sim_integrator.param.watch(update_integrator_controls, "value")
        self.sim_run_dynamic.param.watch(update_integrator_controls, "value")
        update_integrator_controls()
        self.sim_enable_pfac_coupling = pn.widgets.Checkbox(
            name="PFAC coupling", value=False, width=130
        )
        self.sim_enable_interhemispheric_coupling = pn.widgets.Checkbox(
            name="Interhemispheric coupling", value=False, width=190
        )
        self.sim_magnetic_boundary_shielding = pn.widgets.Checkbox(
            name="Boundary shielding", value=False, width=150
        )
        self.sim_sample_equilibrium = pn.widgets.Checkbox(
            name="Equilibrium", value=True, width=130
        )
        self.sim_interhemispheric_coupling_latitude = pn.widgets.FloatInput(
            name="Coupling latitude", value=50.0, width=140
        )
        self.simulation_inputs = {
            key: pn.widgets.Checkbox(name=label, value=False, disabled=True, width=150)
            for key, label in {
                "conductance": "Conductance",
                "boundary_jr": "Boundary current jr",
                "boundary_Br": "Boundary field Br",
                "u": "Wind u",
                "Q_eff": "Q_eff",
                "E_neutral_wind": "Neutral-wind E",
            }.items()
        }
        self.load_inputs_button = pn.widgets.Button(name="Load inputs", width=120)
        self.input_status = pn.pane.Str("Load a prepared input package to choose its inputs.")
        self.run_simulation_button = pn.widgets.Button(
            name="Run or resume", button_type="primary", width=150, disabled=True
        )

    def _build_data_widgets(self):
        pn = self.pn
        self.fill = pn.widgets.Select(
            name="Filled contours",
            options={label: key for key, label in MAP_FILL_OPTIONS.items()},
            value=self.figure_settings.fill,
            width=210,
        )
        self.lines = pn.widgets.Select(
            name="Contour lines",
            options={label: key for key, label in MAP_LINE_OPTIONS.items()},
            value=self.figure_settings.lines,
            width=210,
        )
        self.show_north = pn.widgets.Checkbox(
            name="North", value=self.figure_settings.show_north, width=90
        )
        self.show_south = pn.widgets.Checkbox(
            name="South", value=self.figure_settings.show_south, width=90
        )
        self.min_abs_lat = pn.widgets.FloatInput(
            name="Min |lat|",
            value=self.figure_settings.hemisphere_min_abs_latitude,
            start=0,
            end=89.9,
            width=130,
        )
        self.station = pn.widgets.TextInput(
            name="Station",
            value=self.figure_settings.ground_station,
            placeholder="Station code",
            width=120,
        )
        self.station_data_directory = pn.widgets.TextInput(
            name="Station-data directory",
            value=self.figure_settings.station_data_directory,
            placeholder="Optional local IAGA data directory",
            min_width=280,
        )
        self.show_station_labels = pn.widgets.Checkbox(
            name="Station labels", value=self.figure_settings.show_station_labels, width=130
        )
        self.ground_component = pn.widgets.Select(
            name="Component",
            options={
                "Magnitude": "Magnitude",
                "North": "North",
                "East": "East",
                "Down": "Down",
                "|North|": "AbsNorth",
                "|East|": "AbsEast",
                "|Down|": "AbsDown",
            },
            value=self.figure_settings.ground_component,
            width=150,
        )
        self.ground_quantity = pn.widgets.Select(
            name="Signal",
            options={"dB/dt": "dbdt", "B": "b"},
            value=self.figure_settings.ground_quantity,
            width=110,
        )
        self.include_station_data = pn.widgets.Checkbox(
            name="Measured", value=self.figure_settings.include_station_data, width=95
        )
        self.show_dynamic = pn.widgets.Checkbox(
            name="Dynamic", value=self.figure_settings.show_dynamic, width=100
        )
        self.show_equilibrium = pn.widgets.Checkbox(
            name="Equilibrium", value=self.figure_settings.show_equilibrium, width=130
        )
        self.show_difference = pn.widgets.Checkbox(
            name="Difference", value=self.figure_settings.show_difference, width=120
        )
        self.sim_time_offset = pn.widgets.FloatInput(
            name="Sim shift (s)",
            value=self.figure_settings.simulation_time_offset_seconds,
            width=130,
        )
        self.data_time_offset = pn.widgets.FloatInput(
            name="Data shift (s)", value=self.figure_settings.data_time_offset_seconds, width=130
        )
        self.dbdt_window_points = pn.widgets.IntInput(
            name="dB/dt pts",
            value=int(self.figure_settings.dbdt_window_points),
            start=1,
            end=20,
            width=120,
        )
        self.ground_model_lt_count = pn.widgets.IntInput(
            name="Model LT n",
            value=int(self.figure_settings.ground_model_lt_count),
            start=1,
            end=72,
            width=120,
        )
        self.ground_model_lat_count = pn.widgets.IntInput(
            name="Model lat n",
            value=int(self.figure_settings.ground_model_lat_count),
            start=1,
            end=60,
            width=125,
        )
        self.uniform_ground_longitude_count = pn.widgets.Checkbox(
            name="Uniform longitude count",
            value=self.figure_settings.uniform_ground_longitude_count,
            width=170,
        )
        self.show_pedersen_conductance_overlay = pn.widgets.Checkbox(
            name="Pedersen contours",
            value=self.figure_settings.show_pedersen_conductance_overlay,
            width=145,
        )
        self.show_hall_conductance_overlay = pn.widgets.Checkbox(
            name="Hall contours",
            value=self.figure_settings.show_hall_conductance_overlay,
            width=120,
        )

    def _build_visualization_widgets(self):
        pn = self.pn
        color_min, color_max = manual_color_values(self.figure_settings)
        line_start, line_interval, line_count = manual_line_values(self.figure_settings)
        self.show_reference_line = pn.widgets.Checkbox(
            name="Reference line", value=self.figure_settings.show_reference_line, width=130
        )
        self.reference_time = pn.widgets.TextInput(
            name="Ref. UTC", value=self.figure_settings.reference_time_of_day_utc, width=130
        )
        self.curve_scale_mode = pn.widgets.Select(
            name="Curve scale",
            options={"Manual": "manual", "Automatic": "auto"},
            value=self.figure_settings.curve_scale_mode,
            width=130,
        )
        self.curve_scale = pn.widgets.FloatInput(
            name="Scale value", value=self.figure_settings.curve_scale_value, start=0.01, width=120
        )
        self.time_scale = pn.widgets.FloatInput(
            name="Time x", value=self.figure_settings.curve_time_width_scale, start=0.1, width=110
        )
        self.low_lat_cutoff = pn.widgets.FloatInput(
            name="Low-lat selection",
            value=self.figure_settings.min_abs_dip_latitude,
            start=0.0,
            width=155,
        )
        self.low_lat_scale = pn.widgets.FloatInput(
            name="Low-lat x", value=self.figure_settings.low_latitude_scale, start=0.01, width=110
        )
        self.show_dip_equator_curve = pn.widgets.Checkbox(
            name="Dip equator", value=self.figure_settings.show_dip_equator_curve, width=120
        )
        self.show_low_lat_curve = pn.widgets.Checkbox(
            name="Low-lat curve", value=self.figure_settings.show_low_latitude_curve, width=125
        )
        self.color_scale_mode = pn.widgets.Select(
            name="Color scale",
            options={"Manual": "manual", "Percentile": "percentile"},
            value=self.figure_settings.color_scale_mode,
            width=130,
        )
        self.color_scale_percentile = pn.widgets.FloatInput(
            name="Percentile",
            value=self.figure_settings.color_scale_percentile,
            start=0.0,
            end=100.0,
            width=110,
        )
        self.manual_color_min = pn.widgets.FloatInput(name="Color min", value=color_min, width=150)
        self.manual_color_max = pn.widgets.FloatInput(name="Color max", value=color_max, width=150)
        self.line_first_abs_level = pn.widgets.FloatInput(
            name="First |line|", value=line_start, start=0.0, width=150
        )
        self.line_interval = pn.widgets.FloatInput(
            name="Line spacing", value=line_interval, start=0.0, width=150
        )
        self.line_levels_per_sign = pn.widgets.IntInput(
            name="Lines / sign", value=line_count, start=1, width=130
        )
        self.geo_lat_min = pn.widgets.FloatInput(
            name="Geo lat min", value=self.figure_settings.geo_lat_min, width=130
        )
        self.geo_lat_max = pn.widgets.FloatInput(
            name="Geo lat max", value=self.figure_settings.geo_lat_max, width=130
        )
        self.local_time_min = pn.widgets.FloatInput(
            name="LT min",
            value=self.figure_settings.local_time_min,
            start=0.0,
            end=24.0,
            width=110,
        )
        self.local_time_max = pn.widgets.FloatInput(
            name="LT max",
            value=self.figure_settings.local_time_max,
            start=0.0,
            end=24.0,
            width=110,
        )
        self.zoom_window = pn.widgets.Checkbox(
            name="Zoom window", value=self.figure_settings.zoom_window, width=130
        )
        self._sync_style_control_labels()

    def _sync_style_control_labels(self):
        """Show units for the selected fill and line fields."""
        fill_key = self.fill.value if self.fill.value != "none" else "Br"
        line_keys = map_line_keys(self.lines.value)
        line_key = line_keys[0] if line_keys else "Phi"
        color_units, _ = manual_color_control_units(fill_key)
        line_units = PANEL_LINE_UNITS[line_key]
        self.manual_color_min.label = f"Color min ({color_units})"
        self.manual_color_max.label = f"Color max ({color_units})"
        self.line_first_abs_level.label = f"First |line| ({line_units})"
        self.line_interval.label = f"Line spacing ({line_units})"

    def _reset_manual_color_controls(self):
        """Load the selected filled field's existing preset."""
        field_key = self.fill.value if self.fill.value != "none" else "Br"
        minimum, maximum = manual_color_limits(field_key)
        set_widget_value(self.manual_color_min, manual_color_display_value(field_key, minimum))
        set_widget_value(self.manual_color_max, manual_color_display_value(field_key, maximum))

    def _reset_manual_line_controls(self):
        """Load the selected line field's existing preset."""
        line_keys = map_line_keys(self.lines.value)
        start, interval, count = manual_line_parameters(line_keys[0] if line_keys else "Phi")
        set_widget_value(self.line_first_abs_level, start)
        set_widget_value(self.line_interval, interval)
        set_widget_value(self.line_levels_per_sign, count)

    def _build_output_widgets(self):
        pn = self.pn
        self._pending_overwrite = None
        self.redraw_button = pn.widgets.Button(name="Redraw", button_type="primary", width=95)
        self.save_button = pn.widgets.Button(name="Save figure", button_type="warning", width=120)
        self.save_movie_button = pn.widgets.Button(
            name="Save movie", button_type="warning", width=120
        )
        self.save_defaults_button = pn.widgets.Button(name="Save plot defaults", width=155)
        self.cancel_workflow_button = pn.widgets.Button(
            name="Stop workflow", button_type="danger", disabled=True, width=140
        )
        self.workflow_log = pn.pane.Str(
            "", height=180, styles={"overflow-y": "auto", "white-space": "pre-wrap"}
        )
        self.output_filename = pn.widgets.TextInput(
            name="Figure path", value=str(_absolute_path("pynamit_figure.png")), min_width=360
        )
        self.movie_filename = pn.widgets.TextInput(
            name="Movie path",
            value=str(_absolute_path(self.figure_settings.movie_filename)),
            min_width=360,
        )
        self.movie_fps = pn.widgets.FloatInput(
            name="FPS", value=self.figure_settings.movie_fps, start=0.1, width=90
        )
        self.overwrite_message = pn.pane.Str(
            "",
            styles={"overflow-wrap": "anywhere", "white-space": "pre-wrap"},
            sizing_mode="stretch_width",
        )
        self.confirm_overwrite_button = pn.widgets.Button(
            name="Overwrite", button_type="danger", width=110
        )
        self.cancel_overwrite_button = pn.widgets.Button(name="Cancel", width=90)
        self.overwrite_modal = pn.Modal(
            pn.Column(
                pn.pane.Markdown("### Replace existing file?"),
                self.overwrite_message,
                self._control_row(self.cancel_overwrite_button, self.confirm_overwrite_button),
                sizing_mode="stretch_width",
            ),
            open=False,
            background_close=False,
            show_close_button=False,
            width=620,
            max_width=620,
        )
        self.script_download = pn.widgets.FileDownload(
            label="Download .py",
            filename="pynamit_figure.py",
            callback=self._download_script,
            button_type="success",
            width=130,
        )
        self.figure_settings_download = pn.widgets.FileDownload(
            label="Download settings",
            filename="pynamit_figure.json",
            callback=self._download_settings,
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

    def _bind_callbacks(self):
        self.load_button.on_click(self._load_simulation)
        self.prepare_button.on_click(self._prepare_example_inputs)
        self.run_simulation_button.on_click(self._run_simulation)
        self.redraw_button.on_click(self._redraw)
        self.save_button.on_click(self._save_figure)
        self.save_movie_button.on_click(self._save_movie)
        self.save_defaults_button.on_click(self._save_plot_defaults)
        self.cancel_workflow_button.on_click(self._cancel_workflow)
        self.load_inputs_button.on_click(self._sync_simulation_input_availability)
        self.confirm_overwrite_button.on_click(self._confirm_overwrite)
        self.cancel_overwrite_button.on_click(self._cancel_overwrite)
        self.app_mode.param.watch(self._mode_changed, "value")
        self.simulation_directory.param.watch(self._directory_changed, "value")
        self.simulation_input_directory.param.watch(
            self._sync_simulation_input_availability, "value"
        )
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
            self.station_data_directory,
            self.show_station_labels,
            self.ground_component,
            self.ground_quantity,
            self.include_station_data,
            self.show_dynamic,
            self.show_equilibrium,
            self.show_difference,
            self.sim_time_offset,
            self.data_time_offset,
            self.dbdt_window_points,
            self.ground_model_lt_count,
            self.ground_model_lat_count,
            self.uniform_ground_longitude_count,
            self.show_pedersen_conductance_overlay,
            self.show_hall_conductance_overlay,
            self.show_reference_line,
            self.reference_time,
            self.curve_scale_mode,
            self.curve_scale,
            self.time_scale,
            self.low_lat_cutoff,
            self.low_lat_scale,
            self.show_dip_equator_curve,
            self.show_low_lat_curve,
            self.color_scale_mode,
            self.color_scale_percentile,
            self.manual_color_min,
            self.manual_color_max,
            self.line_first_abs_level,
            self.line_interval,
            self.line_levels_per_sign,
            self.geo_lat_min,
            self.geo_lat_max,
            self.local_time_min,
            self.local_time_max,
            self.zoom_window,
        ):
            widget.param.watch(self._control_changed, "value")

    def _set_status(self, message, *, error=False):
        prefix = "**Error:** " if error else ""
        self.status.object = f"{prefix}{message}" if message else ""

    def _session_destroyed(self, session_context):
        """Release resources when the server expires this session."""
        self._cancel_workflow()
        self._discard_figure()

    def _mode_changed(self, event=None):
        if self.app_mode.value == "run_simulation":
            self._sync_simulation_input_availability()
        self._sync_visibility()

    def _show_error(self, error):
        """Show a concise error; keep the traceback in server logs."""
        logger.exception("PynaMIT GUI operation failed")
        self._set_status(f"{type(error).__name__}: {error}", error=True)

    def _sync_simulation_input_availability(self, event=None):
        """Read the inputs actually present in the chosen package."""
        from pynamit.simulation.input_manifest import available_prepared_inputs

        try:
            directory = _absolute_path(self.simulation_input_directory.value)
            available = set(available_prepared_inputs(directory))
        except (OSError, ValueError) as exc:
            self._loaded_input_directory = None
            for widget in self.simulation_inputs.values():
                widget.disabled = True
                widget.value = False
            self.run_simulation_button.disabled = True
            self.input_status.object = str(exc)
            return False
        changed = directory != self._loaded_input_directory
        for key, widget in self.simulation_inputs.items():
            widget.disabled = key not in available
            if changed or widget.disabled:
                widget.value = key in available
        self._loaded_input_directory = directory
        self.run_simulation_button.disabled = not available
        self.input_status.object = "Available inputs: " + ", ".join(sorted(available))
        return True

    def _selected_simulation_inputs(self):
        return tuple(key for key, widget in self.simulation_inputs.items() if widget.value)

    async def _run_workflow(self, workflow, parameters):
        """Run a scientific workflow outside the Panel process."""
        self.workflow_log.object = ""
        self._workflow_cancelled = False
        backend = get_backend()
        environment = os.environ.copy()
        if backend == "jax":
            import jax

            environment["JAX_ENABLE_X64"] = str(jax.config.x64_enabled)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "pynamit.gui.workflow_runner",
            workflow,
            "--backend",
            backend,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._workflow_process = process
        self.cancel_workflow_button.disabled = False
        self._sync_visibility()
        lines = deque(maxlen=80)
        try:
            process.stdin.write(json.dumps(parameters).encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            async for line in process.stdout:
                lines.append(line.decode("utf-8", errors="replace"))
                self.workflow_log.object = "".join(lines)
            returncode = await process.wait()
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            self._workflow_process = None
            self.cancel_workflow_button.disabled = True
            self._sync_visibility()
        if self._workflow_cancelled:
            self._set_status(
                "Workflow stopped. Completed artifacts and checkpoints were retained."
            )
            return False
        if returncode:
            logger.error("PynaMIT workflow output:\n%s", self.workflow_log.object)
            raise RuntimeError(f"Workflow exited with code {returncode}; see the workflow log.")
        return True

    def _cancel_workflow(self, event=None):
        process = self._workflow_process
        if process is not None and process.returncode is None:
            self._workflow_cancelled = True
            process.terminate()

    async def _prepare_example_inputs(self, event=None):
        if self._busy:
            return
        self._busy = True
        self.prepare_button.loading = True
        try:
            if self.prepare_use_q_eff.value and not self.prepare_use_wind.value:
                raise ValueError("Q_eff from u requires the wind input source.")
            input_directory = _absolute_path(self.prepared_input_directory.value)
            if input_directory.exists() and any(input_directory.iterdir()):
                raise FileExistsError(
                    "Input preparation replaces artifacts. Choose a new or empty input directory."
                )
            parameters = {name: widget.value for name, widget in self.example_parameters.items()}
            parameters["event_time"] = parameters["event_time"].isoformat()
            parameters.update(
                input_directory=str(input_directory),
                Nmax=self.prepare_Nmax.value,
                Mmax=self.prepare_Mmax.value,
                Ncs=self.prepare_Ncs.value,
                use_wind=self.prepare_use_wind.value,
                use_Q_eff=self.prepare_use_q_eff.value,
                use_boundary_jr=self.prepare_use_boundary_jr.value,
                horizontal_basis_kind=self.prepare_horizontal_basis.value,
            )
            self._set_status("Preparing inputs; progress appears in the workflow log.")
            if not await self._run_workflow("prepare", parameters):
                return
            set_widget_value(self.prepared_input_directory, str(input_directory))
            set_widget_value(self.simulation_input_directory, str(input_directory))
            self._sync_simulation_input_availability()
            self._set_status(f"Prepared inputs in {input_directory}.")
        except Exception as exc:
            self._show_error(exc)
        finally:
            self.prepare_button.loading = False
            self._busy = False

    async def _run_simulation(self, event=None):
        if self._busy:
            return
        self._busy = True
        self.run_simulation_button.loading = True
        completed = False
        try:
            if not self._sync_simulation_input_availability():
                raise ValueError("Load a valid prepared input package before running.")
            enabled_inputs = self._selected_simulation_inputs()
            if not enabled_inputs:
                raise ValueError("Select at least one prepared input dataset.")
            input_directory = _absolute_path(self.simulation_input_directory.value)
            simulation_directory = _absolute_path(self.new_simulation_directory.value)
            parameters = dict(
                input_directory=str(input_directory),
                simulation_directory=str(simulation_directory),
                enabled_inputs=enabled_inputs,
                final_time=self.sim_final_time.value,
                dt=self.sim_dt.value
                if self.sim_run_dynamic.value and self.sim_integrator.value == "euler"
                else None,
                output_interval=self.sim_output_interval.value,
                rtol=self.sim_rtol.value,
                atol=self.sim_atol.value,
                samples_per_write=self.sim_samples_per_write.value,
                enable_pfac_coupling=self.sim_enable_pfac_coupling.value,
                enable_interhemispheric_coupling=self.sim_enable_interhemispheric_coupling.value,
                interhemispheric_coupling_latitude=self.sim_interhemispheric_coupling_latitude.value,
                run_dynamic=self.sim_run_dynamic.value,
                sample_equilibrium=self.sim_sample_equilibrium.value,
                integrator=self.sim_integrator.value,
                magnetic_boundary_shielding=self.sim_magnetic_boundary_shielding.value,
            )
            self._set_status("Running simulation; progress appears in the workflow log.")
            completed = await self._run_workflow("simulate", parameters)
            if completed:
                set_widget_value(self.new_simulation_directory, str(simulation_directory))
                set_widget_value(self.simulation_directory, str(simulation_directory))
                set_widget_value(self.app_mode, "visualize")
        except Exception as exc:
            self._show_error(exc)
        finally:
            self.run_simulation_button.loading = False
            self._busy = False
        if completed:
            self._load_simulation()

    def _discard_figure(self):
        """Release the displayed figure and its settings."""
        if self.figure is not None:
            plt.close(self.figure)
        self.figure = None
        self._rendered_settings = None
        self.plot_pane.object = None

    def _directory_changed(self, event=None):
        """Never label previously loaded arrays with a new directory."""
        self.plot_data = None
        self._loaded_simulation_directory = None
        self._discard_figure()
        self.time_label.object = ""
        self._sync_visibility()
        self._set_status("Directory changed. Press Load to read its saved data and plot defaults.")

    def _load_simulation(self, event=None):
        if self._busy:
            return
        self._busy = True
        loaded = False
        try:
            directory = str(_absolute_path(self.simulation_directory.value))
            settings = FigureSettings.from_simulation_directory(directory)
            plot_data = get_plot_data(settings)
            datasets = plot_data.results.datasets
            has_dynamic = "dynamic" in datasets
            has_equilibrium = "equilibrium" in datasets
            available_plots = set(PLOT_TYPE_OPTIONS)
            if not plot_data.has_model_output:
                available_plots = {"input_summary"}
            elif not has_dynamic:
                available_plots -= GROUND_PLOT_TYPES
            if not plot_data.available_inputs:
                available_plots.discard("input_summary")
            plot_type = settings.plot_type
            if plot_type not in available_plots:
                plot_type = "global" if plot_data.has_model_output else "input_summary"
            maximum = max(0, plot_data.n_time - 1)
            start = min(settings.time_range[0], maximum)
            end = max(start, min(settings.time_range[1], maximum))
            settings = replace(
                settings,
                plot_type=plot_type,
                time_index=min(settings.time_index, maximum),
                time_range=(start, end),
                show_dynamic=has_dynamic and settings.show_dynamic,
                show_equilibrium=has_equilibrium
                and (settings.show_equilibrium or not has_dynamic),
                show_difference=has_dynamic and has_equilibrium and settings.show_difference,
            )
            self._discard_figure()
            self.time_index.value = 0
            self.time_range.value = (0, 0)
            self.time_index.end = maximum
            self.time_range.end = maximum
            self.plot_type.options = {
                label: key
                for label, key in PANEL_PLOT_TYPE_OPTIONS.items()
                if key in available_plots
            }
            apply_figure_settings_to_widgets(self, settings)
            set_widget_value(self.movie_filename, str(_absolute_path(self.movie_filename.value)))
            self.figure_settings = settings
            self.plot_data = plot_data
            self._loaded_simulation_directory = directory
            loaded = True
        except Exception as exc:
            self.plot_data = None
            self._loaded_simulation_directory = None
            self._discard_figure()
            self._show_error(exc)
        finally:
            self._busy = False
            self._sync_visibility()
        if loaded:
            self._redraw()

    def _control_changed(self, event=None):
        if self._busy or self._syncing_style_controls:
            return
        if event is not None and event.obj in {self.fill, self.lines}:
            self._syncing_style_controls = True
            try:
                if event.obj is self.fill:
                    self._reset_manual_color_controls()
                else:
                    self._reset_manual_line_controls()
                self._sync_style_control_labels()
            finally:
                self._syncing_style_controls = False
        self._sync_visibility()
        if self.app_mode.value == "visualize" and self.plot_data is not None:
            self._set_status(
                "Controls changed. Press Redraw; figure and script exports still describe the displayed figure."
            )

    def _require_loaded_directory(self):
        directory = str(_absolute_path(self.simulation_directory.value))
        if self.plot_data is None or directory != self._loaded_simulation_directory:
            raise ValueError("Press Load before using this simulation directory.")
        return directory

    def _redraw(self, event=None):
        if self._busy:
            return
        self._busy = True
        try:
            self._require_loaded_directory()
            settings = current_figure_settings(self)
            if settings.plot_type == "ground_timeseries" and not settings.ground_station.strip():
                raise ValueError("Choose a station code for the ground time series.")
            figure = render_figure(settings, plot_data=self.plot_data)
            self._discard_figure()
            self.figure = figure
            self._rendered_settings = settings
            self.figure_settings = settings
            self.plot_pane.object = figure
            time_text = self.plot_data.timestamp_at_index(settings.time_index).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self.time_label.object = f"**{time_text}**"
            self._set_status(f"Loaded {self._loaded_simulation_directory}.")
        except Exception as exc:
            self._show_error(exc)
        finally:
            self._busy = False
            self._sync_visibility()

    def _save_figure(self, event=None):
        if self._busy:
            return
        if self.figure is None:
            self._redraw()
        if self.figure is None:
            return
        try:
            path = self._output_widget_path(self.output_filename)
            self._save_or_confirm_overwrite(path, partial(self._write_figure, figure=self.figure))
        except Exception as exc:
            self._show_error(exc)

    def _write_figure(self, path, *, figure):
        """Save the figure selected when the overwrite was requested."""
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        self._set_status(f"Saved figure to {path}")

    def _save_plot_defaults(self, event=None):
        if self._busy:
            return
        try:
            directory = self._require_loaded_directory()
            settings = current_figure_settings(self)
            path = Path(directory) / FIGURE_DEFAULTS_FILENAME
            self._save_or_confirm_overwrite(
                path, partial(self._write_plot_defaults, settings=settings)
            )
        except Exception as exc:
            self._show_error(exc)

    def _write_plot_defaults(self, path, *, settings):
        settings.save_defaults(path.parent)
        self._set_status(f"Saved plot defaults to {path}")

    def _save_movie(self, event=None):
        if self._busy:
            return
        try:
            self._require_loaded_directory()
            settings = current_figure_settings(self)
            path = self._output_widget_path(self.movie_filename)
            self._save_or_confirm_overwrite(
                path,
                lambda path: self.pn.state.execute(
                    partial(self._write_movie, path, settings=settings)
                ),
            )
        except Exception as exc:
            self._show_error(exc)

    async def _write_movie(self, path, *, settings):
        """Use the settings selected before overwrite confirmation."""
        if self._busy:
            return
        self._busy = True
        self.save_movie_button.loading = True
        try:
            self._set_status("Rendering movie; progress appears in the workflow log.")
            if await self._run_workflow(
                "movie", {"settings": settings.to_dict(), "output_path": str(path)}
            ):
                self._set_status(f"Saved movie to {path}")
        except Exception as exc:
            self._show_error(exc)
        finally:
            self.save_movie_button.loading = False
            self._busy = False

    def _output_widget_path(self, widget):
        """Normalize an output widget to its absolute path."""
        path = _absolute_path(widget.value)
        set_widget_value(widget, str(path))
        return path

    def _save_or_confirm_overwrite(self, path, save):
        """Save a new file or confirm its replacement."""
        if path.exists():
            if not path.is_file():
                raise IsADirectoryError(f"Output path is not a file: {path}")
            self._pending_overwrite = (path, save)
            self.overwrite_message.object = f"{path}\n\nThis file already exists. Overwrite it?"
            self.overwrite_modal.open = True
            self._set_status(f"{path} already exists. Confirm or cancel the overwrite.")
            return
        save(path)

    def _confirm_overwrite(self, event=None):
        """Run the captured save after overwrite confirmation."""
        pending = self._pending_overwrite
        self._pending_overwrite = None
        self.overwrite_modal.open = False
        if pending is not None:
            path, save = pending
            try:
                save(path)
            except Exception as exc:
                self._show_error(exc)

    def _cancel_overwrite(self, event=None):
        """Cancel without touching the existing file."""
        pending = self._pending_overwrite
        self._pending_overwrite = None
        self.overwrite_modal.open = False
        if pending is not None:
            self._set_status(f"Save cancelled; existing file left unchanged: {pending[0]}.")

    def _download_script(self):
        if self._rendered_settings is None:
            raise ValueError("Render a figure before exporting its script.")
        return StringIO(
            publication_script(self._rendered_settings, output_path=self.output_filename.value)
        )

    def _download_settings(self):
        if self._rendered_settings is None:
            raise ValueError("Render a figure before exporting its settings.")
        return StringIO(self._rendered_settings.to_json())

    def panel(self):
        """Return this session's existing Panel layout."""
        return self._layout

    def _build_layout(self):
        """Build controls once, before binding their callbacks."""
        pn = self.pn
        app_controls = self._control_row(pn.pane.Markdown("## PynaMIT", width=150), self.app_mode)
        mode_controls = pn.Card(
            self._control_row(
                self.simulation_directory, self.load_button, self.plot_type, self.redraw_button
            ),
            self._control_row(self.time_index, self.time_label),
            self.time_range,
            title="Simulation data",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        prepare_controls = pn.Card(
            self._control_row(self.prepared_input_directory, self.prepare_button),
            self._control_row(
                self.prepare_Nmax,
                self.prepare_Mmax,
                self.prepare_Ncs,
                self.prepare_horizontal_basis,
            ),
            self._control_row(
                self.prepare_use_boundary_jr, self.prepare_use_wind, self.prepare_use_q_eff
            ),
            pn.Card(
                "Editable example scenario; all times are UTC.",
                self._control_row(*self.example_parameters.values()),
                title="Empirical model inputs",
                collapsed=True,
            ),
            title="Input Preparation",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        simulation_controls = pn.Card(
            self._control_row(self.simulation_input_directory, self.load_inputs_button),
            self.input_status,
            self._control_row(*self.simulation_inputs.values()),
            self._control_row(self.new_simulation_directory),
            self._control_row(self.sim_final_time, self.sim_dt, self.sim_integrator),
            self._control_row(self.sim_output_interval, self.sim_samples_per_write),
            self._control_row(self.sim_rtol, self.sim_atol),
            self._control_row(
                self.sim_enable_pfac_coupling,
                self.sim_enable_interhemispheric_coupling,
                self.sim_magnetic_boundary_shielding,
                self.sim_run_dynamic,
                self.sim_sample_equilibrium,
                self.sim_interhemispheric_coupling_latitude,
            ),
            self._control_row(self.run_simulation_button),
            title="Run or resume a simulation",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        data_controls = pn.Card(
            self._control_row(self.fill, self.lines),
            self._control_row(
                self.show_dynamic,
                self.show_equilibrium,
                self.include_station_data,
                self.show_difference,
            ),
            self._control_row(self.station, self.ground_component, self.ground_quantity),
            self._control_row(self.station_data_directory, self.show_station_labels),
            self._control_row(
                self.sim_time_offset, self.data_time_offset, self.dbdt_window_points
            ),
            self._control_row(
                self.ground_model_lt_count,
                self.ground_model_lat_count,
                self.uniform_ground_longitude_count,
            ),
            self._control_row(
                self.show_pedersen_conductance_overlay, self.show_hall_conductance_overlay
            ),
            self._control_row(self.show_north, self.show_south),
            title="Data",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        visualization_controls = pn.Card(
            self._control_row(
                self.geo_lat_min,
                self.geo_lat_max,
                self.local_time_min,
                self.local_time_max,
                self.zoom_window,
            ),
            self._control_row(
                self.show_dip_equator_curve,
                self.show_low_lat_curve,
                self.low_lat_cutoff,
                self.low_lat_scale,
            ),
            self._control_row(
                self.curve_scale_mode,
                self.curve_scale,
                self.time_scale,
                self.color_scale_mode,
                self.color_scale_percentile,
            ),
            self._control_row(self.manual_color_min, self.manual_color_max),
            self._control_row(
                self.line_first_abs_level, self.line_interval, self.line_levels_per_sign
            ),
            self._control_row(self.min_abs_lat),
            self._control_row(self.show_reference_line, self.reference_time),
            title="Visualization",
            collapsed=True,
            sizing_mode="stretch_width",
        )
        output_controls = pn.Card(
            self._control_row(
                self.output_filename,
                self.save_button,
                self.script_download,
                self.figure_settings_download,
                self.save_defaults_button,
            ),
            self._control_row(self.movie_filename, self.movie_fps, self.save_movie_button),
            title="Output",
            collapsed=True,
            sizing_mode="stretch_width",
        )
        self.mode_controls = mode_controls
        self.prepare_controls = prepare_controls
        self.simulation_controls = simulation_controls
        self.data_controls = data_controls
        self.visualization_controls = visualization_controls
        self.output_controls = output_controls
        self.workflow_controls = pn.Card(
            self.workflow_log, self.cancel_workflow_button, title="Workflow log", collapsed=False
        )
        controls = pn.Column(
            app_controls,
            mode_controls,
            self.status,
            prepare_controls,
            simulation_controls,
            data_controls,
            visualization_controls,
            output_controls,
            self.workflow_controls,
            min_width=320,
            max_width=440,
            sizing_mode="stretch_width",
            styles={"flex": "1 1 380px"},
        )
        self.plot_area = pn.Column(
            self.plot_pane, min_width=360, sizing_mode="stretch_both", styles={"flex": "3 1 600px"}
        )
        self._sync_visibility()
        return pn.Column(
            pn.FlexBox(
                controls,
                self.plot_area,
                flex_direction="row",
                flex_wrap="wrap",
                align_items="flex-start",
                gap="14px",
                sizing_mode="stretch_both",
            ),
            self.overwrite_modal,
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
        app_mode = self.app_mode.value
        is_visualize_mode = app_mode == "visualize"
        is_prepare_mode = app_mode == "prepare_example_inputs"
        is_simulation_mode = app_mode == "run_simulation"
        self.mode_controls.visible = is_visualize_mode
        self.prepare_controls.visible = is_prepare_mode
        self.simulation_controls.visible = is_simulation_mode
        self.data_controls.visible = is_visualize_mode
        self.visualization_controls.visible = is_visualize_mode
        self.output_controls.visible = is_visualize_mode
        self.workflow_controls.visible = (
            not is_visualize_mode
            or self._workflow_process is not None
            or bool(self.workflow_log.object)
        )
        self.plot_area.visible = is_visualize_mode
        loaded = self.plot_data is not None
        self.redraw_button.disabled = not loaded
        self.save_defaults_button.disabled = not loaded
        self.save_movie_button.disabled = not loaded
        self.save_button.disabled = self.figure is None
        self.script_download.disabled = self._rendered_settings is None
        self.figure_settings_download.disabled = self._rendered_settings is None
        datasets = self.plot_data.results.datasets if loaded else {}
        self.show_dynamic.disabled = "dynamic" not in datasets
        self.show_equilibrium.disabled = "equilibrium" not in datasets
        self.show_difference.disabled = (
            self.show_dynamic.disabled or self.show_equilibrium.disabled
        )

        plot_type = self.plot_type.value
        is_map = plot_type in MAP_PLOT_TYPES
        is_input = plot_type == "input_summary"
        is_ground_curve = plot_type == "ground_curve_map"
        is_ground_timeseries = plot_type == "ground_timeseries"
        is_ground = plot_type in GROUND_PLOT_TYPES

        self.time_index.visible = is_map or is_input
        self.time_range.visible = True
        self.fill.visible = is_map
        self.lines.visible = is_map
        self.show_dynamic.visible = is_map or is_ground
        self.show_equilibrium.visible = is_map or is_ground
        self.show_difference.visible = is_map
        self.show_north.visible = plot_type == "hemispheres"
        self.show_south.visible = plot_type == "hemispheres"

        self.station.visible = is_ground_timeseries
        self.station_data_directory.visible = is_ground
        self.show_station_labels.visible = is_ground_curve and self.include_station_data.value
        self.ground_component.visible = is_ground_curve
        self.ground_quantity.visible = is_ground
        self.include_station_data.visible = is_ground
        self.sim_time_offset.visible = is_ground
        self.data_time_offset.visible = is_ground
        self.dbdt_window_points.visible = is_ground and self.ground_quantity.value == "dbdt"
        show_model_grid_controls = is_ground_curve and not self.include_station_data.value
        self.ground_model_lt_count.visible = show_model_grid_controls
        self.ground_model_lat_count.visible = show_model_grid_controls
        self.uniform_ground_longitude_count.visible = show_model_grid_controls
        self.show_pedersen_conductance_overlay.visible = is_ground_curve
        self.show_hall_conductance_overlay.visible = is_ground_curve

        self.show_reference_line.visible = is_ground
        self.reference_time.visible = is_ground and self.show_reference_line.value
        self.min_abs_lat.visible = plot_type in {"hemispheres", "input_summary"}
        self.geo_lat_min.visible = is_ground_curve
        self.geo_lat_max.visible = is_ground_curve
        self.local_time_min.visible = is_ground_curve
        self.local_time_max.visible = is_ground_curve
        self.zoom_window.visible = is_ground_curve
        self.curve_scale_mode.visible = is_ground_curve
        self.curve_scale.visible = is_ground_curve and self.curve_scale_mode.value == "manual"
        self.time_scale.visible = is_ground_curve
        self.low_lat_cutoff.visible = is_ground_curve
        self.low_lat_scale.visible = is_ground_curve
        self.show_dip_equator_curve.visible = is_ground_curve
        self.show_low_lat_curve.visible = is_ground_curve
        self.color_scale_mode.visible = is_map or is_input
        has_color_scale = is_map or is_input
        self.color_scale_percentile.visible = (
            has_color_scale and self.color_scale_mode.value == "percentile"
        )
        show_manual_color = (
            is_map and self.fill.value != "none" and self.color_scale_mode.value == "manual"
        )
        self.manual_color_min.visible = show_manual_color
        self.manual_color_max.visible = show_manual_color
        show_line_controls = is_map and self.lines.value != "none"
        self.line_first_abs_level.visible = show_line_controls
        self.line_interval.visible = show_line_controls
        self.line_levels_per_sign.visible = show_line_controls

        can_make_movie = plot_type in MOVIE_PLOT_TYPES
        self.movie_filename.visible = can_make_movie
        self.movie_fps.visible = can_make_movie
        self.save_movie_button.visible = can_make_movie


def build_gui(simulation_directory=None):
    """Build and return the PynaMIT GUI layout."""
    return PynamitGUI(simulation_directory=simulation_directory).panel()


def servable(simulation_directory=None, title="PynaMIT Plot"):
    """Create a servable Panel app."""
    app = build_gui(simulation_directory=simulation_directory)
    return app.servable(title=title)


__all__ = ["PynamitGUI", "build_gui", "servable"]
