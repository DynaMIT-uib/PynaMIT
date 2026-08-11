"""Panel frontend for saved PynaMIT visualization runs."""

from __future__ import annotations

import traceback
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt

from pynamit.simulation.config import INTEGRATORS
from pynamit.visualization.figure_builder import render_pynamit_figure, save_pynamit_movie
from pynamit.visualization.figure_context import clear_saved_field_view_cache, get_saved_field_view
from pynamit.visualization.figure_specs import (
    MAP_FILL_OPTIONS,
    MAP_LINE_OPTIONS,
    PLOT_TYPE_OPTIONS,
    figure_spec_from_run_defaults,
    publication_script_for_spec,
)
from pynamit.visualization.figure_styles import (
    manual_color_control_units,
    manual_color_display_value,
    manual_color_limits,
    manual_line_parameters,
    map_line_keys,
)
from pynamit.visualization.panel_spec_binding import (
    apply_figure_spec_to_widgets,
    current_figure_spec,
    set_widget_value,
)

PANEL_PLOT_TYPE_OPTIONS = {label: key for key, label in PLOT_TYPE_OPTIONS.items()}
MAP_PLOT_TYPES = {"global", "hemispheres"}
GROUND_PLOT_TYPES = {"ground_curve_map", "ground_timeseries"}
MOVIE_PLOT_TYPES = MAP_PLOT_TYPES | {"input_summary"}
PANEL_LINE_UNITS = {"Phi": "kV", "W": "kV", "Jeq": "A"}


def _manual_color_values(spec):
    """Return manual limits for the selected fill."""
    field_key = spec.fill if spec.fill != "none" else "Br"
    if spec.manual_color_min is not None:
        minimum, maximum = float(spec.manual_color_min), float(spec.manual_color_max)
    else:
        minimum, maximum = manual_color_limits(field_key)
    return (
        manual_color_display_value(field_key, minimum),
        manual_color_display_value(field_key, maximum),
    )


def _manual_line_values(spec):
    """Return manual parameters for the selected overlay."""
    if spec.line_first_abs_level is not None:
        return (
            float(spec.line_first_abs_level),
            float(spec.line_interval),
            int(spec.line_levels_per_sign),
        )
    line_keys = map_line_keys(spec.lines)
    return manual_line_parameters(line_keys[0] if line_keys else "Phi")


def _absolute_output_path(value):
    """Return one user-entered output path as an absolute path."""
    text = str(value).strip()
    if not text:
        raise ValueError("Output path cannot be empty.")
    return Path(text).expanduser().resolve()


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


def _has_pynamit_settings(directory):
    directory = Path(directory)
    return (directory / "settings.zarr").exists() or (directory / "settings.ncdf").exists()


def _default_run_directory():
    cwd = Path(".")
    candidates = [cwd]
    candidates.extend(sorted(cwd.glob("results/*")))
    candidates.extend(sorted(cwd.glob("projections/*")))
    candidates.extend(sorted(cwd.glob("mage_output/*/resolutions/*/runs/*")))
    candidates.extend(sorted(cwd.glob("mage_output/*/resolutions/*/projections/*")))
    candidates.extend([Path("sim_dir"), Path("notebooks/sim_dir")])
    for candidate in candidates:
        if _has_pynamit_settings(candidate):
            return str(candidate)
    return "."


class PynamitPanelApp:
    """Interactive Panel application for saved PynaMIT figures."""

    def __init__(self, run_directory=None):
        self.pn = _panel()
        self.spec = figure_spec_from_run_defaults(str(run_directory or _default_run_directory()))
        self.view = None
        self.figure = None
        self._busy = False
        self._syncing_style_controls = False
        self._loaded_run_directory = None

        self._build_run_widgets()
        default_input_directory = str(
            Path(self.spec.run_directory).expanduser() / "prepared_inputs"
        )
        self._build_input_preparation_widgets(default_input_directory)
        self._build_simulation_widgets(default_input_directory)
        self._build_data_widgets()
        self._build_visualization_widgets()
        self._build_output_widgets()
        self._bind_callbacks()
        self._load_run()

    def _build_run_widgets(self):
        pn = self.pn
        self.app_mode = pn.widgets.Select(
            label="Mode",
            options={
                "Visualize run": "visualize",
                "Prepare inputs": "prepare_inputs",
                "Run simulation": "run_simulation",
            },
            value="visualize",
            width=180,
        )
        self.run_directory = pn.widgets.TextInput(
            label="Run directory", value=self.spec.run_directory, min_width=280
        )
        self.load_button = pn.widgets.Button(label="Load", color="primary", width=85)
        self.plot_type = pn.widgets.Select(
            label="Plot",
            options=PANEL_PLOT_TYPE_OPTIONS,
            value=(
                self.spec.plot_type
                if self.spec.plot_type in PLOT_TYPE_OPTIONS
                else "ground_curve_map"
            ),
            width=180,
        )
        self.time_index = pn.widgets.IntSlider(
            label="Time", start=0, end=0, value=0, step=1, min_width=320
        )
        self.time_label = pn.pane.Markdown("", width=260)
        self.time_range = pn.widgets.IntRangeSlider(
            label="Time range", start=0, end=0, value=(0, 0), step=1, min_width=320
        )

    def _build_input_preparation_widgets(self, default_input_directory):
        pn = self.pn
        self.prepared_input_directory = pn.widgets.TextInput(
            label="Input package", value=default_input_directory, min_width=280
        )
        self.prepare_Nmax = pn.widgets.IntInput(label="Nmax", value=20, start=1, width=90)
        self.prepare_Mmax = pn.widgets.IntInput(label="Mmax", value=20, start=0, width=90)
        self.prepare_Ncs = pn.widgets.IntInput(label="Ncs", value=30, start=4, width=90)
        self.prepare_final_time = pn.widgets.FloatInput(
            label="Input final time", value=100.0, start=0.0, width=140
        )
        self.prepare_horizontal_basis = pn.widgets.Select(
            label="Basis",
            options={"Spherical harmonics": "SH", "Cubed sphere": "CS"},
            value="SH",
            width=170,
        )
        self.prepare_use_boundary_jr = pn.widgets.Checkbox(
            label="Boundary jr", value=True, width=110
        )
        self.prepare_use_wind = pn.widgets.Checkbox(label="u", value=False, width=70)
        self.prepare_use_q_eff = pn.widgets.Checkbox(label="Q_eff from u", value=False, width=120)
        self.prepare_multi_data = pn.widgets.Checkbox(label="multi-time", value=False, width=120)
        self.prepare_button = pn.widgets.Button(
            label="Prepare input package", color="primary", width=180
        )

    def _build_simulation_widgets(self, default_input_directory):
        pn = self.pn
        self.simulation_input_directory = pn.widgets.TextInput(
            label="Input package", value=default_input_directory, min_width=280
        )
        self.simulation_run_directory = pn.widgets.TextInput(
            label="Run output", value=self.spec.run_directory, min_width=280
        )
        self.sim_final_time = pn.widgets.FloatInput(
            label="Final time", value=100.0, start=0.0, width=120
        )
        self.sim_dt = pn.widgets.FloatInput(label="dt", value=5e-4, start=1e-12, width=110)
        self.sim_saving_sample_interval = pn.widgets.IntInput(
            label="Samples per save", value=200, start=1, width=150
        )
        self.sim_integrator = pn.widgets.Select(
            label="Integrator", options=list(INTEGRATORS.values()), value="euler", width=140
        )
        self.sim_enable_pfac_coupling = pn.widgets.Checkbox(
            label="PFAC coupling", value=False, width=130
        )
        self.sim_enable_interhemispheric_coupling = pn.widgets.Checkbox(
            label="Interhemispheric coupling", value=False, width=190
        )
        self.sim_magnetic_boundary_shielding = pn.widgets.Checkbox(
            label="Boundary shielding", value=False, width=150
        )
        self.sim_run_dynamic = pn.widgets.Checkbox(label="Dynamic", value=True, width=110)
        self.sim_run_equilibrium = pn.widgets.Checkbox(label="Equilibrium", value=True, width=130)
        self.sim_interhemispheric_coupling_latitude = pn.widgets.FloatInput(
            label="Coupling latitude", value=50.0, width=140
        )
        self.sim_use_conductance = pn.widgets.Checkbox(label="Conductance", value=True, width=120)
        self.sim_use_boundary_jr = pn.widgets.Checkbox(label="Boundary jr", value=True, width=110)
        self.sim_use_br = pn.widgets.Checkbox(label="Br", value=True, width=70)
        self.sim_use_u = pn.widgets.Checkbox(label="u", value=True, width=70)
        self.sim_use_q_eff = pn.widgets.Checkbox(label="Q_eff", value=True, width=90)
        self.sim_use_e_neutral_wind = pn.widgets.Checkbox(
            label="Neutral-wind E", value=True, width=130
        )
        self.run_simulation_button = pn.widgets.Button(
            label="Run from inputs", color="primary", width=150
        )

    def _build_data_widgets(self):
        pn = self.pn
        self.fill = pn.widgets.Select(
            label="Filled contours",
            options={label: key for key, label in MAP_FILL_OPTIONS.items()},
            value=self.spec.fill if self.spec.fill in MAP_FILL_OPTIONS else "Br",
            width=210,
        )
        self.lines = pn.widgets.Select(
            label="Contour lines",
            options={label: key for key, label in MAP_LINE_OPTIONS.items()},
            value=self.spec.lines if self.spec.lines in MAP_LINE_OPTIONS else "none",
            width=210,
        )
        self.show_north = pn.widgets.Checkbox(label="North", value=self.spec.show_north, width=90)
        self.show_south = pn.widgets.Checkbox(label="South", value=self.spec.show_south, width=90)
        self.min_abs_lat = pn.widgets.FloatInput(
            label="Min |lat|",
            value=self.spec.hemisphere_min_abs_latitude,
            start=0,
            end=89.9,
            width=130,
        )
        self.station = pn.widgets.TextInput(
            label="Station", value=self.spec.ground_station, width=120
        )
        self.ground_component = pn.widgets.Select(
            label="Component",
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
            label="Signal",
            options={"dB/dt": "dbdt", "B": "b"},
            value=(
                self.spec.ground_quantity if self.spec.ground_quantity in {"dbdt", "b"} else "dbdt"
            ),
            width=110,
        )
        self.include_station_data = pn.widgets.Checkbox(
            label="Measured", value=self.spec.include_station_data, width=95
        )
        self.show_inductive = pn.widgets.Checkbox(
            label="Inductive", value=self.spec.show_inductive, width=100
        )
        self.show_noninductive = pn.widgets.Checkbox(
            label="Non-inductive", value=self.spec.show_noninductive, width=130
        )
        self.show_difference = pn.widgets.Checkbox(
            label="Difference", value=self.spec.show_difference, width=120
        )
        self.sim_time_offset = pn.widgets.FloatInput(
            label="Sim shift (s)", value=self.spec.sim_time_offset_seconds, width=130
        )
        self.data_time_offset = pn.widgets.FloatInput(
            label="Data shift (s)", value=self.spec.data_time_offset_seconds, width=130
        )
        self.dbdt_window_points = pn.widgets.IntInput(
            label="dB/dt pts",
            value=max(1, int(self.spec.dbdt_window_points)),
            start=1,
            end=20,
            width=120,
        )
        self.ground_model_lt_count = pn.widgets.IntInput(
            label="Model LT n",
            value=max(1, int(self.spec.ground_model_lt_count)),
            start=1,
            end=72,
            width=120,
        )
        self.ground_model_lat_count = pn.widgets.IntInput(
            label="Model lat n",
            value=max(1, int(self.spec.ground_model_lat_count)),
            start=1,
            end=60,
            width=125,
        )
        self.ground_model_visual_even = pn.widgets.Checkbox(
            label="Visual grid", value=self.spec.ground_model_visual_even, width=110
        )
        self.show_pedersen_conductance_overlay = pn.widgets.Checkbox(
            label="Pedersen contours", value=self.spec.show_pedersen_conductance_overlay, width=145
        )
        self.show_hall_conductance_overlay = pn.widgets.Checkbox(
            label="Hall contours", value=self.spec.show_hall_conductance_overlay, width=120
        )

    def _build_visualization_widgets(self):
        pn = self.pn
        color_min, color_max = _manual_color_values(self.spec)
        line_start, line_interval, line_count = _manual_line_values(self.spec)
        self.show_reference_line = pn.widgets.Checkbox(
            label="Reference line", value=self.spec.show_reference_line, width=130
        )
        self.reference_time = pn.widgets.TextInput(
            label="Ref. UTC", value=self.spec.reference_time_of_day_utc, width=130
        )
        self.curve_scale_mode = pn.widgets.Select(
            label="Curve scale",
            options={"Manual": "manual", "Automatic": "auto"},
            value=self.spec.curve_scale_mode
            if self.spec.curve_scale_mode in {"manual", "auto"}
            else "manual",
            width=130,
        )
        self.curve_scale = pn.widgets.FloatInput(
            label="Scale value", value=self.spec.curve_scale_value, start=0.01, width=120
        )
        self.time_scale = pn.widgets.FloatInput(
            label="Time x", value=self.spec.curve_time_scale, start=0.1, width=110
        )
        self.low_lat_cutoff = pn.widgets.FloatInput(
            label="Low-lat selection", value=self.spec.min_abs_dip_latitude, start=0.0, width=155
        )
        self.low_lat_scale = pn.widgets.FloatInput(
            label="Low-lat x", value=self.spec.low_latitude_scale, start=0.01, width=110
        )
        self.show_dip_equator_curve = pn.widgets.Checkbox(
            label="Dip equator", value=self.spec.show_dip_equator_curve, width=120
        )
        self.show_low_lat_curve = pn.widgets.Checkbox(
            label="Low-lat curve", value=self.spec.show_low_latitude_curve, width=125
        )
        self.color_scale_mode = pn.widgets.Select(
            label="Color scale",
            options={"Manual": "manual", "Percentile": "percentile"},
            value=self.spec.color_scale_mode
            if self.spec.color_scale_mode in {"manual", "percentile"}
            else "manual",
            width=130,
        )
        self.color_scale_percentile = pn.widgets.FloatInput(
            label="Percentile",
            value=self.spec.color_scale_percentile,
            start=0.0,
            end=100.0,
            width=110,
        )
        self.manual_color_min = pn.widgets.FloatInput(
            label="Color min", value=color_min, width=150
        )
        self.manual_color_max = pn.widgets.FloatInput(
            label="Color max", value=color_max, width=150
        )
        self.line_first_abs_level = pn.widgets.FloatInput(
            label="First |line|", value=line_start, start=0.0, width=150
        )
        self.line_interval = pn.widgets.FloatInput(
            label="Line spacing", value=line_interval, start=0.0, width=150
        )
        self.line_levels_per_sign = pn.widgets.IntInput(
            label="Lines / sign", value=line_count, start=1, width=130
        )
        self.geo_lat_min = pn.widgets.FloatInput(
            label="Geo lat min", value=self.spec.geo_lat_min, width=130
        )
        self.geo_lat_max = pn.widgets.FloatInput(
            label="Geo lat max", value=self.spec.geo_lat_max, width=130
        )
        self.local_time_min = pn.widgets.FloatInput(
            label="LT min", value=self.spec.local_time_min, start=0.0, end=24.0, width=110
        )
        self.local_time_max = pn.widgets.FloatInput(
            label="LT max", value=self.spec.local_time_max, start=0.0, end=24.0, width=110
        )
        self.zoom_window = pn.widgets.Checkbox(
            label="Zoom window", value=self.spec.zoom_window, width=130
        )
        self._sync_style_control_labels()

    def _sync_style_control_labels(self):
        """Show units for the selected fill and line fields."""
        fill_key = self.fill.value if self.fill.value != "none" else "Br"
        line_keys = map_line_keys(self.lines.value)
        line_key = line_keys[0] if line_keys else "Phi"
        color_units, _ = manual_color_control_units(fill_key)
        line_units = PANEL_LINE_UNITS[line_key]
        self.manual_color_min.name = f"Color min ({color_units})"
        self.manual_color_max.name = f"Color max ({color_units})"
        self.line_first_abs_level.name = f"First |line| ({line_units})"
        self.line_interval.name = f"Line spacing ({line_units})"

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
        self.redraw_button = pn.widgets.Button(label="Redraw", color="primary", width=95)
        self.save_button = pn.widgets.Button(label="Save figure", color="warning", width=120)
        self.save_movie_button = pn.widgets.Button(label="Save movie", color="warning", width=120)
        self.output_filename = pn.widgets.TextInput(
            label="Figure path",
            value=str(_absolute_output_path("pynamit_figure.png")),
            min_width=360,
        )
        self.movie_filename = pn.widgets.TextInput(
            label="Movie path",
            value=str(_absolute_output_path(self.spec.movie_filename)),
            min_width=360,
        )
        self.movie_fps = pn.widgets.FloatInput(
            label="FPS", value=self.spec.movie_fps, start=0.1, width=90
        )
        self.overwrite_message = pn.pane.Str(
            "",
            styles={"overflow-wrap": "anywhere", "white-space": "pre-wrap"},
            sizing_mode="stretch_width",
        )
        self.confirm_overwrite_button = pn.widgets.Button(
            label="Overwrite", color="danger", width=110
        )
        self.cancel_overwrite_button = pn.widgets.Button(label="Cancel", width=90)
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
            color="success",
            width=130,
        )
        self.spec_download = pn.widgets.FileDownload(
            label="Download spec",
            filename="pynamit_figure.json",
            callback=self._download_spec,
            color="success",
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
        self.load_button.on_click(self._load_run)
        self.prepare_button.on_click(self._prepare_inputs)
        self.run_simulation_button.on_click(self._run_simulation)
        self.redraw_button.on_click(self._redraw)
        self.save_button.on_click(self._save_figure)
        self.save_movie_button.on_click(self._save_movie)
        self.confirm_overwrite_button.on_click(self._confirm_overwrite)
        self.cancel_overwrite_button.on_click(self._cancel_overwrite)
        self.app_mode.param.watch(self._mode_changed, "value")
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
            self.show_difference,
            self.sim_time_offset,
            self.data_time_offset,
            self.dbdt_window_points,
            self.ground_model_lt_count,
            self.ground_model_lat_count,
            self.ground_model_visual_even,
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

    def _mode_changed(self, event=None):
        if self.app_mode.value == "run_simulation":
            self._sync_simulation_input_availability()
        self._sync_visibility()

    def _simulation_input_widgets(self):
        return {
            "conductance": self.sim_use_conductance,
            "boundary_jr": self.sim_use_boundary_jr,
            "boundary_Br": self.sim_use_br,
            "u": self.sim_use_u,
            "Q_eff": self.sim_use_q_eff,
            "E_neutral_wind": self.sim_use_e_neutral_wind,
        }

    def _available_simulation_inputs(self, input_directory):
        from pynamit.simulation.schema import INPUT_DATASET_KEYS
        from pynamit.storage import ArtifactStore

        input_directory = ArtifactStore.require_artifact_directory(
            Path(input_directory).expanduser(), ("settings",)
        )
        artifacts = ArtifactStore(input_directory).scan_artifacts(INPUT_DATASET_KEYS)
        return {key for key in INPUT_DATASET_KEYS if key in artifacts}

    def _sync_simulation_input_availability(self):
        try:
            available = self._available_simulation_inputs(self.simulation_input_directory.value)
        except Exception:
            available = None
        for key, widget in self._simulation_input_widgets().items():
            if available is None:
                widget.disabled = False
                continue
            present = key in available
            widget.disabled = not present
            if not present and widget.value:
                set_widget_value(widget, False)

    def _selected_simulation_inputs(self):
        selected = []
        if self.sim_use_conductance.value:
            selected.append("conductance")
        if self.sim_use_boundary_jr.value:
            selected.append("boundary_jr")
        if self.sim_use_br.value:
            selected.append("boundary_Br")
        if self.sim_use_u.value:
            selected.append("u")
        if self.sim_use_q_eff.value:
            selected.append("Q_eff")
        if self.sim_use_e_neutral_wind.value:
            selected.append("E_neutral_wind")
        return tuple(selected)

    def _prepare_inputs(self, event=None):
        if self._busy:
            return
        self._busy = True
        self.prepare_button.loading = True
        try:
            from pynamit.simulation.workflows.prepared_inputs import prepare_pynamit_inputs

            if self.prepare_use_q_eff.value and not self.prepare_use_wind.value:
                raise ValueError("Q_eff from u requires the wind input source.")
            input_directory = Path(self.prepared_input_directory.value).expanduser()
            simulation = prepare_pynamit_inputs(
                input_directory=input_directory,
                final_time=float(self.prepare_final_time.value),
                Nmax=int(self.prepare_Nmax.value),
                Mmax=int(self.prepare_Mmax.value),
                Ncs=int(self.prepare_Ncs.value),
                use_wind=bool(self.prepare_use_wind.value),
                use_Q_eff=bool(self.prepare_use_q_eff.value),
                use_boundary_jr=bool(self.prepare_use_boundary_jr.value),
                multi_data=bool(self.prepare_multi_data.value),
                horizontal_basis_kind=self.prepare_horizontal_basis.value,
            )
            prepared_path = Path(simulation.run_directory)
            set_widget_value(self.prepared_input_directory, str(prepared_path))
            set_widget_value(self.simulation_input_directory, str(prepared_path))
            self._set_status(f"Prepared inputs in [`{prepared_path}`]({prepared_path}).")
        except Exception:
            self._set_status(traceback.format_exc(limit=8), error=True)
        finally:
            self.prepare_button.loading = False
            self._busy = False

    def _run_simulation(self, event=None):
        if self._busy:
            return
        self._busy = True
        self.run_simulation_button.loading = True
        should_load_run = False
        try:
            from pynamit.simulation.workflows.prepared_inputs import run_pynamit_from_inputs

            self._sync_simulation_input_availability()
            enabled_inputs = self._selected_simulation_inputs()
            if not enabled_inputs:
                raise ValueError("Select at least one prepared input dataset.")
            input_directory = Path(self.simulation_input_directory.value).expanduser()
            run_directory = Path(self.simulation_run_directory.value).expanduser()
            simulation = run_pynamit_from_inputs(
                input_directory,
                run_directory=run_directory,
                enabled_inputs=enabled_inputs,
                final_time=float(self.sim_final_time.value),
                saving_sample_interval=int(self.sim_saving_sample_interval.value),
                dt=float(self.sim_dt.value),
                enable_pfac_coupling=bool(self.sim_enable_pfac_coupling.value),
                enable_interhemispheric_coupling=bool(
                    self.sim_enable_interhemispheric_coupling.value
                ),
                interhemispheric_coupling_latitude=float(
                    self.sim_interhemispheric_coupling_latitude.value
                ),
                run_dynamic=bool(self.sim_run_dynamic.value),
                run_equilibrium=bool(self.sim_run_equilibrium.value),
                integrator=self.sim_integrator.value,
                magnetic_boundary_shielding=bool(self.sim_magnetic_boundary_shielding.value),
            )
            run_path = Path(simulation.run_directory)
            set_widget_value(self.simulation_run_directory, str(run_path))
            set_widget_value(self.run_directory, str(run_path))
            set_widget_value(self.app_mode, "visualize")
            self._set_status(f"Finished run in [`{run_path}`]({run_path}).")
            should_load_run = True
        except Exception:
            self._set_status(traceback.format_exc(limit=8), error=True)
        finally:
            self.run_simulation_button.loading = False
            self._busy = False
        if should_load_run:
            self._load_run()

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
                self.spec = current_figure_spec(self)
            self.view = get_saved_field_view(self.spec)
            if not self.view.has_model_output and self.spec.plot_type != "input_summary":
                spec_data = self.spec.to_dict()
                spec_data["plot_type"] = "input_summary"
                self.spec = self.spec.from_dict(spec_data)
            elif self.view.has_model_output and self.spec.plot_type != "input_summary":
                has_state = "dynamic" in self.view.run_view.datasets
                has_steady = "equilibrium" in self.view.run_view.datasets
                spec_data = self.spec.to_dict()
                if not has_state and spec_data["plot_type"] not in {"global", "hemispheres"}:
                    spec_data["plot_type"] = "global"
                spec_data["show_inductive"] = bool(has_state and spec_data["show_inductive"])
                spec_data["show_noninductive"] = bool(
                    has_steady and (spec_data["show_noninductive"] or not has_state)
                )
                spec_data["show_difference"] = bool(
                    has_state and has_steady and spec_data["show_difference"]
                )
                self.spec = self.spec.from_dict(spec_data)
            self.time_index.end = max(0, self.view.n_time - 1)
            self.time_range.end = max(0, self.view.n_time - 1)
            if self.time_range.value == (0, 0):
                self.time_range.value = (0, min(int(self.time_range.end), 60))
            apply_figure_spec_to_widgets(self, self.spec)
            set_widget_value(
                self.movie_filename, str(_absolute_output_path(self.movie_filename.value))
            )
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
        if self._syncing_style_controls:
            return
        if event is not None and event.obj in {self.fill, self.lines}:
            self._syncing_style_controls = True
            try:
                if event.obj is self.fill:
                    self._reset_manual_color_controls()
                if event.obj is self.lines:
                    self._reset_manual_line_controls()
                self._sync_style_control_labels()
            finally:
                self._syncing_style_controls = False
        if self._busy:
            return
        if self.app_mode.value != "visualize":
            return
        if self.view is None:
            return
        self.spec = current_figure_spec(self)
        self._sync_visibility()
        self._set_status("Controls changed. Press **Redraw** to update the figure.")

    def _redraw(self, event=None):
        if self._busy:
            return
        self._busy = True
        try:
            self.spec = current_figure_spec(self)
            self._sync_visibility()
            view = self.view if self.view is not None else get_saved_field_view(self.spec)
            index = min(max(0, int(self.spec.time_index)), view.n_time - 1)
            time_text = view.timestamp_at_index(index).strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.object = f"**{time_text}**"
            if self.figure is not None:
                plt.close(self.figure)
            self.figure = render_pynamit_figure(self.spec, view=view)
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
            path = self._output_widget_path(self.output_filename)
            self._save_or_confirm_overwrite(path, self._write_figure)
        except Exception:
            self._set_status(traceback.format_exc(limit=6), error=True)

    def _write_figure(self, path):
        """Write the active figure to one confirmed path."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
            self._set_status(f"Saved figure to [{path}]({path})")
        except Exception:
            self._set_status(traceback.format_exc(limit=6), error=True)

    def _save_movie(self, event=None):
        if self._busy:
            return
        try:
            path = self._output_widget_path(self.movie_filename)
            self._save_or_confirm_overwrite(path, self._write_movie)
        except Exception:
            self._set_status(traceback.format_exc(limit=8), error=True)

    def _write_movie(self, path):
        """Render a movie to one confirmed path."""
        self._busy = True
        self.save_movie_button.loading = True
        try:
            spec = current_figure_spec(self)
            path = save_pynamit_movie(
                spec, path, fps=float(self.movie_fps.value), dpi=int(spec.movie_dpi)
            )
            self._set_status(f"Saved movie to [{path}]({path})")
        except Exception:
            self._set_status(traceback.format_exc(limit=8), error=True)
        finally:
            self.save_movie_button.loading = False
            self._busy = False

    def _output_widget_path(self, widget):
        """Normalize an output widget to its absolute path."""
        path = _absolute_output_path(widget.value)
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
            self._set_status(f"`{path}` already exists. Confirm or cancel the overwrite.")
            return
        save(path)

    def _confirm_overwrite(self, event=None):
        """Run the pending save after overwrite confirmation."""
        pending = self._pending_overwrite
        self._pending_overwrite = None
        self.overwrite_modal.open = False
        if pending is None:
            return
        path, save = pending
        save(path)

    def _cancel_overwrite(self, event=None):
        """Cancel without touching the existing file."""
        pending = self._pending_overwrite
        self._pending_overwrite = None
        self.overwrite_modal.open = False
        if pending is not None:
            self._set_status(f"Save cancelled; existing file left unchanged: `{pending[0]}`.")

    def _download_script(self):
        spec = current_figure_spec(self)
        text = publication_script_for_spec(spec, output_path=self.output_filename.value)
        return StringIO(text)

    def _download_spec(self):
        return StringIO(current_figure_spec(self).to_json())

    def panel(self):
        """Return the Panel layout."""
        pn = self.pn
        app_controls = pn.Card(
            self._control_row(self.app_mode),
            title="Mode",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        mode_controls = pn.Card(
            self._control_row(self.run_directory, self.load_button, self.plot_type),
            self._control_row(self.time_index, self.time_label),
            self.time_range,
            title="Run",
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
                self.prepare_final_time,
            ),
            self._control_row(
                self.prepare_use_boundary_jr,
                self.prepare_use_wind,
                self.prepare_use_q_eff,
                self.prepare_multi_data,
            ),
            title="Input Preparation",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        simulation_controls = pn.Card(
            self._control_row(self.simulation_input_directory, self.simulation_run_directory),
            self._control_row(
                self.sim_final_time,
                self.sim_dt,
                self.sim_saving_sample_interval,
                self.sim_integrator,
            ),
            self._control_row(
                self.sim_enable_pfac_coupling,
                self.sim_enable_interhemispheric_coupling,
                self.sim_magnetic_boundary_shielding,
                self.sim_run_dynamic,
                self.sim_run_equilibrium,
                self.sim_interhemispheric_coupling_latitude,
            ),
            self._control_row(
                self.sim_use_conductance,
                self.sim_use_boundary_jr,
                self.sim_use_br,
                self.sim_use_u,
                self.sim_use_q_eff,
                self.sim_use_e_neutral_wind,
            ),
            self._control_row(self.run_simulation_button),
            title="Simulation Run",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        data_controls = pn.Card(
            self._control_row(self.fill, self.lines),
            self._control_row(
                self.show_inductive,
                self.show_noninductive,
                self.include_station_data,
                self.show_difference,
            ),
            self._control_row(self.station, self.ground_component, self.ground_quantity),
            self._control_row(
                self.sim_time_offset, self.data_time_offset, self.dbdt_window_points
            ),
            self._control_row(
                self.ground_model_lt_count,
                self.ground_model_lat_count,
                self.ground_model_visual_even,
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
            self._control_row(self.movie_filename, self.movie_fps, self.save_movie_button),
            title="Output",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        self.app_controls = app_controls
        self.mode_controls = mode_controls
        self.prepare_controls = prepare_controls
        self.simulation_controls = simulation_controls
        self.data_controls = data_controls
        self.visualization_controls = visualization_controls
        self.output_controls = output_controls
        self._sync_visibility()
        controls = pn.Column(
            app_controls,
            mode_controls,
            prepare_controls,
            simulation_controls,
            data_controls,
            visualization_controls,
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
        return pn.Column(
            pn.FlexBox(
                controls,
                plot_area,
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
        is_prepare_mode = app_mode == "prepare_inputs"
        is_run_mode = app_mode == "run_simulation"
        if hasattr(self, "mode_controls"):
            self.mode_controls.visible = is_visualize_mode
            self.prepare_controls.visible = is_prepare_mode
            self.simulation_controls.visible = is_run_mode
            self.data_controls.visible = is_visualize_mode
            self.visualization_controls.visible = is_visualize_mode
            self.output_controls.visible = is_visualize_mode

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
        self.show_inductive.visible = is_map or is_ground
        self.show_noninductive.visible = is_map or is_ground
        self.show_difference.visible = is_map
        self.show_north.visible = plot_type == "hemispheres"
        self.show_south.visible = plot_type == "hemispheres"

        self.station.visible = is_ground_timeseries
        self.ground_component.visible = is_ground_curve
        self.ground_quantity.visible = is_ground
        self.include_station_data.visible = is_ground
        self.sim_time_offset.visible = is_ground
        self.data_time_offset.visible = is_ground
        self.dbdt_window_points.visible = is_ground and self.ground_quantity.value == "dbdt"
        show_model_grid_controls = is_ground_curve and not self.include_station_data.value
        self.ground_model_lt_count.visible = show_model_grid_controls
        self.ground_model_lat_count.visible = show_model_grid_controls
        self.ground_model_visual_even.visible = show_model_grid_controls
        self.show_pedersen_conductance_overlay.visible = is_ground_curve
        self.show_hall_conductance_overlay.visible = is_ground_curve

        self.show_reference_line.visible = is_ground
        self.reference_time.visible = is_ground
        self.min_abs_lat.visible = plot_type in {"hemispheres", "input_summary"}
        self.geo_lat_min.visible = is_ground_curve
        self.geo_lat_max.visible = is_ground_curve
        self.local_time_min.visible = is_ground_curve
        self.local_time_max.visible = is_ground_curve
        self.zoom_window.visible = is_ground_curve
        self.curve_scale_mode.visible = is_ground_curve
        self.curve_scale.visible = is_ground_curve
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


def build_pynamit_panel_app(run_directory=None):
    """Build and return the Panel layout for saved-run plotting."""
    return PynamitPanelApp(run_directory=run_directory).panel()


def servable(run_directory=None, title="PynaMIT Plot"):
    """Create a servable Panel app."""
    app = build_pynamit_panel_app(run_directory=run_directory)
    return app.servable(title=title)


__all__ = ["PynamitPanelApp", "build_pynamit_panel_app", "servable"]
