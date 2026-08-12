"""Serializable settings for PynaMIT plots."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MAP_FILL_OPTIONS = {
    "none": "No fill",
    "Br": "Induced radial B field",
    "jr": "Boundary radial current",
    "joule": "Joule heating",
}

MAP_LINE_OPTIONS = {
    "none": "No contour lines",
    "Phi": "Electric potential",
    "W": "E streamfunction",
    "Phi_W": "Phi + W",
    "Jeq": "Equivalent-current function",
}

PLOT_TYPE_OPTIONS = {
    "ground_curve_map": "Ground curve map",
    "ground_timeseries": "Ground time series",
    "hemispheres": "Hemispheres",
    "global": "Global maps",
    "input_summary": "Input drivers",
}

GROUND_COMPONENT_OPTIONS = {"Magnitude", "North", "East", "Down", "AbsNorth", "AbsEast", "AbsDown"}
GROUND_QUANTITY_OPTIONS = {"b", "dbdt"}
CURVE_SCALE_MODE_OPTIONS = {"auto", "manual"}
COLOR_SCALE_MODE_OPTIONS = {"manual", "percentile"}

_PLOT_DEFAULT_FILENAME = "pynamit_plot_defaults.json"


@dataclass
class FigureSettings:
    """Settings for one PynaMIT figure."""

    simulation_directory: str = "."
    data_directory: str = ""
    plot_type: str = "ground_curve_map"
    time_index: int = 0
    time_range: tuple[int, int] = (0, 0)
    fill: str = "Br"
    lines: str = "none"
    show_north: bool = True
    show_south: bool = True
    hemisphere_min_abs_latitude: float = 40.0
    ground_station: str = "IPM"
    ground_component: str = "Magnitude"
    ground_quantity: str = "dbdt"
    include_station_data: bool = True
    show_station_labels: bool = True
    show_inductive: bool = True
    show_noninductive: bool = True
    show_difference: bool = True
    show_reference_line: bool = True
    reference_time_of_day_utc: str = "18:31:00"
    sim_time_offset_seconds: float = 30.0
    data_time_offset_seconds: float = 0.0
    dbdt_window_points: int = 1
    ground_model_lt_count: int = 8
    ground_model_lat_count: int = 7
    ground_model_visual_even: bool = False
    show_pedersen_conductance_overlay: bool = False
    show_hall_conductance_overlay: bool = False
    min_abs_dip_latitude: float = 65.0
    low_latitude_scale: float = 3.0
    show_dip_equator_curve: bool = True
    show_low_latitude_curve: bool = True
    curve_scale_mode: str = "manual"
    curve_scale_value: float = 10.0
    curve_time_scale: float = 1.0
    color_scale_mode: str = "manual"
    color_scale_percentile: float = 99.8
    manual_color_min: float | None = None
    manual_color_max: float | None = None
    line_first_abs_level: float | None = None
    line_interval: float | None = None
    line_levels_per_sign: int | None = None
    geo_lat_min: float = -90.0
    geo_lat_max: float = 90.0
    local_time_min: float = 0.0
    local_time_max: float = 24.0
    zoom_window: bool = False
    movie_filename: str = "pynamit_movie.gif"
    movie_fps: float = 4.0
    movie_dpi: int = 120
    def __post_init__(self):
        """Normalize and validate renderer-facing options."""
        self.time_range = self._validate_time_range(self.time_range)
        self._validate_integer("time_index", self.time_index, minimum=0)
        self._validate_choice("plot_type", self.plot_type, PLOT_TYPE_OPTIONS)
        self._validate_choice("fill", self.fill, MAP_FILL_OPTIONS)
        self._validate_choice("lines", self.lines, MAP_LINE_OPTIONS)
        self._validate_choice("ground_component", self.ground_component, GROUND_COMPONENT_OPTIONS)
        self._validate_choice("ground_quantity", self.ground_quantity, GROUND_QUANTITY_OPTIONS)
        self._validate_choice("curve_scale_mode", self.curve_scale_mode, CURVE_SCALE_MODE_OPTIONS)
        self._validate_choice("color_scale_mode", self.color_scale_mode, COLOR_SCALE_MODE_OPTIONS)
        self._validate_range(
            "hemisphere_min_abs_latitude", self.hemisphere_min_abs_latitude, 0, 90
        )
        self._validate_range("min_abs_dip_latitude", self.min_abs_dip_latitude, 0, 90)
        self._validate_ordered_range("geo_lat_min", "geo_lat_max", -90, 90)
        self._validate_range("local_time_min", self.local_time_min, 0, 24)
        self._validate_range("local_time_max", self.local_time_max, 0, 24)
        self._validate_integer("dbdt_window_points", self.dbdt_window_points, minimum=1)
        self._validate_integer("ground_model_lt_count", self.ground_model_lt_count, minimum=1)
        self._validate_integer("ground_model_lat_count", self.ground_model_lat_count, minimum=1)
        self._validate_positive("low_latitude_scale", self.low_latitude_scale)
        self._validate_positive("curve_scale_value", self.curve_scale_value)
        self._validate_positive("curve_time_scale", self.curve_time_scale)
        self._validate_positive("movie_fps", self.movie_fps)
        self._validate_integer("movie_dpi", self.movie_dpi, minimum=1)
        self._validate_finite("sim_time_offset_seconds", self.sim_time_offset_seconds)
        self._validate_finite("data_time_offset_seconds", self.data_time_offset_seconds)
        self._validate_range("color_scale_percentile", self.color_scale_percentile, 0, 100)
        self._validate_optional_pair(
            "manual_color_min", self.manual_color_min, "manual_color_max", self.manual_color_max
        )
        if self.manual_color_min is not None and self.manual_color_min >= self.manual_color_max:
            raise ValueError("manual_color_min must be less than manual_color_max.")
        line_values = (self.line_first_abs_level, self.line_interval, self.line_levels_per_sign)
        if any(value is not None for value in line_values):
            if not all(value is not None for value in line_values):
                raise ValueError(
                    "line_first_abs_level, line_interval, and line_levels_per_sign "
                    "must be set together."
                )
            self._validate_positive("line_first_abs_level", self.line_first_abs_level)
            self._validate_positive("line_interval", self.line_interval)
            self._validate_integer("line_levels_per_sign", self.line_levels_per_sign, minimum=1)

    @staticmethod
    def _validate_choice(name, value, options):
        if value not in options:
            allowed = ", ".join(sorted(options))
            raise ValueError(f"{name} must be one of {allowed}; got {value!r}.")

    @staticmethod
    def _validate_time_range(value):
        if len(value) != 2:
            raise ValueError("time_range must contain exactly two indices.")
        start, end = value
        FigureSettings._validate_integer("time_range start", start, minimum=0)
        FigureSettings._validate_integer("time_range end", end, minimum=0)
        start, end = int(start), int(end)
        if start < 0 or end < 0:
            raise ValueError("time_range indices must be non-negative.")
        if end < start:
            raise ValueError("time_range end must be greater than or equal to start.")
        return (start, end)

    @staticmethod
    def _validate_range(name, value, minimum, maximum):
        value = float(value)
        if not math.isfinite(value) or value < float(minimum) or value > float(maximum):
            raise ValueError(f"{name} must be between {minimum} and {maximum}; got {value!r}.")

    def _validate_ordered_range(self, minimum_name, maximum_name, minimum, maximum):
        lower = float(getattr(self, minimum_name))
        upper = float(getattr(self, maximum_name))
        if lower > upper:
            raise ValueError(f"{minimum_name} must be <= {maximum_name}.")
        self._validate_range(minimum_name, lower, minimum, maximum)
        self._validate_range(maximum_name, upper, minimum, maximum)

    @staticmethod
    def _validate_positive(name, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive; got {value!r}.")

    @staticmethod
    def _validate_integer(name, value, *, minimum):
        integer = int(value)
        if isinstance(value, bool) or integer != value or integer < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")

    @staticmethod
    def _validate_finite(name, value):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite; got {value!r}.")

    @classmethod
    def _validate_optional_pair(cls, first_name, first, second_name, second):
        if (first is None) != (second is None):
            raise ValueError(f"{first_name} and {second_name} must be set together.")
        if first is not None:
            cls._validate_finite(first_name, first)
            cls._validate_finite(second_name, second)

    def to_dict(self):
        """Return a JSON-compatible dictionary."""
        data = asdict(self)
        data["time_range"] = list(self.time_range)
        return data

    @classmethod
    def from_dict(cls, data):
        """Build a settings from a JSON-compatible dictionary."""
        values = dict(data)
        if "time_range" in values:
            values["time_range"] = tuple(values["time_range"])
        names = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown figure setting(s): {unknown}.")
        return cls(**values)

    def to_json(self, *, indent=2):
        """Serialize the settings as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        """Deserialize the settings from JSON."""
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_simulation_directory(cls, simulation_directory, **overrides):
        """Load a simulation's optional plotting defaults."""
        simulation_directory = Path(simulation_directory).expanduser()
        path = simulation_directory / _PLOT_DEFAULT_FILENAME
        data = {}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as stream:
                    data = json.load(stream)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse plotting defaults in {path}.") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Plotting defaults in {path} must be a JSON object.")
        data["simulation_directory"] = str(simulation_directory)
        data.update(overrides)
        return cls.from_dict(data)

    def with_simulation_directory(self, simulation_directory):
        """Return a copy with a different simulation directory."""
        data = self.to_dict()
        data["simulation_directory"] = str(simulation_directory)
        return self.from_dict(data)


def publication_script(settings, *, output_path="figure.png"):
    """Return a Jupyter-friendly Python script for one figure."""
    if not isinstance(settings, FigureSettings):
        settings = FigureSettings.from_dict(settings)
    settings_json = settings.to_json(indent=4)
    output_path = str(Path(output_path))
    return (
        "# %%\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n\n"
        "from pynamit.plotting import FigureSettings, render_figure\n\n\n"
        'SETTINGS = FigureSettings.from_json(r"""\n'
        f"{settings_json}\n"
        '""")\n'
        f"OUTPUT_PATH = Path({output_path!r})\n\n\n"
        "# %%\n"
        "fig = render_figure(SETTINGS)\n"
        "fig\n\n\n"
        "# %%\n"
        "OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        'fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")\n'
        'print(f"Saved figure to {OUTPUT_PATH}")\n'
        "plt.close(fig)\n"
    )


__all__ = [
    "MAP_FILL_OPTIONS",
    "MAP_LINE_OPTIONS",
    "PLOT_TYPE_OPTIONS",
    "FigureSettings",
    "publication_script",
]
