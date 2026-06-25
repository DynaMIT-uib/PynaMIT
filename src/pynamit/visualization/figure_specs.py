"""Serializable figure specifications for PynaMIT visualizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path


MAP_FILL_OPTIONS = {
    "none": "No fill",
    "Br": "Radial B-field",
    "jr": "Radial current",
    "joule": "Joule heating",
}

MAP_LINE_OPTIONS = {
    "none": "No contour lines",
    "Phi": "Electric potential",
    "W": "E streamfunction",
    "Phi_W": "Phi + W",
    "Jeq": "Equivalent current",
}

PLOT_TYPE_OPTIONS = {
    "ground_curve_map": "Ground curve map",
    "ground_timeseries": "Ground time series",
    "hemispheres": "Hemispheres",
    "global": "Global maps",
    "input_summary": "Input drivers",
}

RUN_PLOT_DEFAULT_FILENAMES = (
    "pynamit_plot_defaults.json",
    "plot_defaults.json",
    "pynamit_panel_defaults.json",
)


@dataclass
class PynamitFigureSpec:
    """Small, serializable description of one PynaMIT figure."""

    run_directory: str = "sim_dir"
    data_directory: str = ""
    plot_type: str = "ground_curve_map"
    time_index: int = 0
    time_range: tuple[int, int] = (182, 191)
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
    show_noninductive: bool = False
    show_difference: bool = True
    show_reference_line: bool = True
    reference_time_of_day_utc: str = "18:31:00"
    conductance_overlay: str = "none"
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
    color_scale_mode: str = "fixed"
    color_scale_percentile: float = 99.8
    geo_lat_min: float = -90.0
    geo_lat_max: float = 90.0
    local_time_min: float = 0.0
    local_time_max: float = 24.0
    zoom_window: bool = False
    movie_filename: str = "pynamit_movie.gif"
    movie_fps: float = 4.0
    movie_dpi: int = 120
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        """Return a JSON-compatible dictionary."""
        data = asdict(self)
        data["time_range"] = list(self.time_range)
        return data

    @classmethod
    def from_dict(cls, data):
        """Build a spec from a JSON-compatible dictionary."""
        values = dict(data)
        if "time_range" in values:
            values["time_range"] = tuple(values["time_range"])
        names = {item.name for item in fields(cls)}
        extra = dict(values.pop("extra", {}))
        for key in list(values):
            if key not in names:
                extra[key] = values.pop(key)
        if extra:
            values["extra"] = extra
        return cls(**values)

    def to_json(self, *, indent=2):
        """Serialize the spec as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        """Deserialize the spec from JSON."""
        return cls.from_dict(json.loads(text))

    def with_run_directory(self, run_directory):
        """Return a copy with a different run directory."""
        data = self.to_dict()
        data["run_directory"] = str(run_directory)
        return self.from_dict(data)


def publication_script_for_spec(spec, *, output_path="figure.png"):
    """Return a Jupyter-friendly Python script for one figure spec."""
    if not isinstance(spec, PynamitFigureSpec):
        spec = PynamitFigureSpec.from_dict(spec)
    spec_json = spec.to_json(indent=4)
    output_path = str(Path(output_path))
    return (
        "# %%\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n\n"
        "from pynamit.visualization.figure_builder import render_pynamit_figure\n"
        "from pynamit.visualization.figure_specs import PynamitFigureSpec\n\n\n"
        'SPEC = PynamitFigureSpec.from_json(r"""\n'
        f"{spec_json}\n"
        '""")\n'
        f"OUTPUT_PATH = Path({output_path!r})\n\n\n"
        "# %%\n"
        "fig = render_pynamit_figure(SPEC)\n"
        "fig\n\n\n"
        "# %%\n"
        "OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        'fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")\n'
        'print(f"Saved figure to {OUTPUT_PATH}")\n'
        "plt.close(fig)\n"
    )


def find_run_plot_defaults(run_directory, filenames=RUN_PLOT_DEFAULT_FILENAMES):
    """Return the first plotting-defaults file in a run."""
    run_dir = Path(run_directory).expanduser()
    for filename in filenames:
        path = run_dir / filename
        if path.exists():
            return path
    return None


def load_run_plot_defaults(run_directory, filenames=RUN_PLOT_DEFAULT_FILENAMES):
    """Load optional per-run plotting defaults from JSON."""
    path = find_run_plot_defaults(run_directory, filenames=filenames)
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse plotting defaults in {path}.") from exc

    defaults = dict(raw.get("plot_defaults", raw))
    if "data_directory" in raw and "data_directory" not in defaults:
        defaults["data_directory"] = raw["data_directory"]
    return defaults


def figure_spec_from_run_defaults(run_directory, **overrides):
    """Build a figure spec from defaults and overrides."""
    data = load_run_plot_defaults(run_directory)
    data["run_directory"] = str(run_directory)
    data.update(overrides)
    return PynamitFigureSpec.from_dict(data)


__all__ = [
    "MAP_FILL_OPTIONS",
    "MAP_LINE_OPTIONS",
    "PLOT_TYPE_OPTIONS",
    "PynamitFigureSpec",
    "RUN_PLOT_DEFAULT_FILENAMES",
    "figure_spec_from_run_defaults",
    "find_run_plot_defaults",
    "load_run_plot_defaults",
    "publication_script_for_spec",
]
