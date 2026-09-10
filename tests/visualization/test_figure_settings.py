"""Tests for figure settings contracts."""

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_index": -1},
        {"time_range": (0.5, 2)},
        {"dbdt_window_points": 1.5},
        {"reference_time_of_day_utc": "not a time"},
        {"color_scale_mode": "fixed"},
        {"color_scale_percentile": np.nan},
        {"manual_color_min": -1.0},
        {"manual_color_min": 1.0, "manual_color_max": -1.0},
        {"line_first_abs_level": 1.0},
        {"line_first_abs_level": 1.0, "line_interval": 0.0, "line_levels_per_sign": 2},
        {"simulation_time_offset_seconds": np.inf},
    ],
)
def test_figure_settings_reject_invalid_renderer_values(kwargs):
    """Figure settings reject values renderers cannot interpret."""
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")

    with pytest.raises(ValueError):
        figure_settings.FigureSettings(**kwargs)


def test_figure_settings_reject_unknown_options():
    """Unknown settings fail instead of hiding spelling errors."""
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")

    with pytest.raises(ValueError, match="Unknown figure setting.*conductance_overlay"):
        figure_settings.FigureSettings.from_dict({"conductance_overlay": "hall"})
    with pytest.raises(ValueError, match="Unknown figure setting.*run_directory"):
        figure_settings.FigureSettings.from_dict({"run_directory": "."})


def test_simulation_plot_defaults_are_applied(tmp_path):
    """Simulation directories can carry plotting defaults."""
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")

    (tmp_path / "pynamit_plot_defaults.json").write_text(
        """
        {
          "station_data_directory": "/tmp/mag_data",
          "plot_type": "ground_timeseries",
          "ground_station": "OTT",
          "time_range": [4, 12]
        }
        """,
        encoding="utf-8",
    )

    settings = figure_settings.FigureSettings.from_simulation_directory(tmp_path)

    assert settings.simulation_directory == str(tmp_path)
    assert settings.station_data_directory == "/tmp/mag_data"
    assert settings.plot_type == "ground_timeseries"
    assert settings.ground_station == "OTT"
    assert settings.time_range == (4, 12)


def test_figure_defaults_are_event_independent():
    """Ordinary figures do not require a particular observed event."""
    from pynamit.plotting import FigureSettings

    settings = FigureSettings()
    assert settings.plot_type == "global"
    assert settings.simulation_time_offset_seconds == 0
    assert settings.data_time_offset_seconds == 0
    assert not settings.include_station_data
    assert not settings.show_reference_line
    assert settings.ground_station == ""
    assert settings.color_scale_mode == "percentile"


def test_saved_figure_defaults_are_portable(tmp_path):
    """Copied defaults follow their new simulation directory."""
    import json

    from pynamit.plotting import FigureSettings
    from pynamit.plotting.figure_settings import FIGURE_DEFAULTS_FILENAME

    settings = FigureSettings(
        simulation_directory="old-simulation",
        station_data_directory="stations",
        ground_station="OTT",
        time_range=(1, 7),
    )
    path = settings.save_defaults(tmp_path)
    assert path == tmp_path / FIGURE_DEFAULTS_FILENAME
    assert "simulation_directory" not in json.loads(path.read_text())
    loaded = FigureSettings.from_simulation_directory(tmp_path)
    assert loaded.simulation_directory == str(tmp_path)
    assert loaded.station_data_directory == str(tmp_path / "stations")
    assert loaded.ground_station == "OTT"
    assert loaded.time_range == (1, 7)
