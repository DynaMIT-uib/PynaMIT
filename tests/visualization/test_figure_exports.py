"""Tests for figure exports contracts."""

import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("time_range", [(1, 1), (1, 3)])
@pytest.mark.parametrize("overrides", [{}, {"fps": 5.0, "dpi": 50}])
def test_movie_export_honors_exact_frame_range(tmp_path, monkeypatch, time_range, overrides):
    """Render only the requested frames with the requested settings."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from pynamit.plotting import FigureSettings, figure_builder

    settings = FigureSettings(plot_type="global", time_range=time_range)
    plot_data = SimpleNamespace(n_time=5)
    monkeypatch.setattr(figure_builder, "get_plot_data", lambda _: plot_data)
    rendered = []

    def render(frame_settings, *, plot_data):
        rendered.append(
            (frame_settings.time_index, frame_settings.movie_fps, frame_settings.movie_dpi)
        )
        return plt.figure(figsize=(1, 1), facecolor=(frame_settings.time_index / 5.0, 0.0, 0.0))

    monkeypatch.setattr(figure_builder, "render_figure", render)
    output = figure_builder.save_movie(settings, tmp_path / "movie.gif", **overrides)
    fps = overrides.get("fps", settings.movie_fps)
    dpi = overrides.get("dpi", settings.movie_dpi)
    assert rendered == [(index, fps, dpi) for index in range(time_range[0], time_range[1] + 1)]
    with Image.open(output) as movie:
        assert movie.n_frames == len(rendered)
        assert movie.info["duration"] == round(1000.0 / fps)
    assert settings.time_range == time_range
    assert settings.movie_fps == 4.0


@pytest.mark.parametrize("overrides", [{"fps": 0.0}, {"fps": -1.0}, {"dpi": 0}, {"dpi": 1.5}])
def test_movie_export_rejects_invalid_overrides(tmp_path, overrides):
    """Invalid export settings are not replaced with defaults."""
    from pynamit.plotting import FigureSettings, save_movie

    with pytest.raises(ValueError, match="movie_"):
        save_movie(FigureSettings(plot_type="global"), tmp_path / "movie.gif", **overrides)


def test_movie_export_rejects_out_of_range_frames(tmp_path, monkeypatch):
    """Do not silently clip a time range to available data."""
    from pynamit.plotting import FigureSettings, figure_builder

    monkeypatch.setattr(figure_builder, "get_plot_data", lambda _: SimpleNamespace(n_time=3))
    settings = FigureSettings(plot_type="global", time_range=(1, 3))
    with pytest.raises(ValueError, match="time_range.*exceeds"):
        figure_builder.save_movie(settings, tmp_path / "movie.gif")


def test_publication_script_export_is_jupyter_friendly():
    """Figure settings can produce editable publication scripts."""
    figure_settings = importlib.import_module("pynamit.plotting.figure_settings")

    settings = figure_settings.FigureSettings(simulation_directory="run", plot_type="global")
    script = figure_settings.publication_script(settings, output_path="figures/test.png")

    assert script.startswith("# %%")
    assert "render_figure" in script
    assert '"simulation_directory": "run"' in script
    assert "fig.savefig" in script
