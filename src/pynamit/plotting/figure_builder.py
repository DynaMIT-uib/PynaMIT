"""Render Matplotlib figures for saved PynaMIT simulations."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

from pynamit.plotting.field_comparison_figures import FieldComparisonRenderer
from pynamit.plotting.ground_figures import GroundFigureRenderer
from pynamit.plotting.input_driver_figures import InputDriverRenderer
from pynamit.plotting.plot_data import _coerce_figure_settings, get_plot_data


def render_figure(settings, plot_data=None):
    """Render a figure from :class:`pynamit.plotting.FigureSettings`."""
    settings = _coerce_figure_settings(settings)
    if settings.plot_type in {"global", "hemispheres"}:
        return FieldComparisonRenderer(settings, plot_data=plot_data).render()
    if settings.plot_type == "input_summary":
        return InputDriverRenderer(settings, plot_data=plot_data).render()
    if settings.plot_type == "ground_curve_map":
        return GroundFigureRenderer(settings, plot_data=plot_data).render_curve_map()
    if settings.plot_type == "ground_timeseries":
        return GroundFigureRenderer(settings, plot_data=plot_data).render_timeseries()
    raise NotImplementedError(f"{settings.plot_type!r} is not implemented by the figure renderer.")


def save_movie(settings, output_path, *, fps=None, dpi=None):
    """Render the inclusive ``time_range`` as an animated GIF."""
    settings = _coerce_figure_settings(settings)
    if fps is not None or dpi is not None:
        settings = replace(
            settings,
            movie_fps=settings.movie_fps if fps is None else fps,
            movie_dpi=settings.movie_dpi if dpi is None else dpi,
        )
    if settings.plot_type not in {"global", "hemispheres", "input_summary"}:
        raise ValueError("Movie export is currently for global, hemisphere, and input maps.")
    output_path = Path(output_path).expanduser()
    if output_path.suffix.lower() != ".gif":
        raise ValueError("Movie export currently writes animated GIF files; use a .gif path.")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Movie export requires Pillow.") from exc

    plot_data = get_plot_data(settings)
    start, end = settings.time_range
    if end >= plot_data.n_time:
        raise ValueError(f"time_range {settings.time_range} exceeds {plot_data.n_time} samples.")

    duration_ms = int(round(1000.0 / settings.movie_fps))
    frames = []
    try:
        for index in range(start, end + 1):
            fig = render_figure(replace(settings, time_index=index), plot_data=plot_data)
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=settings.movie_dpi, bbox_inches="tight")
            plt.close(fig)
            buffer.seek(0)
            frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0
        )
    finally:
        for frame in frames:
            frame.close()
    return output_path


__all__ = ["render_figure", "save_movie"]
