"""Render Matplotlib figures for saved PynaMIT simulations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

from pynamit.plotting.field_comparison_figures import FieldComparisonRenderer
from pynamit.plotting.figure_context import as_figure_settings, get_grid_fields
from pynamit.plotting.figure_settings import FigureSettings
from pynamit.plotting.ground_figures import GroundFigureRenderer
from pynamit.plotting.input_driver_figures import InputDriverRenderer


def render_figure(settings, grid_fields=None):
    """Render a Matplotlib figure from :class:`FigureSettings`."""
    settings = as_figure_settings(settings)
    if settings.plot_type in {"global", "hemispheres"}:
        return FieldComparisonRenderer(settings, grid_fields=grid_fields).render()
    if settings.plot_type == "input_summary":
        return InputDriverRenderer(settings, grid_fields=grid_fields).render()
    if settings.plot_type == "ground_curve_map":
        return GroundFigureRenderer(settings, grid_fields=grid_fields).render_curve_map()
    if settings.plot_type == "ground_timeseries":
        return GroundFigureRenderer(settings, grid_fields=grid_fields).render_timeseries()
    raise NotImplementedError(
        f"{settings.plot_type!r} is not implemented by the figure renderer."
    )


def save_movie(settings, output_path, *, fps=None, dpi=None):
    """Render a time-index movie as an animated GIF."""
    settings = as_figure_settings(settings)
    if settings.plot_type not in {"global", "hemispheres", "input_summary"}:
        raise ValueError("Movie export is currently for global, hemisphere, and input maps.")
    output_path = Path(output_path).expanduser()
    if output_path.suffix.lower() != ".gif":
        raise ValueError("Movie export currently writes animated GIF files; use a .gif path.")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Movie export requires Pillow.") from exc

    grid_fields = get_grid_fields(settings)
    start, end = [int(value) for value in settings.time_range]
    start = max(0, min(start, grid_fields.n_time - 1))
    end = max(start, min(end, grid_fields.n_time - 1))
    if end == start:
        end = min(
            grid_fields.n_time - 1, start + min(60, max(grid_fields.n_time - 1, 1))
        )

    duration_ms = int(round(1000.0 / max(float(fps or settings.movie_fps), 1e-6)))
    frame_dpi = int(dpi or settings.movie_dpi)
    image_palette = getattr(getattr(Image, "Palette", None), "ADAPTIVE", None)
    if image_palette is None:
        image_palette = getattr(Image, "ADAPTIVE", 1)
    frames = []
    try:
        for index in range(start, end + 1):
            frame_data = settings.to_dict()
            frame_data["time_index"] = index
            fig = render_figure(FigureSettings.from_dict(frame_data), grid_fields=grid_fields)
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=frame_dpi, bbox_inches="tight")
            plt.close(fig)
            buffer.seek(0)
            frames.append(Image.open(buffer).convert("P", palette=image_palette))
        if not frames:
            raise ValueError("No frames were rendered.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0
        )
    finally:
        for frame in frames:
            try:
                frame.close()
            except Exception:
                pass
    return output_path


__all__ = ["render_figure", "save_movie"]
