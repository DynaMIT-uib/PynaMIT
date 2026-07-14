"""Stable Matplotlib figure-builder facade for saved PynaMIT runs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

from pynamit.visualization.field_comparison_figures import FieldComparisonRenderer
from pynamit.visualization.figure_context import as_figure_spec, get_saved_field_view
from pynamit.visualization.figure_specs import PynamitFigureSpec
from pynamit.visualization.ground_figures import GroundFigureRenderer
from pynamit.visualization.input_driver_figures import InputDriverRenderer


def render_pynamit_figure(spec, view=None):
    """Render a Matplotlib figure from a serializable figure spec."""
    spec = as_figure_spec(spec)
    if spec.plot_type in {"global", "hemispheres"}:
        return FieldComparisonRenderer(spec, view=view).render()
    if spec.plot_type == "input_summary":
        return InputDriverRenderer(spec, view=view).render()
    if spec.plot_type == "ground_curve_map":
        return GroundFigureRenderer(spec, view=view).render_curve_map()
    if spec.plot_type == "ground_timeseries":
        return GroundFigureRenderer(spec, view=view).render_timeseries()
    raise NotImplementedError(f"{spec.plot_type!r} is not implemented by the figure renderer.")


def save_pynamit_movie(spec, output_path, *, fps=None, dpi=None):
    """Render a time-index movie as an animated GIF."""
    spec = as_figure_spec(spec)
    if spec.plot_type not in {"global", "hemispheres", "input_summary"}:
        raise ValueError("Movie export is currently for global, hemisphere, and input maps.")
    output_path = Path(output_path).expanduser()
    if output_path.suffix.lower() != ".gif":
        raise ValueError("Movie export currently writes animated GIF files; use a .gif path.")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Movie export requires Pillow.") from exc

    view = get_saved_field_view(spec)
    start, end = [int(value) for value in spec.time_range]
    start = max(0, min(start, view.n_time - 1))
    end = max(start, min(end, view.n_time - 1))
    if end == start:
        end = min(view.n_time - 1, start + min(60, max(view.n_time - 1, 1)))

    duration_ms = int(round(1000.0 / max(float(fps or spec.movie_fps), 1e-6)))
    frame_dpi = int(dpi or spec.movie_dpi)
    image_palette = getattr(getattr(Image, "Palette", None), "ADAPTIVE", None)
    if image_palette is None:
        image_palette = getattr(Image, "ADAPTIVE", 1)
    frames = []
    try:
        for index in range(start, end + 1):
            frame_data = spec.to_dict()
            frame_data["time_index"] = index
            fig = render_pynamit_figure(PynamitFigureSpec.from_dict(frame_data), view=view)
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


__all__ = ["render_pynamit_figure", "save_pynamit_movie"]
