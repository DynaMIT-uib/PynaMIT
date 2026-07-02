"""Shared context for rendering figures from saved PynaMIT runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pynamit.primitives.io import RUN_ARTIFACTS
from pynamit.visualization.figure_specs import PynamitFigureSpec
from pynamit.visualization.artifacts import artifact_path, resolve_xarray_artifact_path
from pynamit.visualization.run_fields import SavedCoefficientFieldView


_CACHE_ARTIFACTS = tuple(sorted(RUN_ARTIFACTS))
_VIEW_CACHE: dict[tuple[str, int, int, tuple], SavedCoefficientFieldView] = {}


def as_figure_spec(spec):
    """Return ``spec`` as a ``PynamitFigureSpec``."""
    if isinstance(spec, PynamitFigureSpec):
        return spec
    return PynamitFigureSpec.from_dict(spec)


def clear_saved_field_view_cache():
    """Clear cached saved-run field views."""
    _VIEW_CACHE.clear()


def _artifact_fingerprint(run_directory):
    run_dir = Path(run_directory).expanduser()
    fingerprint = []
    for name in _CACHE_ARTIFACTS:
        path = Path(resolve_xarray_artifact_path(artifact_path(run_dir, name)))
        if path.exists():
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                mtime = None
            fingerprint.append((name, str(path), mtime))
    return tuple(fingerprint)


def get_saved_field_view(spec):
    """Return a cached coefficient-field view for a figure spec."""
    spec = as_figure_spec(spec)
    key = (str(spec.run_directory), 60, 100, _artifact_fingerprint(spec.run_directory))
    view = _VIEW_CACHE.get(key)
    if view is None:
        view = SavedCoefficientFieldView.from_directory(spec.run_directory)
        _VIEW_CACHE[key] = view
    return view


@dataclass(frozen=True)
class SavedRunFigureContext:
    """Figure spec plus the saved-run field view it renders."""

    spec: PynamitFigureSpec
    view: SavedCoefficientFieldView

    @classmethod
    def from_spec(cls, spec, view=None):
        """Build a context from a spec and optional preloaded view."""
        spec = as_figure_spec(spec)
        return cls(spec=spec, view=get_saved_field_view(spec) if view is None else view)

    @property
    def time_index(self):
        """Clamped selected time index."""
        return int(max(0, min(int(self.spec.time_index), self.view.n_time - 1)))

    @property
    def timestamp(self):
        """Selected timestamp."""
        return self.view.timestamp_at_index(self.time_index)


__all__ = [
    "SavedRunFigureContext",
    "as_figure_spec",
    "clear_saved_field_view_cache",
    "get_saved_field_view",
]
