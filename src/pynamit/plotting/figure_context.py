"""Load and cache the evaluated fields needed for figures."""

from __future__ import annotations

import datetime as dt
import stat
from pathlib import Path

from pynamit.plotting.figure_settings import FigureSettings
from pynamit.plotting.grid_fields import GridFields
from pynamit.simulation.schema import SIMULATION_ARTIFACT_NAMES
from pynamit.storage import ArtifactStore

_CACHE_ARTIFACTS = tuple(sorted(SIMULATION_ARTIFACT_NAMES))
_GRID_FIELDS_CACHE: dict[tuple[str, int, int], tuple[tuple, GridFields]] = {}


def figure_time_string(timestamp):
    """Return a compact title-friendly timestamp label."""
    try:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        if isinstance(timestamp, (int, float)):
            return str(dt.timedelta(seconds=float(timestamp)))
        return str(timestamp)


def as_figure_settings(settings):
    """Return a :class:`FigureSettings` instance."""
    if isinstance(settings, FigureSettings):
        return settings
    return FigureSettings.from_dict(settings)


def clear_grid_fields_cache():
    """Clear cached grid-field evaluators."""
    _GRID_FIELDS_CACHE.clear()


def _path_fingerprint(path):
    """Return a change fingerprint for one file or directory tree."""
    try:
        path_stat = path.stat()
    except OSError:
        return None
    if not path.is_dir():
        return ("file", path_stat.st_mtime_ns, path_stat.st_size)

    latest_mtime = path_stat.st_mtime_ns
    entry_count = 0
    total_file_size = 0
    for child in path.rglob("*"):
        try:
            child_stat = child.stat()
        except OSError:
            continue
        entry_count += 1
        latest_mtime = max(latest_mtime, child_stat.st_mtime_ns)
        if stat.S_ISREG(child_stat.st_mode):
            total_file_size += child_stat.st_size
    return ("tree", latest_mtime, entry_count, total_file_size)


def _artifact_fingerprint(simulation_directory):
    directory = Path(simulation_directory).expanduser()
    artifacts = ArtifactStore(directory)
    fingerprint = []
    for name in _CACHE_ARTIFACTS:
        path = artifacts.existing_artifact_path(name)
        if path is not None:
            fingerprint.append((name, str(path), _path_fingerprint(path)))
    return tuple(fingerprint)


def get_grid_fields(settings):
    """Return cached field evaluators for a figure's simulation."""
    settings = as_figure_settings(settings)
    simulation_directory = str(Path(settings.simulation_directory).expanduser().resolve())
    key = (simulation_directory, 60, 100)
    fingerprint = _artifact_fingerprint(simulation_directory)
    cached = _GRID_FIELDS_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    grid_fields = GridFields.from_directory(simulation_directory)
    _GRID_FIELDS_CACHE[key] = (fingerprint, grid_fields)
    return grid_fields


__all__ = [
    "as_figure_settings",
    "clear_grid_fields_cache",
    "figure_time_string",
    "get_grid_fields",
]
