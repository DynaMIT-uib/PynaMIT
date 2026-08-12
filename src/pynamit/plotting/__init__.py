"""Plot PynaMIT inputs and simulation results."""

_LAZY_EXPORTS = {
    "FigureSettings": ("pynamit.plotting.figure_settings", "FigureSettings"),
    "MapCoordinateContext": ("pynamit.plotting.map_coordinates", "MapCoordinateContext"),
    "GridFields": ("pynamit.plotting.grid_fields", "GridFields"),
    "render_figure": ("pynamit.plotting.figure_builder", "render_figure"),
    "save_movie": ("pynamit.plotting.figure_builder", "save_movie"),
    "plot_global_polar_map": ("pynamit.plotting.diagnostics", "plot_global_polar_map"),
    "plot_output_diagnostics": ("pynamit.plotting.diagnostics", "plot_output_diagnostics"),
}


def __getattr__(name):
    """Load optional plotting dependencies only when requested."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return public plotting attributes including lazy exports."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["FigureSettings", "render_figure", "save_movie"]
