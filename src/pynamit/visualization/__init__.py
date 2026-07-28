"""High-level visualization API for PynaMIT results."""

_LAZY_EXPORTS = {
    "PynamEye": ("pynamit.visualization.pynameye", "PynamEye"),
    "PynamitFigureSpec": ("pynamit.visualization.figure_specs", "PynamitFigureSpec"),
    "PynamitPanelApp": ("pynamit.visualization.panel_app", "PynamitPanelApp"),
    "SavedCoefficientFieldView": ("pynamit.visualization.run_fields", "SavedCoefficientFieldView"),
    "SavedRunView": ("pynamit.visualization.saved_run", "SavedRunView"),
    "build_pynamit_panel_app": ("pynamit.visualization.panel_app", "build_pynamit_panel_app"),
    "evaluate_projected_input": (
        "pynamit.visualization.input_projection",
        "evaluate_projected_input",
    ),
    "plot_input_projection_comparison": (
        "pynamit.visualization.input_projection_comparison",
        "plot_input_projection_comparison",
    ),
    "render_pynamit_figure": ("pynamit.visualization.figure_builder", "render_pynamit_figure"),
    "save_pynamit_movie": ("pynamit.visualization.figure_builder", "save_pynamit_movie"),
    "write_input_projection_diagnostics": (
        "pynamit.visualization.input_projection_comparison",
        "write_input_projection_diagnostics",
    ),
}


def __getattr__(name):
    """Load optional visualization dependencies only when requested."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return public visualization attributes including lazy exports."""
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(_LAZY_EXPORTS)
