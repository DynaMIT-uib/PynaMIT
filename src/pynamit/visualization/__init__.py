"""Visualization tools for PynaMIT.

This package owns rendering and viewer utilities. Numerical read/evaluation
helpers live under ``pynamit.postprocess``.
"""

__all__: list[str] = []

try:
    from .plot_recipes import plot_global_map, plot_simulation_snapshot
except ModuleNotFoundError:
    pass
else:
    __all__.extend(["plot_global_map", "plot_simulation_snapshot"])

try:
    from .simulation_viewer import SimulationViewer
except ModuleNotFoundError:
    pass
else:
    __all__.append("SimulationViewer")


try:
    from .input_vs_interpolated import plot_input_vs_interpolated
except ModuleNotFoundError:
    pass
else:
    __all__.append("plot_input_vs_interpolated")
