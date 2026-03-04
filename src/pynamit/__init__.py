"""
PynaMIT: A Python package for dynamic MIT coupling simulations.

This package includes modules for simulation, visualization, and various
utilities.

Attributes
----------
CSBasis : class
    Class for cubed sphere projections.
Dynamics : class
    Class for simulating ionospheric dynamics.
Field : class
    Unified class for scalar/vector fields.
Grid : class
    Class for grid management.
Mainfield : class
    Class for main field evaluation.
SimulationViewer : class
    Class for saved-run visualization.
plot_global_map : function
    Function for global map plotting.
plot_simulation_snapshot : function
    Function for diagnostic snapshot plotting.
SHBasis : class
    Class for spherical harmonics basis functions.
"""

from .cubed_sphere import CSBasis
from .primitives import (
    Field,
    Grid,
    IO,
    Mainfield,
    Timeseries,
)
from .simulation import Dynamics, SimulationData, State, run_pynamit
try:
    from .visualization import SimulationViewer, plot_global_map, plot_simulation_snapshot
except ImportError:
    # Visualization dependencies (e.g. Cartopy) might be missing in production/test envs
    SimulationViewer = None
    plot_global_map = None
    plot_simulation_snapshot = None
from .spherical_harmonics import SHBasis
from .utils import set_backend

__all__ = [
    "CSBasis",
    "Dynamics",
    "Field",
    "Grid",
    "IO",
    "Mainfield",
    "SimulationData",
    "SimulationViewer",
    "plot_global_map",
    "plot_simulation_snapshot",
    "SHBasis",
    "State",
    "Timeseries",
    "run_pynamit",
    "set_backend",
]
