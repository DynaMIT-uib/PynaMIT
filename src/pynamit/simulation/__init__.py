"""Simulation module.

Core simulation infrastructure for PynaMIT:
- Dynamics: Main simulation orchestrator
- State: Physical state management
- run_pynamit: Entry point function
- settings: Configuration classes (SimulationMode, DynamicsSettings)
"""

from .state import State
from .dynamics import Dynamics
from .runner import run_pynamit
from .settings import SimulationMode, DynamicsSettings, FLOAT_ERROR_MARGIN

__all__ = [
    "Dynamics",
    "State",
    "run_pynamit",
    "SimulationMode",
    "DynamicsSettings",
    "FLOAT_ERROR_MARGIN",
]
