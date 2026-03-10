"""Simulation module.

Core simulation infrastructure for PynaMIT:
- Dynamics: Main simulation orchestrator
- State: Physical state management
- run_pynamit: Entry point function
- settings: Configuration classes (SimulationMode, DynamicsSettings)
- ToroidalSystemMatrices: System matrices for toroidal induction
- PoloidalSystemMatrices: System matrices for poloidal induction
"""

from .state import State
from .dynamics import Dynamics
from .runner import run_pynamit
from .settings import SimulationMode, DynamicsSettings, FLOAT_ERROR_MARGIN
from .induction import PoloidalSystemMatrices, ToroidalSystemMatrices
from .data import SimulationData
from .migration import migrate_run_storage, RunStorageMigrationReport

__all__ = [
    "Dynamics",
    "State",
    "run_pynamit",
    "SimulationMode",
    "DynamicsSettings",
    "FLOAT_ERROR_MARGIN",
    "ToroidalSystemMatrices",
    "PoloidalSystemMatrices",
    "SimulationData",
    "migrate_run_storage",
    "RunStorageMigrationReport",
]
