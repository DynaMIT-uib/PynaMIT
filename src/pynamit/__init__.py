"""Core public API for PynaMIT simulations."""

from kompe.math import get_backend, set_backend

from .geomagnetism import MainField
from .results import SimulationResults
from .simulation.config import SimulationConfig
from .simulation.geometry import SimulationGeometry
from .simulation.input_preparation import InputPreparation
from .simulation.simulation import Simulation

__all__ = [
    "MainField",
    "InputPreparation",
    "SimulationResults",
    "Simulation",
    "SimulationConfig",
    "SimulationGeometry",
    "get_backend",
    "set_backend",
]
