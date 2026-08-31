"""Core public API for PynaMIT simulations."""

from kompe.math import get_backend, set_backend

from .fields import FieldCoefficients, FieldSpace
from .geomagnetism import MainField
from .results import SimulationResults
from .simulation import InputPreparation, Simulation, SimulationConfig

__all__ = [
    "FieldCoefficients",
    "FieldSpace",
    "MainField",
    "InputPreparation",
    "SimulationResults",
    "Simulation",
    "SimulationConfig",
    "get_backend",
    "set_backend",
]
