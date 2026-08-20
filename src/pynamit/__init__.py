"""Core public API for PynaMIT simulations."""

from kompe.math import get_backend, jax_enabled, set_backend, use_jax

from .fields import FieldCoefficients, FieldSpace
from .geomagnetism import MagneticFieldEvaluation, MainField
from .results import SimulationResults
from .simulation import InputPreparation, Simulation, SimulationConfig

__all__ = [
    "FieldCoefficients",
    "FieldSpace",
    "MagneticFieldEvaluation",
    "MainField",
    "InputPreparation",
    "SimulationResults",
    "Simulation",
    "SimulationConfig",
    "get_backend",
    "jax_enabled",
    "set_backend",
    "use_jax",
]
