"""Core public API for PynaMIT simulations."""

from kompe.math import set_backend, use_jax

from .fields import FieldCoefficients, FieldSpace
from .geomagnetism import MagneticFieldEvaluation, MainField
from .simulation import Simulation, SimulationConfig
from .sphere import BasisEvaluator

__all__ = [
    "BasisEvaluator",
    "FieldCoefficients",
    "FieldSpace",
    "MagneticFieldEvaluation",
    "MainField",
    "Simulation",
    "SimulationConfig",
    "set_backend",
    "use_jax",
]
