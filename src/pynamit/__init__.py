"""Core public API for PynaMIT simulations."""

from .sphere import (
    BasisEvaluator,
    CSBasis,
    Grid,
    SHBasis,
    SolidHarmonics,
    SphericalTransform,
)
from .fields import FieldCoefficients, FieldSpace
from .geomagnetism import MagneticFieldEvaluation, MainField
from .simulation.config import SimulationConfig
from .simulation.api import Simulation
from .math import set_backend, use_jax


__all__ = [
    "BasisEvaluator",
    "CSBasis",
    "FieldCoefficients",
    "MagneticFieldEvaluation",
    "FieldSpace",
    "Grid",
    "MainField",
    "SHBasis",
    "Simulation",
    "SimulationConfig",
    "SolidHarmonics",
    "SphericalTransform",
    "set_backend",
    "use_jax",
]
