"""Core public API for PynaMIT simulations."""

from .fields import FieldCoefficients, FieldSpace
from .geomagnetism import MagneticFieldEvaluation, MainField
from .math import set_backend, use_jax
from .simulation import Simulation, SimulationConfig
from .sphere import BasisEvaluator, CSBasis, Grid, SHBasis, SolidHarmonics, SphericalTransform


__all__ = [
    "BasisEvaluator",
    "CSBasis",
    "FieldCoefficients",
    "FieldSpace",
    "Grid",
    "MagneticFieldEvaluation",
    "MainField",
    "SHBasis",
    "Simulation",
    "SimulationConfig",
    "SolidHarmonics",
    "SphericalTransform",
    "set_backend",
    "use_jax",
]
