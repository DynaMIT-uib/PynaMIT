"""Core public API for PynaMIT simulations."""

from .sphere import (
    CSBasis,
    Grid,
    SHBasis,
    SolidHarmonics,
    SphericalTransform,
)
from .primitives.field_coefficients import FieldCoefficients
from .primitives.field_evaluator import FieldEvaluator
from .primitives.field_space import FieldSpace
from .simulation.dynamics import Dynamics
from .simulation.prepared_inputs import (
    prepare_pynamit_inputs,
    run_pynamit_from_inputs,
)
from .simulation.mainfield import Mainfield, mainfield_from_config
from .math import set_backend, use_jax


__all__ = [
    "CSBasis",
    "Dynamics",
    "FieldCoefficients",
    "FieldEvaluator",
    "FieldSpace",
    "Grid",
    "Mainfield",
    "SHBasis",
    "SolidHarmonics",
    "SphericalTransform",
    "mainfield_from_config",
    "prepare_pynamit_inputs",
    "run_pynamit_from_inputs",
    "set_backend",
    "use_jax",
]
