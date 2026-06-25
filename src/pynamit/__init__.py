"""
PynaMIT: A Python package for dynamic MIT coupling simulations.

This package includes modules for simulation, visualization, and various
utilities.

Attributes
----------
SphericalTransform : class
    Class for analysis and synthesis between a surface basis and grid.
BasisEvaluator : class
    Historical alias for SphericalTransform.
CSBasis : class
    Class for cubed sphere projections.
Dynamics : class
    Class for simulating ionospheric dynamics.
FieldCoefficients : class
    Class for storing validated field coefficients.
FieldEvaluator : class
    Class for evaluating fields.
Grid : class
    Class for grid management.
Mainfield : class
    Class for main field evaluation.
PynamEye : class
    Class for visualization.
SHBasis : class
    Class for spherical harmonics basis functions.
SolidHarmonics : class
    Radial solid-harmonic operations wrapping an SHBasis.
FieldSpace : class
    Class for describing field coefficient spaces.
"""

from .sphere import (
    BasisView,
    CSBasis,
    Grid,
    SHBasis,
    SolidHarmonics,
    SphericalBasis,
    SphericalRepresentation,
    SphericalTransform,
    SurfaceOperators,
    basis_kind,
    is_basis_kind,
    is_cs_basis,
    is_sh_basis,
)
from .primitives.basis_evaluator import BasisEvaluator
from .primitives.field_coefficients import FieldCoefficients
from .primitives.field_evaluator import FieldEvaluator
from .primitives.field_space import FieldSpace
from .simulation.dynamics import Dynamics
from .simulation.prepared_inputs import (
    input_dataset_requirements,
    input_geometry_settings,
    input_projection_settings,
    load_prepared_inputs_into_dynamics,
    prepare_pynamit_inputs,
    prepared_input_contract,
    read_input_manifest,
    run_pynamit_from_inputs,
    validate_prepared_input_compatibility,
)
from .simulation.mainfield import Mainfield
from .math import set_backend, use_jax
from .external_inputs import set_input_source, get_input_source

_LAZY_EXPORTS = {
    "PynamEye": ("pynamit.visualization.pynameye", "PynamEye"),
    "evaluate_projected_input": ("pynamit.visualization", "evaluate_projected_input"),
}


def __getattr__(name):
    """Load optional visualization helpers only when requested."""
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return public package attributes including lazy exports."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "BasisView",
    "BasisEvaluator",
    "CSBasis",
    "FieldCoefficients",
    "Dynamics",
    "FieldEvaluator",
    "FieldSpace",
    "Grid",
    "Mainfield",
    "PynamEye",
    "SHBasis",
    "SolidHarmonics",
    "SphericalBasis",
    "SphericalRepresentation",
    "SphericalTransform",
    "SurfaceOperators",
    "basis_kind",
    "evaluate_projected_input",
    "is_basis_kind",
    "is_cs_basis",
    "is_sh_basis",
    "input_dataset_requirements",
    "input_geometry_settings",
    "input_projection_settings",
    "load_prepared_inputs_into_dynamics",
    "prepare_pynamit_inputs",
    "prepared_input_contract",
    "read_input_manifest",
    "run_pynamit_from_inputs",
    "set_backend",
    "use_jax",
    "set_input_source",
    "get_input_source",
    "validate_prepared_input_compatibility",
]
