"""Induction-system assembly and solver helpers."""

from .operator_utils import (
    build_linear_map,
    cached_dense_builder,
    coerce_dense_operator_matrix,
)
from .operators import BlockCoupledOperator, ResistivityTensorOperator
from .poloidal import PoloidalSystemMatrices
from .poloidal_closure import PoloidalClosureProjector, RMCouplingOperators
from .poloidal_solver import PoloidalSolver
from .toroidal import ToroidalSystemMatrices
from .toroidal_closure import ToroidalClosureProjector
from .toroidal_solver import ToroidalSolver

__all__ = [
    "BlockCoupledOperator",
    "ResistivityTensorOperator",
    "PoloidalSystemMatrices",
    "PoloidalClosureProjector",
    "RMCouplingOperators",
    "PoloidalSolver",
    "ToroidalSystemMatrices",
    "ToroidalClosureProjector",
    "ToroidalSolver",
    "build_linear_map",
    "cached_dense_builder",
    "coerce_dense_operator_matrix",
]
