"""Induction-system assembly and solver helpers."""

from pynamit.simulation.operator_api_utils import (
    build_linear_map,
    cached_dense_builder,
    coerce_dense_operator_matrix,
)
from pynamit.simulation.operators import BlockCoupledOperator, ResistivityTensorOperator
from pynamit.simulation.poloidal import PoloidalSystemMatrices
from pynamit.simulation.poloidal_closure import PoloidalClosureProjector, RMCouplingOperators
from pynamit.simulation.poloidal_solver import PoloidalOperatorAPI
from pynamit.simulation.toroidal import ToroidalSystemMatrices
from pynamit.simulation.toroidal_closure import ToroidalClosureProjector
from pynamit.simulation.toroidal_solver import ToroidalOperatorAPI

__all__ = [
    "BlockCoupledOperator",
    "ResistivityTensorOperator",
    "PoloidalSystemMatrices",
    "PoloidalClosureProjector",
    "RMCouplingOperators",
    "PoloidalOperatorAPI",
    "ToroidalSystemMatrices",
    "ToroidalClosureProjector",
    "ToroidalOperatorAPI",
    "build_linear_map",
    "cached_dense_builder",
    "coerce_dense_operator_matrix",
]
