"""Mathematical utilities for PynaMIT.

This module provides mathematical abstractions and utilities including:
- Linear map representations (LinearMap, TensorChain)
- Least squares problem definition and solvers
- Physical constants
- Array utilities
"""

from . import arrayutils
from . import constants
from .linear_map import LinearMap, as_linear_map, diagonal_linear_map, BlockLinearMap
from .tensor_chain import TensorChain
from .least_squares_problem import LeastSquaresProblem
from .least_squares_solver import LeastSquaresSolver

__all__ = [
    "arrayutils",
    "constants",
    "LinearMap",
    "as_linear_map",
    "diagonal_linear_map",
    "BlockLinearMap",
    "TensorChain",
    "LeastSquaresProblem",
    "LeastSquaresSolver",
]
