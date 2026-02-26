"""Mathematical utilities for PynaMIT.

This module provides mathematical abstractions and utilities including:
- Linear map representations (LinearMap, TensorChain)
- Least squares problem definition and solvers
- Physical constants
"""

from . import constants
from .linear_map import LinearMap, as_linear_map, diagonal_linear_map, block_linear_map
from .tensor_chain import TensorChain
from .least_squares_problem import LeastSquaresProblem
from .least_squares_solver import LeastSquaresSolver

__all__ = [
    "constants",
    "LinearMap",
    "as_linear_map",
    "diagonal_linear_map",
    "block_linear_map",
    "TensorChain",
    "LeastSquaresProblem",
    "LeastSquaresSolver",
]
