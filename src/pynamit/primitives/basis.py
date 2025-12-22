"""Basis Function Utilities.

This module contains the abstract Basis class for basis representations
of fields.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pynamit.primitives.basis_evaluator import BasisEvaluator


class Basis(ABC):
    """Abstract class for basis representations of fields.

    Defines the interface for different basis representations of fields,
    including functions for evaluating basis functions and their
    derivatives on grids.

    Attributes
    ----------
    kind : str
        Short identifier for the basis.
    index_names : list[str]
        Names of the indices used in the basis representation.
    index_length : int
        Total number of basis functions.
    index_arrays : list
        Arrays containing the indices used in the basis.
    minimum_phi_sampling : float
        Minimum required sampling points in phi direction.
    caching : bool
        Whether basis evaluations can be cached.

    Notes
    -----
    Subclasses must implement all abstract methods and properties.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Short identifier for the basis."""
        pass

    @property
    @abstractmethod
    def index_names(self) -> list[str]:
        """Names of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def index_length(self) -> int:
        """Total number of basis functions."""
        pass

    @property
    @abstractmethod
    def index_arrays(self) -> list:
        """Arrays of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def minimum_phi_sampling(self) -> float:
        """Minimum required sampling in phi direction."""
        pass

    @property
    @abstractmethod
    def caching(self) -> bool:
        """Whether basis evaluations can be cached."""
        pass

    @abstractmethod
    def to_grid_values(
        self, coeffs: np.ndarray, evaluator: "BasisEvaluator", field_type: str = "scalar"
    ) -> np.ndarray:
        """Evaluate basis on a grid (interpolate coeffs)."""
        pass

    @abstractmethod
    def from_grid_values(
        self, values: np.ndarray, evaluator: "BasisEvaluator", field_type: str = "scalar"
    ) -> np.ndarray:
        """Convert grid values to coefficients."""
        pass

    @abstractmethod
    def regularization_term(
        self, coeffs: np.ndarray, evaluator: "BasisEvaluator", field_type: str = "scalar"
    ) -> float:
        """Compute regularization penalty term."""
        pass

