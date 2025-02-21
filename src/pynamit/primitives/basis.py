"""Basis Function Utilities.

This module defines the abstract Basis class for basis representations
of fields.
"""

from abc import ABC, abstractmethod


class Basis(ABC):
    """Abstract base class for basis representations of fields.

    Defines the interface for different basis representations of fields,
    including functions for evaluating basis functions and their
    derivatives on grids.

    Attributes
    ----------
    short_name : str
        Short identifier for the basis.
    index_names : list of str
        Names of the indices used in the basis representation.
    index_length : int
        Total number of basis functions.
    index_arrays : list of array-like
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
    def short_name(self):
        """Short identifier for the basis."""
        pass

    @property
    @abstractmethod
    def index_names(self):
        """Names of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def index_length(self):
        """Total number of basis functions."""
        pass

    @property
    @abstractmethod
    def index_arrays(self):
        """Arrays of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def minimum_phi_sampling(self):
        """Minimum required sampling in phi direction."""
        pass

    @property
    @abstractmethod
    def caching(self):
        """Whether basis evaluations can be cached."""
        pass

    @abstractmethod
    def get_G(self, coeffs, grid, derivative=None):
        """Evaluate basis coefficients on a grid.

        Parameters
        ----------
        coeffs : array-like
            Coefficients of the field in this basis.
        grid : Grid
            Spatial grid for evaluation.
        derivative : {None, 'theta', 'phi'}, optional
            Type of derivative to evaluate:
            - None: evaluate basis functions (default)
            - 'theta': evaluate derivative with respect to theta
            - 'phi': evaluate derivative with respect to phi

        Returns
        -------
        ndarray
            Field values evaluated on the grid points.
        """
        pass
