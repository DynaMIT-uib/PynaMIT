"""Basis Function Utilities.

This module contains the abstract Basis class for basis representations
of fields.
"""

from abc import ABC, abstractmethod


class Basis(ABC):
    """Abstract class for basis representations of fields.

    Defines the interface for different basis representations of fields,
    including functions for evaluating basis functions and their
    derivatives on grids.

    Attributes
    ----------
    kind : str
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
    def kind(self):
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
