from abc import ABC, abstractmethod

class Basis(ABC):
    """
    Abstract class for basis representations.

    Should include a method to evaluate basis coefficients on a grid.
    Also, should include functions to return derivatives of coefficients,
    represented in the same basis.

    """

    @property
    @abstractmethod
    def short_name(self):
        pass

    @property
    @abstractmethod
    def index_names(self):
        pass

    @property
    @abstractmethod
    def index_length(self):
        pass

    @property
    @abstractmethod
    def indices(self):
        pass

    @property
    @abstractmethod
    def minimum_phi_sampling(self):
        pass

    @property
    @abstractmethod
    def caching(self):
        pass

    @abstractmethod
    def get_G(self, coeffs, grid, derivative = None):
        pass