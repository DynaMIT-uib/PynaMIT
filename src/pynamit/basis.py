from abc import ABC, abstractmethod

class Basis(ABC):
    """
    Abstract class for basis representations.

    Should include a method to evaluate basis coefficients on a grid.
    Also, should include functions to return derivatives of coefficients,
    represented in the same basis.

    """

    @abstractmethod
    def get_G(self, coeffs, grid):
        pass