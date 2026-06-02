"""Compatibility wrapper for basis-grid evaluation."""

import numpy as np

from pynamit.primitives.field_space import FieldSpace
from pynamit.primitives.field_transform import FieldTransform


class BasisEvaluator(FieldTransform):
    """FieldTransform with the historical basis-first constructor."""

    def __init__(self, basis, grid, **kwargs):
        """Initialize from a scalar basis and target grid."""
        super().__init__(
            FieldSpace.from_basis(basis, field_type="scalar"),
            grid,
            **kwargs,
        )

    def basis_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values."""
        return self._coefficients_to_grid(
            np.asarray(coeffs),
            derivative=derivative,
            helmholtz=helmholtz,
        )

    def grid_to_basis(self, grid_values, helmholtz=False):
        """Transform grid values to basis coefficients."""
        if helmholtz:
            return self.least_squares_solution_helmholtz(grid_values)
        return self.least_squares_solution(grid_values)

    def regularization_term(self, coeffs, helmholtz=False):
        """Return the field-space regularization term."""
        if helmholtz:
            return np.tensordot(self.L_helmholtz, np.asarray(coeffs), 2)
        return super().regularization_term(coeffs)
