from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
import numpy as np

@dataclass
class FieldExpansion:
    """Class for representing fields as basis expansions.

    This class stores and manages expansion coefficients for scalar and
    horizontal vector fields in a given basis and provides methods for
    conversion between coefficient and grid representations.
    """
    
    basis: Any
    coeffs: np.ndarray
    field_type: Literal["scalar", "tangential"] = "scalar"
    
    def __post_init__(self):
        if self.field_type not in ["scalar", "tangential"]:
            raise ValueError("field type must be either 'scalar' or 'tangential'.")

    @classmethod
    def from_grid_values(
        cls, 
        basis: Any, 
        basis_evaluator: Any, 
        grid_values: np.ndarray, 
        field_type: Literal["scalar", "tangential"] = "scalar"
    ) -> FieldExpansion:
        """Create a FieldExpansion from grid values."""
        coeffs = cls._calculate_coeffs_from_grid(basis, basis_evaluator, grid_values, field_type)
        return cls(basis, coeffs, field_type)

    @staticmethod
    def _calculate_coeffs_from_grid(basis, basis_evaluator, grid_values, field_type):
        """Helper to compute basis coefficients from grid values."""
        if basis.kind == "GRID":
            return grid_values
        else:
            if field_type == "scalar":
                return basis_evaluator.grid_to_basis(grid_values, helmholtz=False)
            elif field_type == "tangential":
                return basis_evaluator.grid_to_basis(grid_values, helmholtz=True)
            else:
                raise ValueError(f"Unknown field_type: {field_type}")



    def to_grid(self, basis_evaluator):
        """Evaluate field on grid points.

        Parameters
        ----------
        basis_evaluator : BasisEvaluator
            Evaluator for coefficient-grid conversions.

        Returns
        -------
        ndarray
            Field values on grid points.

        Notes
        -----
        For tangential fields, reconstructs vector components from
        Helmholtz decomposition terms evaluated on the grid. For scalar
        fields, directly evaluates basis functions on the grid.
        """
        if self.basis.kind == "GRID":
            # If the basis is a grid, return the grid values as
            # coefficients.
            return self.coeffs
        else:
            if self.field_type == "scalar":
                return basis_evaluator.basis_to_grid(self.coeffs, helmholtz=False)
            elif self.field_type == "tangential":
                return basis_evaluator.basis_to_grid(self.coeffs, helmholtz=True)

    def regularization_term(self, basis_evaluator):
        """Compute regularization penalty term.

        Parameters
        ----------
        basis_evaluator : BasisEvaluator
            Evaluator containing regularization parameters.

        Returns
        -------
        float
            Value of regularization penalty term.

        Notes
        -----
        Form of regularization depends on field type:
        - scalar: Single penalty on scalar field
        - tangential: Separate penalties on Helmholtz components
        """
        if self.basis.kind == "GRID":
            # If the basis is a grid, return the grid values as
            # coefficients.
            return None
        else:
            if self.field_type == "scalar":
                return basis_evaluator.regularization_term(self.coeffs, helmholtz=False)
            elif self.field_type == "tangential":
                return basis_evaluator.regularization_term(self.coeffs, helmholtz=True)
