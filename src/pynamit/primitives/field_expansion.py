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
        coeffs = basis.from_grid_values(grid_values, basis_evaluator, field_type)
        return cls(basis, coeffs, field_type)

    def to_grid_values(self, basis_evaluator):
        """Evaluate the field expansion on a grid."""
        return self.basis.to_grid_values(self.coeffs, basis_evaluator, self.field_type)

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
        return self.basis.regularization_term(self.coeffs, basis_evaluator, self.field_type)
