"""Field expansion module.

This module contains the FieldExpansion class for representing fields as
basis expansions.
"""


class FieldExpansion(object):
    """Class for representing fields as basis expansions.

    This class stores and manages expansion coefficients for scalar and
    horizontal vector fields in a given basis and provides methods for
    conversion between coefficient and grid representations.

    Attributes
    ----------
    basis : Basis
        Basis object representing the basis of the field.
    coeffs : ndarray
        Expansion coefficients of the field for the given basis.
    field_type : str
        Type of the field, either 'scalar' or 'tangential'.
    """

    def __init__(
        self, basis, coeffs=None, basis_evaluator=None, grid_values=None, field_type="scalar"
    ):
        """Initialize the field expansion.

        Initializes the expansion with the provided basis and either
        coefficients or grid values in combination with a basis
        evaluator for conversion to coefficients.

        Parameters
        ----------
        basis : Basis
            Basis object representing the basis of the field.
        coeffs : array-like, optional
            Expansion coefficients of the field for the given basis.
        basis_evaluator : BasisEvaluator, optional
            BasisEvaluator object for converting between basis
            coefficients and grid values.
        grid_values : array-like, optional
            Values of the field on the grid.
        field_type : str, optional
            Type of the field ('scalar' or 'tangential').

        Raises
        ------
        ValueError
            If `field_type` is invalid or if insufficient initialization
            parameters are provided.
        """
        if field_type not in ["scalar", "tangential"]:
            raise ValueError("field type must be either 'scalar' or 'tangential'.")

        self.basis = basis
        self.field_type = field_type

        if coeffs is not None:
            self.coeffs = coeffs
        elif (basis_evaluator is not None) and (grid_values is not None):
            self.coeffs = self.coeffs_from_grid(basis_evaluator, grid_values)
        else:
            raise ValueError("Either coeffs or basis evaluator and grid values must be provided.")

    def coeffs_from_grid(self, basis_evaluator, grid_values):
        """Compute basis coefficients from grid values.

        Parameters
        ----------
        basis_evaluator : BasisEvaluator
            Evaluator for grid-coefficient conversions.
        grid_values : array-like
            Field values on grid points.

        Returns
        -------
        ndarray
            Computed basis coefficients.

        Notes
        -----
        Uses different least squares inversion methods for scalar and
        tangential fields:
        - scalar: Direct inversion
        - tangential: Helmholtz decomposition based inversion
        """
        if self.basis.short_name == "GRID":
            # If the basis is a grid, return the grid values as coefficients.
            return self.coeffs
        else:
            if self.field_type == "scalar":
                return basis_evaluator.grid_to_basis(grid_values, helmholtz=False)
            elif self.field_type == "tangential":
                return basis_evaluator.grid_to_basis(grid_values, helmholtz=True)

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
        if self.basis.short_name == "GRID":
            # If the basis is a grid, return the grid values as coefficients.
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
        if self.basis.short_name == "GRID":
            # If the basis is a grid, return the grid values as coefficients.
            return None
        else:
            if self.field_type == "scalar":
                return basis_evaluator.regularization_term(self.coeffs, helmholtz=False)
            elif self.field_type == "tangential":
                return basis_evaluator.regularization_term(self.coeffs, helmholtz=True)
