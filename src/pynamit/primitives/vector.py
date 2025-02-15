"""Vector field representation in basis expansions.

This module provides the Vector class for representing scalar and vector fields
in terms of basis function expansions with transformations between coefficient
and grid representations.

Classes
-------
Vector
    Represents fields in terms of basis function expansions.
"""


class Vector(object):
    """Represents scalar and vector fields in basis expansions.

    Stores and manages coefficients of scalar or tangential vector fields
    represented in terms of basis functions, with methods for converting
    between coefficient and grid representations.

    Parameters
    ----------
    basis : Basis
        Basis functions for field representation
    coeffs : array-like, optional
        Expansion coefficients, by default None
    basis_evaluator : BasisEvaluator, optional
        Evaluator for grid-coefficient conversions, by default None
    grid_values : array-like, optional
        Field values on grid points, by default None
    type : {'scalar', 'tangential'}, optional
        Type of field representation, by default 'scalar'

    Attributes
    ----------
    basis : Basis
        Basis functions used for representation
    coeffs : ndarray
        Expansion coefficients
    _type : str
        Field type ('scalar' or 'tangential')

    Notes
    -----
    Must provide either coeffs directly or both basis_evaluator and grid_values
    for coefficient computation. The basis_evaluator handles transformations
    between coefficient and grid representations.

    Raises
    ------
    ValueError
        If type is invalid or if insufficient initialization parameters
    """

    def __init__(
        self, basis, coeffs=None, basis_evaluator=None, grid_values=None, type="scalar"
    ):
        """
        Initialize the object for storing the coefficients of a vector
        field in a basis. The `basis` object must be provided, along with
        either the `coeffs` or the `grid_values` in combination with a
        `basis_evaluator` object for converting the grid values to basis
        coefficients.

        Parameters
        ----------
        basis : Basis
            Basis object representing the basis of the field.
        coeffs : array-like, optional
            Coefficients of the field in the basis. Default is None.
        basis_evaluator : BasisEvaluator, optional
            BasisEvaluator object for converting grid values to basis coefficients. Default is None.
        grid_values : array-like, optional
            Values of the field on the grid. Default is None.
        type : str, optional
            Type of the field ('scalar' or 'tangential'). Default is 'scalar'.
        """
        if type not in ["scalar", "tangential"]:
            raise ValueError("Type must be either 'scalar' or 'tangential'.")

        self.basis = basis
        self._type = type

        if coeffs is not None:
            self.coeffs = coeffs
        elif (basis_evaluator is not None) and (grid_values is not None):
            self.coeffs = self.coeffs_from_grid(basis_evaluator, grid_values)
        else:
            raise ValueError(
                "Either coeffs or basis evaluator and grid values must be provided."
            )

    def coeffs_from_grid(self, basis_evaluator, grid_values):
        """Compute basis coefficients from grid values.

        Parameters
        ----------
        basis_evaluator : BasisEvaluator
            Evaluator for grid-coefficient conversions
        grid_values : array-like
            Field values on grid points

        Returns
        -------
        ndarray
            Computed basis coefficients

        Notes
        -----
        Uses different inversion methods for scalar vs tangential fields:
        - scalar: Direct least squares inversion
        - tangential: Helmholtz decomposition based inversion
        """
        if self._type == "scalar":
            return basis_evaluator.grid_to_basis(grid_values, helmholtz=False)
        elif self._type == "tangential":
            return basis_evaluator.grid_to_basis(grid_values, helmholtz=True)

    def to_grid(self, basis_evaluator):
        """Evaluate field on grid points.

        Parameters
        ----------
        basis_evaluator : BasisEvaluator
            Evaluator for coefficient-grid conversions

        Returns
        -------
        ndarray
            Field values on grid points

        Notes
        -----
        For tangential fields, reconstructs vector components
        from Helmholtz decomposition.
        """
        if self._type == "scalar":
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz=False)
        elif self._type == "tangential":
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz=True)

    def regularization_term(self, basis_evaluator):
        """Compute regularization penalty term.

        Parameters
        ----------
        basis_evaluator : BasisEvaluator
            Evaluator containing regularization parameters

        Returns
        -------
        float
            Value of regularization penalty term

        Notes
        -----
        Form of regularization depends on field type:
        - scalar: Smoothness penalty on scalar field
        - tangential: Separate penalties on Helmholtz components
        """
        if self._type == "scalar":
            return basis_evaluator.regularization_term(self.coeffs, helmholtz=False)
        elif self._type == "tangential":
            return basis_evaluator.regularization_term(self.coeffs, helmholtz=True)
