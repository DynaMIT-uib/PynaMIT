class Vector(object):
    """
    Object for storing the coefficients of a scalar or tangential vector
    field represented in a basis. The `basis` object, `coeffs`, and vector
    `type` are stored as attributes of the object.

    """

    def __init__(self, basis, coeffs = None, basis_evaluator = None, grid_values = None, type = 'scalar'):
        """
        Initialize the object for storing the coefficients of a vector
        field in a basis. The `basis` object must be provided, along with
        either the `coeffs` or the `grid_values` in combination with a
        `basis_evaluator` object for converting the grid values to basis
        coefficients.

        """

        if type not in ['scalar', 'tangential']:
            raise ValueError("Type must be either 'scalar' or 'tangential'.")

        self.basis = basis
        self._type = type

        if coeffs is not None:
            self.coeffs = coeffs
        elif (basis_evaluator is not None) and (grid_values is not None):
            self.coeffs = self.coeffs_from_grid(basis_evaluator, grid_values)
        else:
            raise ValueError("Either coeffs or basis evaluator and grid values must be provided.")


    def coeffs_from_grid(self, basis_evaluator, grid_values):
        """
        Perform the inversion to compute the basis expansion coefficients
        from the grid values.

        """

        if self._type == 'scalar':
            return basis_evaluator.grid_to_basis(grid_values, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.grid_to_basis(grid_values, helmholtz = True)


    def to_grid(self, basis_evaluator):
        """
        Evaluate the basis expansion on a grid, where the mapping between
        the basis and the grid is provided by the `basis_evaluator` object.

        """

        if self._type == 'scalar':
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = True)


    def regularization_term(self, basis_evaluator):
        """
        Return the regularization term for the inversion used to compute
        the basis expansion coefficients.

        """

        if self._type == 'scalar':
            return basis_evaluator.regularization_term(self.coeffs, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.regularization_term(self.coeffs, helmholtz = True)