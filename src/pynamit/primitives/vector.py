class Vector(object):

    def __init__(self, basis, coeffs = None, basis_evaluator = None, grid_values = None, type = 'scalar'):
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
        if self._type == 'scalar':
            return basis_evaluator.grid_to_basis(grid_values, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.grid_to_basis(grid_values, helmholtz = True)


    def to_grid(self, basis_evaluator):
        if self._type == 'scalar':
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = True)