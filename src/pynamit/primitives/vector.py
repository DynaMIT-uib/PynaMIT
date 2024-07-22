class Vector(object):

    def __init__(self, basis, coeffs = None, basis_evaluator = None, grid_values = None, type = 'scalar'):
        self.basis = basis
        self._type = type

        if coeffs is not None:
            self.coeffs = coeffs
        elif (basis_evaluator is not None) and (grid_values is not None):
            if self._type == 'scalar':
                self.coeffs = basis_evaluator.grid_to_basis(grid_values, helmholtz = False)
            elif self._type == 'tangential':
                self.coeffs = basis_evaluator.grid_to_basis(grid_values, helmholtz = True)
            else:
                raise ValueError("Type must be either 'scalar' or 'tangential'.")
        else:
            raise ValueError("Either coeffs or grid and grid values must be provided.")
 
    def to_grid(self, basis_evaluator):
        if self._type == 'scalar':
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = True)