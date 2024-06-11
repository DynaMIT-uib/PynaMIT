class Vector(object):

    def __init__(self, basis, coeffs = None, basis_evaluator = None, grid_values = None, helmholtz = False):
        self.basis = basis
        self.helmholtz = helmholtz

        if coeffs is not None:
            self.coeffs = coeffs
        elif (basis_evaluator is not None) and (grid_values is not None):
            self.coeffs = basis_evaluator.grid_to_basis(grid_values, helmholtz = self.helmholtz)
        else:
            raise ValueError("Either coeffs or grid and grid values must be provided.")
 
    def to_grid(self, basis_evaluator):
        return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = self.helmholtz)