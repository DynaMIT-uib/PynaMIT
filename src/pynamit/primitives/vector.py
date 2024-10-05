import numpy as np

class Vector(object):

    def __init__(self, basis, coeffs = None, basis_evaluator = None, grid_values = None, type = 'scalar'):
        if type not in ['scalar', 'tangential']:
            raise ValueError("Type must be either 'scalar' or 'tangential'.")

        self.basis = basis
        self._type = type

        if coeffs is not None:
            self.coeffs = self.split_coeffs(coeffs)
        elif (basis_evaluator is not None) and (grid_values is not None):
            self.coeffs = self.coeffs_from_grid(basis_evaluator, grid_values)
        else:
            raise ValueError("Either coeffs or basis evaluator and grid values must be provided.")


    def coeffs_from_grid(self, basis_evaluator, grid_values):
        if self._type == 'scalar':
            return basis_evaluator.grid_to_basis(grid_values, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.grid_to_basis(grid_values, helmholtz = True)


    def split_coeffs(self, coeffs):
        if self._type == 'scalar':
            return coeffs
        elif self._type == 'tangential':
            return np.moveaxis(np.split(coeffs, 2), 0, 1)


    def to_grid(self, basis_evaluator):
        if self._type == 'scalar':
            return basis_evaluator.basis_to_grid(self.coeffs, helmholtz = False)
        elif self._type == 'tangential':
            return np.hstack(np.moveaxis(basis_evaluator.basis_to_grid(self.coeffs, helmholtz = True), 0, 1))


    def regularization_term(self, basis_evaluator):
        if self._type == 'scalar':
            return basis_evaluator.regularization_term(self.coeffs, helmholtz = False)
        elif self._type == 'tangential':
            return basis_evaluator.regularization_term(self.coeffs, helmholtz = True)