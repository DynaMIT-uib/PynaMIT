class Vector(object):

    def __init__(self, basis, coeffs = None, basis_evaluator = None, grid_values = None, component = None):
        self.basis = basis

        if coeffs is not None:
            self.coeffs = coeffs
        elif (basis_evaluator is not None) and (grid_values is not None):
            if component is not None:
                self.coeffs = basis_evaluator.from_grid(grid_values, component)
            else:
                self.coeffs = basis_evaluator.from_grid(grid_values)
        else:
            raise ValueError("Either coeffs or grid and grid values must be provided.")
 
    def to_grid(self, basis_evaluator):
        return basis_evaluator.to_grid(self.coeffs)