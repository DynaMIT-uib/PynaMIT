import numpy as np
from pynamit.math.tensor_operations import tensor_pinv_positive_semidefinite, tensor_scale_left, tensor_transpose

class LeastSquares(object):
    """
    Class for solving least squares problems of the form ``Ax = b``.

    """
    
    def __init__(self, A, solution_dims, weights = None, reg_lambda = None, reg_L = None, pinv_rtol = 1e-15):
        self.A = self.ensure_list(A)
        self.solution_dims = solution_dims
        self.weights = self.ensure_list(weights)
        self.reg_lambda = self.ensure_list(reg_lambda)
        self.reg_L = self.ensure_list(reg_L)
        self.pinv_rtol = pinv_rtol

        self.b_dims = [len(self.A[i].shape) - solution_dims for i in range(len(self.A))]

    def ensure_list(self, x):
        """
        Ensure that x is a list.

        """

        if not isinstance(x, list):
            x = [x]

        return x

    def solve(self, b):
        """
        Solve the least squares problem.

        """

        b_list = self.ensure_list(b)

        solution = [np.tensordot(self.ATWA_plus_R_inv, np.tensordot(self.ATW[i], b_list[i], self.b_dims[i]), self.solution_dims) for i in range(len(self.A))]

        return solution

    @property
    def ATW(self):
        """
        Return the matrix ``A^T W``.

        """

        if not hasattr(self, '_ATW'):

            if self.weights[0] is not None:
                self._ATW = [tensor_transpose(tensor_scale_left(self.weights[i], self.A[i]), self.b_dims[i]) for i in range(len(self.A))]
            else:
                self._ATW = [tensor_transpose(self.A[i], self.b_dims[i]) for i in range(len(self.A))]

        return self._ATW

    @property
    def ATWA(self):
        """
        Return the matrix ``A^T W A``

        """

        if not hasattr(self, '_ATWA'):
            self._ATWA = sum([np.tensordot(self.ATW[i], self.A[i], self.b_dims[i]) for i in range(len(self.A))])

        return self._ATWA

    @property
    def ATWA_plus_R_inv(self):
        """
        Return the inverse of the matrix ``A^T W A + R``.

        """

        if not hasattr(self, '_ATWA_plus_R_inv'):
            if (self.reg_lambda[0] is not None) and (self.reg_L[0] is not None):
                ATWA_plus_R = self.ATWA + sum([self.reg_lambda[i] * np.tensordot(tensor_transpose(self.reg_L[i], self.solution_dims), self.reg_L[i], self.solution_dims) for i in range(len(self.A))])
            else:
                ATWA_plus_R = self.ATWA

            self._ATWA_plus_R_inv = tensor_pinv_positive_semidefinite(ATWA_plus_R, contracted_dims = self.solution_dims, rtol = self.pinv_rtol)

        return self._ATWA_plus_R_inv