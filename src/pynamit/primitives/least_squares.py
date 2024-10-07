import numpy as np
from pynamit.various.math import tensor_pinv_positive_semidefinite, tensor_scale_left, tensor_transpose

class LeastSquares(object):
    """
    Class for solving least squares problems of the form ``Ax = b``.

    """
    
    def __init__(self, A, contracted_dims, pinv_rtol = 1e-15, weights = None, reg_lambda = None, reg_L = None):
        self.A = A
        self.contracted_dims = contracted_dims
        self.pinv_rtol = pinv_rtol
        self.weights = weights
        self.reg_lambda = reg_lambda
        self.reg_L = reg_L

    def solve(self, b):
        """
        Solve the least squares problem.

        """

        return np.tensordot(self.ATWA_plus_R_inv, np.tensordot(self.ATW, b, self.contracted_dims), self.contracted_dims)

    @property
    def ATW(self):
        """
        Return the matrix ``A^T W``.

        """

        if not hasattr(self, '_ATW'):
            if self.weights is not None:
                self._ATW = tensor_transpose(tensor_scale_left(self.weights, self.A), self.contracted_dims)
            else:
                self._ATW = tensor_transpose(self.A, self.contracted_dims)

        return self._ATW

    @property
    def ATWA(self):
        """
        Return the matrix ``A^T W A``

        """

        if not hasattr(self, '_ATWA'):
            self._ATWA = np.tensordot(self.ATW, self.A, self.contracted_dims)

        return self._ATWA

    @property
    def ATWA_plus_R_inv(self):
        """
        Return the inverse of the matrix ``A^T W A + R``.

        """

        if not hasattr(self, '_ATWA_plus_R_inv'):
            if (self.reg_lambda is not None) and (self.reg_L is not None):
                ATWA_plus_R = self.ATWA + self.reg_lambda * np.tensordot(tensor_transpose(self.reg_L, self.contracted_dims), self.reg_L, self.contracted_dims)
            else:
                ATWA_plus_R = self.ATWA

            self._ATWA_plus_R_inv = tensor_pinv_positive_semidefinite(ATWA_plus_R, contracted_dims = self.contracted_dims, rtol = self.pinv_rtol)

        return self._ATWA_plus_R_inv