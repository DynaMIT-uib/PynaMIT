import numpy as np

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

        self.b_original_shape = [self.A[i].shape[:self.b_dims[i]] for i in range(len(self.A))]
        self.b_compound_size = [np.prod(self.b_original_shape[i]) for i in range(len(self.A))]

        self.solution_original_shape = [self.A[i].shape[self.b_dims[i]:] for i in range(len(self.A))]
        self.solution_compound_size = [np.prod(self.solution_original_shape[i]) for i in range(len(self.A))]

        for i in range(len(self.A)):
            self.A[i] = self.A[i].reshape((self.b_compound_size[i], self.solution_compound_size[i]))
            if self.weights[i] is not None:
                self.weights[i] = self.weights[i].reshape((self.b_compound_size[i], 1))
            if self.reg_L[i] is not None:
                self.reg_L[i] = self.reg_L[i].reshape((self.solution_compound_size[i], self.solution_compound_size[i]))

    def ensure_list(self, x):
        """
        Ensure that x is a list.

        """

        if x is None:
            x = [None] * len(self.A)

        elif not isinstance(x, list):
            x = [x]

        return x

    def solve(self, b):
        """
        Solve the least squares problem.

        """

        b_list = self.ensure_list(b)

        b_remaining_original_shape = [b_list[i].shape[self.b_dims[i]:] if ((b_list[i] is not None) and (len(b_list[i].shape) > self.b_dims[i])) else None for i in range(len(self.A))]
        b_remaining_compound_size = [np.prod(b_remaining_original_shape[i]) if b_remaining_original_shape[i] is not None else None for i in range(len(self.A))]

        for i in range(len(self.A)):
            if b_list[i] is not None:
                if b_remaining_compound_size[i] is not None:
                    b_list[i] = b_list[i].reshape((self.b_compound_size[i], b_remaining_compound_size[i]))
                else:
                    b_list[i] = b_list[i].reshape((self.b_compound_size[i]))
            else:
                b_list[i] = None

        solution = [np.dot(self.ATWA_plus_R_inv_ATW[i], b_list[i]) if b_list[i] is not None else None for i in range(len(self.A))]

        for i in range(len(self.A)):
            if solution[i] is not None:
                if b_remaining_compound_size[i] is not None:
                    solution[i] = solution[i].reshape((self.solution_original_shape[i] + b_remaining_original_shape[i]))
                else:
                    solution[i] = solution[i].reshape((self.solution_original_shape[i]))

        return solution

    @property
    def ATW(self):
        """
        Return the matrix ``A^T W``.

        """

        if not hasattr(self, '_ATW'):
            self._ATW = []
            for i in range(len(self.A)):
                if self.weights[i] is not None:
                    self._ATW.append((self.weights[i] * self.A[i]).T)
                else:
                    self._ATW.append(self.A[i].T)

        return self._ATW

    @property
    def ATWA(self):
        """
        Return the matrix ``A^T W A``

        """

        if not hasattr(self, '_ATWA'):
            self._ATWA = sum([np.dot(self.ATW[i], self.A[i]) for i in range(len(self.A))])

        return self._ATWA

    @property
    def ATWA_plus_R_inv(self):
        """
        Return the inverse of the matrix ``A^T W A + R``.

        """

        if not hasattr(self, '_ATWA_plus_R_inv'):
            ATWA_plus_R = self.ATWA.copy()
            for i in range(len(self.A)):
                if self.reg_lambda[i] is not None:
                    ATWA_plus_R += np.dot(self.reg_L[i].T, self.reg_L[i])

            self._ATWA_plus_R_inv = np.linalg.pinv(ATWA_plus_R, rcond = self.pinv_rtol, hermitian = True)

        return self._ATWA_plus_R_inv
    
    @property
    def ATWA_plus_R_inv_ATW(self):
        """
        Return inverse of the matrix ``A^T W A + R`` multiplied by
        ``A^T W``.

        """

        if not hasattr(self, '_ATWA_plus_R_inv_ATW'):
            self._ATWA_plus_R_inv_ATW = [np.dot(self.ATWA_plus_R_inv, self.ATW[i]) for i in range(len(self.A))]

        return self._ATWA_plus_R_inv_ATW