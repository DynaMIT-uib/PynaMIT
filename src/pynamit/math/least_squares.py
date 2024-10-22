import numpy as np
from pynamit.math.compound_index_array import CompoundIndexArray

class LeastSquares(object):
    """
    Class for solving least squares problems of the form ``Ax = b``.

    """
    
    def __init__(self, A, solution_dims, weights = None, reg_lambda = None, reg_L = None, pinv_rtol = 1e-15):
        self.solution_dims = solution_dims

        if isinstance(A, list):
            self.n_arrays = len(A)
        else:
            self.n_arrays = 1

        self.A = self.compound_index_arrays(A, last_n_compounded = [solution_dims] * self.n_arrays)
        self.weights = self.compound_index_arrays(weights, last_n_compounded = [0] * self.n_arrays)
        self.reg_L = self.compound_index_arrays(reg_L, first_n_compounded = [solution_dims] * self.n_arrays, last_n_compounded = [solution_dims] * self.n_arrays)

        if reg_lambda is None:
            self.reg_lambda = [None] * self.n_arrays
        elif not isinstance(reg_lambda, list):
            self.reg_lambda = [reg_lambda]

        self.pinv_rtol = pinv_rtol

    def compound_index_arrays(self, arrays, first_n_compounded = None, last_n_compounded = None):
        """
        Ensure that arrays is a list of CompoundIndexArray objects.

        """

        if first_n_compounded is None:
            first_n_compounded = [None] * self.n_arrays
        if last_n_compounded is None:
            last_n_compounded = [None] * self.n_arrays

        arrays_compounded = [None] * self.n_arrays

        if arrays is not None:
            if not isinstance(arrays, list):
                arrays = [arrays]

            for i in range(len(arrays)):
                if arrays[i] is not None:
                    arrays_compounded[i] = CompoundIndexArray(arrays[i], first_n_compounded = first_n_compounded[i], last_n_compounded = last_n_compounded[i])

        return arrays_compounded

    def solve(self, b):
        """
        Solve the least squares problem.

        """

        b_list = self.compound_index_arrays(b, first_n_compounded = [len(self.A[i].full_shapes[0]) for i in range(self.n_arrays)])

        solution = [None] * self.n_arrays

        for i in range(self.n_arrays):
            if b_list[i] is not None:
                solution[i] = np.dot(self.ATWA_plus_R_inv_ATW[i], b_list[i].array)

                if len(b_list[i].shapes) == 2:
                    solution[i] = solution[i].reshape(self.A[i].full_shapes[1] + b_list[i].full_shapes[1])
                else:
                    solution[i] = solution[i].reshape(self.A[i].full_shapes[1])

        return solution

    @property
    def ATW(self):
        """
        Return the matrix ``A^T W``.

        """

        if not hasattr(self, '_ATW'):
            self._ATW = []
            for i in range(self.n_arrays):
                if self.weights[i] is not None:
                    self._ATW.append((self.weights[i].array * self.A[i].array).T)
                else:
                    self._ATW.append(self.A[i].array.T)

        return self._ATW

    @property
    def ATWA(self):
        """
        Return the matrix ``A^T W A``

        """

        if not hasattr(self, '_ATWA'):
            self._ATWA = sum([np.dot(self.ATW[i], self.A[i].array) for i in range(self.n_arrays)])

        return self._ATWA

    @property
    def ATWA_plus_R_inv(self):
        """
        Return the inverse of the matrix ``A^T W A + R``.

        """

        if not hasattr(self, '_ATWA_plus_R_inv'):
            ATWA_plus_R = self.ATWA.copy()
            for i in range(self.n_arrays):
                if self.reg_lambda[i] is not None:
                    ATWA_plus_R += np.dot(self.reg_L[i].array.T, self.reg_L[i].array)

            self._ATWA_plus_R_inv = np.linalg.pinv(ATWA_plus_R, rcond = self.pinv_rtol, hermitian = True)

        return self._ATWA_plus_R_inv
    
    @property
    def ATWA_plus_R_inv_ATW(self):
        """
        Return inverse of the matrix ``A^T W A + R`` multiplied by
        ``A^T W``.

        """

        if not hasattr(self, '_ATWA_plus_R_inv_ATW'):
            self._ATWA_plus_R_inv_ATW = [np.dot(self.ATWA_plus_R_inv, self.ATW[i]) for i in range(self.n_arrays)]

        return self._ATWA_plus_R_inv_ATW