"""Least squares module.

This module contains the LeastSquares class for multi-constraint
multi-dimensional least squares.
"""

import numpy as np
from pynamit.math.flattened_array import FlattenedArray


class LeastSquares:
    """Class for multi-constraint multi-dimensional least squares.

    Solves problems of the form::

        min_x Σᵢ (||Wᵢ(Aᵢx - bᵢ)||² + λᵢ||Lᵢx||²)

    Where each constraint i has:
    - Aᵢ : Forward operator
    - bᵢ : Data vector
    - Wᵢ : Weight matrix (optional)
    - λᵢ : Regularization parameter (optional)
    - Lᵢ : Regularization operator (optional)

    Attributes
    ----------
    n_constraints : int
        Number of constraints.
    A : list of FlattenedArray
        Flattened forward operators.
    weights : list of FlattenedArray
        Flattened weight arrays.
    reg_L : list of FlattenedArray
        Flattened regularization operators.
    reg_lambda : list of float
        Regularization parameters.
    ATW : list of ndarray
        ``A^T W`` products for each constraint.
    ATWA : ndarray
        Combined ``A^T W A matrix``.
    ATWA_plus_R_pinv : ndarray
        Inverse of ``A^T W A + λL^T L``.
    ATWA_plus_R_pinv_ATW : list of ndarray
        Solution operators for each constraint.

    Notes
    -----
    The solver handles multiple constraints simultaneously and supports:
    - Array-valued variables and operators
    - Per-constraint weights and regularization
    - Efficient handling of high-dimensional arrays
    """

    def __init__(
        self,
        A,
        solution_dims,
        weights=None,
        reg_lambda=None,
        reg_L=None,
        pinv_rtol=1e-15,
    ):
        """Initialize the least squares solver.

        Parameters
        ----------
        A : list of ndarray or ndarray
            Forward operator array(s). Single array or list of arrays.
        solution_dims : int
            Number of dimensions in solution space.
        weights : list of ndarray or ndarray, optional
            Weight array(s) for each constraint.
        reg_lambda : list of float or float, optional
            Regularization parameter(s) for each constraint.
        reg_L : list of ndarray or ndarray, optional
            Regularization operator array(s).
        pinv_rtol : float, optional
            Relative tolerance for pseudoinverse, by default 1e-15.
        """
        self.solution_dims = solution_dims

        if isinstance(A, list):
            self.n_constraints = len(A)
        else:
            self.n_constraints = 1

        self.A = self.flatten_arrays(
            A, n_trailing_flattened=[solution_dims] * self.n_constraints
        )
        self.weights = self.flatten_arrays(
            weights, n_trailing_flattened=[0] * self.n_constraints
        )
        self.reg_L = self.flatten_arrays(
            reg_L,
            n_leading_flattened=[solution_dims] * self.n_constraints,
            n_trailing_flattened=[solution_dims] * self.n_constraints,
        )

        if reg_lambda is None:
            self.reg_lambda = [None] * self.n_constraints
        elif not isinstance(reg_lambda, list):
            self.reg_lambda = [reg_lambda]

        self.pinv_rtol = pinv_rtol

    def flatten_arrays(
        self, arrays, n_leading_flattened=None, n_trailing_flattened=None
    ):
        """Convert arrays to flattened form.

        Parameters
        ----------
        arrays : list of ndarray or ndarray
            Input arrays to flatten.
        n_leading_flattened : list of int, optional
            Number of leading dimensions to flatten for each array.
        n_trailing_flattened : list of int, optional
            Number of trailing dimensions to flatten for each array.

        Returns
        -------
        list of FlattenedArray
            Containers of input arrays containing flattened indices.

        Notes
        -----
        Creates FlattenedArray objects to efficiently handle
        multidimensional array operations via flattened indices, while
        also preserving the original indices.
        """
        if n_leading_flattened is None:
            n_leading_flattened = [None] * self.n_constraints
        if n_trailing_flattened is None:
            n_trailing_flattened = [None] * self.n_constraints

        arrays_compounded = [None] * self.n_constraints

        if arrays is not None:
            if not isinstance(arrays, list):
                arrays = [arrays]

            for i in range(len(arrays)):
                if arrays[i] is not None:
                    arrays_compounded[i] = FlattenedArray(
                        arrays[i],
                        n_leading_flattened=n_leading_flattened[i],
                        n_trailing_flattened=n_trailing_flattened[i],
                    )

        return arrays_compounded

    def solve(self, b):
        """Solve the least squares system.

        Parameters
        ----------
        b : list of ndarray or ndarray
            Right-hand side array(s) for each constraint.

        Returns
        -------
        list of ndarray
            Solution array(s) with original dimensionality restored.
            Returns list also for single solution.

        Notes
        -----
        For each constraint ``i``, solves::

            (Aᵢ^T Wᵢ Aᵢ + λᵢLᵢ^T Lᵢ)x = Aᵢ^T Wᵢ bᵢ

        The complete solution minimizes the sum of all constraint terms.
        """
        b_list = self.flatten_arrays(
            b,
            n_leading_flattened=[
                len(self.A[i].full_shapes[0]) for i in range(self.n_constraints)
            ],
        )

        solution = [None] * self.n_constraints

        for i in range(self.n_constraints):
            if b_list[i] is not None:
                solution[i] = np.dot(self.ATWA_plus_R_pinv_ATW[i], b_list[i].array)

                if len(b_list[i].shapes) == 2:
                    solution[i] = solution[i].reshape(
                        self.A[i].full_shapes[1] + b_list[i].full_shapes[1]
                    )
                else:
                    solution[i] = solution[i].reshape(self.A[i].full_shapes[1])

        return solution

    @property
    def ATW(self):
        """Compute ``A^T W`` terms for all constraints.

        Returns
        -------
        list of ndarray
            List of ``A^T W`` arrays for each constraint.
        """
        if not hasattr(self, "_ATW"):
            self._ATW = []
            for i in range(self.n_constraints):
                if self.weights[i] is not None:
                    self._ATW.append((self.weights[i].array * self.A[i].array).T)
                else:
                    self._ATW.append(self.A[i].array.T)

        return self._ATW

    @property
    def ATWA(self):
        """Compute ``A^T W A`` term.

        Returns
        -------
        ndarray
            Combined ``A^T W A`` matrix for all constraints.
        """
        if not hasattr(self, "_ATWA"):
            self._ATWA = sum(
                [
                    np.dot(self.ATW[i], self.A[i].array)
                    for i in range(self.n_constraints)
                ]
            )

        return self._ATWA

    @property
    def ATWA_plus_R_pinv(self):
        """Compute pseudoinverse of ``A^T W A + λL^T L``.

        Returns
        -------
        ndarray
            Pseudoinverse of ``A^T W A + λL^T L``.

        Notes
        -----
        Uses pseudoinverse with specified tolerance.
        """
        if not hasattr(self, "_ATWA_plus_R_pinv"):
            ATWA_plus_R = self.ATWA.copy()
            for i in range(self.n_constraints):
                if self.reg_lambda[i] is not None:
                    ATWA_plus_R += np.dot(self.reg_L[i].array.T, self.reg_L[i].array)

            self._ATWA_plus_R_pinv = np.linalg.pinv(
                ATWA_plus_R, rcond=self.pinv_rtol, hermitian=True
            )

        return self._ATWA_plus_R_pinv

    @property
    def ATWA_plus_R_pinv_ATW(self):
        """Compute ``(A^T W A + λL^T L)^† A^T W``.

        Returns
        -------
        list of ndarray
            List of ``(A^T W A + λL^T L)^† A^T W`` for each constraint.
        """
        if not hasattr(self, "_ATWA_plus_R_pinv_ATW"):
            self._ATWA_plus_R_pinv_ATW = [
                np.dot(self.ATWA_plus_R_pinv, self.ATW[i])
                for i in range(self.n_constraints)
            ]

        return self._ATWA_plus_R_pinv_ATW
