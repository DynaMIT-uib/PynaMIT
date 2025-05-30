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
        algorithm="pinv",
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
            Relative tolerance for pseudoinverse.
        algorithm : str, optional
            Algorithm to use for solving the least squares problem.
            Options are 'pinv', 'cg', or 'solve'. Default is 'pinv'.
        """
        self.solution_dims = solution_dims
        self.algorithm = algorithm

        if isinstance(A, list):
            self.n_As = len(A)
        else:
            self.n_As = 1

        self.A = self.flatten_arrays(
            A, self.n_As, n_trailing_flattened=[solution_dims] * self.n_As
        )
        self.weights = self.flatten_arrays(
            weights, self.n_As, n_trailing_flattened=[0] * self.n_As
        )

        self.n_Ls = self.count_elements(reg_L)

        if any(a is None for a in self.A):
            raise ValueError("At least one forward operator (A) is None.")
        if self.n_Ls != self.count_elements(reg_lambda):
            raise ValueError(
                "Number of regularization operators (reg_L) must match number of regularization parameters (reg_lambda)."
            )

        for i in range(self.n_As):
            if self.weights[i] is not None and np.any(self.weights[i].array < 0):
                # W needs to be positive semidefinite
                # (no negative elements for diagonal matrices)
                raise ValueError(f"Weights for constraint {i} contain negative values.")

        weighted_A_stacked = np.vstack(
            [
                self.A[i].array
                if self.weights[i] is None
                else np.sqrt(self.weights[i].array) * self.A[i].array
                for i in range(self.n_As)
            ]
        )

        weighted_A_stacked_scale = np.median(np.sum(weighted_A_stacked**2, axis=0))

        if self.n_Ls > 0:
            if not isinstance(reg_lambda, list):
                reg_lambda = [reg_lambda]

            self.reg_L = self.flatten_arrays(
                reg_L,
                self.n_Ls,
                n_leading_flattened=[solution_dims] * self.n_Ls,
                n_trailing_flattened=[solution_dims] * self.n_Ls,
            )

            L_scales = [
                np.median(np.sum(self.reg_L[i].array ** 2, axis=0)) for i in range(self.n_Ls)
            ]

            reg_stacked = np.vstack(
                [
                    np.sqrt(reg_lambda[i] / L_scales[i] * weighted_A_stacked_scale)
                    * self.reg_L[i].array
                    for i in range(self.n_Ls)
                ]
            )

            self.stacked_arrays = np.vstack((weighted_A_stacked, reg_stacked))

            if any(l is None for l in self.reg_L):
                raise ValueError("At least one regularization operator (L) is None.")
        else:
            self.stacked_arrays = weighted_A_stacked

        self.U, self.S, self.Vh = np.linalg.svd(
            self.stacked_arrays, full_matrices=False, hermitian=False
        )

        self.pinv_rtol = pinv_rtol

    def count_elements(self, argument):
        """Count elements in the argument.

        Parameters
        ----------
        argument : list of ndarray or ndarray
            Input arrays to count elements in.
        Returns
        -------
        int
            Total number of elements across all arrays.
        """
        if argument is None:
            return 0
        if isinstance(argument, list):
            return len(argument)
        else:
            return 1

    def flatten_arrays(
        self, arrays, n_arrays, n_leading_flattened=None, n_trailing_flattened=None
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
            n_leading_flattened = [None] * n_arrays
        if n_trailing_flattened is None:
            n_trailing_flattened = [None] * n_arrays

        arrays_compounded = [None] * n_arrays

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
            self.n_As,
            n_leading_flattened=[len(self.A[i].full_shapes[0]) for i in range(self.n_As)],
        )

        solution = [None] * self.n_As

        traversed_rows = 0

        for i in range(self.n_As):
            if b_list[i] is not None:
                weighted_b = (
                    np.sqrt(self.weights[i].array) * b_list[i].array
                    if self.weights[i] is not None
                    else b_list[i].array
                )

                solution[i] = self.Vh.T.dot(
                    self.U[traversed_rows : traversed_rows + self.A[i].array.shape[0], :]
                    .T.dot(weighted_b)
                    / self.S.reshape((-1, 1))
                )

                if len(b_list[i].shapes) == 2:
                    solution[i] = solution[i].reshape(
                        self.A[i].full_shapes[1] + b_list[i].full_shapes[1]
                    )
                else:
                    solution[i] = solution[i].reshape(self.A[i].full_shapes[1])

            traversed_rows += self.A[i].array.shape[0]

        return solution
