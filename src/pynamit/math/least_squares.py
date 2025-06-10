"""Least squares module.

This module contains the LeastSquares class for multi-constraint
multi-dimensional least squares.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from pynamit.math.flattened_array import FlattenedArray

class LeastSquares:
    """Class for multi-constraint multi-dimensional least squares.

    Solves problems of the form::

        min_x ( Σᵢ ||Wᵢ(Aᵢx - bᵢ)||² + Σⱼ λⱼ||Lⱼx||² )

    Where each data term `i` has:
    - Aᵢ : Forward operator (can be a tensor)
    - bᵢ : Data vector (can be a tensor)
    - Wᵢ : Weight matrix (optional, assumed diagonal)

    And each regularization term `j` has:
    - λⱼ : Regularization parameter
    - Lⱼ : Regularization operator (can be a tensor)

    Attributes
    ----------
    n_data_terms : int
        Number of data-fitting constraints (A, b, W).
    n_reg_terms : int
        Number of regularization constraints (L, λ).
    A : list of FlattenedArray
        Flattened forward operators.
    weights : list of FlattenedArray
        Flattened weight arrays.
    reg_L : list of FlattenedArray
        Flattened regularization operators.
    reg_lambda : list of float
        Regularization parameters.
    solver : str
        The selected solver type ('direct', 'pinv', or 'iterative').
    tolerance : float
        The tolerance used for the iterative solver or pseudoinverse.
    ATWA_plus_R : ndarray
        The fully-formed system matrix for the normal equations.

    Notes
    -----
    The solver handles multiple, decoupled constraints simultaneously and supports:
    - Tensor-valued variables and operators.
    - Different data and scenario dimensions for each constraint.
    - A choice of direct, pseudoinverse, or iterative solvers.
    """
    def __init__(self, A, solution_dims, weights=None, reg_lambda=None, reg_L=None, solver='iterative', tolerance=1e-10):
        """Initialize the least squares solver.

        Parameters
        ----------
        A : list of ndarray or ndarray
            Forward operator array(s).
        solution_dims : int
            Number of dimensions in the shared solution space `x`.
        weights : list of ndarray or ndarray, optional
            Weight array(s) for each data term.
        reg_lambda : list of float or float, optional
            Regularization parameter(s).
        reg_L : list of ndarray or ndarray, optional
            Regularization operator array(s).
        solver : {'iterative', 'direct', 'pinv'}, optional
            The solver to use.
            - 'iterative': Conjugate Gradient on the normal equations. Fast and memory-efficient.
            - 'direct': `numpy.linalg.solve`. Fast for smaller, well-conditioned problems.
            - 'pinv': `numpy.linalg.pinv`. Slower but robust for ill-conditioned problems.
        tolerance : float, optional
            Tolerance for the iterative solver (`atol`, `rtol`) or the
            pseudoinverse cutoff (`rcond`).
        """
        if solver not in ['iterative', 'direct', 'pinv']:
            raise ValueError("Solver must be one of 'iterative', 'direct', or 'pinv'.")
        self.solver = solver
        self.tolerance = tolerance

        A_list = A if isinstance(A, list) else [A]
        self.n_data_terms = len(A_list)
        self.solution_dims = solution_dims

        self.A = self._flatten_arrays(A_list, self.n_data_terms, n_trailing_flattened=[solution_dims] * self.n_data_terms)
        self.weights = self._flatten_arrays(weights, self.n_data_terms, n_trailing_flattened=[0] * self.n_data_terms)

        reg_L_list = reg_L if isinstance(reg_L, list) else [reg_L] if reg_L is not None else []
        self.n_reg_terms = len(reg_L_list)
        n_leading_L = [l.ndim - solution_dims if l is not None else None for l in reg_L_list]
        self.reg_L = self._flatten_arrays(reg_L_list, self.n_reg_terms, n_leading_flattened=n_leading_L)
        if reg_lambda is None: self.reg_lambda = [0.0] * self.n_reg_terms
        elif not isinstance(reg_lambda, list): self.reg_lambda = [reg_lambda] * self.n_reg_terms
        else: self.reg_lambda = reg_lambda

        if self.n_reg_terms != len(self.reg_lambda): raise ValueError("Number of reg_L must match number of reg_lambda.")

    @staticmethod
    def _flatten_arrays(arrays, n_terms, n_leading_flattened=None, n_trailing_flattened=None):
        if arrays is None: return [None] * n_terms
        arr_list = arrays if isinstance(arrays, list) else [arrays]
        if len(arr_list) == 1 and n_terms > 1: arr_list = arr_list * n_terms
        arrays_compounded = []
        for i in range(n_terms):
            arr = arr_list[i] if i < len(arr_list) else None
            if arr is not None:
                lead_dims = n_leading_flattened[i] if n_leading_flattened and i < len(n_leading_flattened) else None
                trail_dims = n_trailing_flattened[i] if n_trailing_flattened and i < len(n_trailing_flattened) else None
                arrays_compounded.append(FlattenedArray(arr, n_leading_flattened=lead_dims, n_trailing_flattened=trail_dims))
            else: arrays_compounded.append(None)
        return arrays_compounded

    @property
    def ATW(self):
        if not hasattr(self, "_ATW"):
            self._ATW = [((w.array.reshape(-1, 1) * a.array).T if w is not None else a.array.T) for a, w in zip(self.A, self.weights)]
        return self._ATW

    @property
    def ATWA(self):
        if not hasattr(self, "_ATWA"):
            self._ATWA = sum([atw @ a.array for atw, a in zip(self.ATW, self.A)])
        return self._ATWA

    @property
    def ATWA_plus_R(self):
        if not hasattr(self, "_ATWA_plus_R"):
            ATWA_scale = np.median(np.diag(self.ATWA))
            R = np.zeros_like(self.ATWA)
            for i in range(self.n_reg_terms):
                if self.reg_lambda[i] > 0 and self.reg_L[i] is not None:
                    Li = self.reg_L[i].array; LTLi = Li.T @ Li
                    if Li.ndim == 2 and Li.shape[0] == Li.shape[1] and np.allclose(Li, np.diag(np.diag(Li))):
                        LTLi_scale = np.median(np.diag(LTLi))
                    else: LTLi_scale = np.median(np.diag(LTLi))
                    if LTLi_scale > 1e-12: R += (self.reg_lambda[i] / LTLi_scale * ATWA_scale) * LTLi
            self._ATWA_plus_R = self.ATWA + R
        return self._ATWA_plus_R

    def as_normal_eq_operator(self, n_scenarios):
        """Creates a LinearOperator for the normal equations matrix (AᵀWA + R)."""
        n_features = self.A[0].array.shape[1]
        shape = (n_features * n_scenarios, n_features * n_scenarios)
        def _matvec(x_flat):
            x_matrix = x_flat.reshape(n_features, n_scenarios)
            res_matrix = self.ATWA_plus_R @ x_matrix
            return res_matrix.flatten()
        return LinearOperator(shape, matvec=_matvec, rmatvec=_matvec)
    
    def solve(self, b, **kwargs):
        """Solve the least squares system using the method specified at initialization.

        Parameters
        ----------
        b : list of ndarray or ndarray
            Right-hand side array(s) for each data term.
        **kwargs :
            Additional keyword arguments passed to the iterative solver (`cg`),
            e.g., `maxiter`.

        Returns
        -------
        list of ndarray
            A list of solution array component(s), each corresponding to a `b`.
            The final solution that minimizes the sum of all terms is `sum(list)`.
        """
        if self.solver == 'iterative':
            return self._solve_iterative(b, **kwargs)
        elif self.solver == 'direct':
            return self._solve_direct(b)
        elif self.solver == 'pinv':
            return self._solve_pinv(b)

    def _calculate_total_ATWb(self, b):
        """Helper to calculate the total right-hand side for the normal equations."""
        b_list = self._flatten_arrays(b, self.n_data_terms, n_leading_flattened=[len(a.full_shapes[0]) for a in self.A])
        rhs_components = []
        for i in range(self.n_data_terms):
            if b_list[i] is None:
                rhs_components.append(None)
                continue
            ATWb_i = self.ATW[i] @ b_list[i].array
            rhs_components.append(ATWb_i)
        return rhs_components, b_list
    
    def _solve_iterative(self, b, **kwargs):
        """Solves for each solution component using an iterative solver."""
        rhs_components, b_list = self._calculate_total_ATWb(b)
        solutions = []
        n_features = self.A[0].array.shape[1]
        solution_shape = self.A[0].full_shapes[1]
        
        cg_kwargs = {'atol': self.tolerance, 'rtol': self.tolerance}
        cg_kwargs.update(kwargs)

        for i, ATWb_i in enumerate(rhs_components):
            if ATWb_i is None:
                solutions.append(None)
                continue
            n_scenarios_i = ATWb_i.shape[1] if ATWb_i.ndim > 1 else 1
            op = self.as_normal_eq_operator(n_scenarios_i)
            rhs_flat = ATWb_i.flatten()
            sol_flat, info = cg(op, rhs_flat, **cg_kwargs)
            if info != 0: print(f"Warning: CG did not converge for data term {i}. Info: {info}")
            sol_matrix = sol_flat.reshape(n_features, n_scenarios_i)
            rhs_shape = b_list[i].full_shapes[1]
            solutions.append(sol_matrix.reshape(solution_shape + rhs_shape))
        return solutions

    def _solve_direct(self, b):
        """Solves for each solution component using a direct solver."""
        rhs_components, b_list = self._calculate_total_ATWb(b)
        solutions = []
        for i, ATWb_i in enumerate(rhs_components):
            if ATWb_i is None:
                solutions.append(None)
                continue
            sol_matrix = np.linalg.solve(self.ATWA_plus_R, ATWb_i)
            solution_shape = self.A[0].full_shapes[1]
            rhs_shape = b_list[i].full_shapes[1]
            solutions.append(sol_matrix.reshape(solution_shape + rhs_shape))
        return solutions

    def _solve_pinv(self, b):
        """Solves for each solution component using the pseudoinverse."""
        rhs_components, b_list = self._calculate_total_ATWb(b)
        M_pinv = np.linalg.pinv(self.ATWA_plus_R, rcond=self.tolerance)
        solutions = []
        for i, ATWb_i in enumerate(rhs_components):
            if ATWb_i is None:
                solutions.append(None)
                continue
            sol_matrix = M_pinv @ ATWb_i
            solution_shape = self.A[0].full_shapes[1]
            rhs_shape = b_list[i].full_shapes[1]
            solutions.append(sol_matrix.reshape(solution_shape + rhs_shape))
        return solutions