"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required
for simulating ionospheric electrodynamics.
"""

from __future__ import annotations
from functools import cached_property
import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

from pynamit.primitives.coefficient_field import CoefficientField
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver, get_default_least_squares_solver
from pynamit.math.linear_map import LinearMap, as_linear_map, diagonal_linear_map
from pynamit.math.tensor_chain import TensorChain
from pynamit.math.backend import (
    block_after_jax_linalg,
    block_until_ready,
    get_array_module,
    to_jax,
    to_numpy,
    use_jax,
    xp,
)
from pynamit.sphere import Basis, CSBasis
from pynamit.simulation.geometry import Geometry
from pynamit.simulation.operators import StateOperators

logger = logging.getLogger(__name__)


class State:
    """Manages the ionospheric electrodynamic state.

    This class encapsulates the physical state (e.g., potentials,
    currents), handles the construction of all necessary numerical
    operators based on the provided geometry and settings, and
    orchestrates the time evolution of the system. It uses a Geometry
    object to manage the underlying grid and mappings.
    """

    def __init__(
        self,
        basis: Basis,
        mainfield: Any,
        cs_basis: CSBasis,
        settings: Any,
        PFAC_matrix: Optional[np.ndarray] = None,
        radial_continuation_basis: Optional[Basis] = None,
    ) -> None:
        """Initialize the State object."""
        self.basis = basis
        self._init_settings(settings)

        # Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(
            basis,
            cs_basis,
            mainfield,
            settings,
            PFAC_matrix,
            radial_continuation_basis=radial_continuation_basis,
        )

        # Operator for mapping velocity field `u` to E-field
        # (independent of conductance)
        self._u_coeffs_to_E_coeffs: Optional[TensorChain] = None

        # The solver is configured here but remains stateless.
        self.m_imp_solver = LeastSquaresSolver(
            solver=self.solver_type, preconditioner=self.preconditioner
        )

        # Initialize state variables
        self.u: Optional[CoefficientField] = None
        self.Br: Optional[CoefficientField] = None
        self.jr: Optional[CoefficientField] = None
        self.etaP: Optional[CoefficientField] = None
        self.etaH: Optional[CoefficientField] = None

        # Invalidate all caches
        self._invalidate_caches()

    # ----- Initialization Helpers -----

    def _init_settings(self, settings: Any) -> None:
        """Extract and store configuration from the settings object."""
        self.solver_type = getattr(
            settings, "least_squares_solver", get_default_least_squares_solver()
        )
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.static_preconditioner = getattr(settings, "static_preconditioner", False)
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        self.connect_hemispheres = bool(settings.connect_hemispheres)

    @staticmethod
    def _project_scalar_with_basis(basis: Any, coeffs: Any) -> Any:
        """Apply a scalar gauge projection when available."""
        projector = getattr(basis, "project_scalar_mean_free", None)
        if not callable(projector):
            return coeffs
        return projector(coeffs)

    def project_scalar_mean_free(self, coeffs: Any) -> Any:
        """Project scalar-potential coefficients to a fixed gauge."""
        projected = self._project_scalar_with_basis(self.basis, coeffs)
        return projected

    def project_helmholtz_mean_free(self, coeffs: Any) -> Any:
        """Project Helmholtz-potential coefficients to a fixed gauge."""
        projector = getattr(self.basis, "project_helmholtz_mean_free", None)
        if not callable(projector):
            return coeffs
        return projector(coeffs)

    def _create_u_to_E_operator(self) -> TensorChain:
        """Operator mapping wind coefficients to E coefficients."""
        G_helmholtz = xp.asarray(self.geometry.field_transform.G_helmholtz)
        return TensorChain(
            component_tensors=[
                xp.asarray(self.geometry.G_helmholtz_pinv),
                xp.asarray(self.geometry.bu),
                G_helmholtz,
            ],
            einsum_string_dense="cmpg,pqg,qgrs->cmrs",
            einsum_string_matvec="cmpg,pqg,qgrs,rs->cm",
            einsum_string_rmatvec="cm,cmpg,pqg,qgrs->rs",
            output_shape=(2, self.basis.index_length),
            input_shape=G_helmholtz.shape[2:],
        )

    @property
    def u_coeffs_to_E_coeffs(self) -> TensorChain:
        """Operator mapping wind coefficients to E coefficients."""
        if self._u_coeffs_to_E_coeffs is None:
            self._u_coeffs_to_E_coeffs = self._create_u_to_E_operator()
        return self._u_coeffs_to_E_coeffs

    def _invalidate_caches(self) -> None:
        """Invalidate all conductance-dependent cached properties."""
        self._M_total_on_grid: Optional[np.ndarray] = None
        self._m_ind_to_E_coeffs: Optional[TensorChain] = None
        self._m_imp_to_E_coeffs: Optional[TensorChain] = None
        self._Br_to_E_coeffs: Optional[TensorChain] = None
        self._m_ind_to_E_coeffs_dense: Optional[np.ndarray] = None
        self._m_imp_to_E_coeffs_dense: Optional[np.ndarray] = None
        self._Br_to_E_coeffs_dense: Optional[np.ndarray] = None
        self._E_map_constraint_operator: Optional[TensorChain] = None
        self._m_ind_to_E_df_matrix: Optional[np.ndarray] = None
        self._m_ind_to_E_df_operator: Optional[LinearMap] = None
        self._E_noind_to_m_ind_steady_matrix: Optional[np.ndarray] = None
        self._E_noind_to_m_ind_steady_operator: Optional[LinearMap] = None
        self._jr_to_m_imp_matrix: Optional[np.ndarray] = None
        self._E_direct_to_m_imp_matrix: Optional[np.ndarray] = None
        self._direct_E_coeffs_to_total_E_coeffs_operator: Optional[LinearMap] = None
        self._direct_E_coeffs_to_E_df_operator: Optional[LinearMap] = None
        self._m_imp_problem: Optional[LeastSquaresProblem] = None
        self._m_imp_preconditioner: Optional[LinearMap] = None
        self._m_imp_preconditioner_ready = False

    # ----- Cached Physical Properties (dependent on conductance) -----

    @property
    def M_total_on_grid(self) -> np.ndarray:
        """Resistance tensor on the spatial grid."""
        if self._M_total_on_grid is None:
            if self.etaP is None or self.etaH is None:
                raise RuntimeError(
                    "Conductance must be set before accessing conductance-dependent properties."
                )
            eta_stacked = xp.stack(
                [xp.asarray(self.etaP.coeffs), xp.asarray(self.etaH.coeffs)], axis=0
            )
            if self.etaP.basis.coefficients_are_compatible_with(self.basis):
                G_eta = xp.asarray(self.geometry.field_transform.G)
            else:
                G_eta = xp.asarray(self.geometry.field_transform_zero_added.G)
            b_stacked = xp.stack(
                [xp.asarray(self.geometry.bP), xp.asarray(self.geometry.bH)], axis=0
            )
            self._M_total_on_grid = xp.einsum(
                "sijk,kp,sp->ijk", b_stacked, G_eta, eta_stacked, optimize=True
            )
        return self._M_total_on_grid

    def _create_E_coeffs_operator(self, G_X_to_JS: Optional[np.ndarray]) -> Optional[TensorChain]:
        if G_X_to_JS is None:
            return None
        tensors = [
            xp.asarray(self.geometry.G_helmholtz_pinv),
            xp.asarray(self.M_total_on_grid),
            xp.asarray(G_X_to_JS),
        ]
        return TensorChain(
            component_tensors=tensors,
            einsum_string_dense="cmpg,pqg,qgl->cml",
            einsum_string_matvec="cmpg,pqg,qgl,l->cm",
            einsum_string_rmatvec="cm,cmpg,pqg,qgl->l",
            output_shape=(2, self.basis.index_length),
            input_shape=G_X_to_JS.shape[2:],
        )

    @property
    def m_ind_to_E_coeffs(self) -> Optional[TensorChain]:
        """Operator mapping m_ind coefficients to E coefficients."""
        if self._m_ind_to_E_coeffs is None:
            self._m_ind_to_E_coeffs = self._create_E_coeffs_operator(self.geometry.G_m_ind_to_JS)
        return self._m_ind_to_E_coeffs

    @property
    def m_imp_to_E_coeffs(self) -> Optional[TensorChain]:
        """Operator mapping m_imp coefficients to E coefficients."""
        if self._m_imp_to_E_coeffs is None:
            self._m_imp_to_E_coeffs = self._create_E_coeffs_operator(self.geometry.G_m_imp_to_JS)
        return self._m_imp_to_E_coeffs

    @property
    def Br_to_E_coeffs(self) -> Optional[TensorChain]:
        """Operator mapping Br coefficients to E coefficients."""
        if self._Br_to_E_coeffs is None:
            self._Br_to_E_coeffs = self._create_E_coeffs_operator(
                getattr(self.geometry, "G_Br_to_JS", None)
            )
        return self._Br_to_E_coeffs

    def _dense_E_coeffs_operator(self, op: Optional[TensorChain]) -> Optional[np.ndarray]:
        """Return a dense E-coefficient operator."""
        if op is None:
            return None
        return op.materialize_dense().reshape(op.output_shape + op.input_shape)

    @property
    def m_ind_to_E_coeffs_dense(self) -> Optional[np.ndarray]:
        """Dense operator mapping m_ind to E coefficients."""
        if self._m_ind_to_E_coeffs_dense is None:
            self._m_ind_to_E_coeffs_dense = self._dense_E_coeffs_operator(self.m_ind_to_E_coeffs)
        return self._m_ind_to_E_coeffs_dense

    @property
    def m_imp_to_E_coeffs_dense(self) -> Optional[np.ndarray]:
        """Dense operator mapping m_imp to E coefficients."""
        if self._m_imp_to_E_coeffs_dense is None:
            self._m_imp_to_E_coeffs_dense = self._dense_E_coeffs_operator(self.m_imp_to_E_coeffs)
        return self._m_imp_to_E_coeffs_dense

    @property
    def Br_to_E_coeffs_dense(self) -> Optional[np.ndarray]:
        """Dense operator mapping Br coefficients to E coefficients."""
        if self._Br_to_E_coeffs_dense is None:
            self._Br_to_E_coeffs_dense = self._dense_E_coeffs_operator(self.Br_to_E_coeffs)
        return self._Br_to_E_coeffs_dense

    @property
    def E_map_constraint_operator(self) -> Optional[TensorChain]:
        """Operator enforcing E-field mapping at low latitudes."""
        if self._E_map_constraint_operator is None:
            inner_chain = self.m_imp_to_E_coeffs
            outer_tensor = self.geometry.E_coeffs_to_E_apex_ll_diff
            if inner_chain is not None and outer_tensor is not None:
                self._E_map_constraint_operator = TensorChain(
                    component_tensors=[outer_tensor] + inner_chain.component_tensors,
                    einsum_string_dense="ticm,cmpg,pqg,qgl->til",
                    einsum_string_matvec="ticm,cmpg,pqg,qgl,l->ti",
                    einsum_string_rmatvec="ti,ticm,cmpg,pqg,qgl->l",
                    output_shape=(2, int(np.sum(self.geometry.ll_mask))),
                    input_shape=inner_chain.input_shape,
                )
        return self._E_map_constraint_operator

    # ----- Solver Setup and Execution -----
    @property
    def m_imp_problem(self) -> LeastSquaresProblem:
        """The least-squares problem definition for `m_imp`."""
        if self._m_imp_problem is None:
            logger.info("Defining new least-squares problem for m_imp.")
            operators, data_shapes = [], []

            # Radial current (jr) must match imposed field.
            op_jr = (
                as_linear_map(self.geometry.jr_coeffs_to_j_apex)
                @ self.geometry.m_imp_to_jr_operator
            )
            operators.append(op_jr)
            data_shapes.append(self.geometry.jr_coeffs_to_j_apex.shape[:-1])

            # E-field must map at low latitudes.
            if self.connect_hemispheres and self.E_map_constraint_operator is not None:
                op_E = self.E_map_constraint_operator.with_scaling(self.ih_constraint_scaling)
                operators.append(op_E)
                data_shapes.append(op_E.output_shape)

            # Add Tikhonov regularization if lambda is set.
            reg_ops, reg_weights = [], []
            if self.m_imp_regularization_lambda > 0:
                n = self.basis.index_length
                reg_ops.append(diagonal_linear_map(np.ones(n)))
                reg_weights.append(self.m_imp_regularization_lambda)

            self._m_imp_problem = LeastSquaresProblem(
                A=operators,
                solution_shape=self.basis.index_length,
                data_shapes=data_shapes,
                regularization_matrices=reg_ops,
                regularization_weights=reg_weights,
            )
        return self._m_imp_problem

    @property
    def m_imp_preconditioner(self) -> Optional[LinearMap]:
        """Preconditioner for the m_imp least-squares problem."""
        return self._get_m_imp_preconditioner()

    def _get_m_imp_preconditioner(self) -> Optional[LinearMap]:
        """Return a cached preconditioner for the base m_imp system."""
        if self.m_imp_solver.solver not in ("lsmr", "cgls"):
            return None
        if not self._m_imp_preconditioner_ready:
            logger.info("Building new preconditioner for m_imp solver.")
            self._m_imp_preconditioner = self.m_imp_solver.build_preconditioner(
                problem=self.m_imp_problem
            )
            self._m_imp_preconditioner_ready = True
        return self._m_imp_preconditioner

    def _solve_m_imp_response(
        self, problem: LeastSquaresProblem, rhs_entries: List[Optional[np.ndarray]]
    ) -> np.ndarray:
        """Solve one response block with a matching preconditioner."""
        preconditioner = None
        if self.m_imp_solver.solver in ("lsmr", "cgls"):
            preconditioner = self._get_m_imp_preconditioner()
        return self.m_imp_solver.solve(
            problem=problem, rhs=rhs_entries, preconditioner=preconditioner
        )

    def _build_m_imp_response_matrices(self) -> None:
        """Construct dense response matrices for the m_imp solve."""
        logger.info("Building dense m_imp response matrices.")
        n = self.basis.index_length
        problem = self.m_imp_problem

        def solver_response(rhs_entries: List[Optional[np.ndarray]]) -> np.ndarray:
            return self._solve_m_imp_response(problem, rhs_entries)

        solve_response = solver_response

        if self.m_imp_solver.solver == "normal_pinv":
            system_matrix = problem.assemble_dense_system_matrix()
            array_module = get_array_module(system_matrix)
            system_matrix_H = system_matrix.T.conj()
            normal_matrix = system_matrix_H @ system_matrix
            normal_pinv = block_after_jax_linalg(
                array_module.linalg.pinv(
                    normal_matrix,
                    rtol=self.m_imp_solver.tolerance,
                    hermitian=True,
                )
            )

            def cached_pinv_response(rhs_entries: List[Optional[np.ndarray]]) -> np.ndarray:
                rhs_block, rhs_shape, _ = problem.assemble_rhs_block(rhs_entries)
                # Finish the cached response application before the next
                # RHS/operator block may be assembled with NumPy.
                solution_block = block_until_ready(normal_pinv @ (system_matrix_H @ rhs_block))
                return solution_block.reshape(problem.solution_shape + rhs_shape)

            solve_response = cached_pinv_response

        jr_rhs = np.asarray(self.geometry.jr_coeffs_to_j_apex).reshape(
            problem.A[0].output_shape + (-1,)
        )
        rhs_entries = [None] * problem.num_data_terms
        rhs_entries[0] = jr_rhs
        jr_to_m_imp = solve_response(rhs_entries)

        E_direct_to_m_imp = None
        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            E_rhs = -self.geometry.E_coeffs_to_E_apex_ll_diff.reshape(
                problem.A[1].output_shape + (2 * n,)
            )
            E_rhs *= self.ih_constraint_scaling
            rhs_entries = [None] * problem.num_data_terms
            rhs_entries[1] = E_rhs
            E_direct_to_m_imp = solve_response(rhs_entries)
            E_direct_to_m_imp = E_direct_to_m_imp.reshape((n, 2, n))

        self._jr_to_m_imp_matrix = to_jax(jr_to_m_imp) if use_jax() else jr_to_m_imp
        if E_direct_to_m_imp is not None:
            self._E_direct_to_m_imp_matrix = (
                to_jax(E_direct_to_m_imp) if use_jax() else E_direct_to_m_imp
            )
        else:
            self._E_direct_to_m_imp_matrix = None

    def _ensure_m_imp_response_matrices(self) -> None:
        """Build m_imp response matrices when needed."""
        if self._jr_to_m_imp_matrix is None:
            self._build_m_imp_response_matrices()

    def _solve_for_m_imp(
        self, jr_coeffs: Optional[np.ndarray], E_direct_coeffs: np.ndarray
    ) -> np.ndarray:
        """Solve for the imposed potential coefficients `m_imp`."""
        self._ensure_m_imp_response_matrices()
        solution = xp.zeros(self.basis.index_length)

        if jr_coeffs is not None:
            solution += self._jr_to_m_imp_matrix @ xp.asarray(jr_coeffs)

        if self._E_direct_to_m_imp_matrix is not None:
            solution += xp.tensordot(
                self._E_direct_to_m_imp_matrix, xp.asarray(E_direct_coeffs), axes=([1, 2], [0, 1])
            )

        return self.project_scalar_mean_free(solution)

    # ----- State Update -----

    def update(self, input_timeseries: Any, time: float, interpolation: bool = False) -> None:
        """Update the state variables based on the current input."""
        conductance_updated = False
        for key, dataset in input_timeseries.datasets.items():
            updated_input = input_timeseries.get_entry_if_changed(key, time, interpolation)
            if updated_input is None:
                continue

            field_space = input_timeseries.get_storage_spec(key)
            if key == "conductance":
                conductance_updated = True
                self.etaP = CoefficientField(field_space, coeffs=updated_input["etaP"])
                self.etaH = CoefficientField(field_space, coeffs=updated_input["etaH"])
            elif key == "jr":
                self.jr = CoefficientField(field_space, coeffs=updated_input["jr"])
            elif key == "Br":
                if self.RM is None:
                    raise ValueError("Br input can only be set if RM is not None.")
                self.Br = CoefficientField(field_space, coeffs=updated_input["Br"])
            elif key == "u":
                self.u = CoefficientField(field_space, coeffs=updated_input["u"].reshape((2, -1)))

        if conductance_updated:
            logger.info("Conductance updated: invalidating caches and problem definition.")
            preconditioner_to_keep = self._m_imp_preconditioner
            preconditioner_ready_to_keep = self._m_imp_preconditioner_ready
            self._invalidate_caches()
            if self.static_preconditioner and preconditioner_ready_to_keep:
                logger.info("...retaining static preconditioner due to setting.")
                self._m_imp_preconditioner = preconditioner_to_keep
                self._m_imp_preconditioner_ready = True

    # ----- State Calculation -----

    def _apply_operator(self, op: Any, coeffs: Any, output_shape: Tuple[int, ...]) -> np.ndarray:
        if op is None or coeffs is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return get_array_module(op, coeffs).zeros(output_shape)

        chain = op if isinstance(op, TensorChain) else getattr(op, "_tensor_chain", None)
        if isinstance(chain, TensorChain):
            array_module = get_array_module(coeffs, *chain.component_tensors)
            coeffs_arr = array_module.asarray(coeffs)
            result = chain.matvec(coeffs_arr.flatten()).reshape(output_shape)
            return result

        if isinstance(op, LinearMap):
            array_module = get_array_module(coeffs)
            coeffs_arr = array_module.asarray(coeffs)
            result = op.matvec(coeffs_arr.flatten()).reshape(output_shape)
            return result

        if isinstance(op, LinearOperator):
            coeffs_np = to_numpy(coeffs)
            result = as_linear_map(op).matvec(coeffs_np.flatten()).reshape(output_shape)
            return to_jax(result) if use_jax() else result

        array_module = get_array_module(op, coeffs)
        op_arr = array_module.asarray(op)
        coeffs_arr = array_module.asarray(coeffs)
        res = array_module.tensordot(op_arr, coeffs_arr, axes=coeffs_arr.ndim)
        return res.reshape(output_shape) if res.shape != output_shape else res

    def _calculate_total_E_field(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        E_direct_coeffs = self.project_helmholtz_mean_free(E_direct_coeffs)
        m_imp = self._solve_for_m_imp(jr_coeffs, E_direct_coeffs)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs_dense, m_imp, E_shape)
        return self.project_helmholtz_mean_free(E_direct_coeffs + E_imp), m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction effects."""
        E_shape = (2, self.basis.index_length)
        u_coeffs = 0 if self.u is None else xp.asarray(self.u.coeffs)
        E_direct = self._apply_operator(self.u_coeffs_to_E_coeffs, u_coeffs, E_shape)
        if self.Br is not None:
            E_direct += self._apply_operator(
                self.Br_to_E_coeffs_dense, xp.asarray(self.Br.coeffs), E_shape
            )

        jr_coeffs = None if self.jr is None else xp.asarray(self.jr.coeffs)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total E-field coefficients."""
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(
            self.m_ind_to_E_coeffs_dense,
            xp.asarray(self.project_scalar_mean_free(m_ind)),
            E_shape,
        )
        return self._calculate_total_E_field(E_direct_ind, None)

    # ----- Time Evolution -----

    @property
    def m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Dense matrix mapping m_ind to div-free E-field."""
        if self._m_ind_to_E_df_matrix is None:
            self._build_m_ind_to_E_df_matrix()
        return self._m_ind_to_E_df_matrix

    @property
    def m_ind_to_E_df_operator(self) -> LinearMap:
        """Linear map from m_ind to divergence-free E potential."""
        if self._m_ind_to_E_df_operator is None:
            self._m_ind_to_E_df_operator = self._create_m_ind_to_E_df_operator()
        return self._m_ind_to_E_df_operator

    @property
    def E_noind_to_m_ind_steady_matrix(self) -> np.ndarray:
        """Dense matrix mapping no-induction E-field to steady m_ind."""
        if self._E_noind_to_m_ind_steady_matrix is None:
            array_module = get_array_module(self.m_ind_to_E_df_matrix)
            self._E_noind_to_m_ind_steady_matrix = -block_after_jax_linalg(
                array_module.linalg.pinv(self.m_ind_to_E_df_matrix, rtol=1e-15)
            )
        return self._E_noind_to_m_ind_steady_matrix

    @property
    def E_noind_to_m_ind_steady_operator(self) -> LinearMap:
        """Linear map from no-induction E_df to steady m_ind."""
        if self._E_noind_to_m_ind_steady_operator is None:
            self._E_noind_to_m_ind_steady_operator = as_linear_map(
                self.E_noind_to_m_ind_steady_matrix
            )
        return self._E_noind_to_m_ind_steady_operator

    @cached_property
    def operators(self) -> StateOperators:
        """Simulation model operator accessors."""
        return StateOperators(self)

    def _create_m_ind_to_E_df_operator(self) -> LinearMap:
        """Construct matrix-free m_ind -> E_df map."""
        n = self.basis.index_length
        m_ind_to_E = self.m_ind_to_E_coeffs
        E_df_operator = self.geometry.helmholtz_divergence_free_potential_operator
        if m_ind_to_E is None:
            raise RuntimeError("m_ind_to_E_coeffs is not available.")

        def E_total_block(m_ind_block: Any) -> Any:
            array_module = get_array_module(m_ind_block, *m_ind_to_E.component_tensors)
            m_ind_block = array_module.asarray(m_ind_block).reshape(n, -1)
            E_direct = m_ind_to_E.matmat(m_ind_block).reshape(2, n, -1)
            E_total = E_direct

            if self.connect_hemispheres and self.E_map_constraint_operator is not None:
                self._ensure_m_imp_response_matrices()
                if self._E_direct_to_m_imp_matrix is not None:
                    E_direct_to_m_imp = array_module.asarray(self._E_direct_to_m_imp_matrix)
                    m_imp_block = array_module.tensordot(
                        E_direct_to_m_imp,
                        E_direct,
                        axes=([1, 2], [0, 1]),
                    )
                    m_imp_to_E = self.m_imp_to_E_coeffs
                    if m_imp_to_E is None:
                        raise RuntimeError("m_imp_to_E_coeffs is not available.")
                    E_imp = m_imp_to_E.matmat(m_imp_block).reshape(2, n, -1)
                    E_total = E_total + E_imp

            return E_total.reshape(2 * n, -1)

        def matmat(block: Any) -> Any:
            return E_df_operator.matmat(E_total_block(block))

        def rmatmat(block: Any) -> Any:
            matrix = self.m_ind_to_E_df_matrix
            array_module = get_array_module(matrix, block)
            matrix = array_module.asarray(matrix)
            block = array_module.asarray(block).reshape(n, -1)
            return matrix.T.conj() @ block

        def matvec(vec: Any) -> Any:
            array_module = get_array_module(vec, *m_ind_to_E.component_tensors)
            return matmat(array_module.asarray(vec).reshape(n, 1)).reshape(n)

        def rmatvec(vec: Any) -> Any:
            array_module = get_array_module(vec, *m_ind_to_E.component_tensors)
            return rmatmat(array_module.asarray(vec).reshape(n, 1)).reshape(n)

        return LinearMap(
            shape=(n, n),
            dtype=np.result_type(m_ind_to_E.dtype, E_df_operator.dtype),
            _matvec=matvec,
            _rmatvec=rmatvec,
            _matmat=matmat,
            _rmatmat=rmatmat,
        )

    def _build_m_ind_to_E_df_matrix(self) -> None:
        """Construct the dense matrix for the induction operator."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        self._m_ind_to_E_df_matrix = block_until_ready(
            self.m_ind_to_E_df_operator.materialize_dense()
        )
        logger.info("Dense induction operator built.")

    def _calculate_d_m_ind_dt(self, m_ind: np.ndarray, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculate the time derivative of the induced potential.

        This is the right-hand side of the ODE: d(m_ind)/dt = f(m_ind).
        The non-induced E-field is treated as a constant parameter for
        the ODE.
        """
        # Total divergence-free E-field is the sum of induced and
        # non-induced parts.
        E_df_total = self.m_ind_to_E_df_operator.matvec(m_ind)
        E_df_total += self.geometry.helmholtz_divergence_free_potential_operator.matvec(
            E_coeffs_noind
        )

        # Calculate the time derivative using the geometry operator.
        d_m_ind_dt = self.geometry.E_df_to_d_m_ind_dt * E_df_total
        return d_m_ind_dt

    def evolve_m_ind(
        self,
        m_ind: np.ndarray,
        dt: float,
        E_coeffs_noind: np.ndarray,
        steady_state_m_ind: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evolves the induced potential `m_ind` forward in time.

        Uses the integration scheme specified by `self.integrator`.
        Supports 'euler', 'exponential', and any method supported by
        `scipy.solve_ivp`.
        """
        backend_m_ind = xp.asarray(m_ind)
        backend_E_noind = xp.asarray(E_coeffs_noind)

        if self.integrator == "euler":
            d_m_ind_dt = self._calculate_d_m_ind_dt(backend_m_ind, backend_E_noind)
            return self.project_scalar_mean_free(backend_m_ind + dt * d_m_ind_dt)

        elif self.integrator == "exponential":
            # The exponential integrator requires the dense operator
            # matrix.
            op_A = xp.asarray(self.geometry.E_df_to_d_m_ind_dt * self.m_ind_to_E_df_matrix)

            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(backend_E_noind)
            diff = backend_m_ind - xp.asarray(steady_state_m_ind)

            evolved = expm(dt * to_numpy(op_A)) @ to_numpy(diff) + to_numpy(steady_state_m_ind)
            return self.project_scalar_mean_free(evolved)

        else:
            # Fallback to scipy.solve_ivp for other integrators
            logger.debug(f"Using scipy.solve_ivp with method='{self.integrator}'.")

            m_ind_to_E_df_matrix = to_numpy(self.m_ind_to_E_df_matrix)
            E_noind_df = to_numpy(
                self.geometry.helmholtz_divergence_free_potential_operator.matvec(
                    backend_E_noind
                )
            )
            rhs_scale = float(self.geometry.E_df_to_d_m_ind_dt)

            def rhs(t, y):
                del t
                return rhs_scale * (m_ind_to_E_df_matrix @ y + E_noind_df)

            # Integrate from t=0 to t=dt. The ODE is autonomous
            # (not t-dependent).
            sol = solve_ivp(
                fun=rhs,
                t_span=(0, dt),
                y0=to_numpy(backend_m_ind),
                method=self.integrator,
                t_eval=[dt],  # We only need the final state
                dense_output=False,
            )

            if not sol.success:
                logger.warning(
                    f"solve_ivp integrator '{self.integrator}' failed with "
                    f"status {sol.status}: {sol.message}"
                )

            # The result shape is (n_vars, n_times),
            # so we take the last time point.
            result = sol.y[:, -1]
            result = self.project_scalar_mean_free(result)
            return to_jax(result) if use_jax() else result

    def steady_state_m_ind(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculate the steady-state induced potential."""
        E_noind_df = self.geometry.helmholtz_divergence_free_potential_operator.matvec(
            E_coeffs_noind
        )
        steady = self.E_noind_to_m_ind_steady_operator.matvec(E_noind_df)
        return self.project_scalar_mean_free(steady)
