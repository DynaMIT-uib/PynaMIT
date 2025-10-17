"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required for
simulating ionospheric electrodynamics.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.tensor_chain import TensorChain
from pynamit.simulation.geometry import Geometry
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.utils import to_numpy, to_jax, use_jax, xp

logger = logging.getLogger(__name__)


class State:
    """Manages the ionospheric electrodynamic state and associated operators.

    This class encapsulates the physical state (e.g., potentials, currents),
    handles the construction of all necessary numerical operators based on the
    provided geometry and settings, and orchestrates the time evolution of
    the system. It uses a Geometry object to manage the underlying grid
    and mappings.
    """

    def __init__(
        self,
        basis: SHBasis,
        mainfield: Any,
        cs_basis: SHBasis,
        settings: Any,
        PFAC_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the State object."""
        self.basis = basis
        self._init_settings(settings)

        # Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(basis, cs_basis, mainfield, settings, PFAC_matrix)

        # Operator for mapping velocity field `u` to E-field, independent of conductance
        self.u_coeffs_to_E_coeffs = self._create_u_to_E_operator()

        # The solver is configured here but remains stateless.
        self.m_imp_solver = LeastSquaresSolver(
            solver=self.solver_type, preconditioner=self.preconditioner
        )

        # Initialize state variables
        self.u: Optional[FieldExpansion] = None
        self.Br: Optional[FieldExpansion] = None
        self.jr: Optional[FieldExpansion] = None
        self.etaP: Optional[FieldExpansion] = None
        self.etaH: Optional[FieldExpansion] = None

        # Invalidate all caches
        self._invalidate_caches()

    # ----- Initialization Helpers -----

    def _init_settings(self, settings: Any) -> None:
        """Extract and store configuration from the settings object."""
        self.solver_type = getattr(settings, "least_squares_solver", "lsmr")
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.static_preconditioner = getattr(settings, "static_preconditioner", False)
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        self.connect_hemispheres = bool(settings.connect_hemispheres)

    def _create_u_to_E_operator(self) -> np.ndarray:
        """Create the operator mapping velocity coefficients to E-field coefficients."""
        bu = xp.asarray(self.geometry.bu)
        G_helmholtz = xp.asarray(self.geometry.basis_evaluator.G_helmholtz)
        G_u_to_uxB_grid = xp.einsum(
            "ijk,jklm->iklm",
            bu,
            G_helmholtz,
            optimize=True,
        )
        G_helmholtz_pinv = xp.asarray(self.geometry.G_helmholtz_pinv)
        return xp.tensordot(G_helmholtz_pinv, G_u_to_uxB_grid, axes=2)

    def _invalidate_caches(self) -> None:
        """Invalidate all cached properties that depend on conductance."""
        self._M_total_on_grid: Optional[np.ndarray] = None
        self._m_ind_to_E_coeffs: Optional[TensorChain] = None
        self._m_imp_to_E_coeffs: Optional[TensorChain] = None
        self._Br_to_E_coeffs: Optional[TensorChain] = None
        self._E_map_constraint_operator: Optional[TensorChain] = None
        self._m_ind_to_E_df_matrix: Optional[np.ndarray] = None
        self._m_imp_problem: Optional[LeastSquaresProblem] = None
        self._m_imp_preconditioner: Optional[LinearOperator] = None

    # ----- Cached Physical Properties (dependent on conductance) -----

    @property
    def M_total_on_grid(self) -> np.ndarray:
        if self._M_total_on_grid is None:
            if self.etaP is None or self.etaH is None:
                raise RuntimeError(
                    "Conductance must be set before accessing conductance-dependent properties."
                )
            eta_stacked = xp.stack(
                [xp.asarray(self.etaP.coeffs), xp.asarray(self.etaH.coeffs)], axis=0
            )
            G_eta = xp.asarray(self.geometry.basis_evaluator_zero_added.G)
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
        if self._m_ind_to_E_coeffs is None:
            self._m_ind_to_E_coeffs = self._create_E_coeffs_operator(self.geometry.G_m_ind_to_JS)
        return self._m_ind_to_E_coeffs

    @property
    def m_imp_to_E_coeffs(self) -> Optional[TensorChain]:
        if self._m_imp_to_E_coeffs is None:
            self._m_imp_to_E_coeffs = self._create_E_coeffs_operator(self.geometry.G_m_imp_to_JS)
        return self._m_imp_to_E_coeffs

    @property
    def Br_to_E_coeffs(self) -> Optional[TensorChain]:
        if self._Br_to_E_coeffs is None:
            self._Br_to_E_coeffs = self._create_E_coeffs_operator(
                getattr(self.geometry, "G_Br_to_JS", None)
            )
        return self._Br_to_E_coeffs

    @property
    def E_map_constraint_operator(self) -> Optional[TensorChain]:
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
        """The least-squares problem definition for the imposed potential `m_imp`."""
        if self._m_imp_problem is None:
            logger.info("Defining new least-squares problem for m_imp.")
            operators, data_shapes = [], []

            # Constraint 1: Radial current (jr) must match imposed field.
            op_jr = self.geometry.jr_coeffs_to_j_apex * self.geometry.m_imp_to_jr.reshape((1, -1))
            operators.append(op_jr)
            data_shapes.append(op_jr.shape[:-1])

            # Constraint 2: E-fields in conjugate hemispheres must match at low latitudes.
            if self.connect_hemispheres and self.E_map_constraint_operator is not None:
                op_E = self.E_map_constraint_operator.with_scaling(self.ih_constraint_scaling)
                operators.append(op_E)
                data_shapes.append(op_E.output_shape)

            # Regularization: Add Tikhonov regularization if lambda is set.
            reg_ops, reg_weights = [], []
            if self.m_imp_regularization_lambda > 0:
                n = self.basis.index_length
                identity_op = LinearOperator((n, n), matvec=lambda x: x)
                reg_ops.append(identity_op)
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
    def m_imp_preconditioner(self) -> Optional[LinearOperator]:
        """The pre-computed preconditioner for the m_imp least-squares problem."""
        if self._m_imp_preconditioner is None:
            logger.info("Building new preconditioner for m_imp solver.")
            self._m_imp_preconditioner = self.m_imp_solver.build_preconditioner(
                problem=self.m_imp_problem, num_scenarios=1
            )
        return self._m_imp_preconditioner

    def _solve_for_m_imp(
        self, jr_coeffs: Optional[np.ndarray], E_direct_coeffs: np.ndarray
    ) -> np.ndarray:
        """Solves for the imposed potential coefficients `m_imp`."""
        problem = self.m_imp_problem
        preconditioner = self.m_imp_preconditioner

        rhs_list = []
        b_jr = (
            np.dot(self.geometry.jr_coeffs_to_j_apex, to_numpy(jr_coeffs))
            if jr_coeffs is not None
            else None
        )
        rhs_list.append(b_jr)

        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            E_map_op = self.geometry.E_coeffs_to_E_apex_ll_diff
            b_E = -np.einsum("cikl,kl->ci", E_map_op, to_numpy(E_direct_coeffs)).flatten()
            rhs_list.append(b_E * self.ih_constraint_scaling)

        solution = self.m_imp_solver.solve(
            problem=problem, rhs=rhs_list, preconditioner=preconditioner
        )
        if solution is None:
            solution = np.zeros(self.basis.index_length)
        return to_jax(solution) if use_jax() else solution

    # ----- State Update -----

    def update(self, input_timeseries: Any, time: float, interpolation: bool = False) -> None:
        conductance_updated = False
        for key, dataset in input_timeseries.datasets.items():
            updated_input = input_timeseries.get_entry_if_changed(key, time, interpolation)
            if updated_input is None:
                continue

            storage_base = input_timeseries.storage_bases.get(key)
            if key == "conductance":
                conductance_updated = True
                self.etaP = FieldExpansion(storage_base, coeffs=updated_input["etaP"])
                self.etaH = FieldExpansion(storage_base, coeffs=updated_input["etaH"])
            elif key == "jr":
                self.jr = FieldExpansion(storage_base, coeffs=updated_input["jr"])
            elif key == "Br":
                if self.RM is None:
                    raise ValueError("Br input can only be set if RM is not None.")
                self.Br = FieldExpansion(storage_base, coeffs=updated_input["Br"])
            elif key == "u":
                self.u = FieldExpansion(storage_base, coeffs=updated_input["u"].reshape((2, -1)))

        if conductance_updated:
            logger.info("Conductance updated: invalidating caches and problem definition.")
            preconditioner_to_keep = (
                self._m_imp_preconditioner if self.static_preconditioner else None
            )
            self._invalidate_caches()
            if preconditioner_to_keep is not None:
                logger.info("...retaining static preconditioner due to setting.")
                self._m_imp_preconditioner = preconditioner_to_keep

    # ----- State Calculation -----

    def _apply_operator(self, op: Any, coeffs: Any, output_shape: Tuple[int, ...]) -> np.ndarray:
        if op is None or coeffs is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return xp.zeros(output_shape)

        coeffs_np = to_numpy(coeffs)
        if isinstance(op, TensorChain):
            linop = op.as_linear_operator()
            res_np = linop.matvec(coeffs_np.flatten()).reshape(output_shape)
            return to_jax(res_np) if use_jax() else res_np

        if isinstance(op, LinearOperator):
            res_np = op.matvec(coeffs_np.flatten()).reshape(output_shape)
            return to_jax(res_np) if use_jax() else res_np

        module = xp
        op_arr = module.asarray(op)
        coeffs_arr = module.asarray(coeffs)
        res = module.tensordot(op_arr, coeffs_arr, axes=coeffs_arr.ndim)
        return res.reshape(output_shape) if res.shape != output_shape else res

    def _calculate_total_E_field(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        m_imp = self._solve_for_m_imp(jr_coeffs, E_direct_coeffs)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct_coeffs + E_imp, m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        u_coeffs = 0 if self.u is None else xp.asarray(self.u.coeffs)
        E_direct = self._apply_operator(self.u_coeffs_to_E_coeffs, u_coeffs, E_shape)
        if self.Br is not None:
            E_direct += self._apply_operator(self.Br_to_E_coeffs, xp.asarray(self.Br.coeffs), E_shape)

        jr_coeffs = None if self.jr is None else xp.asarray(self.jr.coeffs)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, xp.asarray(m_ind), E_shape)
        return self._calculate_total_E_field(E_direct_ind, None)

    # ----- Time Evolution -----

    @property
    def m_ind_to_E_df_matrix(self) -> np.ndarray:
        """The dense matrix mapping induced potential to divergence-free E-field."""
        if self._m_ind_to_E_df_matrix is None:
            self._build_m_ind_to_E_df_matrix()
        return self._m_ind_to_E_df_matrix

    def _build_m_ind_to_E_df_matrix(self) -> None:
        """Constructs the dense matrix for the induction operator using matvec."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self.basis.index_length
        identity = np.eye(n)

        # Apply forward operator to each column of identity matrix
        columns = []
        for vec in identity:
            backend_vec = to_jax(vec) if use_jax() else vec
            E_ind_coeffs, _ = self.calculate_ind_coeffs(backend_vec)
            columns.append(E_ind_coeffs[1])

        self._m_ind_to_E_df_matrix = xp.stack(columns, axis=1)
        logger.info("Dense induction operator built.")

    def _calculate_d_m_ind_dt(self, m_ind: np.ndarray, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculates the time derivative of the induced potential.

        This is the right-hand side of the ODE: d(m_ind)/dt = f(m_ind).
        The non-induced E-field is treated as a constant parameter for the ODE.
        """
        # Calculate the E-field contribution from the current induced potential.
        E_ind_coeffs, _ = self.calculate_ind_coeffs(m_ind)
        E_df_ind = E_ind_coeffs[1]

        # Total divergence-free E-field is the sum of induced and non-induced parts.
        E_df_total = E_df_ind + E_coeffs_noind[1]

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
        """Evolves the induced potential `m_ind` forward in time by `dt`.

        Uses the integration scheme specified by `self.integrator`. Supports 'euler',
        'exponential', and any method supported by `scipy.solve_ivp`.
        """
        backend_m_ind = xp.asarray(m_ind)
        backend_E_noind = xp.asarray(E_coeffs_noind)

        if self.integrator == "euler":
            d_m_ind_dt = self._calculate_d_m_ind_dt(backend_m_ind, backend_E_noind)
            return backend_m_ind + dt * d_m_ind_dt

        elif self.integrator == "exponential":
            # The exponential integrator requires the dense operator matrix.
            op_A = xp.asarray(self.geometry.E_df_to_d_m_ind_dt * self.m_ind_to_E_df_matrix)

            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(backend_E_noind)
            diff = backend_m_ind - xp.asarray(steady_state_m_ind)

            if use_jax():
                from jax.scipy.linalg import expm as jax_expm

                evolved = jax_expm(dt * op_A) @ diff + xp.asarray(steady_state_m_ind)
                return evolved

            evolved = expm(dt * to_numpy(op_A)) @ diff + xp.asarray(steady_state_m_ind)
            return evolved

        else:
            # Fallback to scipy.solve_ivp for other specified integrators
            logger.debug(f"Using scipy.solve_ivp with method='{self.integrator}'.")

            # Define the right-hand side of the ODE for the solver.
            # The non-induced part is constant, so it's captured from the outer scope.
            def rhs(t, y):
                y_backend = to_jax(y) if use_jax() else y
                dy = self._calculate_d_m_ind_dt(y_backend, backend_E_noind)
                return to_numpy(dy)

            # Integrate from t=0 to t=dt. The ODE is autonomous (not t-dependent).
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

            # The result shape is (n_vars, n_times), so we take the last time point.
            result = sol.y[:, -1]
            return to_jax(result) if use_jax() else result

    def steady_state_m_ind(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculates the steady-state induced potential."""
        # This operation requires solving a linear system, which is most
        # robustly done with the dense matrix form of the operator.
        op_A = xp.asarray(self.m_ind_to_E_df_matrix)
        vec_b = -xp.asarray(E_coeffs_noind[1])
        return xp.linalg.solve(op_A, vec_b)
