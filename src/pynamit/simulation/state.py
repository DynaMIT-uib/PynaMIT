"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required
for simulating ionospheric electrodynamics.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Any, List

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.tensor_chain import TensorChain
from pynamit.math.linear_map import as_linear_map
from pynamit.simulation.geometry import Geometry
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.utils import asarray, use_jax, xp

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

        # Operator for mapping velocity field `u` to E-field
        # (independent of conductance)
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
        self.solver_type = getattr(settings, "least_squares_solver", "cg")
        self.preconditioner = getattr(settings, "least_squares_preconditioner", "pinv")
        self.static_preconditioner = getattr(settings, "static_preconditioner", False)
        self.integrator = settings.integrator
        self.m_imp_regularization_lambda = getattr(settings, "m_imp_regularization_lambda", 0.0)
        self.RI = settings.RI
        self.RM = None if settings.RM == 0 else settings.RM
        self.ih_constraint_scaling = settings.ih_constraint_scaling
        self.connect_hemispheres = bool(settings.connect_hemispheres)

    def _create_u_to_E_operator(self) -> np.ndarray:
        """Operator mapping wind coefficients to E coefficients."""
        bu = asarray(self.geometry.bu)
        G_helmholtz = asarray(self.geometry.basis_evaluator.G_helmholtz)
        G_u_to_uxB_grid = xp.einsum("ijk,jklm->iklm", bu, G_helmholtz, optimize=True)
        G_helmholtz_pinv = asarray(self.geometry.G_helmholtz_pinv)
        return xp.tensordot(G_helmholtz_pinv, G_u_to_uxB_grid, axes=2)

    def _invalidate_caches(self) -> None:
        """Invalidate all conductance-dependent cached properties."""
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
        """Resistance tensor on the spatial grid."""
        if self._M_total_on_grid is None:
            if self.etaP is None or self.etaH is None:
                raise RuntimeError(
                    "Conductance must be set before accessing conductance-dependent properties."
                )
            eta_stacked = xp.stack([asarray(self.etaP.coeffs), asarray(self.etaH.coeffs)], axis=0)
            G_eta = asarray(self.geometry.basis_evaluator_zero_added.G)
            b_stacked = xp.stack(
                [asarray(self.geometry.bP), asarray(self.geometry.bH)], axis=0
            )
            self._M_total_on_grid = xp.einsum(
                "sijk,kp,sp->ijk", b_stacked, G_eta, eta_stacked, optimize=True
            )
        return self._M_total_on_grid

    def _create_E_coeffs_operator(self, G_X_to_JS: Optional[np.ndarray]) -> Optional[TensorChain]:
        if G_X_to_JS is None:
            return None
        tensors = [
            asarray(self.geometry.G_helmholtz_pinv),
            asarray(self.M_total_on_grid),
            asarray(G_X_to_JS),
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
            op_jr = self.geometry.jr_coeffs_to_j_apex * self.geometry.m_imp_to_jr.reshape((1, -1))
            operators.append(op_jr)
            data_shapes.append(op_jr.shape[:-1])

            # E-field must map at low latitudes.
            if self.connect_hemispheres and self.E_map_constraint_operator is not None:
                op_E = self.E_map_constraint_operator.with_scaling(self.ih_constraint_scaling)
                operators.append(op_E)
                data_shapes.append(op_E.output_shape)

            # Add Tikhonov regularizationif lambda is set.
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
        """Preconditioner for the m_imp least-squares problem."""
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

        rhs_entries: List[Optional[Any]] = [None] * problem.num_data_terms
        if jr_coeffs is not None:
            jr_matrix = asarray(self.geometry.jr_coeffs_to_j_apex)
            rhs_entries[0] = jr_matrix @ asarray(jr_coeffs).reshape(-1)

        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            E_map_op = asarray(self.geometry.E_coeffs_to_E_apex_ll_diff)
            b_E = -xp.einsum("cikl,kl->ci", E_map_op, asarray(E_direct_coeffs))
            rhs_entries[1] = self.ih_constraint_scaling * xp.reshape(b_E, (-1,))

        solver = self.m_imp_solver
        solution = solver.solve(problem=problem, rhs=rhs_entries, preconditioner=preconditioner)
        if solution is None:
            solution = np.zeros(self.basis.index_length)
        return asarray(solution)

    # ----- State Update -----

    def update(self, input_timeseries: Any, time: float, interpolation: bool = False) -> None:
        """Update the state variables based on the current input."""
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

        linear_map = as_linear_map(op)
        flat_in = linear_map.shape[1]
        backend_coeffs = asarray(coeffs).reshape(flat_in)
        res_flat = linear_map.matvec(backend_coeffs)
        res_backend = asarray(res_flat).reshape(output_shape)
        return res_backend

    def _calculate_total_E_field(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        m_imp = self._solve_for_m_imp(jr_coeffs, E_direct_coeffs)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct_coeffs + E_imp, m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction effects."""
        E_shape = (2, self.basis.index_length)
        u_coeffs = 0 if self.u is None else asarray(self.u.coeffs)
        E_direct = self._apply_operator(self.u_coeffs_to_E_coeffs, u_coeffs, E_shape)
        if self.Br is not None:
            E_direct += self._apply_operator(
                self.Br_to_E_coeffs, asarray(self.Br.coeffs), E_shape
            )

        jr_coeffs = None if self.jr is None else asarray(self.jr.coeffs)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total E-field coefficients."""
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, asarray(m_ind), E_shape)
        return self._calculate_total_E_field(E_direct_ind, None)

    # ----- Time Evolution -----

    @property
    def m_ind_to_E_df_matrix(self) -> np.ndarray:
        """Dense matrix mapping m_ind to div-free E-field."""
        if self._m_ind_to_E_df_matrix is None:
            self._build_m_ind_to_E_df_matrix()
        return self._m_ind_to_E_df_matrix

    def _build_m_ind_to_E_df_matrix(self) -> None:
        """Construct the dense matrix for the induction operator."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        n = self.basis.index_length
        if self.m_ind_to_E_coeffs is None:
            self._m_ind_to_E_df_matrix = xp.zeros((n, n))
            logger.info("Dense induction operator built (degenerate: no mapping available).")
            return

        # Direct contribution from induced potential (without imposed solver feedback)
        E_direct_dense = asarray(self.m_ind_to_E_coeffs.to_dense()).reshape(2, n, n)
        E_direct_dense_np = np.asarray(E_direct_dense)

        problem = self.m_imp_problem
        rhs_entries = [None] * problem.num_data_terms if problem.num_data_terms > 0 else []

        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            E_map_op = asarray(self.geometry.E_coeffs_to_E_apex_ll_diff)
            # Compute RHS blocks for all basis vectors simultaneously
            b_E_block = -xp.einsum("cikl,klj->cij", E_map_op, E_direct_dense)
            if len(rhs_entries) > 1:
                rhs_entries[1] = self.ih_constraint_scaling * b_E_block

        rhs_block, _, num_scenarios = problem.assemble_rhs_block(rhs_entries)
        if rhs_block is None:
            op_rows = problem.get_system_operator().shape[0]
            rhs_block = np.zeros((op_rows, n), dtype=E_direct_dense_np.dtype)
            num_scenarios = n
        rhs_block = np.asarray(rhs_block)

        # Solve in batch using cached SVD decomposition
        u, s, vt = problem.svd
        if s.size == 0:
            m_imp_block = np.zeros((problem.solution_size, num_scenarios), dtype=rhs_block.dtype)
        else:
            tol = getattr(self.m_imp_solver, "tolerance", 0.0)
            cutoff = tol * s[0] if tol > 0 else 0.0
            s_inv = np.zeros_like(s)
            mask = s > cutoff
            s_inv[mask] = 1.0 / s[mask]
            tmp = u.T.conj() @ rhs_block
            tmp = s_inv[:, None] * tmp
            m_imp_block = vt.T.conj() @ tmp

        if num_scenarios != n:
            raise RuntimeError(
                f"Expected {n} scenarios when building induction operator, got {num_scenarios}."
            )

        # Map imposed potential response back to E-field coefficients
        if self.m_imp_to_E_coeffs is not None:
            m_imp_flat = xp.asarray(m_imp_block)
            E_imp_flat = self.m_imp_to_E_coeffs.matmat(m_imp_flat)
            E_imp_block = xp.asarray(E_imp_flat).reshape(2, n, n)
        else:
            E_imp_block = xp.zeros_like(E_direct_dense)

        total_E = E_direct_dense_np + np.asarray(E_imp_block)
        self._m_ind_to_E_df_matrix = asarray(total_E[1])
        logger.info("Dense induction operator built.")

    def _calculate_d_m_ind_dt(self, m_ind: np.ndarray, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculate the time derivative of the induced potential.

        This is the right-hand side of the ODE: d(m_ind)/dt = f(m_ind).
        The non-induced E-field is treated as a constant parameter for
        the ODE.
        """
        # Calculate the E-field contribution from the current induced
        # potential.
        E_ind_coeffs, _ = self.calculate_ind_coeffs(m_ind)
        E_df_ind = E_ind_coeffs[1]

        # Total divergence-free E-field is the sum of induced and
        # non-induced parts.
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
        """Evolves the induced potential `m_ind` forward in time.

        Uses the integration scheme specified by `self.integrator`.
        Supports 'euler', 'exponential', and any method supported by
        `scipy.solve_ivp`.
        """
        backend_m_ind = asarray(m_ind)
        backend_E_noind = asarray(E_coeffs_noind)

        if self.integrator == "euler":
            d_m_ind_dt = self._calculate_d_m_ind_dt(backend_m_ind, backend_E_noind)
            return backend_m_ind + dt * d_m_ind_dt

        elif self.integrator == "exponential":
            # The exponential integrator requires the dense operator
            # matrix.
            op_A = asarray(self.geometry.E_df_to_d_m_ind_dt * self.m_ind_to_E_df_matrix)

            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(backend_E_noind)
            diff = backend_m_ind - asarray(steady_state_m_ind)

            if use_jax():
                from jax.scipy.linalg import expm as jax_expm

                evolved = jax_expm(dt * op_A) @ diff + asarray(steady_state_m_ind)
                return evolved

            from scipy.linalg import expm

            evolved = expm(dt * np.asarray(op_A)) @ np.asarray(diff)
            return asarray(evolved) + asarray(steady_state_m_ind)

        else:
            # Fallback to scipy.solve_ivp for other integrators
            logger.debug(f"Using scipy.solve_ivp with method='{self.integrator}'.")

            def rhs_numpy(t, y):
                y_backend = asarray(y)
                dy = self._calculate_d_m_ind_dt(y_backend, backend_E_noind)
                return np.asarray(dy)

            sol = solve_ivp(
                fun=rhs_numpy,
                t_span=(0, dt),
                y0=np.asarray(backend_m_ind),
                method=self.integrator,
                t_eval=[dt],
                dense_output=False,
            )

            if not sol.success:
                logger.warning(
                    f"solve_ivp integrator '{self.integrator}' failed with "
                    f"status {sol.status}: {sol.message}"
                )

            result = sol.y[:, -1]
            return asarray(result)

    def steady_state_m_ind(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculate the steady-state induced potential."""
        # This operation requires solving a linear system, which is most
        # robustly done with the dense matrix form of the operator.
        op_A = asarray(self.m_ind_to_E_df_matrix)
        vec_b = -asarray(E_coeffs_noind[1])
        return xp.linalg.solve(op_A, vec_b)
