"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required for
simulating ionospheric electrodynamics.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Any

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.tensor_chain import TensorChain
from pynamit.simulation.geometry import Geometry
from pynamit.spherical_harmonics.sh_basis import SHBasis

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
        G_u_to_uxB_grid = np.einsum(
            "ijk,jklm->iklm",
            self.geometry.bu,
            self.geometry.basis_evaluator.G_helmholtz,
            optimize=True,
        )
        return np.tensordot(self.geometry.G_helmholtz_pinv, G_u_to_uxB_grid, axes=2)

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
            eta_stacked = np.stack([self.etaP.coeffs, self.etaH.coeffs], axis=0)
            G_eta = self.geometry.basis_evaluator_zero_added.G
            b_stacked = np.stack([self.geometry.bP, self.geometry.bH], axis=0)
            self._M_total_on_grid = np.einsum(
                "sijk,kp,sp->ijk", b_stacked, G_eta, eta_stacked, optimize=True
            )
        return self._M_total_on_grid

    def _create_E_coeffs_operator(self, G_X_to_JS: Optional[np.ndarray]) -> Optional[TensorChain]:
        if G_X_to_JS is None:
            return None
        return TensorChain(
            component_tensors=[self.geometry.G_helmholtz_pinv, self.M_total_on_grid, G_X_to_JS],
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
            np.dot(self.geometry.jr_coeffs_to_j_apex, jr_coeffs) if jr_coeffs is not None else None
        )
        rhs_list.append(b_jr)

        if self.connect_hemispheres and self.E_map_constraint_operator is not None:
            E_map_op = self.geometry.E_coeffs_to_E_apex_ll_diff
            b_E = -np.einsum("cikl,kl->ci", E_map_op, E_direct_coeffs).flatten()
            rhs_list.append(b_E * self.ih_constraint_scaling)

        solution = self.m_imp_solver.solve(
            problem=problem, rhs=rhs_list, preconditioner=preconditioner
        )
        return solution if solution is not None else np.zeros(self.basis.index_length)

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
        if op is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            return np.zeros(output_shape)

        coeffs_arr = np.asarray(coeffs)
        if isinstance(op, (TensorChain, LinearOperator)):
            linop = op if isinstance(op, LinearOperator) else op.as_linear_operator()
            return linop.matvec(coeffs_arr.flatten()).reshape(output_shape)

        return np.tensordot(np.ascontiguousarray(op), coeffs_arr, axes=coeffs_arr.ndim)

    def _calculate_total_E_field(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        m_imp = self._solve_for_m_imp(jr_coeffs, E_direct_coeffs)
        E_imp = self._apply_operator(self.m_imp_to_E_coeffs, m_imp, E_shape)
        return E_direct_coeffs + E_imp, m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        E_direct = self._apply_operator(
            self.u_coeffs_to_E_coeffs, getattr(self.u, "coeffs", 0), E_shape
        )
        if self.Br is not None:
            E_direct += self._apply_operator(self.Br_to_E_coeffs, self.Br.coeffs, E_shape)

        jr_coeffs = getattr(self.jr, "coeffs", None)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(self.m_ind_to_E_coeffs, m_ind, E_shape)
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
        E_df_columns = [self.calculate_ind_coeffs(v)[0][1] for v in identity]
        self._m_ind_to_E_df_matrix = np.array(E_df_columns).T
        logger.info("Dense induction operator built.")

    def evolve_m_ind(
        self,
        m_ind: np.ndarray,
        dt: float,
        E_coeffs_noind: np.ndarray,
        steady_state_m_ind: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evolves the induced potential `m_ind` forward in time by `dt`."""
        if self.integrator == "euler":
            # For Euler, avoid densifying the operator. Use matvec operations directly.
            # Calculate the E-field from the current induced potential.
            E_ind_coeffs, _ = self.calculate_ind_coeffs(m_ind)
            E_df_ind = E_ind_coeffs[1]

            # Total divergence-free E-field is the sum of induced and non-induced parts.
            E_df_total = E_df_ind + E_coeffs_noind[1]

            # Calculate the time derivative and perform the Euler step.
            d_m_ind_dt = self.geometry.E_df_to_d_m_ind_dt * E_df_total
            return m_ind + dt * d_m_ind_dt

        if self.integrator == "exponential":
            # The exponential integrator requires the dense operator matrix.
            # Accessing self.m_ind_to_E_df_matrix will build it if not cached.
            op_A = self.geometry.E_df_to_d_m_ind_dt * self.m_ind_to_E_df_matrix

            if steady_state_m_ind is None:
                steady_state_m_ind = self.steady_state_m_ind(E_coeffs_noind)
            diff = m_ind - steady_state_m_ind
            return expm(dt * op_A) @ diff + steady_state_m_ind

        raise ValueError(f"Unknown integrator: {self.integrator}")

    def steady_state_m_ind(self, E_coeffs_noind: np.ndarray) -> np.ndarray:
        """Calculates the steady-state induced potential."""
        # This operation requires solving a linear system, which is most
        # robustly done with the dense matrix form of the operator.
        op_A = self.m_ind_to_E_df_matrix
        vec_b = -E_coeffs_noind[1]
        return np.linalg.solve(op_A, vec_b)
