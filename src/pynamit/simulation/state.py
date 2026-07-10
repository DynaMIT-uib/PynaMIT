"""State module for ionospheric electrodynamics.

This module contains the State class, which manages the physical state
variables (potentials, currents, etc.) and numerical operators required
for simulating ionospheric electrodynamics.
"""

from __future__ import annotations
from functools import cached_property
import logging
from typing import Any, Optional, Tuple

import numpy as np

from pynamit.primitives.field_coefficients import FieldCoefficients
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver, get_default_least_squares_solver
from pynamit.math.linear_map import (
    LinearMap,
    as_linear_map,
    identity_linear_map,
)
from pynamit.math.backend import (
    block_after_jax_linalg,
    get_array_module,
    to_jax,
    use_jax,
    xp,
)
from pynamit.sphere import CSBasis, SolidHarmonics, SurfaceOperators
from pynamit.simulation.geometry import Geometry
from pynamit.simulation import ionospheric_closure
from pynamit.simulation.operators import StateOperators
from pynamit.simulation.config import setting_value

logger = logging.getLogger(__name__)


class State:
    """Own the mutable ionospheric state and its cached operators.

    The state stores the current input coefficients and assembles the
    run-specific electrodynamic operators that depend on geometry and
    conductance. Time integration itself belongs to ``induction`` and
    run scheduling belongs to ``EvolutionRunner``.
    """

    def __init__(
        self,
        basis: SurfaceOperators,
        mainfield: Any,
        cs_basis: CSBasis,
        settings: Any,
        PFAC_matrix: Optional[np.ndarray] = None,
        solid_harmonics: Optional[SolidHarmonics] = None,
    ) -> None:
        """Initialize the State object."""
        self.basis = basis
        self._init_settings(settings)

        # Encapsulate all geometry, mappings, and evaluators
        self.geometry = Geometry(
            basis, cs_basis, mainfield, settings, PFAC_matrix, solid_harmonics=solid_harmonics
        )

        # Operator for mapping velocity field `u` to E-field
        # (independent of conductance)
        self._u_coeffs_to_E_coeffs_cache: Optional[LinearMap] = None
        self._Q_eff_synthesis_operator_cache: dict[Any, LinearMap] = {}
        self._E_source_to_E_operator_cache: dict[Any, LinearMap] = {}

        # Initialize state variables
        self.u: Optional[FieldCoefficients] = None
        self.Q_eff: Optional[FieldCoefficients] = None
        self.E_source: Optional[FieldCoefficients] = None
        self.Br: Optional[FieldCoefficients] = None
        self.jr: Optional[FieldCoefficients] = None
        self.etaP: Optional[FieldCoefficients] = None
        self.etaH: Optional[FieldCoefficients] = None

        # Invalidate all caches
        self._invalidate_caches()

    # ----- Initialization Helpers -----

    def _init_settings(self, settings: Any) -> None:
        """Extract and store configuration from the settings object."""
        self.m_imp_solver = LeastSquaresSolver(
            solver=setting_value(
                settings, "least_squares_solver", get_default_least_squares_solver()
            ),
            preconditioner=setting_value(
                settings, "least_squares_preconditioner", "pinv"
            ),
        )
        self.static_preconditioner = setting_value(settings, "static_preconditioner", False)
        self.integrator = setting_value(settings, "integrator")
        self.m_imp_regularization_lambda = setting_value(
            settings, "m_imp_regularization_lambda", 0.0
        )
        self.ih_constraint_scaling = setting_value(settings, "ih_constraint_scaling")
        self.connect_hemispheres = bool(setting_value(settings, "connect_hemispheres"))

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

    def _create_u_to_E_operator(self) -> LinearMap:
        """Operator mapping wind coefficients to E coefficients."""
        helmholtz_synthesis = xp.asarray(
            self.geometry.spherical_transform.helmholtz_coeffs_to_gridded_vector
        )
        return ionospheric_closure.wind_to_E_coeffs_operator(
            self.geometry.helmholtz_analysis_matrix,
            self.geometry.bu,
            helmholtz_synthesis,
        )

    @property
    def u_coeffs_to_E_coeffs(self) -> LinearMap:
        """Linear map from wind coefficients to E coefficients."""
        if self._u_coeffs_to_E_coeffs_cache is None:
            self._u_coeffs_to_E_coeffs_cache = self._create_u_to_E_operator()
        return self._u_coeffs_to_E_coeffs_cache

    def _Q_eff_synthesis_operator_for_representation(self, representation) -> LinearMap:
        """Return Q_eff coefficient synthesis to the model grid."""
        cache_key = getattr(representation, "coefficient_space_signature", None)
        if cache_key is None:
            cache_key = getattr(representation, "signature", id(representation))
        if cache_key not in self._Q_eff_synthesis_operator_cache:
            get_operator = getattr(representation, "get_helmholtz_synthesis_operator", None)
            if not callable(get_operator):
                raise ValueError(
                    "Q_eff storage basis cannot evaluate tangential fields on "
                    "the state/model grid."
                )
            self._Q_eff_synthesis_operator_cache[cache_key] = get_operator(self.geometry.grid)
        return self._Q_eff_synthesis_operator_cache[cache_key]

    def _create_Q_eff_to_E_operator_for_representation(self, representation) -> LinearMap:
        """Map effective-current coefficients to E coefficients."""
        q_synthesis = self._Q_eff_synthesis_operator_for_representation(representation)

        return ionospheric_closure.tangential_current_to_E_coeffs_operator(
            xp.asarray(self.geometry.helmholtz_analysis_matrix),
            xp.asarray(self.resistance_tensor_on_grid),
            q_synthesis,
        )

    @property
    def Q_eff_to_E_coeffs(self) -> Optional[LinearMap]:
        """Linear map from effective-current coeffs to E coeffs."""
        if getattr(self, "Q_eff", None) is None:
            return None
        if self._Q_eff_to_E_coeffs_cache is None:
            self._Q_eff_to_E_coeffs_cache = self._create_Q_eff_to_E_operator_for_representation(
                self.Q_eff.representation
            )
        return self._Q_eff_to_E_coeffs_cache

    def _create_E_source_to_E_operator_for_representation(self, representation) -> LinearMap:
        """Map direct E-source coefficients to model E coefficients."""
        if representation.coefficients_are_compatible_with(self.basis):
            return identity_linear_map((2, self.basis.index_length))

        cache_key = getattr(representation, "coefficient_space_signature", None)
        if cache_key is None:
            cache_key = getattr(representation, "signature", id(representation))
        if cache_key not in self._E_source_to_E_operator_cache:
            get_operator = getattr(representation, "get_helmholtz_synthesis_operator", None)
            if not callable(get_operator):
                raise ValueError(
                    "E_source storage basis cannot evaluate tangential fields on "
                    "the state/model grid."
                )
            source_synthesis = get_operator(self.geometry.grid)
            grid_to_coeffs = as_linear_map(
                xp.asarray(self.geometry.helmholtz_analysis_matrix),
                input_shape=(2, self.geometry.grid.size),
                output_shape=(2, self.basis.index_length),
            )
            self._E_source_to_E_operator_cache[cache_key] = grid_to_coeffs @ source_synthesis
        return self._E_source_to_E_operator_cache[cache_key]

    @property
    def E_source_to_E_coeffs(self) -> Optional[LinearMap]:
        """Linear map from direct E-source coeffs to E coeffs."""
        if getattr(self, "E_source", None) is None:
            return None
        if self._E_source_to_E_coeffs_cache is None:
            self._E_source_to_E_coeffs_cache = (
                self._create_E_source_to_E_operator_for_representation(
                    self.E_source.representation
                )
            )
        return self._E_source_to_E_coeffs_cache

    def _invalidate_caches(self) -> None:
        """Invalidate all conductance-dependent cached properties."""
        self._resistance_tensor_on_grid: Optional[np.ndarray] = None
        self._m_ind_to_E_coeffs_cache: Optional[LinearMap] = None
        self._m_imp_to_E_coeffs_cache: Optional[LinearMap] = None
        self._Br_to_E_coeffs_cache: Optional[LinearMap] = None
        self._Q_eff_to_E_coeffs_cache: Optional[LinearMap] = None
        self._E_source_to_E_coeffs_cache: Optional[LinearMap] = None
        self._m_ind_to_E_coeffs_runtime_cache: Optional[LinearMap] = None
        self._m_imp_to_E_coeffs_runtime_cache: Optional[LinearMap] = None
        self._Br_to_E_coeffs_runtime_cache: Optional[LinearMap] = None
        self._Q_eff_to_E_coeffs_runtime_cache: Optional[LinearMap] = None
        self._E_source_to_E_coeffs_runtime_cache: Optional[LinearMap] = None
        self._E_map_constraint_cache: Optional[LinearMap] = None
        self._m_ind_to_E_df_matrix: Optional[np.ndarray] = None
        self._m_ind_to_E_df_operator: Optional[LinearMap] = None
        self._E_noind_to_m_ind_steady_matrix: Optional[np.ndarray] = None
        self._E_noind_to_m_ind_steady_operator: Optional[LinearMap] = None
        self._jr_to_m_imp_matrix: Optional[np.ndarray] = None
        self._E_direct_to_m_imp_matrix: Optional[np.ndarray] = None
        self._jr_to_m_imp_operator: Optional[LinearMap] = None
        self._E_direct_to_m_imp_operator: Optional[LinearMap] = None
        self._direct_E_coeffs_to_total_E_coeffs_operator: Optional[LinearMap] = None
        self._direct_E_coeffs_to_E_df_operator: Optional[LinearMap] = None
        self._m_imp_problem: Optional[LeastSquaresProblem] = None
        self._m_imp_preconditioner: Optional[LinearMap] = None
        self._m_imp_preconditioner_ready = False

    # ----- Cached Physical Properties (dependent on resistance) -----

    @property
    def resistance_tensor_on_grid(self) -> np.ndarray:
        """Resistance tensor on the spatial grid."""
        if self._resistance_tensor_on_grid is None:
            if self.etaP is None or self.etaH is None:
                raise RuntimeError(
                    "Resistance or conductance must be set before accessing "
                    "closure-dependent properties."
                )
            eta_stacked = xp.stack(
                [xp.asarray(self.etaP.array), xp.asarray(self.etaH.array)], axis=0
            )
            resistance_synthesis = self._resistance_synthesis_operator()
            resistance_on_grid = xp.asarray(
                resistance_synthesis.matmat(xp.swapaxes(eta_stacked, 0, 1))
            )
            resistance_on_grid = xp.swapaxes(resistance_on_grid, 0, 1)
            self._resistance_tensor_on_grid = ionospheric_closure.resistance_tensor_on_grid(
                resistance_on_grid[0],
                resistance_on_grid[1],
                self.geometry.bP,
                self.geometry.bH,
            )
        return self._resistance_tensor_on_grid

    def _resistance_storage_basis(self):
        """Return the shared resistance storage basis."""
        basis = self.etaP.representation
        hall_basis = self.etaH.representation
        if hall_basis is not basis:
            compatible = getattr(hall_basis, "coefficients_are_compatible_with", None)
            if not callable(compatible) or not compatible(basis):
                raise ValueError(
                    "Pedersen and Hall resistance storage bases must be coefficient-compatible."
                )
        return basis

    def _compatible_resistance_transform(self, basis) -> Optional[Any]:
        """Return a compatible resistance synthesis transform."""
        if basis.coefficients_are_compatible_with(self.basis):
            return self.geometry.spherical_transform

        zero_added_transform = self.geometry.spherical_transform_zero_added
        if basis.coefficients_are_compatible_with(zero_added_transform.source):
            return zero_added_transform

        return None

    def _resistance_synthesis_operator(self) -> LinearMap:
        """Return stored-resistance synthesis to the model grid."""
        basis = self._resistance_storage_basis()

        get_operator = getattr(basis, "get_scalar_evaluation_operator", None)
        if callable(get_operator):
            return get_operator(self.geometry.grid)

        transform = self._compatible_resistance_transform(basis)
        if transform is not None:
            return transform.scalar_coeffs_to_grid_operator

        get_matrix = getattr(basis, "get_scalar_evaluation_matrix", None)
        if callable(get_matrix):
            return as_linear_map(get_matrix(self.geometry.grid))

        raise ValueError("Resistance storage basis cannot be evaluated on the state/model grid.")

    def _create_E_coeffs_operator(self, source_to_JS: Optional[np.ndarray]) -> Optional[LinearMap]:
        if source_to_JS is None:
            return None
        return ionospheric_closure.source_to_E_coeffs_operator(
            self.geometry.helmholtz_analysis_matrix,
            self.resistance_tensor_on_grid,
            source_to_JS,
            self.basis.index_length,
        )

    @property
    def m_ind_to_E_coeffs(self) -> Optional[LinearMap]:
        """Linear map from m_ind coefficients to E coefficients."""
        if self._m_ind_to_E_coeffs_cache is None:
            self._m_ind_to_E_coeffs_cache = self._create_E_coeffs_operator(
                self.geometry.m_ind_to_gridded_JS()
            )
        return self._m_ind_to_E_coeffs_cache

    @property
    def m_imp_to_E_coeffs(self) -> Optional[LinearMap]:
        """Linear map from m_imp coefficients to E coefficients."""
        if self._m_imp_to_E_coeffs_cache is None:
            self._m_imp_to_E_coeffs_cache = self._create_E_coeffs_operator(
                self.geometry.m_imp_to_gridded_JS()
            )
        return self._m_imp_to_E_coeffs_cache

    @property
    def Br_to_E_coeffs(self) -> Optional[LinearMap]:
        """Linear map from Br coefficients to E coefficients."""
        if self._Br_to_E_coeffs_cache is None:
            self._Br_to_E_coeffs_cache = self._create_E_coeffs_operator(
                self.geometry.Br_to_gridded_JS()
            )
        return self._Br_to_E_coeffs_cache

    def _runtime_E_coeffs_operator(self, op: Optional[LinearMap]) -> Optional[LinearMap]:
        """Return an E-coefficient operator for repeated applies."""
        if op is None:
            return None
        _ = op.array
        return op

    @property
    def _m_ind_to_E_coeffs_runtime(self) -> Optional[LinearMap]:
        """Runtime operator mapping m_ind to E coefficients."""
        if getattr(self, "_m_ind_to_E_coeffs_runtime_cache", None) is None:
            self._m_ind_to_E_coeffs_runtime_cache = self._runtime_E_coeffs_operator(
                self.m_ind_to_E_coeffs
            )
        return self._m_ind_to_E_coeffs_runtime_cache

    @property
    def _m_imp_to_E_coeffs_runtime(self) -> Optional[LinearMap]:
        """Runtime operator mapping m_imp to E coefficients."""
        if getattr(self, "_m_imp_to_E_coeffs_runtime_cache", None) is None:
            self._m_imp_to_E_coeffs_runtime_cache = self._runtime_E_coeffs_operator(
                self.m_imp_to_E_coeffs
            )
        return self._m_imp_to_E_coeffs_runtime_cache

    @property
    def _Br_to_E_coeffs_runtime(self) -> Optional[LinearMap]:
        """Runtime map from Br coefficients to E coefficients."""
        if getattr(self, "_Br_to_E_coeffs_runtime_cache", None) is None:
            self._Br_to_E_coeffs_runtime_cache = self._runtime_E_coeffs_operator(
                self.Br_to_E_coeffs
            )
        return self._Br_to_E_coeffs_runtime_cache

    @property
    def _Q_eff_to_E_coeffs_runtime(self) -> Optional[LinearMap]:
        """Runtime map from effective-current coeffs to E coeffs."""
        if getattr(self, "_Q_eff_to_E_coeffs_runtime_cache", None) is None:
            self._Q_eff_to_E_coeffs_runtime_cache = self._runtime_E_coeffs_operator(
                self.Q_eff_to_E_coeffs
            )
        return self._Q_eff_to_E_coeffs_runtime_cache

    @property
    def _E_source_to_E_coeffs_runtime(self) -> Optional[LinearMap]:
        """Runtime map from direct E-source coeffs to E coeffs."""
        if getattr(self, "_E_source_to_E_coeffs_runtime_cache", None) is None:
            self._E_source_to_E_coeffs_runtime_cache = self._runtime_E_coeffs_operator(
                self.E_source_to_E_coeffs
            )
        return self._E_source_to_E_coeffs_runtime_cache

    @property
    def _E_map_constraint(self) -> Optional[LinearMap]:
        """Linear map enforcing the E-field low-latitude constraint."""
        if self._E_map_constraint_cache is None:
            inner_map = self.m_imp_to_E_coeffs
            outer_map = self.geometry.E_coeffs_to_E_apex_ll_diff_operator
            if inner_map is not None and outer_map is not None:
                self._E_map_constraint_cache = outer_map @ inner_map
        return self._E_map_constraint_cache

    # ----- Solver Setup and Execution -----
    @property
    def m_imp_problem(self) -> LeastSquaresProblem:
        """The least-squares problem definition for `m_imp`."""
        if self._m_imp_problem is None:
            logger.info("Defining new least-squares problem for m_imp.")
            operators, data_shapes = [], []

            # Radial current (jr) must match imposed field.
            op_jr = self.geometry.jr_coeffs_to_j_apex_operator @ self.geometry.m_imp_to_jr_operator
            operators.append(op_jr)
            data_shapes.append(op_jr.output_shape)

            # E-field must map at low latitudes.
            if self.connect_hemispheres:
                E_map_constraint = self._E_map_constraint
                if E_map_constraint is not None:
                    op_E = self.ih_constraint_scaling * E_map_constraint
                    operators.append(op_E)
                    data_shapes.append(op_E.output_shape)

            # Add Tikhonov regularization if lambda is set.
            reg_ops, reg_weights = [], []
            if self.m_imp_regularization_lambda > 0:
                n = self.basis.index_length
                reg_ops.append(identity_linear_map((n,)))
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
        if self.m_imp_solver.solver not in ("lsmr", "cgls"):
            return None
        if not self._m_imp_preconditioner_ready:
            logger.info("Building new preconditioner for m_imp solver.")
            self._m_imp_preconditioner = self.m_imp_solver.build_preconditioner(
                problem=self.m_imp_problem
            )
            self._m_imp_preconditioner_ready = True
        return self._m_imp_preconditioner

    def _build_m_imp_response_matrices(self) -> None:
        """Construct dense response matrices for the m_imp solve."""
        logger.info("Building dense m_imp response matrices.")
        n = self.basis.index_length
        problem = self.m_imp_problem
        solve_response = self.m_imp_solver.build_response_solver(
            problem, preconditioner=self.m_imp_preconditioner
        )

        jr_rhs = np.asarray(self.geometry.jr_coeffs_to_j_apex).reshape(
            problem.A[0].output_shape + (-1,)
        )
        rhs_entries = [None] * problem.num_data_terms
        rhs_entries[0] = jr_rhs
        jr_to_m_imp = solve_response(rhs_entries)

        E_direct_to_m_imp = None
        if self.connect_hemispheres and self._E_map_constraint is not None:
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
        solution = xp.zeros(self.basis.index_length)

        if jr_coeffs is not None:
            solution += self.operators.jr_to_m_imp.matvec(xp.asarray(jr_coeffs))

        if self.connect_hemispheres:
            E_direct_to_m_imp = self.operators.E_direct_to_m_imp
            if E_direct_to_m_imp is not None:
                solution += E_direct_to_m_imp.matvec(xp.asarray(E_direct_coeffs))

        return self.project_scalar_mean_free(solution)

    # ----- State Update -----

    def update(self, input_timeseries: Any, time: float, interpolation: bool = False) -> None:
        """Update the state variables based on the current input."""
        conductance_updated = False
        for key, dataset in input_timeseries.datasets.items():
            updated_input = input_timeseries.get_entry_if_changed(key, time, interpolation)
            if updated_input is None:
                continue

            field_space = input_timeseries.get_field_space(key)
            if key == "conductance":
                conductance_updated = True
                self.etaP = FieldCoefficients(field_space, coeffs=updated_input["etaP"])
                self.etaH = FieldCoefficients(field_space, coeffs=updated_input["etaH"])
            elif key == "jr":
                self.jr = FieldCoefficients(field_space, coeffs=updated_input["jr"])
            elif key == "Br":
                if self.geometry.RM is None:
                    raise ValueError("Br input can only be set if RM is not None.")
                self.Br = FieldCoefficients(field_space, coeffs=updated_input["Br"])
            elif key == "u":
                self.u = FieldCoefficients(field_space, coeffs=updated_input["u"])
            elif key == "Q_eff":
                self.Q_eff = FieldCoefficients(field_space, coeffs=updated_input["Q_eff"])
            elif key == "E_source":
                self.E_source = FieldCoefficients(field_space, coeffs=updated_input["E_source"])

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

    def _apply_operator(
        self, op: Optional[LinearMap], coeffs: Any, output_shape: Tuple[int, ...]
    ) -> Any:
        if op is None or coeffs is None or (isinstance(coeffs, (int, float)) and coeffs == 0):
            array_module = op.array_module(coeffs) if op is not None else get_array_module(coeffs)
            return array_module.zeros(output_shape)

        array_module = op.array_module(coeffs)
        coeffs_arr = array_module.asarray(coeffs)
        return op.matvec(coeffs_arr.reshape(-1)).reshape(output_shape)

    def _calculate_total_E_field(
        self, E_direct_coeffs: np.ndarray, jr_coeffs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        E_shape = (2, self.basis.index_length)
        E_direct_coeffs = self.project_helmholtz_mean_free(E_direct_coeffs)
        m_imp = self._solve_for_m_imp(jr_coeffs, E_direct_coeffs)
        E_imp = self._apply_operator(self._m_imp_to_E_coeffs_runtime, m_imp, E_shape)
        return self.project_helmholtz_mean_free(E_direct_coeffs + E_imp), m_imp

    def calculate_noind_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate E-field coefficients without induction effects."""
        E_shape = (2, self.basis.index_length)
        active_wind_sources = [
            name for name in ("u", "Q_eff", "E_source") if getattr(self, name, None) is not None
        ]
        if len(active_wind_sources) > 1:
            raise ValueError(
                "Neutral wind input 'u', effective-current input 'Q_eff', and "
                "direct electric-field input 'E_source' are mutually exclusive; "
                "use only one wind forcing representation."
            )
        u_coeffs = 0 if self.u is None else xp.asarray(self.u.array)
        E_direct = self._apply_operator(self.u_coeffs_to_E_coeffs, u_coeffs, E_shape)
        if getattr(self, "E_source", None) is not None:
            E_direct += self._apply_operator(
                self._E_source_to_E_coeffs_runtime, xp.asarray(self.E_source.array), E_shape
            )
        if self.Br is not None:
            E_direct += self._apply_operator(
                self._Br_to_E_coeffs_runtime, xp.asarray(self.Br.array), E_shape
            )
        if getattr(self, "Q_eff", None) is not None:
            E_direct += self._apply_operator(
                self._Q_eff_to_E_coeffs_runtime, xp.asarray(self.Q_eff.array), E_shape
            )

        jr_coeffs = None if self.jr is None else xp.asarray(self.jr.array)
        return self._calculate_total_E_field(E_direct, jr_coeffs)

    def calculate_ind_coeffs(self, m_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate total E-field coefficients."""
        E_shape = (2, self.basis.index_length)
        E_direct_ind = self._apply_operator(
            self._m_ind_to_E_coeffs_runtime,
            xp.asarray(self.project_scalar_mean_free(m_ind)),
            E_shape,
        )
        return self._calculate_total_E_field(E_direct_ind, None)

    # ----- Induction Operators -----

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
        """Construct the m_ind -> total E_df map."""
        m_ind_to_E = self.m_ind_to_E_coeffs
        if m_ind_to_E is None:
            raise RuntimeError("m_ind_to_E_coeffs is not available.")
        return self.operators.direct_E_to_E_df @ m_ind_to_E

    def _build_m_ind_to_E_df_matrix(self) -> None:
        """Construct the dense matrix for the induction operator."""
        logger.info("Building dense induction operator matrix (m_ind -> E_df)...")
        self._m_ind_to_E_df_matrix = self.m_ind_to_E_df_operator.to_matrix()
        logger.info("Dense induction operator built.")
