"""Instantaneous electrodynamic response model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pynamit.fields import FieldCoefficients
from pynamit.math import content_fingerprint
from pynamit.math.backend import block_after_jax_linalg, get_array_module, to_jax, use_jax, xp
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver
from pynamit.math.linear_map import LinearMap, MatrixBackend, as_linear_map, identity_linear_map
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.geometry import SimulationGeometry

logger = logging.getLogger(__name__)

_ACTIVE_INPUT_NAMES = frozenset(
    {
        "u",
        "Q_eff",
        "E_neutral_wind",
        "Br",
        "jr",
        "log_conductance_magnitude",
        "log_hall_to_pedersen_ratio",
    }
)


class ElectrodynamicResponse:
    """Compile and evaluate the instantaneous electrodynamic response.

    The model stores inputs active at one simulation time and assembles
    their resistance-dependent response operators. The runner holds
    the evolving ``m_ind`` trajectory and passes it into response
    calculations.
    Time integration belongs to
    ``electrodynamics.induction``; run scheduling belongs to
    ``SimulationRunner``.
    """

    def __init__(self, geometry: SimulationGeometry, config: SimulationConfig) -> None:
        """Initialize the response model."""
        if not isinstance(geometry, SimulationGeometry):
            raise TypeError("ElectrodynamicResponse requires a SimulationGeometry object.")
        if not isinstance(config, SimulationConfig):
            raise TypeError("ElectrodynamicResponse requires a validated SimulationConfig.")
        self.config = config
        self.geometry = geometry
        self._m_imp_solver = LeastSquaresSolver(
            solver=config.least_squares_solver, preconditioner=config.least_squares_preconditioner
        )

        # Operator for mapping velocity field `u` to E-field
        # (independent of resistance)
        self._u_coeffs_to_E_coeffs_cache: LinearMap | None = None
        self._Q_eff_synthesis_operator_cache: LinearMap | None = None
        self._E_neutral_wind_to_E_coeffs_cache: LinearMap | None = None

        # Inputs active at the current simulation time.
        self.u: FieldCoefficients | None = None
        self.Q_eff: FieldCoefficients | None = None
        self.E_neutral_wind: FieldCoefficients | None = None
        self.Br: FieldCoefficients | None = None
        self.jr: FieldCoefficients | None = None
        self.log_conductance_magnitude: FieldCoefficients | None = None
        self.log_hall_to_pedersen_ratio: FieldCoefficients | None = None

        # Initialize closure-dependent caches.
        self._invalidate_closure_caches()

    def project_surface_scalar_mean_free(self, coeffs: Any) -> Any:
        """Project surface-potential coefficients to a fixed gauge."""
        projector = getattr(self.geometry.horizontal_basis, "project_scalar_mean_free", None)
        return coeffs if not callable(projector) else projector(coeffs)

    def project_helmholtz_mean_free(self, coeffs: Any) -> Any:
        """Project Helmholtz-potential coefficients to a fixed gauge."""
        projector = getattr(self.geometry.horizontal_basis, "project_helmholtz_mean_free", None)
        if not callable(projector):
            return coeffs
        return projector(coeffs)

    def _create_u_to_E_operator(self) -> LinearMap:
        """Operator mapping wind coefficients to E coefficients."""
        helmholtz_synthesis = (
            self.geometry.horizontal_transform.helmholtz_coeffs_to_gridded_vector_operator
        )
        return ionospheric_closure.wind_to_E_coeffs_operator(
            self.geometry.helmholtz_analysis_operator,
            self.geometry.wind_motional_E_tensor,
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
        if self._Q_eff_synthesis_operator_cache is None:
            get_operator = getattr(representation, "get_helmholtz_synthesis_operator", None)
            if not callable(get_operator):
                raise ValueError(
                    "Q_eff storage basis cannot evaluate tangential fields on "
                    "the state/model grid."
                )
            self._Q_eff_synthesis_operator_cache = get_operator(self.geometry.model_grid)
        return self._Q_eff_synthesis_operator_cache

    def _create_Q_eff_to_E_operator_for_representation(self, representation) -> LinearMap:
        """Map effective-current coefficients to E coefficients."""
        q_synthesis = self._Q_eff_synthesis_operator_for_representation(representation)

        return ionospheric_closure.tangential_current_to_E_coeffs_operator(
            self.geometry.helmholtz_analysis_operator,
            xp.asarray(self.resistance_tensor_on_grid),
            q_synthesis,
        )

    @property
    def Q_eff_to_E_coeffs(self) -> LinearMap | None:
        """Linear map from effective-current coeffs to E coeffs."""
        if self.Q_eff is None:
            return None
        if self._Q_eff_to_E_coeffs_cache is None:
            self._Q_eff_to_E_coeffs_cache = self._create_Q_eff_to_E_operator_for_representation(
                self.Q_eff.field_space.representation
            )
        return self._Q_eff_to_E_coeffs_cache

    def _create_E_neutral_wind_to_E_operator_for_representation(self, representation) -> LinearMap:
        """Map neutral-wind E coefficients to model E coefficients."""
        if representation.coefficients_are_compatible_with(self.geometry.horizontal_basis):
            return identity_linear_map((2, self.geometry.horizontal_basis.index_length))

        get_operator = getattr(representation, "get_helmholtz_synthesis_operator", None)
        if not callable(get_operator):
            raise ValueError(
                "E_neutral_wind storage basis cannot evaluate tangential "
                "fields on the state/model grid."
            )
        source_synthesis = get_operator(self.geometry.model_grid)
        grid_to_coeffs = self.geometry.helmholtz_analysis_operator
        return grid_to_coeffs @ source_synthesis

    @property
    def E_neutral_wind_to_E_coeffs(self) -> LinearMap | None:
        """Map neutral-wind E coefficients to model E coefficients."""
        if self.E_neutral_wind is None:
            return None
        if self._E_neutral_wind_to_E_coeffs_cache is None:
            self._E_neutral_wind_to_E_coeffs_cache = (
                self._create_E_neutral_wind_to_E_operator_for_representation(
                    self.E_neutral_wind.field_space.representation
                )
            )
        return self._E_neutral_wind_to_E_coeffs_cache

    def _invalidate_closure_caches(self) -> None:
        """Invalidate resistance-dependent cached properties."""
        self._conductance_fingerprint_cache: str | None = None
        self._resistance_tensor_on_grid: np.ndarray | None = None
        self._m_ind_to_E_coeffs_cache: LinearMap | None = None
        self._m_imp_to_E_coeffs_cache: LinearMap | None = None
        self._Br_to_E_coeffs_cache: LinearMap | None = None
        self._Q_eff_to_E_coeffs_cache: LinearMap | None = None
        self._runtime_m_ind_to_E_coeffs_cache: LinearMap | None = None
        self._runtime_m_imp_to_E_coeffs_cache: LinearMap | None = None
        self._runtime_Br_to_E_coeffs_cache: LinearMap | None = None
        self._runtime_Q_eff_to_E_coeffs_cache: LinearMap | None = None
        self._interhemispheric_electric_field_constraint_cache: LinearMap | None = None
        self._m_ind_feedback_matrix: np.ndarray | None = None
        self._m_ind_feedback_operator: LinearMap | None = None
        self._m_ind_to_E_df_operator_cache: LinearMap | None = None
        self._noninductive_E_df_to_steady_m_ind_matrix: np.ndarray | None = None
        self._noninductive_E_df_to_steady_m_ind_operator: LinearMap | None = None
        self._jr_to_m_imp_matrix: np.ndarray | None = None
        self._driving_E_to_m_imp_matrix: np.ndarray | None = None
        self._jr_to_m_imp_operator: LinearMap | None = None
        self._driving_E_to_m_imp_operator: LinearMap | None = None
        self._driving_E_to_total_E_operator: LinearMap | None = None
        self._driving_E_to_E_df_operator: LinearMap | None = None
        self._m_imp_problem_cache: LeastSquaresProblem | None = None
        self._m_imp_response_solver_cache = None
        self._m_imp_preconditioner_cache: LinearMap | None = None
        self._m_imp_preconditioner_ready = False

    # ----- Cached Physical Properties (dependent on resistance) -----

    @property
    def conductance_fingerprint(self) -> str:
        """Return the exact identity of the active conductance field."""
        if self.log_conductance_magnitude is None or self.log_hall_to_pedersen_ratio is None:
            raise RuntimeError(
                "Resistance or conductance must be set before it can be fingerprinted."
            )
        if self._conductance_fingerprint_cache is None:
            self._conductance_fingerprint_cache = content_fingerprint(
                {
                    "field_space": self.log_conductance_magnitude.field_space.signature,
                    "log_conductance_magnitude": np.asarray(self.log_conductance_magnitude.array),
                    "log_hall_to_pedersen_ratio": np.asarray(
                        self.log_hall_to_pedersen_ratio.array
                    ),
                }
            )
        return self._conductance_fingerprint_cache

    @property
    def resistance_tensor_on_grid(self) -> np.ndarray:
        """Resistance tensor on the spatial grid."""
        if self._resistance_tensor_on_grid is None:
            if self.log_conductance_magnitude is None or self.log_hall_to_pedersen_ratio is None:
                raise RuntimeError(
                    "Resistance or conductance must be set before accessing "
                    "closure-dependent properties."
                )
            log_coordinates = xp.stack(
                [
                    xp.asarray(self.log_conductance_magnitude.array),
                    xp.asarray(self.log_hall_to_pedersen_ratio.array),
                ],
                axis=0,
            )
            conductance_synthesis = self._conductance_synthesis_operator()
            log_coordinates_on_grid = xp.asarray(
                conductance_synthesis.matmat(xp.swapaxes(log_coordinates, 0, 1))
            )
            log_coordinates_on_grid = xp.swapaxes(log_coordinates_on_grid, 0, 1)
            etaP, etaH = ionospheric_closure.resistance_from_log_conductance_coordinates(
                log_coordinates_on_grid[0], log_coordinates_on_grid[1]
            )
            self._resistance_tensor_on_grid = ionospheric_closure.resistance_tensor_on_grid(
                etaP,
                etaH,
                self.geometry.pedersen_geometry_tensor,
                self.geometry.hall_geometry_tensor,
            )
        return self._resistance_tensor_on_grid

    def _conductance_storage_basis(self):
        """Return the shared log-conductance storage basis."""
        basis = self.log_conductance_magnitude.field_space.representation
        ratio_basis = self.log_hall_to_pedersen_ratio.field_space.representation
        if ratio_basis is not basis:
            compatible = getattr(ratio_basis, "coefficients_are_compatible_with", None)
            if not callable(compatible) or not compatible(basis):
                raise ValueError(
                    "Conductance log-coordinate storage bases must be coefficient-compatible."
                )
        return basis

    def _conductance_synthesis_operator(self) -> LinearMap:
        """Return log-conductance synthesis to the model grid."""
        basis = self._conductance_storage_basis()

        get_operator = getattr(basis, "get_scalar_evaluation_operator", None)
        if callable(get_operator):
            return get_operator(self.geometry.model_grid)

        get_matrix = getattr(basis, "get_scalar_evaluation_matrix", None)
        if callable(get_matrix):
            return as_linear_map(get_matrix(self.geometry.model_grid))

        raise ValueError("Conductance storage basis cannot be evaluated on the state/model grid.")

    def _sheet_current_source_to_E_coeffs_operator(self, source_to_JS: LinearMap) -> LinearMap:
        """Map a magnetic source through derived sheet current to E."""
        return ionospheric_closure.tangential_current_to_E_coeffs_operator(
            self.geometry.helmholtz_analysis_operator, self.resistance_tensor_on_grid, source_to_JS
        )

    @property
    def m_ind_to_E_coeffs(self) -> LinearMap:
        """Linear map from m_ind coefficients to E coefficients."""
        if self._m_ind_to_E_coeffs_cache is None:
            self._m_ind_to_E_coeffs_cache = self._sheet_current_source_to_E_coeffs_operator(
                self.geometry.m_ind_to_gridded_JS_operator()
            )
        return self._m_ind_to_E_coeffs_cache

    @property
    def m_imp_to_E_coeffs(self) -> LinearMap:
        """Linear map from m_imp coefficients to E coefficients."""
        if self._m_imp_to_E_coeffs_cache is None:
            self._m_imp_to_E_coeffs_cache = self._sheet_current_source_to_E_coeffs_operator(
                self.geometry.m_imp_to_gridded_JS_operator()
            )
        return self._m_imp_to_E_coeffs_cache

    @property
    def Br_to_E_coeffs(self) -> LinearMap | None:
        """Linear map from Br coefficients to E coefficients."""
        if self._Br_to_E_coeffs_cache is None:
            Br_to_JS = self.geometry.Br_to_gridded_JS_operator()
            if Br_to_JS is None:
                return None
            self._Br_to_E_coeffs_cache = self._sheet_current_source_to_E_coeffs_operator(Br_to_JS)
        return self._Br_to_E_coeffs_cache

    def _optimize_repeated_E_operator(
        self, op: LinearMap | None, *, compact_input: bool
    ) -> LinearMap | None:
        """Prepare an E-coefficient map for repeated runtime use."""
        if op is None:
            return None
        spaces_coincide = self.geometry.horizontal_basis is self.geometry.poloidal_basis
        if compact_input or spaces_coincide:
            _ = op.array
        return op

    @property
    def _runtime_m_ind_to_E_coeffs(self) -> LinearMap:
        """Runtime map from m_ind to E coefficients."""
        if self._runtime_m_ind_to_E_coeffs_cache is None:
            self._runtime_m_ind_to_E_coeffs_cache = self._optimize_repeated_E_operator(
                self.m_ind_to_E_coeffs, compact_input=True
            )
        return self._runtime_m_ind_to_E_coeffs_cache

    @property
    def _runtime_m_imp_to_E_coeffs(self) -> LinearMap:
        """Runtime map from m_imp to E coefficients."""
        if self._runtime_m_imp_to_E_coeffs_cache is None:
            self._runtime_m_imp_to_E_coeffs_cache = self._optimize_repeated_E_operator(
                self.m_imp_to_E_coeffs, compact_input=False
            )
        return self._runtime_m_imp_to_E_coeffs_cache

    @property
    def _runtime_Br_to_E_coeffs(self) -> LinearMap | None:
        """Runtime map from Br coefficients to E coefficients."""
        if self._runtime_Br_to_E_coeffs_cache is None:
            self._runtime_Br_to_E_coeffs_cache = self._optimize_repeated_E_operator(
                self.Br_to_E_coeffs, compact_input=True
            )
        return self._runtime_Br_to_E_coeffs_cache

    @property
    def _runtime_Q_eff_to_E_coeffs(self) -> LinearMap | None:
        """Runtime map from effective-current coefficients to E."""
        if self._runtime_Q_eff_to_E_coeffs_cache is None:
            self._runtime_Q_eff_to_E_coeffs_cache = self._optimize_repeated_E_operator(
                self.Q_eff_to_E_coeffs, compact_input=False
            )
        return self._runtime_Q_eff_to_E_coeffs_cache

    @property
    def _interhemispheric_electric_field_constraint(self) -> LinearMap | None:
        """Linear map enforcing the E-field low-latitude constraint."""
        if self._interhemispheric_electric_field_constraint_cache is None:
            outer_map = self.geometry.interhemispheric_electric_field_difference_operator
            if outer_map is not None:
                self._interhemispheric_electric_field_constraint_cache = (
                    outer_map @ self.m_imp_to_E_coeffs
                )
        return self._interhemispheric_electric_field_constraint_cache

    # ----- Solver Setup and Execution -----
    @property
    def _m_imp_problem(self) -> LeastSquaresProblem:
        """The least-squares problem definition for `m_imp`."""
        if self._m_imp_problem_cache is None:
            logger.info("Defining new least-squares problem for m_imp.")
            operators, data_shapes = [], []

            # Radial current (jr) must match imposed field.
            radial_current_operator = (
                self.geometry.radial_current_constraint_operator
                @ self.geometry.m_imp_to_jr_operator
            )
            operators.append(radial_current_operator)
            data_shapes.append(radial_current_operator.output_shape)

            # E-field must map at low latitudes.
            if self.config.enable_interhemispheric_coupling:
                electric_field_constraint = self._interhemispheric_electric_field_constraint
                if electric_field_constraint is not None:
                    electric_field_operator = (
                        self.config.interhemispheric_electric_field_weight
                        * electric_field_constraint
                    )
                    operators.append(electric_field_operator)
                    data_shapes.append(electric_field_operator.output_shape)

            # CS potentials include one constant gauge. Constrain that
            # coefficient direction exactly instead of regularizing it.
            if self.geometry.surface_gauge_operator is not None:
                operators.append(self.geometry.surface_gauge_operator)
                data_shapes.append(self.geometry.surface_gauge_operator.output_shape)

            # Add Tikhonov regularization if lambda is set.
            reg_ops, reg_weights = [], []
            if self.config.m_imp_regularization_lambda > 0:
                n = self.geometry.horizontal_basis.index_length
                reg_ops.append(identity_linear_map((n,)))
                reg_weights.append(self.config.m_imp_regularization_lambda)

            self._m_imp_problem_cache = LeastSquaresProblem(
                A=operators,
                solution_shape=self.geometry.horizontal_basis.index_length,
                data_shapes=data_shapes,
                regularization_matrices=reg_ops,
                regularization_weights=reg_weights,
            )
        return self._m_imp_problem_cache

    @property
    def _m_imp_preconditioner(self) -> LinearMap | None:
        """Preconditioner for the m_imp least-squares problem."""
        if self._m_imp_solver.solver not in ("lsmr", "cgls"):
            return None
        if not self._m_imp_preconditioner_ready:
            logger.info("Building new preconditioner for m_imp solver.")
            self._m_imp_preconditioner_cache = self._m_imp_solver.build_preconditioner(
                problem=self._m_imp_problem
            )
            self._m_imp_preconditioner_ready = True
        return self._m_imp_preconditioner_cache

    def _solve_m_imp_response(self, rhs_entries):
        """Solve imposed-potential response right-hand sides."""
        if self._m_imp_response_solver_cache is None:
            self._m_imp_response_solver_cache = self._m_imp_solver.build_response_solver(
                self._m_imp_problem, preconditioner=self._m_imp_preconditioner
            )
        return self._m_imp_response_solver_cache(rhs_entries)

    def _build_jr_to_m_imp_matrix(self) -> None:
        """Construct the explicit radial-current response matrix."""
        logger.info("Building dense jr-to-m_imp response matrix.")
        problem = self._m_imp_problem

        radial_current_matrix = self.geometry.radial_current_constraint_operator.to_matrix(
            backend="numpy"
        )
        radial_current_rhs = np.asarray(radial_current_matrix).reshape(
            problem.A[0].output_shape + (-1,)
        )
        rhs_entries = [None] * problem.num_data_terms
        rhs_entries[0] = radial_current_rhs
        jr_to_m_imp = self._solve_m_imp_response(rhs_entries)
        self._jr_to_m_imp_matrix = to_jax(jr_to_m_imp) if use_jax() else jr_to_m_imp

    def _build_driving_E_to_m_imp_matrix(self) -> None:
        """Construct the explicit interhemispheric E-response matrix."""
        if (
            not self.config.enable_interhemispheric_coupling
            or self._interhemispheric_electric_field_constraint is None
        ):
            self._driving_E_to_m_imp_matrix = None
            return

        logger.info("Building dense driving-E-to-m_imp response matrix.")
        n = self.geometry.horizontal_basis.index_length
        problem = self._m_imp_problem
        electric_field_rhs = (
            -self.geometry.interhemispheric_electric_field_difference_matrix.reshape(
                problem.A[1].output_shape + (2 * n,)
            )
        )
        electric_field_rhs *= self.config.interhemispheric_electric_field_weight
        rhs_entries = [None] * problem.num_data_terms
        rhs_entries[1] = electric_field_rhs
        driving_E_to_m_imp = self._solve_m_imp_response(rhs_entries).reshape((n, 2, n))
        self._driving_E_to_m_imp_matrix = (
            to_jax(driving_E_to_m_imp) if use_jax() else driving_E_to_m_imp
        )

    def _m_imp_rhs_entries(
        self, jr_coeffs: np.ndarray | None, driving_E: np.ndarray
    ) -> list[np.ndarray | None] | None:
        """Assemble physical right-hand sides for one m_imp solve."""
        problem = self._m_imp_problem
        rhs_entries = [None] * problem.num_data_terms
        has_rhs = False

        if jr_coeffs is not None:
            rhs_entries[0] = self.geometry.radial_current_constraint_operator.matvec(jr_coeffs)
            has_rhs = True

        if (
            self.config.enable_interhemispheric_coupling
            and self._interhemispheric_electric_field_constraint is not None
        ):
            electric_field_difference = (
                self.geometry.interhemispheric_electric_field_difference_operator.matvec(driving_E)
            )
            rhs_entries[1] = (
                -self.config.interhemispheric_electric_field_weight * electric_field_difference
            )
            has_rhs = True

        return rhs_entries if has_rhs else None

    # ----- Response Operators -----

    @property
    def jr_to_m_imp_operator(self) -> LinearMap:
        """Linear map from radial current to imposed potential."""
        if self._jr_to_m_imp_operator is None:
            if self._jr_to_m_imp_matrix is None:
                self._build_jr_to_m_imp_matrix()
            self._jr_to_m_imp_operator = as_linear_map(
                self._jr_to_m_imp_matrix,
                input_shape=(self.geometry.horizontal_basis.index_length,),
                output_shape=(self.geometry.horizontal_basis.index_length,),
            )
        return self._jr_to_m_imp_operator

    @property
    def driving_E_to_m_imp_operator(self) -> LinearMap | None:
        """Map driving E to imposed potential."""
        if self._driving_E_to_m_imp_operator is None:
            if self._driving_E_to_m_imp_matrix is None:
                self._build_driving_E_to_m_imp_matrix()
            if self._driving_E_to_m_imp_matrix is None:
                return None
            self._driving_E_to_m_imp_operator = as_linear_map(
                self._driving_E_to_m_imp_matrix,
                input_shape=(2, self.geometry.horizontal_basis.index_length),
                output_shape=(self.geometry.horizontal_basis.index_length,),
            )
        return self._driving_E_to_m_imp_operator

    @property
    def driving_E_to_total_E_operator(self) -> LinearMap:
        """Map driving E to total model E."""
        if self._driving_E_to_total_E_operator is None:
            self._driving_E_to_total_E_operator = self._create_driving_E_to_total_E_operator()
        return self._driving_E_to_total_E_operator

    @property
    def driving_E_to_E_df_operator(self) -> LinearMap:
        """Map driving E to total E_df."""
        if self._driving_E_to_E_df_operator is None:
            self._driving_E_to_E_df_operator = self._create_driving_E_to_E_df_operator()
        return self._driving_E_to_E_df_operator

    def _m_imp_feedback_maps(self) -> tuple[LinearMap, LinearMap] | None:
        """Return maps for electric-field feedback through m_imp."""
        if (
            not self.config.enable_interhemispheric_coupling
            or self._interhemispheric_electric_field_constraint is None
        ):
            return None

        driving_E_to_m_imp = self.driving_E_to_m_imp_operator
        if driving_E_to_m_imp is None:
            return None

        return self.m_imp_to_E_coeffs, driving_E_to_m_imp

    def _create_driving_E_to_total_E_operator(self) -> LinearMap:
        """Construct the driving-E to total-E map."""
        identity = identity_linear_map((2, self.geometry.horizontal_basis.index_length))
        feedback = self._m_imp_feedback_maps()
        if feedback is None:
            return identity

        m_imp_to_E, driving_E_to_m_imp = feedback
        return identity + (m_imp_to_E @ driving_E_to_m_imp)

    def _create_driving_E_to_E_df_operator(self) -> LinearMap:
        """Construct the driving-E to divergence-free E map."""
        return (
            self.geometry.helmholtz_divergence_free_potential_operator
            @ self.driving_E_to_total_E_operator
        )

    def _create_driving_source_to_E_df_operator(self, source_to_driving_E: LinearMap) -> LinearMap:
        """Complete a driving source through m_imp and extract E_df."""
        total_E_from_source = source_to_driving_E
        if (
            self.config.enable_interhemispheric_coupling
            and self._interhemispheric_electric_field_constraint is not None
        ):
            problem = self._m_imp_problem
            electric_field_rhs_operator = (
                -self.config.interhemispheric_electric_field_weight
                * self.geometry.interhemispheric_electric_field_difference_operator
                @ source_to_driving_E
            )
            electric_field_rhs = electric_field_rhs_operator.to_matrix().reshape(
                problem.A[1].output_shape + source_to_driving_E.input_shape
            )
            rhs_entries = [None] * problem.num_data_terms
            rhs_entries[1] = electric_field_rhs
            m_imp_from_source = as_linear_map(
                self._solve_m_imp_response(rhs_entries),
                input_shape=source_to_driving_E.input_shape,
                output_shape=(self.geometry.horizontal_basis.index_length,),
            )
            total_E_from_source = source_to_driving_E + self.m_imp_to_E_coeffs @ m_imp_from_source
        return self.geometry.helmholtz_divergence_free_potential_operator @ total_E_from_source

    @property
    def m_ind_to_E_df_operator(self) -> LinearMap:
        """Map poloidal state to its full divergence-free E response."""
        if self._m_ind_to_E_df_operator_cache is None:
            self._m_ind_to_E_df_operator_cache = self._create_driving_source_to_E_df_operator(
                self.m_ind_to_E_coeffs
            )
        return self._m_ind_to_E_df_operator_cache

    def _solve_for_m_imp(self, jr_coeffs: np.ndarray | None, driving_E: np.ndarray) -> np.ndarray:
        """Solve for the imposed potential coefficients `m_imp`."""
        jr_coeffs = None if jr_coeffs is None else xp.asarray(jr_coeffs)
        driving_E = xp.asarray(driving_E)
        rhs_entries = self._m_imp_rhs_entries(jr_coeffs, driving_E)
        if rhs_entries is None:
            return xp.zeros(self.geometry.horizontal_basis.index_length)

        solution = self._solve_m_imp_response(rhs_entries)
        return self.project_surface_scalar_mean_free(solution)

    # ----- Active Input Update -----

    def activate_inputs_at_time(
        self, input_series: Any, time: float, interpolation: bool = False
    ) -> None:
        """Update inputs active at the requested simulation time."""
        previous_conductance_fingerprint = (
            None
            if (self.log_conductance_magnitude is None or self.log_hall_to_pedersen_ratio is None)
            else self.conductance_fingerprint
        )
        conductance_updated = False
        for key in input_series.datasets:
            updated_input = input_series.get_entry_if_changed(key, time, interpolation)
            if updated_input is None:
                continue

            field_space = input_series.get_field_space(key)
            if key == "Br" and self.geometry.RM is None:
                raise ValueError("Br input can only be set if RM is not None.")

            unknown_variables = set(updated_input) - _ACTIVE_INPUT_NAMES
            if unknown_variables:
                raise ValueError(
                    f"Unsupported active input variables: {sorted(unknown_variables)}."
                )
            for variable, coefficients in updated_input.items():
                setattr(
                    self,
                    variable,
                    FieldCoefficients(field_space, coeffs=coefficients, name=variable),
                )
            conductance_updated |= key == "conductance"

        if conductance_updated:
            self._conductance_fingerprint_cache = None
            active_conductance_fingerprint = self.conductance_fingerprint
            if active_conductance_fingerprint == previous_conductance_fingerprint:
                logger.info("Conductance coefficients unchanged: retaining closure caches.")
                return

            logger.info("Conductance updated: invalidating closure caches and problem definition.")
            preconditioner_to_keep = self._m_imp_preconditioner_cache
            preconditioner_ready_to_keep = self._m_imp_preconditioner_ready
            self._invalidate_closure_caches()
            self._conductance_fingerprint_cache = active_conductance_fingerprint
            if self.config.reuse_preconditioner and preconditioner_ready_to_keep:
                logger.info("...reusing preconditioner due to configuration.")
                self._m_imp_preconditioner_cache = preconditioner_to_keep
                self._m_imp_preconditioner_ready = True

    # ----- Response Calculation -----

    @staticmethod
    def _apply_operator(op: LinearMap | None, coeffs: Any, output_shape: tuple[int, ...]) -> Any:
        if op is None or coeffs is None:
            array_module = op.array_module(coeffs) if op is not None else get_array_module(coeffs)
            return array_module.zeros(output_shape)

        array_module = op.array_module(coeffs)
        coeffs_arr = array_module.asarray(coeffs)
        return op.matvec(coeffs_arr.reshape(-1)).reshape(output_shape)

    def _complete_electric_response(
        self, driving_E: np.ndarray, jr_coeffs: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add the m_imp response required by the model constraints."""
        driving_E = self.project_helmholtz_mean_free(driving_E)
        m_imp = self._solve_for_m_imp(jr_coeffs, driving_E)
        E_from_m_imp = self._apply_operator(
            self._runtime_m_imp_to_E_coeffs,
            m_imp,
            (2, self.geometry.horizontal_basis.index_length),
        )
        return self.project_helmholtz_mean_free(driving_E + E_from_m_imp), m_imp

    def calculate_noninductive_response(self) -> tuple[np.ndarray, np.ndarray]:
        """Return E and imposed-potential responses without m_ind."""
        E_shape = (2, self.geometry.horizontal_basis.index_length)
        active_wind_forcings = [
            name for name in ("u", "Q_eff", "E_neutral_wind") if getattr(self, name) is not None
        ]
        if len(active_wind_forcings) > 1:
            representations = ", ".join(repr(name) for name in active_wind_forcings)
            raise ValueError(
                f"Wind-forcing representations {representations} are mutually "
                "exclusive; use only one."
            )
        driving_E = xp.zeros(E_shape)
        if self.u is not None:
            driving_E += self._apply_operator(
                self.u_coeffs_to_E_coeffs, xp.asarray(self.u.array), E_shape
            )
        if self.E_neutral_wind is not None:
            driving_E += self._apply_operator(
                self.E_neutral_wind_to_E_coeffs, xp.asarray(self.E_neutral_wind.array), E_shape
            )
        if self.Br is not None:
            driving_E += self._apply_operator(
                self._runtime_Br_to_E_coeffs, xp.asarray(self.Br.array), E_shape
            )
        if self.Q_eff is not None:
            driving_E += self._apply_operator(
                self._runtime_Q_eff_to_E_coeffs, xp.asarray(self.Q_eff.array), E_shape
            )

        jr_coeffs = None if self.jr is None else xp.asarray(self.jr.array)
        return self._complete_electric_response(driving_E, jr_coeffs)

    def calculate_inductive_response(self, m_ind: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return E and imposed-potential responses caused by m_ind."""
        E_shape = (2, self.geometry.horizontal_basis.index_length)
        driving_E = self._apply_operator(
            self._runtime_m_ind_to_E_coeffs, xp.asarray(m_ind), E_shape
        )
        return self._complete_electric_response(driving_E, None)

    # ----- Induction Operators -----

    @property
    def m_ind_feedback_matrix(self) -> np.ndarray:
        """Return induced-state feedback before Faraday scaling."""
        if self._m_ind_feedback_matrix is None:
            self._build_m_ind_feedback_matrix()
        return self._m_ind_feedback_matrix

    @property
    def m_ind_feedback_operator(self) -> LinearMap:
        """Map induced state to Faraday-driving W in poloidal space."""
        if self._m_ind_feedback_operator is None:
            self._m_ind_feedback_operator = self._create_m_ind_feedback_operator()
        return self._m_ind_feedback_operator

    @property
    def noninductive_E_df_to_steady_m_ind_matrix(self) -> np.ndarray:
        """Return the explicit steady-state response matrix."""
        if self._noninductive_E_df_to_steady_m_ind_matrix is None:
            self._noninductive_E_df_to_steady_m_ind_matrix = (
                self.noninductive_E_df_to_steady_m_ind_operator.to_matrix()
            )
        return self._noninductive_E_df_to_steady_m_ind_matrix

    @property
    def noninductive_E_df_to_steady_m_ind_operator(self) -> LinearMap:
        """Linear map from non-inductive E_df to steady m_ind."""
        if self._noninductive_E_df_to_steady_m_ind_operator is None:
            if self._noninductive_E_df_to_steady_m_ind_matrix is not None:
                self._noninductive_E_df_to_steady_m_ind_operator = as_linear_map(
                    self._noninductive_E_df_to_steady_m_ind_matrix
                )
                return self._noninductive_E_df_to_steady_m_ind_operator

            array_module = get_array_module(self.m_ind_feedback_matrix)
            feedback_pinv = block_after_jax_linalg(
                array_module.linalg.pinv(self.m_ind_feedback_matrix, rtol=1e-15)
            )
            poloidal_size = self.geometry.poloidal_basis.index_length
            steady_poloidal_response = as_linear_map(
                -feedback_pinv, input_shape=(poloidal_size,), output_shape=(poloidal_size,)
            )
            self._noninductive_E_df_to_steady_m_ind_operator = (
                steady_poloidal_response @ self.geometry.surface_to_poloidal_operator
            )
        return self._noninductive_E_df_to_steady_m_ind_operator

    def _create_m_ind_feedback_operator(self) -> LinearMap:
        """Construct induced-state feedback in poloidal space."""
        return self.geometry.surface_to_poloidal_operator @ self.m_ind_to_E_df_operator

    def _build_m_ind_feedback_matrix(self) -> None:
        """Construct the dense induced-state feedback matrix."""
        logger.info("Building dense induced-state feedback matrix...")
        self._m_ind_feedback_matrix = self.m_ind_feedback_operator.to_matrix()
        logger.info("Dense induction operator built.")

    def E_df_operators(
        self,
        *,
        include_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
    ) -> dict[str, LinearMap]:
        """Return named input/state to total E_df operators."""
        operators = {
            "E_df_from_u": self.driving_E_to_E_df_operator @ self.u_coeffs_to_E_coeffs,
            "E_df_from_jr": (
                self.geometry.helmholtz_divergence_free_potential_operator
                @ self.m_imp_to_E_coeffs
                @ self.jr_to_m_imp_operator
            ),
            "E_df_from_m_ind": self.m_ind_to_E_df_operator,
        }

        if include_Br and self.Br_to_E_coeffs is not None:
            operators["E_df_from_Br"] = self.driving_E_to_E_df_operator @ self.Br_to_E_coeffs
        if include_Q_eff and self.Q_eff_to_E_coeffs is not None:
            operators["E_df_from_Q_eff"] = self.driving_E_to_E_df_operator @ self.Q_eff_to_E_coeffs
        if include_E_neutral_wind and self.E_neutral_wind_to_E_coeffs is not None:
            operators["E_df_from_neutral_wind"] = (
                self.driving_E_to_E_df_operator @ self.E_neutral_wind_to_E_coeffs
            )

        return operators

    def m_ind_rate_operators(
        self,
        *,
        include_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
    ) -> dict[str, LinearMap]:
        """Return named input/state to d(m_ind)/dt operators."""
        faraday = (
            float(self.geometry.faraday_rate_scale) * self.geometry.surface_to_poloidal_operator
        )
        return {
            key.replace("E_df_from_", "d_m_ind_dt_from_"): faraday @ operator
            for key, operator in self.E_df_operators(
                include_Br=include_Br,
                include_Q_eff=include_Q_eff,
                include_E_neutral_wind=include_E_neutral_wind,
            ).items()
        }

    def E_df_matrices(
        self,
        *,
        include_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
        backend: MatrixBackend | None = None,
    ) -> dict[str, Any]:
        """Return E_df maps as explicit matrices."""
        return {
            key: operator.to_matrix(backend=backend)
            for key, operator in self.E_df_operators(
                include_Br=include_Br,
                include_Q_eff=include_Q_eff,
                include_E_neutral_wind=include_E_neutral_wind,
            ).items()
        }

    def m_ind_rate_matrices(
        self,
        *,
        include_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
        backend: MatrixBackend | None = None,
    ) -> dict[str, Any]:
        """Return d(m_ind)/dt maps as explicit matrices."""
        return {
            key: operator.to_matrix(backend=backend)
            for key, operator in self.m_ind_rate_operators(
                include_Br=include_Br,
                include_Q_eff=include_Q_eff,
                include_E_neutral_wind=include_E_neutral_wind,
            ).items()
        }


__all__ = ["ElectrodynamicResponse"]
