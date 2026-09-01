"""Instantaneous electrodynamic response model."""

from __future__ import annotations

import logging
from typing import Any

from kompe.math import (
    ArrayBackend,
    LeastSquaresProblem,
    LeastSquaresSolver,
    LinearMap,
    as_linear_map,
    content_fingerprint,
    get_array_module,
    identity_linear_map,
    synchronize_linalg_result,
    to_numpy,
)

from pynamit.fields import FieldCoefficients
from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.geometry import SimulationGeometry

logger = logging.getLogger(__name__)


class ElectrodynamicResponse:
    """Compile and evaluate the instantaneous electrodynamic response.

    The model stores inputs active at one simulation time and assembles
    their resistance-dependent response operators. ``_TimeEvolution``
    holds the evolving ``induced_Br`` trajectory and passes it into
    response calculations. Poloidal and toroidal magnetic potentials
    remain private numerical coordinates.

    The tangential electric field uses the Helmholtz convention
    ``E = -grad(Phi) + rhat x grad(W)``; ``W`` is therefore the
    divergence-free electric-potential coefficient vector that drives
    Faraday induction.

    Time integration belongs to
    ``electrodynamics.induction``; evolution scheduling belongs to
    ``_TimeEvolution``.
    """

    def __init__(self, geometry: SimulationGeometry, config: SimulationConfig) -> None:
        """Initialize the response model."""
        self.config = config
        self.geometry = geometry
        self._toroidal_potential_solver = LeastSquaresSolver(
            solver=config.least_squares_solver, preconditioner=config.least_squares_preconditioner
        )

        # Operator for mapping velocity field `u` to E-field
        # (independent of resistance)
        self._u_coeffs_to_E_coeffs_operator_cache: LinearMap | None = None
        self._Q_eff_synthesis_operator_cache: LinearMap | None = None
        self._E_neutral_wind_to_E_coeffs_operator_cache: LinearMap | None = None

        # Inputs active at the current simulation time.
        self.u: FieldCoefficients | None = None
        self.Q_eff: FieldCoefficients | None = None
        self.E_neutral_wind: FieldCoefficients | None = None
        self.boundary_Br: FieldCoefficients | None = None
        self.boundary_jr: FieldCoefficients | None = None
        self.log_conductance_magnitude: FieldCoefficients | None = None
        self.log_hall_to_pedersen_ratio: FieldCoefficients | None = None

        # Initialize closure-dependent caches.
        self._invalidate_closure_caches()

    def __repr__(self):
        """Summarize inputs without materializing operators."""
        input_names = (
            "u",
            "Q_eff",
            "E_neutral_wind",
            "boundary_Br",
            "boundary_jr",
            "log_conductance_magnitude",
            "log_hall_to_pedersen_ratio",
        )
        active = ", ".join(name for name in input_names if getattr(self, name) is not None)
        return (
            f"ElectrodynamicResponse(horizontal_basis={self.geometry.horizontal_basis!r}, "
            f"active_inputs=[{active or 'none'}])"
        )

    @property
    def u_coeffs_to_E_coeffs_operator(self) -> LinearMap:
        """Linear map from wind coefficients to E coefficients."""
        if self._u_coeffs_to_E_coeffs_operator_cache is None:
            self._u_coeffs_to_E_coeffs_operator_cache = (
                ionospheric_closure.wind_to_E_coeffs_operator(
                    self.geometry.helmholtz_analysis_operator,
                    self.geometry.wind_motional_E_tensor,
                    self.geometry.horizontal_transform.helmholtz_synthesis_operator,
                )
            )
        return self._u_coeffs_to_E_coeffs_operator_cache

    @property
    def Q_eff_to_E_coeffs_operator(self) -> LinearMap | None:
        """Linear map from effective-current coeffs to E coeffs."""
        if self.Q_eff is None:
            return None
        if self._Q_eff_to_E_coeffs_operator_cache is None:
            if self._Q_eff_synthesis_operator_cache is None:
                basis = self.Q_eff.field_space.basis
                self._Q_eff_synthesis_operator_cache = basis.helmholtz_synthesis_operator(
                    self.geometry.model_grid
                )
            resistance_tensor = self.resistance_tensor_on_grid
            xp = get_array_module(resistance_tensor)
            self._Q_eff_to_E_coeffs_operator_cache = (
                ionospheric_closure.tangential_current_to_E_coeffs_operator(
                    self.geometry.helmholtz_analysis_operator,
                    xp.asarray(resistance_tensor),
                    self._Q_eff_synthesis_operator_cache,
                )
            )
        return self._Q_eff_to_E_coeffs_operator_cache

    @property
    def E_neutral_wind_to_E_coeffs_operator(self) -> LinearMap | None:
        """Map neutral-wind E coefficients to model E coefficients."""
        if self.E_neutral_wind is None:
            return None
        if self._E_neutral_wind_to_E_coeffs_operator_cache is None:
            basis = self.E_neutral_wind.field_space.basis
            if basis.coefficients_are_compatible_with(self.geometry.horizontal_basis):
                self._E_neutral_wind_to_E_coeffs_operator_cache = identity_linear_map(
                    (2, self.geometry.horizontal_basis.index_length)
                )
            else:
                self._E_neutral_wind_to_E_coeffs_operator_cache = (
                    self.geometry.helmholtz_analysis_operator
                    @ basis.helmholtz_synthesis_operator(self.geometry.model_grid)
                )
        return self._E_neutral_wind_to_E_coeffs_operator_cache

    def _invalidate_closure_caches(self) -> None:
        """Invalidate resistance-dependent cached properties."""
        self._conductance_fingerprint_cache: str | None = None
        self._resistance_tensor_on_grid: Any | None = None
        self._induced_poloidal_potential_to_E_coeffs_operator_cache: LinearMap | None = None
        self._induced_Br_to_E_coeffs_operator_cache: LinearMap | None = None
        self._toroidal_potential_to_E_coeffs_operator_cache: LinearMap | None = None
        self._boundary_Br_to_E_coeffs_operator_cache: LinearMap | None = None
        self._Q_eff_to_E_coeffs_operator_cache: LinearMap | None = None
        self._runtime_induced_Br_to_E_coeffs_cache: LinearMap | None = None
        self._runtime_toroidal_potential_to_E_coeffs_cache: LinearMap | None = None
        self._runtime_boundary_Br_to_E_coeffs_cache: LinearMap | None = None
        self._runtime_Q_eff_to_E_coeffs_cache: LinearMap | None = None
        self._interhemispheric_electric_field_constraint_cache: LinearMap | None = None
        self._induced_poloidal_potential_feedback_operator: LinearMap | None = None
        self._induced_poloidal_potential_to_W_operator_cache: LinearMap | None = None
        self._induced_Br_to_W_operator_cache: LinearMap | None = None
        self._noninductive_W_to_equilibrium_induced_poloidal_potential_operator: (
            LinearMap | None
        ) = None
        self._noninductive_W_to_equilibrium_induced_Br_operator: LinearMap | None = None
        self._boundary_jr_to_toroidal_potential_operator: LinearMap | None = None
        self._driving_E_to_toroidal_potential_operator: LinearMap | None = None
        self._driving_E_to_total_E_operator: LinearMap | None = None
        self._driving_E_to_W_operator: LinearMap | None = None
        self._toroidal_potential_problem_cache: LeastSquaresProblem | None = None
        self._toroidal_potential_response_solver_cache = None
        self._toroidal_potential_preconditioner_cache: LinearMap | None = None
        self._toroidal_potential_preconditioner_ready = False

    # ----- Cached Physical Properties (dependent on resistance) -----

    @staticmethod
    def _fingerprint_conductance(field_space: Any, log_magnitude: Any, log_ratio: Any) -> str:
        """Fingerprint stored conductance coefficients."""
        return content_fingerprint(
            {
                "field_space": field_space.signature,
                "log_conductance_magnitude": log_magnitude,
                "log_hall_to_pedersen_ratio": log_ratio,
            }
        )

    @property
    def conductance_fingerprint(self) -> str:
        """Return the exact identity of the active conductance field."""
        if self.log_conductance_magnitude is None or self.log_hall_to_pedersen_ratio is None:
            raise RuntimeError(
                "Resistance or conductance must be set before it can be fingerprinted."
            )
        if self._conductance_fingerprint_cache is None:
            self._conductance_fingerprint_cache = self._fingerprint_conductance(
                self.log_conductance_magnitude.field_space,
                to_numpy(self.log_conductance_magnitude.array),
                to_numpy(self.log_hall_to_pedersen_ratio.array),
            )
        return self._conductance_fingerprint_cache

    @property
    def resistance_tensor_on_grid(self) -> Any:
        """Resistance tensor on the spatial grid."""
        if self._resistance_tensor_on_grid is None:
            if self.log_conductance_magnitude is None or self.log_hall_to_pedersen_ratio is None:
                raise RuntimeError(
                    "Resistance or conductance must be set before accessing "
                    "closure-dependent properties."
                )
            magnitude_coefficients = self.log_conductance_magnitude.array
            ratio_coefficients = self.log_hall_to_pedersen_ratio.array
            xp = get_array_module(magnitude_coefficients, ratio_coefficients)
            log_coordinate_coefficients = xp.stack(
                [xp.asarray(magnitude_coefficients), xp.asarray(ratio_coefficients)], axis=1
            )
            magnitude_basis = self.log_conductance_magnitude.field_space.basis
            ratio_basis = self.log_hall_to_pedersen_ratio.field_space.basis
            if (
                ratio_basis is not magnitude_basis
                and not ratio_basis.coefficients_are_compatible_with(magnitude_basis)
            ):
                raise ValueError(
                    "Conductance log-coordinate storage bases must be coefficient-compatible."
                )
            conductance_synthesis = magnitude_basis.scalar_evaluation_operator(
                self.geometry.model_grid
            )
            log_coordinates_on_grid = xp.asarray(
                conductance_synthesis.matmat(log_coordinate_coefficients)
            )
            etaP, etaH = ionospheric_closure.resistance_from_log_conductance_coordinates(
                log_coordinates_on_grid[:, 0], log_coordinates_on_grid[:, 1]
            )
            self._resistance_tensor_on_grid = ionospheric_closure.resistance_tensor_on_grid(
                etaP,
                etaH,
                self.geometry.pedersen_geometry_tensor,
                self.geometry.hall_geometry_tensor,
            )
        return self._resistance_tensor_on_grid

    def _sheet_current_source_to_E_coeffs_operator(self, source_to_JS: LinearMap) -> LinearMap:
        """Map a magnetic source through derived sheet current to E."""
        return ionospheric_closure.tangential_current_to_E_coeffs_operator(
            self.geometry.helmholtz_analysis_operator, self.resistance_tensor_on_grid, source_to_JS
        )

    @property
    def induced_poloidal_potential_to_E_coeffs_operator(self) -> LinearMap:
        """Map private induced-potential coordinates to E."""
        if self._induced_poloidal_potential_to_E_coeffs_operator_cache is None:
            potential_to_sheet_current = (
                self.geometry.induced_Br_to_gridded_JS_operator()
                @ self.geometry.induced_poloidal_potential_to_Br_operator
            )
            self._induced_poloidal_potential_to_E_coeffs_operator_cache = (
                self._sheet_current_source_to_E_coeffs_operator(potential_to_sheet_current)
            )
        return self._induced_poloidal_potential_to_E_coeffs_operator_cache

    @property
    def induced_Br_to_E_coeffs_operator(self) -> LinearMap:
        """Map physical induced ``Br(RI)`` coefficients to E."""
        if self._induced_Br_to_E_coeffs_operator_cache is None:
            self._induced_Br_to_E_coeffs_operator_cache = (
                self.induced_poloidal_potential_to_E_coeffs_operator
                @ self.geometry.induced_Br_to_poloidal_potential_operator
            )
        return self._induced_Br_to_E_coeffs_operator_cache

    @property
    def toroidal_potential_to_E_coeffs_operator(self) -> LinearMap:
        """Map private toroidal-potential coefficients to E."""
        if self._toroidal_potential_to_E_coeffs_operator_cache is None:
            self._toroidal_potential_to_E_coeffs_operator_cache = (
                self._sheet_current_source_to_E_coeffs_operator(
                    self.geometry.toroidal_potential_to_gridded_JS_operator()
                )
            )
        return self._toroidal_potential_to_E_coeffs_operator_cache

    @property
    def boundary_Br_to_E_coeffs_operator(self) -> LinearMap | None:
        """Map prescribed outer-boundary Br to E coefficients."""
        if self._boundary_Br_to_E_coeffs_operator_cache is None:
            boundary_Br_to_JS = self.geometry.boundary_Br_to_gridded_JS_operator()
            if boundary_Br_to_JS is None:
                return None
            self._boundary_Br_to_E_coeffs_operator_cache = (
                self._sheet_current_source_to_E_coeffs_operator(boundary_Br_to_JS)
            )
        return self._boundary_Br_to_E_coeffs_operator_cache

    def _prepare_repeated_E_operator(
        self, op: LinearMap | None, *, compact_input: bool
    ) -> LinearMap | None:
        """Use an explicit array when it reduces repeated work."""
        if op is None:
            return None
        spaces_coincide = self.geometry.horizontal_basis is self.geometry.poloidal_basis
        if not (compact_input or spaces_coincide) or op.is_diagonal:
            return op
        return as_linear_map(
            op.to_array(), input_shape=op.input_shape, output_shape=op.output_shape
        )

    @property
    def _runtime_induced_Br_to_E_coeffs(self) -> LinearMap:
        """Runtime map from physical induced Br to E coefficients."""
        if self._runtime_induced_Br_to_E_coeffs_cache is None:
            self._runtime_induced_Br_to_E_coeffs_cache = self._prepare_repeated_E_operator(
                self.induced_Br_to_E_coeffs_operator, compact_input=True
            )
        return self._runtime_induced_Br_to_E_coeffs_cache

    @property
    def _runtime_toroidal_potential_to_E_coeffs(self) -> LinearMap:
        """Runtime map from toroidal potential to E coefficients."""
        if self._runtime_toroidal_potential_to_E_coeffs_cache is None:
            self._runtime_toroidal_potential_to_E_coeffs_cache = self._prepare_repeated_E_operator(
                self.toroidal_potential_to_E_coeffs_operator, compact_input=False
            )
        return self._runtime_toroidal_potential_to_E_coeffs_cache

    @property
    def _runtime_boundary_Br_to_E_coeffs(self) -> LinearMap | None:
        """Runtime map from Br coefficients to E coefficients."""
        if self._runtime_boundary_Br_to_E_coeffs_cache is None:
            self._runtime_boundary_Br_to_E_coeffs_cache = self._prepare_repeated_E_operator(
                self.boundary_Br_to_E_coeffs_operator, compact_input=True
            )
        return self._runtime_boundary_Br_to_E_coeffs_cache

    @property
    def _runtime_Q_eff_to_E_coeffs(self) -> LinearMap | None:
        """Runtime map from effective-current coefficients to E."""
        if self._runtime_Q_eff_to_E_coeffs_cache is None:
            self._runtime_Q_eff_to_E_coeffs_cache = self._prepare_repeated_E_operator(
                self.Q_eff_to_E_coeffs_operator, compact_input=False
            )
        return self._runtime_Q_eff_to_E_coeffs_cache

    @property
    def _interhemispheric_electric_field_constraint(self) -> LinearMap | None:
        """Linear map enforcing the E-field low-latitude constraint."""
        if self._interhemispheric_electric_field_constraint_cache is None:
            outer_map = self.geometry.interhemispheric_electric_field_difference_operator
            if outer_map is not None:
                self._interhemispheric_electric_field_constraint_cache = (
                    outer_map @ self.toroidal_potential_to_E_coeffs_operator
                )
        return self._interhemispheric_electric_field_constraint_cache

    # ----- Solver Setup and Execution -----
    @property
    def _toroidal_potential_problem(self) -> LeastSquaresProblem:
        """Return the toroidal-potential least-squares problem."""
        if self._toroidal_potential_problem_cache is None:
            logger.info("Defining new toroidal-potential least-squares problem.")
            operators, data_shapes = [], []

            # Boundary radial current must match the prescribed field.
            radial_current_operator = (
                self.geometry.radial_current_constraint_operator
                @ self.geometry.toroidal_potential_to_boundary_jr_operator
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
            if self.config.toroidal_potential_regularization_lambda > 0:
                n = self.geometry.horizontal_basis.index_length
                reg_ops.append(identity_linear_map((n,)))
                reg_weights.append(self.config.toroidal_potential_regularization_lambda)

            self._toroidal_potential_problem_cache = LeastSquaresProblem(
                A=operators,
                solution_shape=self.geometry.horizontal_basis.index_length,
                data_shapes=data_shapes,
                regularization_operators=reg_ops,
                regularization_strengths=reg_weights,
            )
        return self._toroidal_potential_problem_cache

    @property
    def _toroidal_potential_preconditioner(self) -> LinearMap | None:
        """Return the toroidal-potential preconditioner."""
        if self._toroidal_potential_solver.solver not in ("lsmr", "cgls"):
            return None
        if not self._toroidal_potential_preconditioner_ready:
            logger.info("Building new toroidal-potential solver preconditioner.")
            self._toroidal_potential_preconditioner_cache = (
                self._toroidal_potential_solver.build_preconditioner(
                    problem=self._toroidal_potential_problem
                )
            )
            self._toroidal_potential_preconditioner_ready = True
        return self._toroidal_potential_preconditioner_cache

    def _solve_toroidal_potential_response(self, rhs_entries):
        """Solve toroidal-potential response right-hand sides."""
        if self._toroidal_potential_response_solver_cache is None:
            self._toroidal_potential_response_solver_cache = (
                self._toroidal_potential_solver.build_response_solver(
                    self._toroidal_potential_problem,
                    preconditioner=self._toroidal_potential_preconditioner,
                )
            )
        return self._toroidal_potential_response_solver_cache(rhs_entries)

    def _toroidal_potential_rhs_entries(
        self, boundary_jr_coeffs: Any | None, driving_E: Any
    ) -> list[Any | None] | None:
        """Assemble the physical right-hand sides for one solve."""
        problem = self._toroidal_potential_problem
        rhs_entries = [None] * len(problem.data_operators)
        has_rhs = False

        if boundary_jr_coeffs is not None:
            rhs_entries[0] = self.geometry.radial_current_constraint_operator.matvec(
                boundary_jr_coeffs
            )
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
    def boundary_jr_to_toroidal_potential_operator(self) -> LinearMap:
        """Map boundary current to the private toroidal potential."""
        if self._boundary_jr_to_toroidal_potential_operator is None:
            logger.info("Building dense boundary-jr to toroidal-potential response matrix.")
            problem = self._toroidal_potential_problem
            radial_current_array = self.geometry.radial_current_constraint_operator.to_array()
            xp = get_array_module(radial_current_array)
            radial_current_rhs = xp.asarray(radial_current_array).reshape(
                problem.data_operators[0].output_shape + (-1,)
            )
            rhs_entries = [None] * len(problem.data_operators)
            rhs_entries[0] = radial_current_rhs
            response = self._solve_toroidal_potential_response(rhs_entries)
            self._boundary_jr_to_toroidal_potential_operator = as_linear_map(
                response,
                input_shape=(self.geometry.horizontal_basis.index_length,),
                output_shape=(self.geometry.horizontal_basis.index_length,),
            )
        return self._boundary_jr_to_toroidal_potential_operator

    @property
    def driving_E_to_toroidal_potential_operator(self) -> LinearMap | None:
        """Map driving E to the private toroidal potential."""
        if (
            not self.config.enable_interhemispheric_coupling
            or self._interhemispheric_electric_field_constraint is None
        ):
            return None
        if self._driving_E_to_toroidal_potential_operator is None:
            logger.info("Building dense driving-E-to-toroidal-potential response matrix.")
            n = self.geometry.horizontal_basis.index_length
            problem = self._toroidal_potential_problem
            electric_field_rhs = (
                -self.geometry.interhemispheric_electric_field_difference_array.reshape(
                    problem.data_operators[1].output_shape + (2 * n,)
                )
            )
            electric_field_rhs *= self.config.interhemispheric_electric_field_weight
            rhs_entries = [None] * len(problem.data_operators)
            rhs_entries[1] = electric_field_rhs
            response = self._solve_toroidal_potential_response(rhs_entries)
            self._driving_E_to_toroidal_potential_operator = as_linear_map(
                response,
                input_shape=(2, self.geometry.horizontal_basis.index_length),
                output_shape=(self.geometry.horizontal_basis.index_length,),
            )
        return self._driving_E_to_toroidal_potential_operator

    @property
    def driving_E_to_total_E_operator(self) -> LinearMap:
        """Map driving E to total model E."""
        if self._driving_E_to_total_E_operator is None:
            identity = identity_linear_map((2, self.geometry.horizontal_basis.index_length))
            if (
                not self.config.enable_interhemispheric_coupling
                or self._interhemispheric_electric_field_constraint is None
            ):
                self._driving_E_to_total_E_operator = identity
            else:
                driving_E_to_toroidal = self.driving_E_to_toroidal_potential_operator
                self._driving_E_to_total_E_operator = (
                    identity + self.toroidal_potential_to_E_coeffs_operator @ driving_E_to_toroidal
                )
        return self._driving_E_to_total_E_operator

    @property
    def driving_E_to_W_operator(self) -> LinearMap:
        """Map driving-E coefficients to total ``W`` coefficients."""
        if self._driving_E_to_W_operator is None:
            self._driving_E_to_W_operator = (
                self.geometry.helmholtz_divergence_free_potential_operator
                @ self.driving_E_to_total_E_operator
            )
        return self._driving_E_to_W_operator

    def _create_driving_source_to_W_operator(self, source_to_driving_E: LinearMap) -> LinearMap:
        """Complete a driving source and extract divergence-free ``W``.

        The returned coefficients use the horizontal basis.
        """
        total_E_from_source = source_to_driving_E
        if (
            self.config.enable_interhemispheric_coupling
            and self._interhemispheric_electric_field_constraint is not None
        ):
            problem = self._toroidal_potential_problem
            electric_field_rhs_operator = (
                -self.config.interhemispheric_electric_field_weight
                * self.geometry.interhemispheric_electric_field_difference_operator
                @ source_to_driving_E
            )
            electric_field_rhs = electric_field_rhs_operator.to_array()
            rhs_entries = [None] * len(problem.data_operators)
            rhs_entries[1] = electric_field_rhs
            toroidal_potential_from_source = as_linear_map(
                self._solve_toroidal_potential_response(rhs_entries),
                input_shape=source_to_driving_E.input_shape,
                output_shape=(self.geometry.horizontal_basis.index_length,),
            )
            total_E_from_source = (
                source_to_driving_E
                + self.toroidal_potential_to_E_coeffs_operator @ toroidal_potential_from_source
            )
        return self.geometry.helmholtz_divergence_free_potential_operator @ total_E_from_source

    @property
    def induced_poloidal_potential_to_W_operator(self) -> LinearMap:
        """Map private induced potential to total ``W`` coefficients."""
        if self._induced_poloidal_potential_to_W_operator_cache is None:
            self._induced_poloidal_potential_to_W_operator_cache = (
                self._create_driving_source_to_W_operator(
                    self.induced_poloidal_potential_to_E_coeffs_operator
                )
            )
        return self._induced_poloidal_potential_to_W_operator_cache

    @property
    def induced_Br_to_W_operator(self) -> LinearMap:
        """Map physical induced Br to total ``W`` coefficients."""
        if self._induced_Br_to_W_operator_cache is None:
            self._induced_Br_to_W_operator_cache = (
                self.induced_poloidal_potential_to_W_operator
                @ self.geometry.induced_Br_to_poloidal_potential_operator
            )
        return self._induced_Br_to_W_operator_cache

    def _solve_for_toroidal_potential(self, boundary_jr_coeffs: Any | None, driving_E: Any) -> Any:
        """Solve the private toroidal-potential response."""
        xp = get_array_module(boundary_jr_coeffs, driving_E)
        boundary_jr_coeffs = None if boundary_jr_coeffs is None else xp.asarray(boundary_jr_coeffs)
        driving_E = xp.asarray(driving_E)
        rhs_entries = self._toroidal_potential_rhs_entries(boundary_jr_coeffs, driving_E)
        if rhs_entries is None:
            return xp.zeros(self.geometry.horizontal_basis.index_length)

        solution = self._solve_toroidal_potential_response(rhs_entries)
        return self.geometry.horizontal_basis.project_scalar_mean_free(solution)

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
        active_conductance_fingerprint = None
        for key in input_series.datasets:
            updated_input = input_series.get_entry_if_changed(key, time, interpolation)
            if updated_input is None:
                continue

            field_space = input_series.get_field_space(key)
            if key == "conductance":
                active_conductance_fingerprint = self._fingerprint_conductance(
                    field_space,
                    updated_input["log_conductance_magnitude"],
                    updated_input["log_hall_to_pedersen_ratio"],
                )
            for variable, coefficients in updated_input.items():
                setattr(
                    self,
                    variable,
                    FieldCoefficients(field_space, coeffs=coefficients, name=variable),
                )

        if active_conductance_fingerprint is not None:
            self._conductance_fingerprint_cache = active_conductance_fingerprint
            if active_conductance_fingerprint == previous_conductance_fingerprint:
                logger.info("Conductance coefficients unchanged: retaining closure caches.")
                return

            logger.info("Conductance updated: invalidating closure caches and problem definition.")
            preconditioner_to_keep = self._toroidal_potential_preconditioner_cache
            preconditioner_ready_to_keep = self._toroidal_potential_preconditioner_ready
            self._invalidate_closure_caches()
            self._conductance_fingerprint_cache = active_conductance_fingerprint
            if self.config.reuse_preconditioner and preconditioner_ready_to_keep:
                logger.info("...reusing preconditioner due to configuration.")
                self._toroidal_potential_preconditioner_cache = preconditioner_to_keep
                self._toroidal_potential_preconditioner_ready = True

    # ----- Response Calculation -----

    @staticmethod
    def _apply_operator(op: LinearMap, coeffs: Any, output_shape: tuple[int, ...]) -> Any:
        """Apply a required response operator and restore its shape."""
        array_module = op.array_module(coeffs)
        coeffs_arr = array_module.asarray(coeffs)
        return op.matvec(coeffs_arr.reshape(-1)).reshape(output_shape)

    def _solve_electric_closure(
        self, driving_E: Any, boundary_jr_coeffs: Any | None
    ) -> tuple[Any, Any]:
        """Complete E and return the resulting boundary current."""
        driving_E = self.geometry.horizontal_basis.project_helmholtz_mean_free(driving_E)
        toroidal_potential = self._solve_for_toroidal_potential(boundary_jr_coeffs, driving_E)
        E_from_toroidal_potential = self._apply_operator(
            self._runtime_toroidal_potential_to_E_coeffs,
            toroidal_potential,
            (2, self.geometry.horizontal_basis.index_length),
        )
        solved_boundary_jr = self.geometry.toroidal_potential_to_boundary_jr_operator.matvec(
            toroidal_potential
        )
        return (
            self.geometry.horizontal_basis.project_helmholtz_mean_free(
                driving_E + E_from_toroidal_potential
            ),
            solved_boundary_jr,
        )

    def solve_noninductive_response(self) -> tuple[Any, Any]:
        """Return E and boundary-jr responses without induced Br."""
        E_shape = (2, self.geometry.horizontal_basis.index_length)
        active_arrays = [
            field.array
            for field in (
                self.u,
                self.Q_eff,
                self.E_neutral_wind,
                self.boundary_Br,
                self.boundary_jr,
                self.log_conductance_magnitude,
                self.log_hall_to_pedersen_ratio,
            )
            if field is not None
        ]
        xp = get_array_module(*active_arrays)
        driving_E = xp.zeros(E_shape)
        if self.u is not None:
            driving_E += self._apply_operator(
                self.u_coeffs_to_E_coeffs_operator, xp.asarray(self.u.array), E_shape
            )
        if self.E_neutral_wind is not None:
            driving_E += self._apply_operator(
                self.E_neutral_wind_to_E_coeffs_operator,
                xp.asarray(self.E_neutral_wind.array),
                E_shape,
            )
        if self.boundary_Br is not None:
            driving_E += self._apply_operator(
                self._runtime_boundary_Br_to_E_coeffs, xp.asarray(self.boundary_Br.array), E_shape
            )
        if self.Q_eff is not None:
            driving_E += self._apply_operator(
                self._runtime_Q_eff_to_E_coeffs, xp.asarray(self.Q_eff.array), E_shape
            )

        boundary_jr_coeffs = (
            None if self.boundary_jr is None else xp.asarray(self.boundary_jr.array)
        )
        return self._solve_electric_closure(driving_E, boundary_jr_coeffs)

    def solve_induced_response(self, induced_Br: Any) -> tuple[Any, Any]:
        """Return E and boundary-jr responses caused by induced Br."""
        xp = get_array_module(induced_Br)
        E_shape = (2, self.geometry.horizontal_basis.index_length)
        driving_E = self._apply_operator(
            self._runtime_induced_Br_to_E_coeffs, xp.asarray(induced_Br), E_shape
        )
        return self._solve_electric_closure(driving_E, None)

    # ----- Induction Operators -----

    @property
    def induced_poloidal_potential_feedback_matrix(self) -> Any:
        """Return private feedback before Faraday scaling."""
        return self.induced_poloidal_potential_feedback_operator.to_matrix()

    @property
    def induced_poloidal_potential_feedback_operator(self) -> LinearMap:
        """Map private induced potential to Faraday-driving W."""
        if self._induced_poloidal_potential_feedback_operator is None:
            self._induced_poloidal_potential_feedback_operator = (
                self.geometry.surface_to_poloidal_operator
                @ self.induced_poloidal_potential_to_W_operator
            )
        return self._induced_poloidal_potential_feedback_operator

    @property
    def noninductive_W_to_equilibrium_induced_poloidal_potential_matrix(self) -> Any:
        """Return the private equilibrium potential response."""
        return self.noninductive_W_to_equilibrium_induced_poloidal_potential_operator.to_matrix()

    @property
    def noninductive_W_to_equilibrium_induced_poloidal_potential_operator(self) -> LinearMap:
        """Map non-inductive W to private equilibrium potential."""
        if self._noninductive_W_to_equilibrium_induced_poloidal_potential_operator is None:
            feedback_matrix = self.induced_poloidal_potential_feedback_matrix
            array_module = get_array_module(feedback_matrix)
            feedback_pinv = synchronize_linalg_result(
                array_module.linalg.pinv(feedback_matrix, rtol=1e-15)
            )
            poloidal_size = self.geometry.poloidal_basis.index_length
            equilibrium_poloidal_response = as_linear_map(
                -feedback_pinv, input_shape=(poloidal_size,), output_shape=(poloidal_size,)
            )
            self._noninductive_W_to_equilibrium_induced_poloidal_potential_operator = (
                equilibrium_poloidal_response @ self.geometry.surface_to_poloidal_operator
            )
        return self._noninductive_W_to_equilibrium_induced_poloidal_potential_operator

    @property
    def noninductive_W_to_equilibrium_induced_Br_operator(self) -> LinearMap:
        """Map non-inductive W to equilibrium induced Br."""
        if self._noninductive_W_to_equilibrium_induced_Br_operator is None:
            self._noninductive_W_to_equilibrium_induced_Br_operator = (
                self.geometry.induced_poloidal_potential_to_Br_operator
                @ self.noninductive_W_to_equilibrium_induced_poloidal_potential_operator
            )
        return self._noninductive_W_to_equilibrium_induced_Br_operator

    def source_to_W_operators(
        self,
        *,
        include_boundary_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
    ) -> dict[str, LinearMap]:
        """Return physical-source maps to total ``W`` coefficients."""
        operators = {
            "u": self.driving_E_to_W_operator @ self.u_coeffs_to_E_coeffs_operator,
            "boundary_jr": (
                self.geometry.helmholtz_divergence_free_potential_operator
                @ self.toroidal_potential_to_E_coeffs_operator
                @ self.boundary_jr_to_toroidal_potential_operator
            ),
            "induced_Br": self.induced_Br_to_W_operator,
        }

        if include_boundary_Br and self.boundary_Br_to_E_coeffs_operator is not None:
            operators["boundary_Br"] = (
                self.driving_E_to_W_operator @ self.boundary_Br_to_E_coeffs_operator
            )
        if include_Q_eff and self.Q_eff_to_E_coeffs_operator is not None:
            operators["Q_eff"] = self.driving_E_to_W_operator @ self.Q_eff_to_E_coeffs_operator
        if include_E_neutral_wind and self.E_neutral_wind_to_E_coeffs_operator is not None:
            operators["E_neutral_wind"] = (
                self.driving_E_to_W_operator @ self.E_neutral_wind_to_E_coeffs_operator
            )

        return operators

    def source_to_induced_Br_rate_operators(
        self,
        *,
        include_boundary_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
    ) -> dict[str, LinearMap]:
        """Return named physical-source maps to ``d(induced_Br)/dt``."""
        faraday = (
            float(self.geometry.induced_poloidal_potential_faraday_rate_scale)
            * self.geometry.induced_poloidal_potential_to_Br_operator
            @ self.geometry.surface_to_poloidal_operator
        )
        return {
            source: faraday @ operator
            for source, operator in self.source_to_W_operators(
                include_boundary_Br=include_boundary_Br,
                include_Q_eff=include_Q_eff,
                include_E_neutral_wind=include_E_neutral_wind,
            ).items()
        }

    def source_to_W_matrices(
        self,
        *,
        include_boundary_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
        backend: ArrayBackend | None = None,
    ) -> dict[str, Any]:
        """Return W maps as explicit matrices."""
        return {
            key: operator.to_matrix(backend=backend)
            for key, operator in self.source_to_W_operators(
                include_boundary_Br=include_boundary_Br,
                include_Q_eff=include_Q_eff,
                include_E_neutral_wind=include_E_neutral_wind,
            ).items()
        }

    def source_to_induced_Br_rate_matrices(
        self,
        *,
        include_boundary_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
        backend: ArrayBackend | None = None,
    ) -> dict[str, Any]:
        """Return ``d(induced_Br)/dt`` maps as explicit matrices."""
        return {
            key: operator.to_matrix(backend=backend)
            for key, operator in self.source_to_induced_Br_rate_operators(
                include_boundary_Br=include_boundary_Br,
                include_Q_eff=include_Q_eff,
                include_E_neutral_wind=include_E_neutral_wind,
            ).items()
        }


__all__ = ["ElectrodynamicResponse"]
