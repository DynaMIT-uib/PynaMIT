"""Instantaneous electrodynamic response model."""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np
from kompe.math import (
    LeastSquaresProblem,
    LeastSquaresSolver,
    LinearMap,
    as_linear_map,
    content_fingerprint,
    get_array_module,
    identity_linear_map,
    relative_regularization,
    synchronize_linalg_result,
    vstack_linear_maps,
)

from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.geometry import SimulationGeometry

logger = logging.getLogger(__name__)


class ElectrodynamicResponse:
    """Compile and evaluate the instantaneous electrodynamic response.

    Geometry and conductance are fixed for this response. Boundary
    forcing and winds are explicit arguments to its calculations;
    ``_TimeEvolution`` owns input selection and the evolving trajectory.
    Poloidal and toroidal magnetic potentials remain private numerical
    coordinates.

    The tangential electric field uses the Helmholtz convention
    ``E = -grad(Phi) + rhat x grad(W)``; ``W`` is therefore the
    divergence-free electric-potential coefficient vector that drives
    Faraday induction.

    Time integration belongs to
    ``electrodynamics.induction``; evolution scheduling belongs to
    ``_TimeEvolution``.
    """

    def __init__(
        self,
        geometry: SimulationGeometry,
        config: SimulationConfig,
        *,
        conductance_space=None,
        conductance_values=None,
    ) -> None:
        """Initialize the response model."""
        self.config = config
        self.geometry = geometry
        self._toroidal_potential_solver = LeastSquaresSolver(
            method=config.least_squares_solver,
            tolerance=config.least_squares_tolerance,
            preconditioner=config.least_squares_preconditioner,
        )

        # Own the CPU snapshot used for exact closure identity.
        # Numerical coefficients below are independent backend values.
        if conductance_values is not None:
            conductance_values = {
                name: np.array(values, copy=True) for name, values in conductance_values.items()
            }
            for values in conductance_values.values():
                if values.size != conductance_space.size:
                    raise ValueError("A conductance response requires one field per coordinate.")
                values.setflags(write=False)
        self._conductance_space = conductance_space
        self._conductance_values = conductance_values
        self.log_conductance_magnitude = (
            None
            if conductance_values is None
            else conductance_space.project_mean_free(
                conductance_values["log_conductance_magnitude"]
            ).reshape(conductance_space.shape)
        )
        self.log_hall_to_pedersen_ratio = (
            None
            if conductance_values is None
            else conductance_space.project_mean_free(
                conductance_values["log_hall_to_pedersen_ratio"]
            ).reshape(conductance_space.shape)
        )
        for values in (self.log_conductance_magnitude, self.log_hall_to_pedersen_ratio):
            if isinstance(values, np.ndarray):
                values.setflags(write=False)

    @classmethod
    def from_conductance(cls, geometry, config, space, values, *, previous=None):
        """Construct a fixed closure, reusing an unchanged previous one.

        Only conductance determines this response. Selection of input
        times belongs to the caller; unchanged CPU coefficients need
        neither another device transfer nor another operator build.
        """
        reusable = (
            previous is not None and previous.geometry is geometry and previous.config is config
        )
        if reusable and (
            (values is None and previous._conductance_values is None)
            or (
                values is not None
                and previous._conductance_values is not None
                and space.signature == previous._conductance_space.signature
                and all(
                    np.array_equal(values[name], previous._conductance_values[name])
                    for name in values
                )
            )
        ):
            return previous
        response = cls(geometry, config, conductance_space=space, conductance_values=values)
        if reusable:
            # Without conjugate E constraints, the entire toroidal fit
            # is geometric. Otherwise only an explicitly reusable
            # preconditioner may survive a conductance change.
            shared = []
            if geometry.interhemispheric_electric_field_difference_operator is None:
                shared.extend(
                    (
                        "_toroidal_potential_problem",
                        "_toroidal_potential_response_solver",
                        "boundary_jr_to_toroidal_potential_operator",
                    )
                )
            if (
                config.reuse_preconditioner
                or geometry.interhemispheric_electric_field_difference_operator is None
            ):
                shared.append("_toroidal_potential_preconditioner")
            # Transfer only already evaluated properties.
            response.__dict__.update(
                (name, previous.__dict__[name]) for name in shared if name in previous.__dict__
            )
        return response

    def __repr__(self):
        """Summarize inputs without materializing operators."""
        return (
            f"ElectrodynamicResponse(horizontal_basis={self.geometry.horizontal_basis!r}, "
            f"has_conductance={self.log_conductance_magnitude is not None})"
        )

    @cached_property
    def Q_eff_to_E_coeffs_operator(self) -> LinearMap:
        """Linear map from effective-current coeffs to E coeffs."""
        resistance_tensor = self.resistance_tensor_on_grid
        xp = get_array_module(resistance_tensor)
        return self._prepare_repeated_E_operator(
            ionospheric_closure.tangential_current_to_E_coeffs_operator(
                self.geometry.helmholtz_analysis_operator,
                xp.asarray(resistance_tensor),
                self.geometry.horizontal_transform.helmholtz_synthesis_operator,
            ),
            compact_input=False,
        )

    @cached_property
    def conductance_fingerprint(self) -> str:
        """Return the exact identity of the active conductance field."""
        if self.log_conductance_magnitude is None or self.log_hall_to_pedersen_ratio is None:
            raise RuntimeError(
                "Resistance or conductance must be set before it can be fingerprinted."
            )
        return content_fingerprint(
            {"field_space": self._conductance_space.signature, **self._conductance_values}
        )

    @cached_property
    def resistance_tensor_on_grid(self) -> Any:
        """Resistance tensor on the spatial grid."""
        if self.log_conductance_magnitude is None or self.log_hall_to_pedersen_ratio is None:
            raise RuntimeError(
                "Resistance or conductance must be set before accessing "
                "closure-dependent properties."
            )
        magnitude_coefficients = self.log_conductance_magnitude
        ratio_coefficients = self.log_hall_to_pedersen_ratio
        xp = get_array_module(magnitude_coefficients, ratio_coefficients)
        log_coordinate_coefficients = xp.stack(
            [xp.asarray(magnitude_coefficients), xp.asarray(ratio_coefficients)], axis=1
        )
        conductance_synthesis = self._conductance_space.basis.scalar_evaluation_operator(
            self.geometry.model_grid
        )
        log_coordinates_on_grid = xp.asarray(
            conductance_synthesis.matmat(log_coordinate_coefficients)
        )
        etaP, etaH = ionospheric_closure.resistance_from_log_conductance_coordinates(
            log_coordinates_on_grid[:, 0], log_coordinates_on_grid[:, 1]
        )
        return ionospheric_closure.resistance_tensor_on_grid(
            etaP, etaH, self.geometry.pedersen_geometry_tensor, self.geometry.hall_geometry_tensor
        )

    def _sheet_current_source_to_E_coeffs_operator(self, source_to_JS: LinearMap) -> LinearMap:
        """Map a magnetic source through derived sheet current to E."""
        return ionospheric_closure.tangential_current_to_E_coeffs_operator(
            self.geometry.helmholtz_analysis_operator, self.resistance_tensor_on_grid, source_to_JS
        )

    @cached_property
    def induced_poloidal_potential_to_E_coeffs_operator(self) -> LinearMap:
        """Map private induced-potential coordinates to E."""
        potential_to_sheet_current = (
            self.geometry.induced_Br_to_gridded_JS_operator()
            @ self.geometry.induced_poloidal_potential_to_Br_operator
        )
        return self._sheet_current_source_to_E_coeffs_operator(potential_to_sheet_current)

    @cached_property
    def induced_Br_to_E_coeffs_operator(self) -> LinearMap:
        """Map physical induced ``Br(RI)`` coefficients to E."""
        return self._prepare_repeated_E_operator(
            self.induced_poloidal_potential_to_E_coeffs_operator
            @ self.geometry.induced_Br_to_poloidal_potential_operator,
            compact_input=True,
        )

    @cached_property
    def toroidal_potential_to_E_coeffs_operator(self) -> LinearMap:
        """Map private toroidal-potential coefficients to E."""
        return self._prepare_repeated_E_operator(
            self._sheet_current_source_to_E_coeffs_operator(
                self.geometry.toroidal_potential_to_gridded_JS_operator()
            ),
            compact_input=False,
        )

    @cached_property
    def boundary_Br_to_E_coeffs_operator(self) -> LinearMap | None:
        """Map prescribed outer-boundary Br to E coefficients."""
        boundary_Br_to_JS = self.geometry.boundary_Br_to_gridded_JS_operator()
        if boundary_Br_to_JS is None:
            return None
        return self._prepare_repeated_E_operator(
            self._sheet_current_source_to_E_coeffs_operator(boundary_Br_to_JS), compact_input=True
        )

    def _prepare_repeated_E_operator(self, op: LinearMap, *, compact_input: bool) -> LinearMap:
        """Use an explicit array when it reduces repeated work."""
        if op.is_diagonal:
            return op
        if compact_input or self.geometry.horizontal_basis is self.geometry.poloidal_basis:
            op.to_matrix()
        return op

    @cached_property
    def _interhemispheric_electric_field_constraint(self) -> LinearMap | None:
        """Linear map enforcing the E-field low-latitude constraint."""
        outer_map = self.geometry.interhemispheric_electric_field_difference_operator
        if outer_map is not None:
            return outer_map @ self.toroidal_potential_to_E_coeffs_operator
        return None

    # ----- Solver Setup and Execution -----
    @cached_property
    def _toroidal_potential_problem(self) -> LeastSquaresProblem:
        """Return the toroidal-potential least-squares problem."""
        logger.info("Defining new toroidal-potential least-squares problem.")
        # Boundary radial current must match the prescribed field.
        operators = [
            self.geometry.radial_current_constraint_operator
            @ self.geometry.toroidal_potential_to_boundary_jr_operator
        ]

        # E-field must map at low latitudes.
        if self.config.enable_interhemispheric_coupling:
            electric_field_constraint = self._interhemispheric_electric_field_constraint
            if electric_field_constraint is not None:
                operators.append(
                    self.config.interhemispheric_electric_field_weight * electric_field_constraint
                )

        # Add Tikhonov regularization if lambda is set.
        regularization = None
        if self.config.toroidal_potential_regularization_lambda > 0:
            n = self.geometry.horizontal_basis.coefficient_count
            regularization = relative_regularization(
                vstack_linear_maps(operators),
                identity_linear_map((n,)),
                self.config.toroidal_potential_regularization_lambda,
            )

        # Fix the unobservable offset in coefficient space. It must not
        # enter the residual, its cutoff, or its regularization scale.
        basis = self.geometry.horizontal_basis
        constraints = None if basis.omits_constant_mode() else basis.scalar_mean_weights[None, :]
        return LeastSquaresProblem(
            A=operators, regularization=regularization, constraints=constraints
        )

    @cached_property
    def _toroidal_potential_preconditioner(self) -> LinearMap | None:
        """Return the toroidal-potential preconditioner."""
        if self._toroidal_potential_solver.method not in ("lsmr", "cgls"):
            return None
        logger.info("Building new toroidal-potential solver preconditioner.")
        return self._toroidal_potential_solver.build_preconditioner(
            problem=self._toroidal_potential_problem
        )

    @cached_property
    def _toroidal_potential_response_solver(self):
        """Prepare one solver for all toroidal right-hand sides."""
        return self._toroidal_potential_solver.prepare(
            self._toroidal_potential_problem,
            preconditioner=self._toroidal_potential_preconditioner,
        )

    def _toroidal_potential_rhs_entries(
        self, boundary_jr_coeffs: Any | None, driving_E: Any
    ) -> list[Any | None] | None:
        """Assemble the physical right-hand sides for one solve."""
        problem = self._toroidal_potential_problem
        rhs_entries = [None] * len(problem.data_operators)
        has_rhs = False

        if boundary_jr_coeffs is not None:
            rhs_entries[0] = self.geometry.radial_current_constraint_operator(boundary_jr_coeffs)
            has_rhs = True

        if (
            self.config.enable_interhemispheric_coupling
            and self._interhemispheric_electric_field_constraint is not None
        ):
            electric_field_difference = (
                self.geometry.interhemispheric_electric_field_difference_operator(driving_E)
            )
            rhs_entries[1] = (
                -self.config.interhemispheric_electric_field_weight * electric_field_difference
            )
            has_rhs = True

        return rhs_entries if has_rhs else None

    # ----- Response Operators -----

    @cached_property
    def boundary_jr_to_toroidal_potential_operator(self) -> LinearMap:
        """Map boundary current to the private toroidal potential."""
        logger.info("Building dense boundary-jr to toroidal-potential response matrix.")
        problem = self._toroidal_potential_problem
        radial_current_array = self.geometry.radial_current_constraint_operator.to_array()
        xp = get_array_module(radial_current_array)
        radial_current_rhs = xp.asarray(radial_current_array).reshape(
            problem.data_operators[0].output_shape + (-1,)
        )
        rhs_entries = [None] * len(problem.data_operators)
        rhs_entries[0] = radial_current_rhs
        response = self._toroidal_potential_response_solver(rhs_entries)
        return as_linear_map(
            response,
            input_shape=(self.geometry.horizontal_basis.coefficient_count,),
            output_shape=(self.geometry.horizontal_basis.coefficient_count,),
        )

    @cached_property
    def driving_E_to_toroidal_potential_operator(self) -> LinearMap | None:
        """Map driving E to the private toroidal potential."""
        if (
            not self.config.enable_interhemispheric_coupling
            or self._interhemispheric_electric_field_constraint is None
        ):
            return None
        logger.info("Building dense driving-E-to-toroidal-potential response matrix.")
        n = self.geometry.horizontal_basis.coefficient_count
        problem = self._toroidal_potential_problem
        electric_field_rhs = (
            -self.geometry.interhemispheric_electric_field_difference_array.reshape(
                problem.data_operators[1].output_shape + (2 * n,)
            )
        )
        electric_field_rhs *= self.config.interhemispheric_electric_field_weight
        rhs_entries = [None] * len(problem.data_operators)
        rhs_entries[1] = electric_field_rhs
        response = self._toroidal_potential_response_solver(rhs_entries)
        return as_linear_map(
            response,
            input_shape=(2, self.geometry.horizontal_basis.coefficient_count),
            output_shape=(self.geometry.horizontal_basis.coefficient_count,),
        )

    @cached_property
    def driving_E_to_total_E_operator(self) -> LinearMap:
        """Map driving E to total model E."""
        identity = identity_linear_map((2, self.geometry.horizontal_basis.coefficient_count))
        if (
            not self.config.enable_interhemispheric_coupling
            or self._interhemispheric_electric_field_constraint is None
        ):
            return identity
        driving_E_to_toroidal = self.driving_E_to_toroidal_potential_operator
        return identity + self.toroidal_potential_to_E_coeffs_operator @ driving_E_to_toroidal

    @cached_property
    def driving_E_to_W_operator(self) -> LinearMap:
        """Map driving-E coefficients to total ``W`` coefficients."""
        return (
            self.geometry.helmholtz_divergence_free_potential_operator
            @ self.driving_E_to_total_E_operator
        )

    @cached_property
    def _induced_poloidal_potential_response_operators(self):
        """Compile induced E and boundary current from one compact fit.

        Retain the toroidal response needed for Faraday feedback.
        Outputs use that fixed response without solving the closure
        again for each snapshot. Columns span the poloidal
        space, not the potentially much larger CS surface space.
        """
        source_to_driving_E = self.induced_poloidal_potential_to_E_coeffs_operator
        total_E_from_source = source_to_driving_E
        boundary_jr_from_source = None
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
                self._toroidal_potential_response_solver(rhs_entries),
                input_shape=source_to_driving_E.input_shape,
                output_shape=(self.geometry.horizontal_basis.coefficient_count,),
            )
            total_E_from_source = (
                source_to_driving_E
                + self.toroidal_potential_to_E_coeffs_operator @ toroidal_potential_from_source
            )
            boundary_jr_from_source = (
                self.geometry.toroidal_potential_to_boundary_jr_operator
                @ toroidal_potential_from_source
            )
        # Materialize the compact map once for repeated outputs and
        # feedback, without repeating its grid contractions.
        total_E_from_source = self._prepare_repeated_E_operator(
            total_E_from_source, compact_input=True
        )
        return total_E_from_source, boundary_jr_from_source

    @cached_property
    def induced_poloidal_potential_to_W_operator(self) -> LinearMap:
        """Map private induced potential to total ``W`` coefficients."""
        total_E, _ = self._induced_poloidal_potential_response_operators
        return self.geometry.helmholtz_divergence_free_potential_operator @ total_E

    @cached_property
    def induced_Br_to_W_operator(self) -> LinearMap:
        """Map physical induced Br to total ``W`` coefficients."""
        return (
            self.induced_poloidal_potential_to_W_operator
            @ self.geometry.induced_Br_to_poloidal_potential_operator
        )

    def _solve_toroidal_potential(self, boundary_jr_coeffs: Any | None, driving_E: Any) -> Any:
        """Solve the private toroidal-potential response."""
        xp = get_array_module(boundary_jr_coeffs, driving_E)
        boundary_jr_coeffs = None if boundary_jr_coeffs is None else xp.asarray(boundary_jr_coeffs)
        driving_E = xp.asarray(driving_E)
        rhs_entries = self._toroidal_potential_rhs_entries(boundary_jr_coeffs, driving_E)
        if rhs_entries is None:
            return xp.zeros(driving_E.shape[1:], dtype=driving_E.dtype)

        return self._toroidal_potential_response_solver(rhs_entries)

    # ----- Response Calculation -----

    def _solve_electric_closure(
        self, driving_E: Any, boundary_jr_coeffs: Any | None
    ) -> tuple[Any, Any]:
        """Complete E and return the resulting boundary current."""
        driving_E = self.geometry.horizontal_basis.project_helmholtz_mean_free(driving_E)
        toroidal_potential = self._solve_toroidal_potential(boundary_jr_coeffs, driving_E)
        E_from_toroidal_potential = self.toroidal_potential_to_E_coeffs_operator(
            toroidal_potential
        )
        solved_boundary_jr = self.geometry.toroidal_potential_to_boundary_jr_operator(
            toroidal_potential
        )
        return (
            self.geometry.horizontal_basis.project_helmholtz_mean_free(
                driving_E + E_from_toroidal_potential
            ),
            solved_boundary_jr,
        )

    def solve_noninductive_response(
        self, *, u=None, Q_eff=None, E_neutral_wind=None, boundary_Br=None, boundary_jr=None
    ) -> tuple[Any, Any]:
        """Solve forcing in the simulation's coefficient spaces.

        Tangential arrays contain two Helmholtz coefficient blocks.
        Boundary Br uses poloidal SH coefficients; all other forcing
        uses the horizontal basis. Input preparation owns projection
        into these spaces. Trailing batch axes describe independent
        forcing cases sharing this conductance response. Batch axes
        broadcast; scientific coefficient axes must match exactly.
        """
        E_shape = (2, self.geometry.horizontal_basis.coefficient_count)
        xp = get_array_module(u, Q_eff, E_neutral_wind, boundary_Br, boundary_jr)
        sources = dict(
            u=u,
            Q_eff=Q_eff,
            E_neutral_wind=E_neutral_wind,
            boundary_Br=boundary_Br,
            boundary_jr=boundary_jr,
        )
        shapes = {name: E_shape for name in ("u", "Q_eff", "E_neutral_wind")}
        shapes.update(
            boundary_Br=(self.geometry.poloidal_basis.coefficient_count,),
            boundary_jr=(self.geometry.horizontal_basis.coefficient_count,),
        )
        sources = {name: xp.asarray(value) for name, value in sources.items() if value is not None}
        for name, value in sources.items():
            shape = shapes[name]
            if value.shape[: len(shape)] != shape:
                raise ValueError(
                    f"{name} must have coefficient shape {shape}, followed by batch axes."
                )
        batch_shape = np.broadcast_shapes(
            *(value.shape[len(shapes[name]) :] for name, value in sources.items())
        )
        for name, value in sources.items():
            shape = shapes[name]
            batch = value.shape[len(shape) :]
            sources[name] = value.reshape(shape + (1,) * (len(batch_shape) - len(batch)) + batch)
        driving_E = xp.zeros(E_shape + batch_shape)
        if u is not None:
            driving_E += self.geometry.u_coeffs_to_E_coeffs_operator(sources["u"])
        if E_neutral_wind is not None:
            driving_E += sources["E_neutral_wind"]
        if boundary_Br is not None:
            if self.boundary_Br_to_E_coeffs_operator is None:
                raise ValueError("Boundary Br requires magnetospheric radius RM.")
            driving_E += self.boundary_Br_to_E_coeffs_operator(sources["boundary_Br"])
        if Q_eff is not None:
            driving_E += self.Q_eff_to_E_coeffs_operator(sources["Q_eff"])
        boundary_jr = sources.get("boundary_jr")
        if boundary_jr is not None:
            boundary_jr = xp.broadcast_to(boundary_jr, shapes["boundary_jr"] + batch_shape)
        return self._solve_electric_closure(driving_E, boundary_jr)

    def solve_induced_response(self, induced_Br: Any) -> tuple[Any, Any]:
        """Evaluate the induced response shared by Faraday feedback."""
        potential = self.geometry.induced_Br_to_poloidal_potential_operator(induced_Br)
        E_operator, jr_operator = self._induced_poloidal_potential_response_operators
        E = E_operator(potential)
        xp = get_array_module(E)
        if not self.geometry.horizontal_basis.mean_free:
            E = self.geometry.horizontal_basis.project_helmholtz_mean_free(E)
        boundary_jr = (
            xp.zeros(E.shape[1:], dtype=E.dtype) if jr_operator is None else jr_operator(potential)
        )
        return E, boundary_jr

    def output_coefficients(self, induced_Br, E_coeffs_noninductive, boundary_jr_noninductive):
        """Combine induced and prescribed responses with batch axes."""
        E_induced, jr_induced = self.solve_induced_response(induced_Br)
        Phi, W = self.geometry.horizontal_basis.project_helmholtz_mean_free(
            E_coeffs_noninductive + E_induced
        )
        # Both currents came from gauge-fixed toroidal potentials;
        # retain the Laplacian's range without projecting again.
        return {
            "induced_Br": induced_Br,
            "boundary_jr": boundary_jr_noninductive + jr_induced,
            "Phi": Phi,
            "W": W,
        }

    # ----- Induction Operators -----

    @cached_property
    def induced_poloidal_potential_feedback_operator(self) -> LinearMap:
        """Map private induced potential to Faraday-driving W."""
        return (
            self.geometry.surface_to_poloidal_operator
            @ self.induced_poloidal_potential_to_W_operator
        )

    @cached_property
    def noninductive_W_to_equilibrium_induced_poloidal_potential_operator(self) -> LinearMap:
        """Map non-inductive W to private equilibrium potential."""
        feedback_matrix = self.induced_poloidal_potential_feedback_operator.to_matrix()
        xp = get_array_module(feedback_matrix)
        feedback_pinv = synchronize_linalg_result(xp.linalg.pinv(feedback_matrix, rtol=1e-15))
        poloidal_size = self.geometry.poloidal_basis.coefficient_count
        equilibrium_poloidal_response = as_linear_map(
            -feedback_pinv, input_shape=(poloidal_size,), output_shape=(poloidal_size,)
        )
        return equilibrium_poloidal_response @ self.geometry.surface_to_poloidal_operator

    @cached_property
    def noninductive_W_to_equilibrium_induced_Br_operator(self) -> LinearMap:
        """Map non-inductive W to equilibrium induced Br."""
        return (
            self.geometry.induced_poloidal_potential_to_Br_operator
            @ self.noninductive_W_to_equilibrium_induced_poloidal_potential_operator
        )

    def source_to_W_operators(
        self,
        *,
        include_boundary_Br: bool = True,
        include_Q_eff: bool = True,
        include_E_neutral_wind: bool = True,
    ) -> dict[str, LinearMap]:
        """Return physical-source maps to total ``W`` coefficients."""
        operators = {
            "u": self.driving_E_to_W_operator @ self.geometry.u_coeffs_to_E_coeffs_operator,
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
        if include_Q_eff:
            operators["Q_eff"] = self.driving_E_to_W_operator @ self.Q_eff_to_E_coeffs_operator
        if include_E_neutral_wind:
            operators["E_neutral_wind"] = self.driving_E_to_W_operator

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


__all__ = ["ElectrodynamicResponse"]
