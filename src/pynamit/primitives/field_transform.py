"""Field transform module.

This module contains the FieldTransform class for converting between
field-space coefficients and grid values.
"""

import numpy as np

from pynamit.math.backend import get_array_module
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver, get_default_least_squares_solver
from pynamit.primitives.coefficient_field import CoefficientField
from pynamit.primitives.field_space import FieldSpace
from pynamit.sphere.core import SurfaceOperators, is_grid_basis

FLOAT_ERROR_MARGIN = 1e-6


def grid_sqrt_area_weights(grid):
    """Return default sqrt area weights for a spherical grid."""
    if hasattr(grid, "area_weights"):
        xp = get_array_module(grid.area_weights)
        weights = xp.asarray(grid.area_weights, dtype=float)
    else:
        xp = get_array_module(grid.theta)
        theta = xp.asarray(grid.theta, dtype=float)
        weights = xp.sin(xp.deg2rad(theta))
    weights = xp.clip(weights, 0.0, None)
    return xp.sqrt(weights)


def resolve_sqrt_weights(grid, sqrt_weights=None, area_weighted=False, vector=False):
    """Resolve explicit or default grid sqrt weights."""
    if sqrt_weights is not None:
        return sqrt_weights
    if not area_weighted:
        return None
    weights = grid_sqrt_area_weights(grid)
    xp = get_array_module(weights)
    return xp.tile(weights, (2, 1)) if vector else weights


class FieldTransform(object):
    """Two-way transform between one ``FieldSpace`` and one grid.

    This class owns both synthesis (coefficients to grid values) and
    analysis (grid values to coefficients). It also handles batched
    projection from external input grids when a target grid basis is
    supplied.
    """

    def __init__(
        self,
        field_space,
        grid,
        *,
        grid_basis=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        area_weighted=False,
    ):
        """Initialize the FieldTransform object."""
        if not isinstance(field_space, FieldSpace):
            raise TypeError("FieldTransform requires a FieldSpace.")
        if not isinstance(field_space.basis, SurfaceOperators):
            raise TypeError(
                "FieldTransform requires a FieldSpace whose basis implements SurfaceOperators."
            )
        self.field_space = field_space
        self.basis = field_space.basis
        self.basis.validate_metadata()
        self.grid = grid
        self.grid_basis = grid_basis
        self.explicit_sqrt_weights = sqrt_weights is not None
        self.area_weighted = bool(area_weighted)
        self.sqrt_weights = resolve_sqrt_weights(
            grid, sqrt_weights=sqrt_weights, area_weighted=area_weighted
        )
        self.helmholtz_sqrt_weights = resolve_sqrt_weights(
            grid,
            sqrt_weights=sqrt_weights,
            area_weighted=area_weighted,
            vector=True,
        )
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

        self._least_squares_problem = None
        self._least_squares_problem_helmholtz = None
        self._input_transform = None

    @property
    def scalar_coeffs_to_grid(self):
        """Matrix mapping scalar coefficients to grid values."""
        if not hasattr(self, "_scalar_coeffs_to_grid"):
            if self.basis.caching:
                if not hasattr(self, "_cache"):
                    self._scalar_coeffs_to_grid, self._cache = self.basis.evaluate_on_grid(
                        self.grid, cache_out=True
                    )
                else:
                    self._scalar_coeffs_to_grid = self.basis.evaluate_on_grid(
                        self.grid, cache_in=self._cache
                    )
            else:
                self._scalar_coeffs_to_grid = self.basis.get_scalar_evaluation_matrix(
                    self.grid
                )
        return self._scalar_coeffs_to_grid

    @property
    def scalar_coeffs_to_gridded_theta_derivative(self):
        """Matrix evaluating the theta derivative."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_theta_derivative"):
            if hasattr(self, "_scalar_coeffs_to_gridded_gradient"):
                self._scalar_coeffs_to_gridded_theta_derivative = (
                    self._scalar_coeffs_to_gridded_gradient[0]
                )
            elif self.basis.caching:
                if not hasattr(self, "_cache"):
                    (
                        self._scalar_coeffs_to_gridded_theta_derivative,
                        self._cache,
                    ) = self.basis.evaluate_on_grid(
                        self.grid, derivative="theta", cache_out=True
                    )
                else:
                    (
                        self._scalar_coeffs_to_gridded_theta_derivative,
                        self._cache,
                    ) = self.basis.evaluate_on_grid(
                        self.grid,
                        derivative="theta",
                        cache_in=self._cache,
                        cache_out=True,
                    )
            else:
                self._scalar_coeffs_to_gridded_theta_derivative = (
                    self.scalar_coeffs_to_gridded_gradient[0]
                )
        return self._scalar_coeffs_to_gridded_theta_derivative

    @property
    def scalar_coeffs_to_gridded_phi_derivative(self):
        """Matrix evaluating the phi derivative."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_phi_derivative"):
            if hasattr(self, "_scalar_coeffs_to_gridded_gradient"):
                self._scalar_coeffs_to_gridded_phi_derivative = (
                    self._scalar_coeffs_to_gridded_gradient[1]
                )
            elif self.basis.caching:
                if not hasattr(self, "_cache"):
                    (
                        self._scalar_coeffs_to_gridded_phi_derivative,
                        self._cache,
                    ) = self.basis.evaluate_on_grid(
                        self.grid, derivative="phi", cache_out=True
                    )
                else:
                    (
                        self._scalar_coeffs_to_gridded_phi_derivative,
                        self._cache,
                    ) = self.basis.evaluate_on_grid(
                        self.grid,
                        derivative="phi",
                        cache_in=self._cache,
                        cache_out=True,
                    )
            else:
                self._scalar_coeffs_to_gridded_phi_derivative = (
                    self.scalar_coeffs_to_gridded_gradient[1]
                )
        return self._scalar_coeffs_to_gridded_phi_derivative

    @property
    def scalar_coeffs_to_gridded_gradient(self):
        """Matrix evaluating the horizontal gradient."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_gradient"):
            self._scalar_coeffs_to_gridded_gradient = (
                self.basis.get_surface_gradient_matrix(self.grid)
            )
        return self._scalar_coeffs_to_gridded_gradient

    @property
    def scalar_coeffs_to_gridded_rhat_cross_gradient(self):
        """Matrix evaluating r-hat x horizontal gradient."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_rhat_cross_gradient"):
            self._scalar_coeffs_to_gridded_rhat_cross_gradient = (
                self.basis.get_rhat_cross_gradient_matrix(self.grid)
            )
        return self._scalar_coeffs_to_gridded_rhat_cross_gradient

    @property
    def helmholtz_coeffs_to_gridded_vector(self):
        """Matrix evaluating horizontal vector field expansions."""
        if not hasattr(self, "_helmholtz_coeffs_to_gridded_vector"):
            if hasattr(self, "_scalar_coeffs_to_gridded_gradient") or hasattr(
                self,
                "_scalar_coeffs_to_gridded_rhat_cross_gradient",
            ):
                gradient = self.scalar_coeffs_to_gridded_gradient
                rotated_gradient = self.scalar_coeffs_to_gridded_rhat_cross_gradient
                xp = get_array_module(gradient, rotated_gradient)
                self._helmholtz_coeffs_to_gridded_vector = xp.stack(
                    [-xp.asarray(gradient), xp.asarray(rotated_gradient)],
                    axis=2,
                )
            else:
                self._helmholtz_coeffs_to_gridded_vector = (
                    self.basis.get_helmholtz_synthesis_matrix(self.grid)
                )
        return self._helmholtz_coeffs_to_gridded_vector

    @property
    def L(self):
        """Degree-weighted regularization matrix for scalar fields."""
        if not hasattr(self, "_L"):
            if self.reg_lambda is None:
                self._L = None
            else:
                if not hasattr(self.basis, "n"):
                    raise NotImplementedError(
                        "Degree-weighted scalar regularization requires basis.n."
                    )
                self._L = np.diag(self.basis.n)
        return self._L

    @property
    def L_helmholtz(self):
        """Degree-weighted regularization for Helmholtz fields."""
        if not hasattr(self, "_L_helmholtz"):
            if self.reg_lambda is None:
                self._L_helmholtz = None
            else:
                if not hasattr(self.basis, "n"):
                    raise NotImplementedError(
                        "Degree-weighted Helmholtz regularization requires basis.n."
                    )
                curl_free_selector = np.asarray(
                    self.basis.get_helmholtz_curl_free_potential_matrix()
                )
                divergence_free_selector = np.asarray(
                    self.basis.get_helmholtz_divergence_free_potential_matrix()
                )
                # The weights are the existing SH spectral penalties.
                # The selector matrices keep the Helmholtz component
                # semantics explicit without moving this policy onto
                # the basis implementation.
                curl_free_weight = np.diag(
                    self.basis.n * (self.basis.n + 1) / (2 * self.basis.n + 1)
                )
                divergence_free_weight = np.diag((self.basis.n + 1) / 2)
                L_cf = np.tensordot(
                    curl_free_weight, curl_free_selector, axes=([1], [0])
                )
                L_df = np.tensordot(
                    divergence_free_weight,
                    divergence_free_selector,
                    axes=([1], [0]),
                )
                self._L_helmholtz = np.stack([L_cf, L_df], axis=0)
        return self._L_helmholtz

    @property
    def least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for scalar fields."""
        if self._least_squares_problem is None:
            self._least_squares_problem = LeastSquaresProblem(
                A=self.scalar_coeffs_to_grid,
                solution_shape=self.basis.index_length,
                data_shapes=self.grid.size,
                sqrt_weights=self.sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L,
            )
        return self._least_squares_problem

    @property
    def least_squares_problem_helmholtz(self) -> LeastSquaresProblem:
        """Least squares problem for horizontal vector fields."""
        if self._least_squares_problem_helmholtz is None:
            self._least_squares_problem_helmholtz = LeastSquaresProblem(
                A=self.helmholtz_coeffs_to_gridded_vector,
                solution_shape=(2, self.basis.index_length),
                data_shapes=(2, self.grid.size),
                sqrt_weights=self.helmholtz_sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L_helmholtz,
            )
        return self._least_squares_problem_helmholtz

    def least_squares_solution(self, grid_values, solver_type=None):
        """Least squares decomposition of a scalar field."""
        return self._solve_least_squares(self.least_squares_problem, grid_values, solver_type)

    def least_squares_solution_helmholtz(self, grid_values, solver_type=None):
        """Least squares decomposition of a horizontal vector field."""
        solution = self._solve_least_squares(
            self.least_squares_problem_helmholtz, grid_values, solver_type
        )
        projector = getattr(self.basis, "project_helmholtz_mean_free", None)
        return projector(solution) if callable(projector) else solution

    def _solve_least_squares(self, problem, grid_values, solver_type=None):
        """Solve one configured least-squares problem."""
        solver_type = solver_type or get_default_least_squares_solver()
        solver = LeastSquaresSolver(solver=solver_type, tolerance=self.pinv_rtol)
        return solver.solve(problem=problem, rhs=grid_values)

    def to_grid(self, coeffs, derivative=None):
        """Transform coefficients in this field space to grid values."""
        coeff_array = self._coefficient_array(coeffs)
        if self.field_space.field_type == "tangential":
            return self._coefficients_to_grid(coeff_array, helmholtz=True)
        return self._coefficients_to_grid(
            coeff_array, derivative=derivative, helmholtz=False
        )

    def to_coefficients(self, grid_values, solver_type=None):
        """Transform grid values to validated field coefficients."""
        if is_grid_basis(self.basis):
            coeffs = grid_values
        elif self.field_space.field_type == "tangential":
            coeffs = self.least_squares_solution_helmholtz(grid_values, solver_type)
        else:
            coeffs = self.least_squares_solution(grid_values, solver_type)
        return CoefficientField(self.field_space, coeffs).coeffs

    def regularization_term(self, coeffs):
        """Return the field-space regularization term."""
        coeff_array = self._coefficient_array(coeffs)
        if self.field_space.field_type == "tangential":
            return np.tensordot(self.L_helmholtz, coeff_array, 2)
        return np.dot(coeff_array, np.dot(self.L, coeff_array))

    def project(
        self,
        values,
        *,
        input_grid,
        projection_basis,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Project grid values to coefficient rows."""
        value_batch = self._normalize_value_batch(values, input_grid)
        direct_projection = self._basis_can_project_directly(projection_basis)

        coeff_rows = []
        if direct_projection:
            self._validate_direct_projection_basis(projection_basis)
            input_transform = self._get_input_transform(
                projection_basis,
                input_grid,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )
            for time_index in range(value_batch.shape[0]):
                coeff_rows.append(input_transform.to_coefficients(value_batch[time_index]))
        else:
            for time_index in range(value_batch.shape[0]):
                grid_values = self._interpolate_to_grid(value_batch[time_index], input_grid)
                coeff_rows.append(self.to_coefficients(grid_values))

        return np.asarray(
            [
                CoefficientField(self.field_space, row).coeffs.reshape(-1)
                for row in coeff_rows
            ]
        )

    def _coefficient_array(self, coeffs):
        """Return validated coefficient values."""
        if isinstance(coeffs, CoefficientField):
            if coeffs.field_space != self.field_space:
                raise ValueError("CoefficientField field_space does not match transform.")
            coeffs = coeffs.coeffs
        return self.field_space.validate_coefficients(coeffs)

    def _coefficients_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values."""
        if derivative == "theta":
            return np.dot(self.scalar_coeffs_to_gridded_theta_derivative, coeffs)
        elif derivative == "phi":
            return np.dot(self.scalar_coeffs_to_gridded_phi_derivative, coeffs)
        elif helmholtz:
            return np.tensordot(self.helmholtz_coeffs_to_gridded_vector, coeffs, 2)
        else:
            return np.dot(self.scalar_coeffs_to_grid, coeffs)

    def contract_scalar_coeffs_to_grid(self, operator):
        """Contract the scalar-grid matrix with an operator."""
        if not isinstance(operator, LinearMap) and getattr(operator, "ndim", None) not in (
            None,
            1,
            2,
        ):
            raise ValueError("operator must be a vector, matrix, or LinearMap.")
        try:
            op = as_linear_map(operator, output_shape=(self.basis.index_length,))
        except ValueError as exc:
            raise ValueError("operator must be a vector, matrix, or LinearMap.") from exc

        xp = get_array_module(self.scalar_coeffs_to_grid, *op.backend_context)
        scalar_coeffs_to_grid = xp.asarray(self.scalar_coeffs_to_grid)
        if op.shape[0] == op.shape[1]:
            try:
                diagonal = xp.asarray(op.diagonal()).reshape((1, -1))
                return scalar_coeffs_to_grid * diagonal
            except ValueError:
                pass
        return scalar_coeffs_to_grid @ xp.asarray(op.dense())

    def _normalize_value_batch(self, values, input_grid):
        """Return values with canonical time-first layout."""
        n_points = int(input_grid.size)
        array = np.asarray(values)

        if self.field_space.field_type == "scalar":
            if array.ndim == 1:
                if array.size != n_points:
                    raise ValueError(
                        f"Scalar field has {array.size} points, expected {n_points}."
                    )
                return array.reshape(1, n_points)
            if array.ndim == 2:
                if array.shape[-1] == n_points:
                    return array
                if array.shape[0] == n_points:
                    return array.T
            raise ValueError(
                "Scalar projection expects shape (N,), (T, N), or (N, T); "
                f"got {array.shape} for grid size {n_points}."
            )

        if array.ndim == 2:
            if array.shape == (2, n_points):
                return array.reshape(1, 2, n_points)
            if array.shape == (n_points, 2):
                return array.T.reshape(1, 2, n_points)
        elif array.ndim == 3:
            if array.shape[1:] == (2, n_points):
                return array
            if array.shape[:2] == (2, n_points):
                return np.moveaxis(array, -1, 0)
            if array.shape[1:] == (n_points, 2):
                return np.moveaxis(array, -1, 1)

        raise ValueError(
            "Tangential projection expects shape (2, N), (T, 2, N), "
            f"(N, 2), or (T, N, 2); got {array.shape} for grid size {n_points}."
        )

    def _basis_can_project_directly(self, projection_basis):
        """Return whether the input basis should be fitted directly."""
        return isinstance(projection_basis, SurfaceOperators) and not is_grid_basis(
            projection_basis
        )

    def _validate_direct_projection_basis(self, projection_basis):
        """Raise if direct-fit coefficients would not match storage."""
        if self.basis is projection_basis:
            return
        compatible = getattr(self.basis, "coefficients_are_compatible_with", None)
        if callable(compatible) and compatible(projection_basis):
            return
        raise ValueError(
            "Direct projection basis is not coefficient-compatible with the "
            "target field space."
        )

    def _get_input_transform(
        self,
        projection_basis,
        input_grid,
        *,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Return transform for direct input-grid projection."""
        transform = self._input_transform
        if transform is not None and self._input_transform_matches(
            transform,
            projection_basis,
            input_grid,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        ):
            return transform

        transform = FieldTransform(
            FieldSpace.from_basis(
                projection_basis,
                field_type=self.field_space.field_type,
                mean_free=getattr(projection_basis, "mean_free", self.field_space.mean_free),
            ),
            input_grid,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
            area_weighted=self.area_weighted,
        )
        self._input_transform = transform
        return transform

    def _input_transform_matches(
        self,
        transform,
        projection_basis,
        input_grid,
        *,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Return whether a cached input transform can be reused."""
        if transform.basis is not projection_basis:
            return False
        if sqrt_weights is not None or transform.explicit_sqrt_weights:
            return False
        return (
            input_grid.theta.shape == transform.grid.theta.shape
            and input_grid.phi.shape == transform.grid.phi.shape
            and transform.reg_lambda == reg_lambda
            and transform.pinv_rtol == pinv_rtol
            and transform.area_weighted == self.area_weighted
            and np.allclose(
                input_grid.theta,
                transform.grid.theta,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
            and np.allclose(
                input_grid.phi,
                transform.grid.phi,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
        )

    def _interpolate_to_grid(self, values, input_grid):
        """Interpolate one field slice to this transform's grid."""
        if self.grid_basis is None:
            raise ValueError("grid_basis is required for grid interpolation.")

        if self.field_space.field_type == "scalar":
            return self.grid_basis.interpolate_scalar(
                values,
                input_grid.theta,
                input_grid.phi,
                self.grid_basis.arr_theta,
                self.grid_basis.arr_phi,
            )

        interpolated_east, interpolated_north, _ = (
            self.grid_basis.interpolate_vector_components(
                values[1],
                -values[0],
                np.zeros_like(values[0]),
                input_grid.theta,
                input_grid.phi,
                self.grid_basis.arr_theta,
                self.grid_basis.arr_phi,
            )
        )
        return np.vstack((-interpolated_north, interpolated_east))
