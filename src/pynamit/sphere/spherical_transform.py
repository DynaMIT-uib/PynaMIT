"""Spherical transform module.

This module contains the SphericalTransform class for converting between
spherical-basis coefficients and grid values.
"""

import numpy as np

from pynamit.math.backend import get_array_module
from pynamit.math.linear_map import LinearMap, as_linear_map
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver, get_default_least_squares_solver
from pynamit.sphere.core import SurfaceOperators, is_grid_basis
from pynamit.sphere.grid import Grid


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


class SphericalTransform:
    """Two-way transform between a spherical basis and a grid.

    This class owns both synthesis (coefficients to grid values) and
    analysis (grid values to coefficients) for scalar and tangential
    Helmholtz fields. It also handles batched projection from external
    input grids when an interpolation basis is supplied.
    """

    def __init__(
        self,
        source,
        target,
        *,
        interpolation_basis=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        area_weighted=False,
    ):
        """Initialize a transform from ``source`` to ``target``."""
        if not isinstance(source, SurfaceOperators):
            raise TypeError("SphericalTransform source must implement SurfaceOperators.")
        if not isinstance(target, Grid):
            raise TypeError("SphericalTransform target must be a Grid.")
        source.validate_metadata()
        self.source = source
        self.target = target
        self.interpolation_basis = interpolation_basis
        self.explicit_sqrt_weights = sqrt_weights is not None
        self.area_weighted = bool(area_weighted)
        self.sqrt_weights = resolve_sqrt_weights(
            target, sqrt_weights=sqrt_weights, area_weighted=area_weighted
        )
        self.helmholtz_sqrt_weights = resolve_sqrt_weights(
            target,
            sqrt_weights=sqrt_weights,
            area_weighted=area_weighted,
            vector=True,
        )
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

        self._scalar_least_squares_problem = None
        self._helmholtz_least_squares_problem = None
        self._input_transform = None

    @property
    def scalar_coeffs_to_grid(self):
        """Matrix mapping scalar coefficients to grid values."""
        if not hasattr(self, "_scalar_coeffs_to_grid"):
            if self.source.caching:
                if not hasattr(self, "_cache"):
                    self._scalar_coeffs_to_grid, self._cache = self.source.evaluate_on_grid(
                        self.target, cache_out=True
                    )
                else:
                    self._scalar_coeffs_to_grid = self.source.evaluate_on_grid(
                        self.target, cache_in=self._cache
                    )
            else:
                self._scalar_coeffs_to_grid = self.source.get_scalar_evaluation_matrix(
                    self.target
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
            elif self.source.caching:
                if not hasattr(self, "_cache"):
                    (
                        self._scalar_coeffs_to_gridded_theta_derivative,
                        self._cache,
                    ) = self.source.evaluate_on_grid(
                        self.target, derivative="theta", cache_out=True
                    )
                else:
                    (
                        self._scalar_coeffs_to_gridded_theta_derivative,
                        self._cache,
                    ) = self.source.evaluate_on_grid(
                        self.target,
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
            elif self.source.caching:
                if not hasattr(self, "_cache"):
                    (
                        self._scalar_coeffs_to_gridded_phi_derivative,
                        self._cache,
                    ) = self.source.evaluate_on_grid(
                        self.target, derivative="phi", cache_out=True
                    )
                else:
                    (
                        self._scalar_coeffs_to_gridded_phi_derivative,
                        self._cache,
                    ) = self.source.evaluate_on_grid(
                        self.target,
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
                self.source.get_surface_gradient_matrix(self.target)
            )
        return self._scalar_coeffs_to_gridded_gradient

    @property
    def scalar_coeffs_to_gridded_rhat_cross_gradient(self):
        """Matrix evaluating r-hat x horizontal gradient."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_rhat_cross_gradient"):
            self._scalar_coeffs_to_gridded_rhat_cross_gradient = (
                self.source.get_rhat_cross_gradient_matrix(self.target)
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
                    self.source.get_helmholtz_synthesis_matrix(self.target)
                )
        return self._helmholtz_coeffs_to_gridded_vector

    @property
    def G(self):
        """Scalar coefficient-to-grid synthesis matrix."""
        return self.scalar_coeffs_to_grid

    @property
    def G_th(self):
        """Gridded theta-derivative synthesis matrix."""
        return self.scalar_coeffs_to_gridded_theta_derivative

    @property
    def G_ph(self):
        """Gridded phi-derivative synthesis matrix."""
        return self.scalar_coeffs_to_gridded_phi_derivative

    @property
    def G_grad(self):
        """Gridded surface-gradient synthesis matrix."""
        return self.scalar_coeffs_to_gridded_gradient

    @property
    def G_rxgrad(self):
        """Gridded r-hat-cross-gradient synthesis matrix."""
        return self.scalar_coeffs_to_gridded_rhat_cross_gradient

    @property
    def G_helmholtz(self):
        """Tangential Helmholtz synthesis tensor."""
        return self.helmholtz_coeffs_to_gridded_vector

    @property
    def L(self):
        """Degree-weighted regularization matrix for scalar fields."""
        if not hasattr(self, "_L"):
            if self.reg_lambda is None:
                self._L = None
            else:
                if not hasattr(self.source, "n"):
                    raise NotImplementedError(
                        "Degree-weighted scalar regularization requires basis.n."
                    )
                self._L = np.diag(self.source.n)
        return self._L

    @property
    def L_helmholtz(self):
        """Degree-weighted regularization for Helmholtz fields."""
        if not hasattr(self, "_L_helmholtz"):
            if self.reg_lambda is None:
                self._L_helmholtz = None
            else:
                if not hasattr(self.source, "n"):
                    raise NotImplementedError(
                        "Degree-weighted Helmholtz regularization requires basis.n."
                    )
                curl_free_selector = np.asarray(
                    self.source.get_helmholtz_curl_free_potential_matrix()
                )
                divergence_free_selector = np.asarray(
                    self.source.get_helmholtz_divergence_free_potential_matrix()
                )
                # The weights are the existing SH spectral penalties.
                # The selector matrices keep the Helmholtz component
                # semantics explicit without moving this policy onto
                # the basis implementation.
                curl_free_weight = np.diag(
                    self.source.n * (self.source.n + 1) / (2 * self.source.n + 1)
                )
                divergence_free_weight = np.diag((self.source.n + 1) / 2)
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
    def scalar_least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for scalar fields."""
        if self._scalar_least_squares_problem is None:
            self._scalar_least_squares_problem = LeastSquaresProblem(
                A=self.scalar_coeffs_to_grid,
                solution_shape=self.source.index_length,
                data_shapes=self.target.size,
                sqrt_weights=self.sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L,
            )
        return self._scalar_least_squares_problem

    @property
    def helmholtz_least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for horizontal vector fields."""
        if self._helmholtz_least_squares_problem is None:
            self._helmholtz_least_squares_problem = LeastSquaresProblem(
                A=self.helmholtz_coeffs_to_gridded_vector,
                solution_shape=(2, self.source.index_length),
                data_shapes=(2, self.target.size),
                sqrt_weights=self.helmholtz_sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L_helmholtz,
            )
        return self._helmholtz_least_squares_problem

    def _solve_least_squares(self, problem, grid_values, solver_type=None):
        """Solve one configured least-squares problem."""
        solver_type = solver_type or get_default_least_squares_solver()
        solver = LeastSquaresSolver(solver=solver_type, tolerance=self.pinv_rtol)
        return solver.solve(problem=problem, rhs=grid_values)

    def synthesize_scalar(self, coeffs, derivative=None):
        """Synthesize scalar coefficients on the target grid."""
        coeff_array = self._coefficient_array(coeffs, preserve_backend=True)
        return self._coefficients_to_grid(coeff_array, derivative=derivative)

    def synthesize_helmholtz(self, coeffs):
        """Synthesize Helmholtz coefficients on the target grid."""
        coeff_array = self._coefficient_array(
            coeffs, helmholtz=True, preserve_backend=True
        )
        return self._coefficients_to_grid(coeff_array, helmholtz=True)

    def analyze_scalar(self, grid_values, solver_type=None):
        """Analyze scalar grid values into source coefficients."""
        if is_grid_basis(self.source):
            return grid_values
        return self._solve_least_squares(
            self.scalar_least_squares_problem, grid_values, solver_type
        )

    def analyze_helmholtz(self, grid_values, solver_type=None):
        """Analyze grid values into Helmholtz coefficients."""
        if is_grid_basis(self.source):
            return grid_values
        return self._solve_least_squares(
            self.helmholtz_least_squares_problem, grid_values, solver_type
        )

    def scalar_regularization_term(self, coeffs):
        """Return the scalar regularization term."""
        coeff_array = self._coefficient_array(coeffs)
        return np.dot(coeff_array, np.dot(self.L, coeff_array))

    def helmholtz_regularization_term(self, coeffs):
        """Return the Helmholtz regularization term."""
        coeff_array = self._coefficient_array(coeffs, helmholtz=True)
        return np.tensordot(self.L_helmholtz, coeff_array, 2)

    def project_scalar(
        self,
        values,
        *,
        input_grid,
        projection_basis,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Project scalar grid values to source-coefficient rows."""
        return self._project(
            values,
            input_grid=input_grid,
            projection_basis=projection_basis,
            helmholtz=False,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def project_helmholtz(
        self,
        values,
        *,
        input_grid,
        projection_basis,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Project grid values to Helmholtz-coefficient rows."""
        return self._project(
            values,
            input_grid=input_grid,
            projection_basis=projection_basis,
            helmholtz=True,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def _project(
        self,
        values,
        *,
        input_grid,
        projection_basis,
        helmholtz,
        sqrt_weights,
        reg_lambda,
        pinv_rtol,
    ):
        """Project one scalar or Helmholtz field batch."""
        value_batch = self._normalize_value_batch(
            values, input_grid, helmholtz=helmholtz
        )
        direct_projection = self._basis_can_project_directly(projection_basis)
        analyze = "analyze_helmholtz" if helmholtz else "analyze_scalar"

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
                coeff_rows.append(
                    getattr(input_transform, analyze)(value_batch[time_index])
                )
        else:
            for time_index in range(value_batch.shape[0]):
                grid_values = self._interpolate_to_grid(
                    value_batch[time_index], input_grid, helmholtz=helmholtz
                )
                coeff_rows.append(getattr(self, analyze)(grid_values))

        return np.asarray([np.asarray(row).reshape(-1) for row in coeff_rows])

    def _coefficient_array(self, coeffs, *, helmholtz=False, preserve_backend=False):
        """Return validated coefficient values."""
        values = getattr(coeffs, "coeffs", coeffs)
        shape = (2, self.source.index_length) if helmholtz else (self.source.index_length,)
        array = np.asarray(values)
        if array.size != int(np.prod(shape)):
            field_type = "Helmholtz" if helmholtz else "scalar"
            raise ValueError(
                f"{field_type} coefficients have length {array.size}, "
                f"expected {int(np.prod(shape))}."
            )
        if preserve_backend and "jax" in type(values).__module__:
            return get_array_module(values).asarray(values).reshape(shape)
        return array.reshape(shape)

    def _cached_coefficients_to_grid_operator(
        self,
        cache_name,
        matrix,
        *,
        input_shape,
        output_shape,
    ):
        """Return a cached coefficient-to-grid ``LinearMap``."""
        operator = getattr(self, cache_name, None)
        if operator is None:
            operator = as_linear_map(
                matrix,
                input_shape=input_shape,
                output_shape=output_shape,
            )
            setattr(self, cache_name, operator)
        return operator

    def _coefficients_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values."""
        if derivative == "theta":
            matrix = self.scalar_coeffs_to_gridded_theta_derivative
            operator = self._cached_coefficients_to_grid_operator(
                "_scalar_coeffs_to_gridded_theta_derivative_operator",
                matrix,
                input_shape=(self.source.index_length,),
                output_shape=matrix.shape[:-1],
            )
        elif derivative == "phi":
            matrix = self.scalar_coeffs_to_gridded_phi_derivative
            operator = self._cached_coefficients_to_grid_operator(
                "_scalar_coeffs_to_gridded_phi_derivative_operator",
                matrix,
                input_shape=(self.source.index_length,),
                output_shape=matrix.shape[:-1],
            )
        elif helmholtz:
            matrix = self.helmholtz_coeffs_to_gridded_vector
            operator = self._cached_coefficients_to_grid_operator(
                "_helmholtz_coeffs_to_gridded_vector_operator",
                matrix,
                input_shape=(2, self.source.index_length),
                output_shape=matrix.shape[:2],
            )
        else:
            matrix = self.scalar_coeffs_to_grid
            operator = self._cached_coefficients_to_grid_operator(
                "_scalar_coeffs_to_grid_operator",
                matrix,
                input_shape=(self.source.index_length,),
                output_shape=matrix.shape[:-1],
            )

        return operator.matvec(coeffs).reshape(operator.output_shape)

    def contract_scalar_coeffs_to_grid(self, operator):
        """Contract the scalar-grid matrix with an operator."""
        if not isinstance(operator, LinearMap) and getattr(operator, "ndim", None) not in (
            None,
            1,
            2,
        ):
            raise ValueError("operator must be a vector, matrix, or LinearMap.")
        try:
            op = as_linear_map(operator, output_shape=(self.source.index_length,))
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

    def normalize_scalar_value_batch(self, values, input_grid):
        """Return scalar values with canonical time-first layout."""
        return self._normalize_value_batch(values, input_grid, helmholtz=False)

    def normalize_helmholtz_value_batch(self, values, input_grid):
        """Return tangential values with canonical time-first layout."""
        return self._normalize_value_batch(values, input_grid, helmholtz=True)

    def _normalize_value_batch(self, values, input_grid, *, helmholtz):
        """Return values with canonical time-first layout."""
        n_points = int(input_grid.size)
        array = np.asarray(values)

        if not helmholtz:
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
        if self.source is projection_basis:
            return
        compatible = getattr(self.source, "coefficients_are_compatible_with", None)
        if callable(compatible) and compatible(projection_basis):
            return
        raise ValueError(
            "Direct projection basis is not coefficient-compatible with the "
            "transform source."
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

        transform = SphericalTransform(
            projection_basis,
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
        if transform.source is not projection_basis:
            return False
        if sqrt_weights is not None or transform.explicit_sqrt_weights:
            return False
        return (
            input_grid.same_as(transform.target)
            and transform.reg_lambda == reg_lambda
            and transform.pinv_rtol == pinv_rtol
            and transform.area_weighted == self.area_weighted
        )

    def _interpolate_to_grid(self, values, input_grid, *, helmholtz):
        """Interpolate one field slice to this transform's grid."""
        if self.interpolation_basis is None:
            raise ValueError("interpolation_basis is required for grid interpolation.")

        if not helmholtz:
            return self.interpolation_basis.interpolate_scalar(
                values,
                input_grid.theta,
                input_grid.phi,
                self.interpolation_basis.arr_theta,
                self.interpolation_basis.arr_phi,
            )

        interpolated_east, interpolated_north, _ = (
            self.interpolation_basis.interpolate_vector_components(
                values[1],
                -values[0],
                np.zeros_like(values[0]),
                input_grid.theta,
                input_grid.phi,
                self.interpolation_basis.arr_theta,
                self.interpolation_basis.arr_phi,
            )
        )
        return np.vstack((-interpolated_north, interpolated_east))
