"""Spherical transform module.

This module contains the SphericalTransform class for converting between
spherical-basis coefficients and grid values.
"""

import numpy as np

from pynamit.math.backend import get_array_module, to_numpy
from pynamit.math.linear_map import (
    LinearMap,
    as_linear_map,
    diagonal_linear_map,
    is_noop_linear_map,
)
from pynamit.math.least_squares_problem import LeastSquaresProblem
from pynamit.math.least_squares_solver import LeastSquaresSolver, get_default_least_squares_solver
from pynamit.sphere.core import SurfaceOperators
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


def _representation_signature(representation):
    """Return a cache key for an evaluable representation."""
    signature = getattr(representation, "signature", None)
    if signature is not None:
        return signature
    return getattr(representation, "coefficient_space_signature", id(representation))


class SphericalTransform:
    """Two-way transform between a spherical basis and a grid.

    This class owns both synthesis (coefficients to grid values) and
    analysis (grid values to coefficients) for scalar and tangential
    Helmholtz fields. It also handles batched projection from external
    input grids when a grid-remap basis is supplied.
    """

    def __init__(
        self,
        source,
        target,
        *,
        grid_remap_basis=None,
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
        self.grid_remap_basis = grid_remap_basis
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
        self._input_transforms = {}

    def _evaluate_source_on_target(self, derivative=None):
        """Evaluate the source on the target grid."""
        return self.source.get_scalar_evaluation_matrix(
            self.target,
            derivative=derivative,
        )

    @property
    def scalar_coeffs_to_grid(self):
        """Matrix mapping scalar coefficients to grid values."""
        if not hasattr(self, "_scalar_coeffs_to_grid"):
            self._scalar_coeffs_to_grid = self._evaluate_source_on_target()
        return self._scalar_coeffs_to_grid

    @property
    def scalar_coeffs_to_grid_operator(self):
        """Operator mapping scalar coefficients to grid values."""
        if not hasattr(self, "_scalar_coeffs_to_grid_operator"):
            self._scalar_coeffs_to_grid_operator = (
                self.source.get_scalar_evaluation_operator(self.target)
            )
        return self._scalar_coeffs_to_grid_operator

    @property
    def scalar_coeffs_to_gridded_theta_derivative(self):
        """Matrix evaluating the theta derivative."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_theta_derivative"):
            if hasattr(self, "_scalar_coeffs_to_gridded_gradient"):
                self._scalar_coeffs_to_gridded_theta_derivative = (
                    self._scalar_coeffs_to_gridded_gradient[0]
                )
            else:
                self._scalar_coeffs_to_gridded_theta_derivative = (
                    self._evaluate_source_on_target(derivative="theta")
                )
        return self._scalar_coeffs_to_gridded_theta_derivative

    @property
    def scalar_coeffs_to_gridded_theta_derivative_operator(self):
        """Operator evaluating the theta derivative."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_theta_derivative_operator"):
            self._scalar_coeffs_to_gridded_theta_derivative_operator = (
                self.source.get_scalar_evaluation_operator(
                    self.target,
                    derivative="theta",
                )
            )
        return self._scalar_coeffs_to_gridded_theta_derivative_operator

    @property
    def scalar_coeffs_to_gridded_phi_derivative(self):
        """Matrix evaluating the phi derivative."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_phi_derivative"):
            if hasattr(self, "_scalar_coeffs_to_gridded_gradient"):
                self._scalar_coeffs_to_gridded_phi_derivative = (
                    self._scalar_coeffs_to_gridded_gradient[1]
                )
            else:
                self._scalar_coeffs_to_gridded_phi_derivative = (
                    self._evaluate_source_on_target(derivative="phi")
                )
        return self._scalar_coeffs_to_gridded_phi_derivative

    @property
    def scalar_coeffs_to_gridded_phi_derivative_operator(self):
        """Operator evaluating the phi derivative."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_phi_derivative_operator"):
            self._scalar_coeffs_to_gridded_phi_derivative_operator = (
                self.source.get_scalar_evaluation_operator(
                    self.target,
                    derivative="phi",
                )
            )
        return self._scalar_coeffs_to_gridded_phi_derivative_operator

    @property
    def scalar_coeffs_to_gridded_gradient(self):
        """Matrix evaluating the horizontal gradient."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_gradient"):
            self._scalar_coeffs_to_gridded_gradient = (
                self.source.get_surface_gradient_matrix(self.target)
            )
        return self._scalar_coeffs_to_gridded_gradient

    @property
    def scalar_coeffs_to_gridded_gradient_operator(self):
        """Operator evaluating the horizontal gradient."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_gradient_operator"):
            self._scalar_coeffs_to_gridded_gradient_operator = (
                self.source.get_surface_gradient_operator(self.target)
            )
        return self._scalar_coeffs_to_gridded_gradient_operator

    @property
    def scalar_coeffs_to_gridded_rhat_cross_gradient(self):
        """Matrix evaluating r-hat x horizontal gradient."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_rhat_cross_gradient"):
            self._scalar_coeffs_to_gridded_rhat_cross_gradient = (
                self.source.get_rhat_cross_gradient_matrix(self.target)
            )
        return self._scalar_coeffs_to_gridded_rhat_cross_gradient

    @property
    def scalar_coeffs_to_gridded_rhat_cross_gradient_operator(self):
        """Operator evaluating r-hat x horizontal gradient."""
        if not hasattr(self, "_scalar_coeffs_to_gridded_rhat_cross_gradient_operator"):
            self._scalar_coeffs_to_gridded_rhat_cross_gradient_operator = (
                self.source.get_rhat_cross_gradient_operator(self.target)
            )
        return self._scalar_coeffs_to_gridded_rhat_cross_gradient_operator

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
    def helmholtz_coeffs_to_gridded_vector_operator(self):
        """Operator evaluating horizontal vector field expansions."""
        if not hasattr(self, "_helmholtz_coeffs_to_gridded_vector_operator"):
            self._helmholtz_coeffs_to_gridded_vector_operator = (
                self.source.get_helmholtz_synthesis_operator(self.target)
            )
        return self._helmholtz_coeffs_to_gridded_vector_operator

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
                self._L = np.asarray(
                    self.L_operator.to_matrix(backend="numpy")
                )
        return self._L

    @property
    def L_operator(self):
        """Degree-weighted regularization operator for scalar fields."""
        if not hasattr(self, "_L_operator"):
            if self.reg_lambda is None:
                self._L_operator = None
            else:
                if not hasattr(self.source, "n"):
                    raise NotImplementedError(
                        "Degree-weighted scalar regularization requires basis.n."
                    )
                self._L_operator = diagonal_linear_map(
                    np.asarray(self.source.n),
                    input_shape=(self.source.index_length,),
                    output_shape=(self.source.index_length,),
                )
        return self._L_operator

    @property
    def L_helmholtz(self):
        """Degree-weighted regularization for Helmholtz fields."""
        if not hasattr(self, "_L_helmholtz"):
            if self.reg_lambda is None:
                self._L_helmholtz = None
            else:
                self._L_helmholtz = to_numpy(self.L_helmholtz_operator.array)
        return self._L_helmholtz

    @property
    def L_helmholtz_operator(self):
        """Degree-weighted Helmholtz regularization operator."""
        if not hasattr(self, "_L_helmholtz_operator"):
            if self.reg_lambda is None:
                self._L_helmholtz_operator = None
            else:
                if not hasattr(self.source, "n"):
                    raise NotImplementedError(
                        "Degree-weighted Helmholtz regularization requires basis.n."
                    )
                n = np.asarray(self.source.n)
                weights = np.stack(
                    [
                        n * (n + 1) / (2 * n + 1),
                        (n + 1) / 2,
                    ],
                    axis=0,
                )
                self._L_helmholtz_operator = diagonal_linear_map(
                    weights.reshape(-1),
                    input_shape=(2, self.source.index_length),
                    output_shape=(2, self.source.index_length),
                )
        return self._L_helmholtz_operator

    @property
    def scalar_least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for scalar fields."""
        if self._scalar_least_squares_problem is None:
            self._scalar_least_squares_problem = LeastSquaresProblem(
                A=self.scalar_coeffs_to_grid_operator,
                solution_shape=self.source.index_length,
                data_shapes=self.target.size,
                sqrt_weights=self.sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L_operator,
            )
        return self._scalar_least_squares_problem

    @property
    def helmholtz_least_squares_problem(self) -> LeastSquaresProblem:
        """Least squares problem for horizontal vector fields."""
        if self._helmholtz_least_squares_problem is None:
            self._helmholtz_least_squares_problem = LeastSquaresProblem(
                A=self.helmholtz_coeffs_to_gridded_vector_operator,
                solution_shape=(2, self.source.index_length),
                data_shapes=(2, self.target.size),
                sqrt_weights=self.helmholtz_sqrt_weights,
                regularization_weights=self.reg_lambda,
                regularization_matrices=self.L_helmholtz_operator,
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
        if self._scalar_analysis_is_noop():
            return grid_values
        return self._solve_least_squares(
            self.scalar_least_squares_problem, grid_values, solver_type
        )

    def analyze_helmholtz(self, grid_values, solver_type=None):
        """Analyze grid values into Helmholtz coefficients."""
        return self._solve_least_squares(
            self.helmholtz_least_squares_problem, grid_values, solver_type
        )

    def scalar_regularization_term(self, coeffs):
        """Return the scalar regularization term."""
        coeff_array = self._coefficient_array(coeffs)
        return np.dot(coeff_array, self.L_operator.matvec(coeff_array))

    def helmholtz_regularization_term(self, coeffs):
        """Return the Helmholtz regularization term."""
        coeff_array = self._coefficient_array(coeffs, helmholtz=True)
        return self.L_helmholtz_operator.matvec(coeff_array.reshape(-1)).reshape(
            coeff_array.shape
        )

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

        if direct_projection:
            self._validate_direct_projection_basis(projection_basis)
            analysis_transform = self._get_input_transform(
                projection_basis,
                input_grid,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )
            grid_values = value_batch
        else:
            analysis_transform = self
            grid_values = (
                value_batch
                if input_grid.same_as(self.target)
                else self._remap_batch_to_grid(
                    value_batch, input_grid, helmholtz=helmholtz
                )
            )

        coeffs = getattr(analysis_transform, analyze)(grid_values)
        return self._analysis_coefficients_to_rows(
            coeffs,
            batch_size=value_batch.shape[0],
            helmholtz=helmholtz,
        )

    def _analysis_coefficients_to_rows(self, coeffs, *, batch_size, helmholtz):
        """Return analysis coefficients in time-row layout."""
        array = np.asarray(coeffs)
        if not helmholtz and self._scalar_analysis_is_noop():
            return array.reshape(batch_size, -1)
        if batch_size == 1:
            return array.reshape(1, -1)
        return np.moveaxis(array, -1, 0).reshape(batch_size, -1)

    def _scalar_analysis_is_noop(self):
        """Return whether scalar analysis is a no-op."""
        return is_noop_linear_map(
            self.scalar_coeffs_to_grid_operator,
            input_shape=(self.source.index_length,),
            output_shape=(self.target.size,),
        )

    def _coefficient_array(self, coeffs, *, helmholtz=False, preserve_backend=False):
        """Return validated coefficient values."""
        values = getattr(coeffs, "array", coeffs)
        shape = (
            (2, self.source.index_length)
            if helmholtz
            else (self.source.index_length,)
        )
        expected_size = int(np.prod(shape))
        size = getattr(values, "size", None)
        if size is None:
            array = np.asarray(values)
            size = array.size
        if int(size) != expected_size:
            field_type = "Helmholtz" if helmholtz else "scalar"
            raise ValueError(
                f"{field_type} coefficients have length {int(size)}, "
                f"expected {expected_size}."
            )
        if preserve_backend:
            xp = get_array_module(values)
            if xp is not np:
                return xp.asarray(values).reshape(shape)
        return np.asarray(values).reshape(shape)

    def _coefficients_to_grid(self, coeffs, derivative=None, helmholtz=False):
        """Transform basis coefficients to grid values."""
        if derivative == "theta":
            operator = self.scalar_coeffs_to_gridded_theta_derivative_operator
        elif derivative == "phi":
            operator = self.scalar_coeffs_to_gridded_phi_derivative_operator
        elif helmholtz:
            operator = self.helmholtz_coeffs_to_gridded_vector_operator
        else:
            operator = self.scalar_coeffs_to_grid_operator

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
        product_adjoint = op.rmatmat(
            xp.swapaxes(xp.conjugate(scalar_coeffs_to_grid), -2, -1)
        )
        return xp.swapaxes(xp.conjugate(product_adjoint), -2, -1)

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
        return isinstance(
            projection_basis,
            SurfaceOperators,
        ) and not callable(getattr(projection_basis, "scalar_grid_remap_operator", None))

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
        if sqrt_weights is not None:
            return SphericalTransform(
                projection_basis,
                input_grid,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
                area_weighted=self.area_weighted,
            )

        cache_key = (
            _representation_signature(projection_basis),
            input_grid.signature,
            reg_lambda,
            pinv_rtol,
            self.area_weighted,
        )
        if cache_key not in self._input_transforms:
            self._input_transforms[cache_key] = SphericalTransform(
                projection_basis,
                input_grid,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
                area_weighted=self.area_weighted,
            )
        return self._input_transforms[cache_key]

    def _grid_remap_operator(self, method_name, input_grid, *, input_shape, output_shape):
        """Return the required grid-remap operator."""
        if self.grid_remap_basis is None:
            raise ValueError("grid_remap_basis is required for grid remapping.")
        remap_operator = getattr(self.grid_remap_basis, method_name, None)
        if not callable(remap_operator):
            raise TypeError(
                "Grid-to-grid projection requires grid_remap_basis to provide "
                f"{method_name}()."
            )
        operator = remap_operator(input_grid, self.target)
        try:
            return as_linear_map(
                operator,
                input_shape=input_shape,
                output_shape=output_shape,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{type(self.grid_remap_basis).__name__}.{method_name}() "
                "must return an operator convertible to LinearMap."
            ) from exc

    def _scalar_batch_remap_to_grid(self, value_batch, input_grid):
        """Apply the scalar grid remap to field rows."""
        operator = self._grid_remap_operator(
            "scalar_grid_remap_operator",
            input_grid,
            input_shape=(input_grid.size,),
            output_shape=(self.target.size,),
        )
        interpolated = operator.matmat(np.asarray(value_batch).T)
        return np.asarray(interpolated).reshape(self.target.size, -1).T

    def _helmholtz_batch_remap_to_grid(self, value_batch, input_grid):
        """Apply the tangential grid remap to field rows."""
        values = np.asarray(value_batch)
        operator = self._grid_remap_operator(
            "tangential_grid_remap_operator",
            input_grid,
            input_shape=(2, input_grid.size),
            output_shape=(2, self.target.size),
        )
        interpolated = operator.matmat(values.reshape(values.shape[0], -1).T)
        return np.moveaxis(
            np.asarray(interpolated).reshape(2, self.target.size, -1),
            -1,
            0,
        )

    def _remap_batch_to_grid(self, value_batch, input_grid, *, helmholtz):
        """Apply grid remap operators to field slices."""
        if not helmholtz:
            return self._scalar_batch_remap_to_grid(value_batch, input_grid)
        return self._helmholtz_batch_remap_to_grid(value_batch, input_grid)
