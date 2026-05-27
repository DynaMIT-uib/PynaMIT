"""Projection of gridded fields into coefficient spaces."""

import numpy as np

from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.coefficient_field import CoefficientField
from pynamit.primitives.field_space import FieldSpace
from pynamit.sphere import Grid
from pynamit.sphere.core import SurfaceOperators, is_grid_basis

FLOAT_ERROR_MARGIN = 1e-6


class FieldProjector:
    """Project gridded field values into one coefficient space.

    The projector owns operational details such as evaluator caching,
    grid interpolation, weights, and regularization. It does not choose
    a storage basis or know simulation input keys.
    """

    def __init__(
        self,
        field_space,
        target_grid_basis=None,
        *,
        area_weighted_least_squares=False,
    ):
        """Initialize a field projector."""
        self.field_space = FieldSpace.from_basis(
            field_space, field_type=getattr(field_space, "field_type", "scalar")
        )
        self.target_grid_basis = target_grid_basis
        self.area_weighted_least_squares = bool(area_weighted_least_squares)
        self._input_basis_evaluator = None
        self._storage_basis_evaluator = None

        if target_grid_basis is not None:
            self.target_grid = Grid(
                theta=target_grid_basis.arr_theta,
                phi=target_grid_basis.arr_phi,
                area_weights=getattr(target_grid_basis, "unit_area", None),
            )
        else:
            self.target_grid = None

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
        """Project gridded values to ``(time, coefficient)`` arrays."""
        value_batch = self._normalize_value_batch(values, input_grid)
        direct_projection = self._basis_can_project_directly(projection_basis)

        coeff_rows = []
        if direct_projection:
            self._validate_direct_projection_basis(projection_basis)
            basis_evaluator = self._get_input_basis_evaluator(
                projection_basis,
                input_grid,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )
            for time_index in range(value_batch.shape[0]):
                coeff_rows.append(
                    self._grid_to_coefficients(
                        basis_evaluator, value_batch[time_index]
                    )
                )
        else:
            basis_evaluator = self._get_storage_basis_evaluator(
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )
            for time_index in range(value_batch.shape[0]):
                grid_values = self._interpolate_to_target_grid(
                    value_batch[time_index], input_grid
                )
                coeff_rows.append(self._grid_to_coefficients(basis_evaluator, grid_values))

        return np.asarray(
            [
                CoefficientField(self.field_space, row).coeffs.reshape(-1)
                for row in coeff_rows
            ]
        )

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
        target_basis = self.field_space.basis
        if target_basis is projection_basis:
            return
        compatible = getattr(target_basis, "coefficients_are_compatible_with", None)
        if callable(compatible) and compatible(projection_basis):
            return
        raise ValueError(
            "Direct projection basis is not coefficient-compatible with the "
            "target field space."
        )

    def _get_storage_basis_evaluator(self, *, reg_lambda=None, pinv_rtol=1e-15):
        """Return evaluator for target-grid fits."""
        if self.target_grid is None:
            raise ValueError("target_grid_basis is required for grid interpolation.")

        evaluator = self._storage_basis_evaluator
        if (
            evaluator is None
            or evaluator.reg_lambda != reg_lambda
            or evaluator.pinv_rtol != pinv_rtol
            or evaluator.area_weighted != self.area_weighted_least_squares
        ):
            evaluator = BasisEvaluator(
                self.field_space.basis,
                self.target_grid,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
                area_weighted=self.area_weighted_least_squares,
            )
            self._storage_basis_evaluator = evaluator
        return evaluator

    def _get_input_basis_evaluator(
        self,
        projection_basis,
        input_grid,
        *,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Return evaluator for direct input-grid projection."""
        evaluator = self._input_basis_evaluator
        if evaluator is not None and self._input_evaluator_matches(
            evaluator,
            projection_basis,
            input_grid,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        ):
            return evaluator

        evaluator = BasisEvaluator(
            projection_basis,
            input_grid,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
            area_weighted=self.area_weighted_least_squares,
        )
        self._input_basis_evaluator = evaluator
        return evaluator

    def _input_evaluator_matches(
        self,
        evaluator,
        projection_basis,
        input_grid,
        *,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Return whether a cached input evaluator can be reused."""
        if evaluator.basis is not projection_basis:
            return False
        if sqrt_weights is not None or evaluator.explicit_sqrt_weights:
            return False
        return (
            input_grid.theta.shape == evaluator.grid.theta.shape
            and input_grid.phi.shape == evaluator.grid.phi.shape
            and evaluator.reg_lambda == reg_lambda
            and evaluator.pinv_rtol == pinv_rtol
            and evaluator.area_weighted == self.area_weighted_least_squares
            and np.allclose(
                input_grid.theta,
                evaluator.grid.theta,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
            and np.allclose(
                input_grid.phi,
                evaluator.grid.phi,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
        )

    def _interpolate_to_target_grid(self, values, input_grid):
        """Interpolate one field slice to the target grid."""
        if self.target_grid_basis is None:
            raise ValueError("target_grid_basis is required for interpolation.")

        if self.field_space.field_type == "scalar":
            return self.target_grid_basis.interpolate_scalar(
                values,
                input_grid.theta,
                input_grid.phi,
                self.target_grid_basis.arr_theta,
                self.target_grid_basis.arr_phi,
            )

        interpolated_east, interpolated_north, _ = (
            self.target_grid_basis.interpolate_vector_components(
                values[1],
                -values[0],
                np.zeros_like(values[0]),
                input_grid.theta,
                input_grid.phi,
                self.target_grid_basis.arr_theta,
                self.target_grid_basis.arr_phi,
            )
        )
        return np.vstack((-interpolated_north, interpolated_east))

    def _grid_to_coefficients(self, basis_evaluator, grid_values):
        """Fit one grid-value slice to the target field space."""
        if is_grid_basis(self.field_space.basis):
            return grid_values
        if self.field_space.field_type == "scalar":
            return basis_evaluator.grid_to_basis(grid_values, helmholtz=False)
        if self.field_space.field_type == "tangential":
            return basis_evaluator.grid_to_basis(grid_values, helmholtz=True)
        raise ValueError("field type must be either 'scalar' or 'tangential'.")
