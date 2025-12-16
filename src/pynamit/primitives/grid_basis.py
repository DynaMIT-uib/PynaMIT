"""Grid Basis module."""

from __future__ import annotations
from typing import Any, Tuple, Optional, TYPE_CHECKING
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from pynamit.math import arrayutils
from pynamit.math import cs_math

if TYPE_CHECKING:
    from pynamit.primitives.grid import Grid
    from pynamit.primitives.basis_evaluator import BasisEvaluator


class GridBasis:
    """Basis representing values defined on a grid, utilizing generic spherical interpolation.

    This class serves as the fundamental object for grid-based fields, providing
    robust spherical interpolation capabilities via projection to cubed sphere faces.
    Other specific grid bases (e.g. CSBasis) can inherit from this.
    """

    def __init__(self, grid: Optional[Grid] = None):
        self.grid = grid
        self.index_length = grid.size if grid else 0
        self.caching = False
        self.kind = "GRID"

    @property
    def theta(self):
        return self.grid.theta if self.grid else None

    @property
    def phi(self):
        return self.grid.phi if self.grid else None

    def to_grid_values(
        self, coeffs: np.ndarray, evaluator: BasisEvaluator, field_type: str = "scalar"
    ) -> np.ndarray:
        """Evaluate basis on a grid (interpolate coeffs)."""
        target_grid = evaluator.grid

        # 1. Identity check
        if self.grid and target_grid is self.grid:
            return coeffs
        if self.grid and (
            target_grid.size == self.grid.size
            and np.allclose(target_grid.theta, self.grid.theta)
            and np.allclose(target_grid.phi, self.grid.phi)
        ):
            return coeffs

        # 2. Interpolate using robust spherical logic
        if not self.grid:
            # If grid is None (e.g. incomplete CSBasis subclassing), cannot interpolate FROM it.
            # However, CSBasis usually has grid points defined.
            raise ValueError("Cannot interpolate from GridBasis without a source grid.")

        th_src = self.grid.theta
        ph_src = self.grid.phi
        th_tgt = target_grid.theta
        ph_tgt = target_grid.phi

        if field_type == "scalar":
            return self.interpolate_scalar(coeffs, th_src, ph_src, th_tgt, ph_tgt)

        elif field_type == "tangential":
            vals = coeffs.reshape(2, -1)
            v2, v3 = vals[0], vals[1]
            # Map PynaMIT (South, East) to CS (East, North)
            u_north = -v2
            u_east = v3
            u_r = np.zeros_like(v2)

            u_east_int, u_north_int, _ = self.interpolate_vector_components(
                u_east, u_north, u_r, th_src, ph_src, th_tgt, ph_tgt
            )
            return np.vstack([-u_north_int, u_east_int])

        elif field_type == "vector":
            vals = coeffs.reshape(3, -1)
            v1, v2, v3 = vals[0], vals[1], vals[2]
            u_r = v1
            u_north = -v2
            u_east = v3
            u_east_int, u_north_int, u_r_int = self.interpolate_vector_components(
                u_east, u_north, u_r, th_src, ph_src, th_tgt, ph_tgt
            )
            return np.vstack([u_r_int, -u_north_int, u_east_int])

        else:
            raise ValueError(f"Unknown field_type: {field_type}")

    def regularization_term(self, coeffs, evaluator, field_type):
        return None

    def from_grid_values(
        self, values: np.ndarray, evaluator: Any, field_type: str = "scalar"
    ) -> np.ndarray:
        input_grid = evaluator.grid
        if self.grid and input_grid is self.grid:
            return values
        if not self.grid:
            return values

        if field_type == "scalar":
            return self.interpolate_scalar(
                values.flatten(), input_grid.theta, input_grid.phi, self.grid.theta, self.grid.phi
            )
        raise NotImplementedError("from_grid_values (interpolation to basis) only impl for scalar")

    def interpolate_scalar(self, scalar, theta, phi, theta_target, phi_target, **kwargs):
        """Interpolate scalar values."""
        # Use configured interpolator if available AND compatible with input data
        if hasattr(self, "_interpolator") and self._interpolator:
            # Check compatibility using the interpolator's own logic
            if self._interpolator.is_compatible(scalar):
                return self._interpolator.interpolate_scalar(
                    scalar, theta_target, phi_target, **kwargs
                )

        if self.grid and (theta is self.grid.theta) and (phi is self.grid.phi):
            if not hasattr(self, "_cached_generic_interpolator"):
                from pynamit.interpolation import create_interpolator

                self._cached_generic_interpolator = create_interpolator(theta, phi)
            return self._cached_generic_interpolator.interpolate_scalar(
                scalar, theta_target, phi_target, **kwargs
            )

        # Fallback using factory (non-cached for arbitrary grids)
        from pynamit.interpolation import create_interpolator

        interp = create_interpolator(theta, phi)
        return interp.interpolate_scalar(scalar, theta_target, phi_target, **kwargs)

    def interpolate_vector_components(
        self, u_east, u_north, u_r, theta, phi, theta_target, phi_target, **kwargs
    ):
        """Interpolate vector components."""
        if hasattr(self, "_interpolator") and self._interpolator:
            # Check compatibility
            if self._interpolator.is_compatible(u_east):
                return self._interpolator.interpolate_vector(
                    u_east, u_north, u_r, theta_target, phi_target, **kwargs
                )

        if self.grid and (theta is self.grid.theta) and (phi is self.grid.phi):
            if not hasattr(self, "_cached_generic_interpolator"):
                from pynamit.interpolation import create_interpolator

                self._cached_generic_interpolator = create_interpolator(theta, phi)
            return self._cached_generic_interpolator.interpolate_vector(
                u_east, u_north, u_r, theta_target, phi_target, **kwargs
            )

        from pynamit.interpolation import create_interpolator

        interp = create_interpolator(theta, phi)
        return interp.interpolate_vector(u_east, u_north, u_r, theta_target, phi_target, **kwargs)
