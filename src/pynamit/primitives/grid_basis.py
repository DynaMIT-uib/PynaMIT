"""Grid Basis module."""

from __future__ import annotations
from typing import Any, Tuple, TYPE_CHECKING, Optional
from abc import ABC
import numpy as np
from pynamit.math import arrayutils
from pynamit.math import cs_math
from pynamit.primitives.basis import Basis
from pynamit.interpolation import create_interpolator

if TYPE_CHECKING:
    from pynamit.primitives.grid import Grid
    from pynamit.primitives.basis_evaluator import BasisEvaluator


class GridBasis(Basis, ABC):
    """Abstract Basis representing values defined on a grid.
    
    This class serves as the fundamental object for grid-based fields, providing
    common functionality for grid management and spherical interpolation.
    """

    # Storage for the grid
    _grid: Optional[Grid] = None
    _cached_interpolator = None

    @property
    def grid(self) -> "Grid":
        """Get the grid, ensuring it is valid."""
        if self._grid is None:
            raise ValueError(f"{self.__class__.__name__} must have a defined grid.")
        return self._grid

    @grid.setter
    def grid(self, value: "Grid"):
        """Set the grid."""
        self._grid = value
        self._cached_interpolator = None

    @property
    def kind(self) -> str:
        """Default kind for grid bases."""
        return "GRID"



    @property
    def index_names(self) -> list[str]:
        """Default index name."""
        return ["point_index"]

    @property
    def index_length(self) -> int:
        """Index length matches storage grid size."""
        return self.grid.size

    @property
    def index_arrays(self) -> list:
        """Index arrays match flattened grid."""
        return [np.arange(self.grid.size)]

    @property
    def minimum_phi_sampling(self) -> float:
        """Default sampling requirement."""
        return 1.0

    @property
    def theta(self):
        """Get theta coordinates of grid points."""
        return self.grid.theta

    @property
    def phi(self):
        """Get phi coordinates of grid points."""
        return self.grid.phi

    def to_grid_values(
        self,
        coeffs: np.ndarray,
        evaluator: "BasisEvaluator",
        field_type: str = "scalar",
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

        # 2. Generic interpolation
        if not self.grid:
            # If grid is None (e.g. incomplete CSBasis subclassing), cannot interpolate FROM it.
            # However, CSBasis usually has grid points defined.
            raise ValueError("Cannot interpolate from GridBasis without a source grid.")

        th_src = self.grid.theta
        ph_src = self.grid.phi
        th_tgt = target_grid.theta
        ph_tgt = target_grid.phi
        
        if self._cached_interpolator is None:
            self._cached_interpolator = create_interpolator(th_src, ph_src)
        interp = self._cached_interpolator

        if field_type == "scalar":
            return interp.interpolate_scalar(coeffs, th_tgt, ph_tgt)

        elif field_type == "tangential":
            vals = coeffs.reshape(2, -1)
            v2, v3 = vals[0], vals[1]
            # Map PynaMIT (South, East) to CS (East, North)
            u_north = -v2
            u_east = v3
            u_r = np.zeros_like(v2)

            u_east_int, u_north_int, _ = interp.interpolate_vector(
                u_east, u_north, u_r, th_tgt, ph_tgt
            )
            return np.vstack([-u_north_int, u_east_int])

        elif field_type == "vector":
            vals = coeffs.reshape(3, -1)
            v1, v2, v3 = vals[0], vals[1], vals[2]
            u_r = v1
            u_north = -v2
            u_east = v3
            u_east_int, u_north_int, u_r_int = interp.interpolate_vector(
                u_east, u_north, u_r, th_tgt, ph_tgt
            )
            return np.vstack([u_r_int, -u_north_int, u_east_int])

        else:
            raise ValueError(f"Unknown field type: {field_type}")

    def from_grid_values(
        self, values: np.ndarray, evaluator: "BasisEvaluator", field_type: str = "scalar"
    ) -> np.ndarray:
        """Convert grid values to coefficients.
        
        For GridBasis, the coefficients ARE the grid values (potentially interpolated).
        """
        # If the evaluator grid matches our storage grid, return values directly
        if self.grid and evaluator.grid is self.grid:
            return values
            
        # If sizes match and coords close, assume same grid
        if self.grid and (
            evaluator.grid.size == self.grid.size 
            and np.allclose(evaluator.grid.theta, self.grid.theta)
            and np.allclose(evaluator.grid.phi, self.grid.phi)
        ):
             return values
             
        # Otherwise, we need to map values FROM the evaluator grid TO our storage grid.
        # This is the inverse of to_grid_values. 
        # For now, we reuse the interpolation logic if supported, or raise error if ambiguous
        
        # NOTE: In many cases for GridBasis, "coeffs" are just values on self.grid.
        # So "from_grid_values" means taking values defined on `evaluator.grid` and 
        # resampling them to `self.grid`.
        
        if not self.grid:
             raise ValueError("Cannot convert to GridBasis coefficients without a defined storage grid.")

        th_src = evaluator.grid.theta
        ph_src = evaluator.grid.phi
        th_tgt = self.grid.theta
        ph_tgt = self.grid.phi
        
        interp = create_interpolator(th_src, ph_src)
        
        # Generic interpolation mapping src -> tgt
        if field_type == "scalar":
             return interp.interpolate_scalar(values, th_tgt, ph_tgt)
        
        elif field_type == "tangential":
            vals = values.reshape(2, -1)
            v2, v3 = vals[0], vals[1]
            u_north = -v2
            u_east = v3
            u_r = np.zeros_like(v2)
            
            u_east_int, u_north_int, _ = interp.interpolate_vector(
                u_east, u_north, u_r, th_tgt, ph_tgt
            )
            return np.vstack([-u_north_int, u_east_int])
            
        elif field_type == "vector":
            vals = values.reshape(3, -1)
            v1, v2, v3 = vals[0], vals[1], vals[2]
            u_r = v1
            u_north = -v2
            u_east = v3
            
            u_east_int, u_north_int, u_r_int = interp.interpolate_vector(
                u_east, u_north, u_r, th_tgt, ph_tgt
            )
            return np.vstack([u_r_int, -u_north_int, u_east_int])
            
        else:
             # Fallback (legacy or unknown)
             return interp.interpolate_scalar(values, th_tgt, ph_tgt)

    def regularization_term(
        self, coeffs: np.ndarray, evaluator: "BasisEvaluator", field_type: str = "scalar"
    ) -> float:
        """Compute regularization penalty term.
        
        For GridBasis, regularization is typically not applied (or handled externally).
        Returns 0.0.
        """
        return 0.0
