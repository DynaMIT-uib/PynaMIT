"""Unified Field primitives module.

This module contains the consolidated Field abstraction:
- Field: The main class representing vector/scalar fields (discrete, expanded, or computed).
- ComponentField: Helper for accessing single components.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Literal, TYPE_CHECKING, Union
import numpy as np

# Imports
from pynamit.primitives.grid import Grid

if TYPE_CHECKING:
    from pynamit.primitives.basis_evaluator import BasisEvaluator


from pynamit.primitives.grid_basis import GridBasis


class _FieldImpl(ABC):
    """Internal implementation interface for Field strategies."""

    @abstractmethod
    def evaluate(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def basis_vectors(self, r, theta, phi):
        raise NotImplementedError


class _ExpansionImpl(_FieldImpl):
    """Implementation for basis expansion fields (including GridBasis)."""

    def __init__(self, basis, coeffs, field_type):
        self.basis = basis
        self.coeffs = coeffs
        self.field_type = field_type

    # --- Property Accessors for seamless integration ---
    @property
    def v1(self):
        if self.field_type == "vector":
            return self.coeffs[0]
        if self.field_type == "scalar":
            return self.coeffs
        return None

    @property
    def v2(self):
        if self.field_type == "vector":
            return self.coeffs[1]
        return None

    @property
    def v3(self):
        if self.field_type == "vector":
            return self.coeffs[2]
        return None

    def evaluate(self, r, theta, phi):
        from pynamit.primitives.basis_evaluator import BasisEvaluator

        g = Grid(theta=theta, phi=phi)
        evaluator = BasisEvaluator(self.basis, g)

        # Basis handles interpolation/evaluation
        values = self.basis.to_grid_values(self.coeffs, evaluator, self.field_type)

        if self.field_type == "scalar":
            return values, np.zeros_like(values), np.zeros_like(values)
        elif self.field_type == "tangential":
            # Tangential -> (v2, v3)
            return np.zeros_like(values[0]), values[0], values[1]
        elif self.field_type == "vector":
            # Vector -> (v1, v2, v3)
            return values[0], values[1], values[2]

        raise ValueError(f"Unknown field_type: {self.field_type}")


class _DiscreteImpl(_FieldImpl):
    """Implementation for discrete grid fields."""
    
    def __init__(self, grid, v1, v2=None, v3=None):
        self.grid = grid
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
        
    @property
    def v1(self): return self._v1
    @property
    def v2(self): return self._v2
    @property
    def v3(self): return self._v3
    
    def evaluate(self, r, theta, phi):
        # Generic interpolation from source grid to target points
        # Using PynaMIT built-in spherical interpolator
        from pynamit.interpolation import create_interpolator
        
        interp = create_interpolator(self.grid.theta, self.grid.phi)
        
        # Check if vector field
        if self._v1 is not None and self._v2 is not None and self._v3 is not None:
             # Map Field components to spherical vector components
             # v1 = u_r
             # v2 = u_theta = -u_north
             # v3 = u_phi = u_east
             u_r = self._v1
             u_north = -self._v2
             u_east = self._v3
             
             u_east_int, u_north_int, u_r_int = interp.interpolate_vector(
                 u_east, u_north, u_r, theta, phi
             )
             
             # Map back to Field components
             v1_int = u_r_int
             v2_int = -u_north_int
             v3_int = u_east_int
             return v1_int, v2_int, v3_int
        
        vals = []
        for v in [self._v1, self._v2, self._v3]:
            if v is None:
                vals.append(np.zeros_like(theta))
                continue
                
            res = interp.interpolate_scalar(v, theta, phi)
            vals.append(res)
            
        return tuple(vals)


class _ComponentImpl(_FieldImpl):
    """Implementation for single component fields."""

    def __init__(self, parent_field, component_index):
        self.parent_field = parent_field
        self.component_index = component_index

    def evaluate(self, r, theta, phi):
        v1, v2, v3 = self.parent_field.evaluate(r, theta, phi)
        if self.component_index == 0:
            val = v1
        elif self.component_index == 1:
            val = v2
        else:
            val = v3
        return val, np.zeros_like(val), np.zeros_like(val)

    def basis_vectors(self, r, theta, phi):
        return self.parent_field.basis_vectors(r, theta, phi)


class Field(ABC):
    """Unified Field class utilizing the Bridge pattern for implementation."""

    def __init__(
        self,
        # Discrete args
        grid: Optional[Grid] = None,
        v1=None,
        v2=None,
        v3=None,
        r_loc=None,
        source_field=None,
        # Expansion args
        basis: Optional[Any] = None,
        coeffs=None,
        field_type="scalar",
        # Component args
        parent_field=None,
        component_index=None,
        **kwargs,
    ):
        self._impl: Optional[_FieldImpl] = None

        # Metadata storage (exposed by properties)
        self._r_loc = r_loc
        self._source_field = source_field

        # Determine strategy
        # Determine strategy
        if v1 is not None and grid is not None:
            # Discrete case
            self._impl = _DiscreteImpl(grid, v1, v2, v3)

        elif coeffs is not None and basis is not None:
            # Standard Expansion
            self._impl = _ExpansionImpl(basis, coeffs, field_type)

        elif parent_field is not None:
            # Component
            self._impl = _ComponentImpl(parent_field, component_index)

    @property
    def vec(self) -> "VectorAccessor":
        return VectorAccessor(self)

    @property
    def scalar(self) -> Any:
        return self.vec.v1

    # --- Property Delegation ---
    @property
    def grid(self):
        # Check if impl has direct grid (Discrete)
        if hasattr(self._impl, "grid"):
            return self._impl.grid
        # Check if impl has basis with grid (Expansion)
        return getattr(getattr(self._impl, "basis", None), "grid", None)

    @property
    def v1(self):
        return getattr(self._impl, "v1", None)

    @property
    def v2(self):
        return getattr(self._impl, "v2", None)

    @property
    def v3(self):
        return getattr(self._impl, "v3", None)

    @property
    def basis(self):
        return getattr(self._impl, "basis", None)

    @property
    def coeffs(self):
        return getattr(self._impl, "coeffs", None)

    @property
    def field_type(self):
        return getattr(self._impl, "field_type", None)

    @property
    def component_index(self):
        return getattr(self._impl, "component_index", None)

    @property
    def r_loc(self):
        return self._r_loc

    @property
    def source_field(self):
        return self._source_field

    @property
    def magnitude(self) -> Optional[np.ndarray]:
        v1 = self.v1
        if v1 is not None and self.v2 is not None and self.v3 is not None:
            return np.linalg.norm(np.vstack([v1, self.v2, self.v3]), axis=0)
        return None

    # --- Factory Methods ---
    @classmethod
    def from_values(
        cls,
        grid: Grid,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        r_loc: float = None,
        source_field: Field = None,
    ) -> "Field":
        return cls(grid=grid, v1=v1, v2=v2, v3=v3, r_loc=r_loc, source_field=source_field)

    @classmethod
    def from_coefficients(
        cls, basis: Any, coeffs: np.ndarray, field_type: str = "scalar"
    ) -> "Field":
        return cls(basis=basis, coeffs=coeffs, field_type=field_type)

    @classmethod
    def from_grid_values_expansion(
        cls, basis: Any, basis_evaluator: Any, grid_values: np.ndarray, field_type: str = "scalar"
    ) -> "Field":
        coeffs = basis.from_grid_values(grid_values, basis_evaluator, field_type)
        return cls(basis=basis, coeffs=coeffs, field_type=field_type)

    # --- Core Methods ---
    def evaluate(self, r: Any, theta: Any, phi: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._impl:
            return self._impl.evaluate(r, theta, phi)
        raise NotImplementedError("evaluate() called on Field with no implementation.")

    def basis_vectors(self, r: Any, theta: Any, phi: Any):
        if self.source_field:
            return self.source_field.basis_vectors(r, theta, phi)
        if self._impl:
            return self._impl.basis_vectors(r, theta, phi)
        raise NotImplementedError("basis_vectors() called on Field with no implementation.")

    def discretize(self, grid: Any, r: Any) -> "Field":
        v1, v2, v3 = self.evaluate(r, grid.theta, grid.phi)
        return Field.from_values(
            grid,
            np.asarray(v1).flatten(),
            np.asarray(v2).flatten(),
            np.asarray(v3).flatten(),
            r_loc=r,
            source_field=self,
        )

    def to_grid_values(self, basis_evaluator: "BasisEvaluator"):
        # Only for Expansion impl; others will raise/fail naturally calls to basis
        if hasattr(self._impl, "basis") and self._impl.basis:
            return self._impl.basis.to_grid_values(
                self._impl.coeffs, basis_evaluator, self._impl.field_type
            )
        raise NotImplementedError("to_grid_values valid only for Expansion fields.")

    def regularization_term(self, basis_evaluator: "BasisEvaluator"):
        if hasattr(self._impl, "basis") and self._impl.basis:
            return self._impl.basis.regularization_term(
                self._impl.coeffs, basis_evaluator, self._impl.field_type
            )
        raise NotImplementedError("regularization_term valid only for Expansion fields.")


class VectorAccessor:
    """Helper class for semantic vector component access."""

    def __init__(self, field: "Field"):
        self._field = field

    def _get_component(self, idx: int):
        """Retrieve component by index (0=v1, 1=v2, 2=v3)."""
        attr_name = f"v{idx + 1}"

        # 1. Try accessing property (works for Discrete mode directly)
        val = getattr(self._field, attr_name, None)
        if val is not None:
            return val

        # 2. Return Field in component mode for lazy eval
        return Field(parent_field=self._field, component_index=idx)

    @property
    def v1(self):
        return self._get_component(0)

    @property
    def v2(self):
        return self._get_component(1)

    @property
    def v3(self):
        return self._get_component(2)

    @property
    def r(self):
        return self.v1

    @property
    def theta(self):
        return self.v2

    @property
    def phi(self):
        return self.v3
