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
from pynamit.primitives.grid.interpolation import create_interpolator

if TYPE_CHECKING:
    from pynamit.math.least_squares_problem import LeastSquaresProblem


class _FieldImpl(ABC):
    """Internal implementation interface for Field strategies."""

    @abstractmethod
    def evaluate(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def basis_vectors(self, r, theta, phi):
        raise NotImplementedError


class _ExpansionImpl(_FieldImpl):
    """Implementation for basis expansion fields (including GridBasis)."""

    def __init__(self, basis, coeffs, field_type, weights=None, reg_lambda=None, pinv_rtol=1e-15):
        self.basis = basis
        self.coeffs = coeffs
        self.field_type = field_type
        self.weights = weights
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol
        
        # Cache for LeastSquaresProblem (mapping this field's config to grid hashes)
        self._problem_cache: dict[Any, "LeastSquaresProblem"] = {}

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
        g = Grid(theta=theta, phi=phi)
        
        # Basis handles internal delegation
        values = self.basis.evaluate(self.coeffs, g, self.field_type)

        if self.field_type == "scalar":
            return values, np.zeros_like(values), np.zeros_like(values)
        elif self.field_type == "tangential":
            # Tangential -> (v2, v3)
            return np.zeros_like(values[0]), values[0], values[1]
        elif self.field_type == "vector":
            # Vector -> (v1, v2, v3)
            return values[0], values[1], values[2]

        raise ValueError(f"Unknown field_type: {self.field_type}")

    def curl(self) -> "ExpansionImpl":
        """Compute the radial curl of the horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("curl() only valid for vector or tangential fields.")
        
        # Mapping: [Poloidal; Toroidal] -> Scalar
        grid = getattr(self.basis, "grid", None)
        op = self.basis.get_vector_curl_operator(grid)
        new_coeffs = op.matvec(self.coeffs.reshape(-1))
        
        return _ExpansionImpl(
            self.basis, new_coeffs, field_type="scalar", 
            weights=self.weights, reg_lambda=self.reg_lambda, pinv_rtol=self.pinv_rtol
        )

    def div(self) -> "_ExpansionImpl":
        """Compute the divergence of the horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("div() only valid for vector or tangential fields.")
            
        grid = getattr(self.basis, "grid", None)
        op = self.basis.get_vector_divergence_operator(grid)
        new_coeffs = op.matvec(self.coeffs.reshape(-1))
        
        return _ExpansionImpl(
            self.basis, new_coeffs, field_type="scalar", 
            weights=self.weights, reg_lambda=self.reg_lambda, pinv_rtol=self.pinv_rtol
        )

    def toroidal_potential(self) -> "_ExpansionImpl":
        """Extract the toroidal potential from a horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("toroidal_potential() only valid for vector or tangential fields.")
        
        psi_coeffs = self.basis.get_toroidal_potential_coeffs(self.coeffs)
        
        return _ExpansionImpl(
            self.basis, psi_coeffs, field_type="scalar",
            weights=self.weights, reg_lambda=self.reg_lambda, pinv_rtol=self.pinv_rtol
        )

    def poloidal_potential(self) -> "_ExpansionImpl":
        """Extract the poloidal potential from a horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("poloidal_potential() only valid for vector or tangential fields.")
        
        phi_coeffs = self.basis.get_poloidal_potential_coeffs(self.coeffs)
        
        return _ExpansionImpl(
            self.basis, phi_coeffs, field_type="scalar",
            weights=self.weights, reg_lambda=self.reg_lambda, pinv_rtol=self.pinv_rtol
        )


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
        # Solve params
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        **kwargs,
    ):
        self._impl: Optional[_FieldImpl] = None

        # Metadata storage (exposed by properties)
        self._r_loc = r_loc
        self._source_field = source_field
        self._weights = weights
        self._reg_lambda = reg_lambda
        self._pinv_rtol = pinv_rtol

        # Determine strategy
        if v1 is not None and grid is not None:
            # Discrete case
            self._impl = _DiscreteImpl(grid, v1, v2, v3)

        elif coeffs is not None and basis is not None:
            # Standard Expansion
            self._impl = _ExpansionImpl(
                basis, coeffs, field_type, 
                weights=weights, reg_lambda=reg_lambda, pinv_rtol=pinv_rtol
            )

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
    def weights(self):
        return self._weights

    @property
    def reg_lambda(self):
        return self._reg_lambda

    @property
    def pinv_rtol(self):
        return self._pinv_rtol

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
        cls, 
        basis: Any, 
        coeffs: np.ndarray, 
        field_type: str = "scalar",
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ) -> "Field":
        return cls(
            basis=basis, coeffs=coeffs, field_type=field_type,
            weights=weights, reg_lambda=reg_lambda, pinv_rtol=pinv_rtol
        )

    @classmethod
    def from_grid_values_expansion(
        cls, basis: Any, grid_values: np.ndarray, grid: Grid, field_type: str = "scalar", 
        weights=None, reg_lambda=None, pinv_rtol=1e-15,
        **kwargs
    ) -> "Field":
        coeffs = basis.from_grid_values(
            grid_values, grid, field_type, 
            weights=weights, reg_lambda=reg_lambda, pinv_rtol=pinv_rtol,
            **kwargs
        )
        return cls(
            basis=basis, coeffs=coeffs, field_type=field_type,
            weights=weights, reg_lambda=reg_lambda, pinv_rtol=pinv_rtol
        )

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

    def to_grid_values(self, grid: Grid):
        """Evaluate the field on a grid."""
        if hasattr(self._impl, "basis") and self._impl.basis:
            return self._impl.basis.evaluate(
                self._impl.coeffs, 
                grid, 
                self._impl.field_type,
            )
        raise NotImplementedError("to_grid_values valid only for Expansion fields.")

    def regularization_term(self, grid: Grid):
        """Compute the regularization penalty term."""
        if hasattr(self._impl, "basis") and self._impl.basis:
            return self._impl.basis.regularization_term(
                self._impl.coeffs, 
                grid, 
                self._impl.field_type,
                reg_lambda=self.reg_lambda,
            )
        raise NotImplementedError("regularization_term valid only for Expansion fields.")

    def curl(self) -> "Field":
        """Compute the radial curl of the field."""
        if hasattr(self._impl, "curl"):
            res_impl = self._impl.curl()
            return Field.from_coefficients(
                basis=res_impl.basis,
                coeffs=res_impl.coeffs,
                field_type=res_impl.field_type,
                weights=res_impl.weights,
                reg_lambda=res_impl.reg_lambda,
                pinv_rtol=res_impl.pinv_rtol
            )
        raise NotImplementedError("curl() only supported for Expansion fields for now.")

    def div(self) -> "Field":
        """Compute the divergence of the field."""
        if hasattr(self._impl, "div"):
            res_impl = self._impl.div()
            return Field.from_coefficients(
                basis=res_impl.basis,
                coeffs=res_impl.coeffs,
                field_type=res_impl.field_type,
                weights=res_impl.weights,
                reg_lambda=res_impl.reg_lambda,
                pinv_rtol=res_impl.pinv_rtol
            )
        raise NotImplementedError("div() only supported for Expansion fields for now.")

    def toroidal_potential(self) -> "Field":
        """Extract the toroidal potential of the field."""
        if hasattr(self._impl, "toroidal_potential"):
            res_impl = self._impl.toroidal_potential()
            return Field.from_coefficients(
                basis=res_impl.basis,
                coeffs=res_impl.coeffs,
                field_type=res_impl.field_type,
                weights=res_impl.weights,
                reg_lambda=res_impl.reg_lambda,
                pinv_rtol=res_impl.pinv_rtol
            )
        raise NotImplementedError("toroidal_potential() only supported for Expansion fields for now.")

    def poloidal_potential(self) -> "Field":
        """Extract the poloidal potential of the field."""
        if hasattr(self._impl, "poloidal_potential"):
            res_impl = self._impl.poloidal_potential()
            return Field.from_coefficients(
                basis=res_impl.basis,
                coeffs=res_impl.coeffs,
                field_type=res_impl.field_type,
                weights=res_impl.weights,
                reg_lambda=res_impl.reg_lambda,
                pinv_rtol=res_impl.pinv_rtol
            )
        raise NotImplementedError("poloidal_potential() only supported for Expansion fields for now.")


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
