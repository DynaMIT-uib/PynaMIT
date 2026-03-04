"""Field value objects and field backends.

``Field`` is the user-facing facade for realized or evaluable fields. The
underlying representation can vary internally between coefficient-backed,
sampled/grid-backed, and analytic/provider-backed forms.

Structural metadata about coefficient-backed fields lives in ``FieldSpec``.

``Field`` and analytic providers such as ``Mainfield`` share a small internal
evaluation mixin and a structural typing protocol. That shared layer is an
implementation detail, not a public object model.
"""

from __future__ import annotations
from functools import cached_property
from typing import Protocol, Tuple, Any, Optional
import numpy as np

# Imports
from pynamit.primitives.field_spec import FieldSpec
from pynamit.primitives.grid import Grid
from pynamit.primitives.grid.interpolation import create_interpolator


class SupportsFieldEvaluation(Protocol):
    """Structural protocol for objects that can evaluate field values."""

    def evaluate(self, r: Any, theta: Any, phi: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the field in spherical components."""
        ...

    def basis_vectors(self, r: Any, theta: Any, phi: Any):
        """Return basis vectors if the source field defines them."""
        ...


class _EvaluableMixin:
    """Internal helper for objects that support field evaluation."""

    def discretize(self, grid: Any, r: Any) -> "Field":
        """Sample the field on ``grid`` at radius ``r`` and return a discrete Field."""
        v1, v2, v3 = self.evaluate(r, grid.theta, grid.phi)
        return Field.from_grid_values(
            grid,
            np.asarray(v1).flatten(),
            np.asarray(v2).flatten(),
            np.asarray(v3).flatten(),
            r_loc=r,
            source_field=self,
        )

    @cached_property
    def vec(self) -> "VectorAccessor":
        """Return semantic vector-component accessors."""
        return VectorAccessor(self)

    @property
    def scalar(self) -> Any:
        """Scalar alias for the first component."""
        return self.vec.v1


class _FieldBackend(Protocol):
    """Backend interface used by ``Field``."""

    def evaluate(self, r, theta, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    def basis_vectors(self, r, theta, phi):
        ...


class _CoefficientFieldBackend(_FieldBackend):
    """Coefficient-backed field backend."""

    def __init__(
        self,
        basis,
        coeffs,
        field_type,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        mean_free: Optional[bool] = None,
    ):
        if mean_free is None:
            mean_free = getattr(basis, "mean_free", False)
        self.spec = FieldSpec(
            basis=basis,
            field_type=field_type,
            mean_free=bool(mean_free),
        )
        self.coeffs = coeffs
        self.weights = weights
        self.reg_lambda = reg_lambda
        self.pinv_rtol = pinv_rtol

    @property
    def basis(self):
        return self.spec.basis

    @property
    def field_type(self):
        return self.spec.field_type

    @property
    def mean_free(self):
        return self.spec.mean_free

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

        values = self.spec.evaluate(self.coeffs, g, self.field_type)

        if self.field_type == "scalar":
            return values, np.zeros_like(values), np.zeros_like(values)
        elif self.field_type == "tangential":
            # Tangential -> (v2, v3)
            return np.zeros_like(values[0]), values[0], values[1]
        elif self.field_type == "vector":
            # Vector -> (v1, v2, v3)
            return values[0], values[1], values[2]

        raise ValueError(f"Unknown field_type: {self.field_type}")

    def curl(self) -> "_CoefficientFieldBackend":
        """Compute the radial curl of the horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("curl() only valid for vector or tangential fields.")
        
        # Mapping: [Poloidal; Toroidal] -> Scalar
        grid = getattr(self.spec, "grid", None)
        op = self.spec.get_vector_curl_operator(grid)
        new_coeffs = op.matvec(self.coeffs.reshape(-1))
        
        return _CoefficientFieldBackend(
            self.basis, new_coeffs, field_type="scalar", 
            weights=self.weights,
            reg_lambda=self.reg_lambda,
            pinv_rtol=self.pinv_rtol,
            mean_free=self.mean_free,
        )

    def div(self) -> "_CoefficientFieldBackend":
        """Compute the divergence of the horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("div() only valid for vector or tangential fields.")
            
        grid = getattr(self.spec, "grid", None)
        op = self.spec.get_vector_divergence_operator(grid)
        new_coeffs = op.matvec(self.coeffs.reshape(-1))
        
        return _CoefficientFieldBackend(
            self.basis, new_coeffs, field_type="scalar", 
            weights=self.weights,
            reg_lambda=self.reg_lambda,
            pinv_rtol=self.pinv_rtol,
            mean_free=self.mean_free,
        )

    def toroidal_potential(self) -> "_CoefficientFieldBackend":
        """Extract the toroidal potential from a horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("toroidal_potential() only valid for vector or tangential fields.")
        
        psi_coeffs = self.spec.get_toroidal_potential_coeffs(self.coeffs)
        
        return _CoefficientFieldBackend(
            self.basis, psi_coeffs, field_type="scalar",
            weights=self.weights,
            reg_lambda=self.reg_lambda,
            pinv_rtol=self.pinv_rtol,
            mean_free=self.mean_free,
        )

    def poloidal_potential(self) -> "_CoefficientFieldBackend":
        """Extract the poloidal potential from a horizontal vector field."""
        if self.field_type not in ["vector", "tangential"]:
            raise ValueError("poloidal_potential() only valid for vector or tangential fields.")
        
        phi_coeffs = self.spec.get_poloidal_potential_coeffs(self.coeffs)
        
        return _CoefficientFieldBackend(
            self.basis, phi_coeffs, field_type="scalar",
            weights=self.weights,
            reg_lambda=self.reg_lambda,
            pinv_rtol=self.pinv_rtol,
            mean_free=self.mean_free,
        )


class _SampledFieldBackend:
    """Sampled/grid-backed field backend."""
    
    def __init__(self, grid, v1, v2=None, v3=None, *, r_loc=None, source_field: Optional[SupportsFieldEvaluation] = None):
        self.grid = grid
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
        self.r_loc = r_loc
        self.source_field = source_field
        
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

    def basis_vectors(self, r, theta, phi):
        if self.source_field is None:
            raise NotImplementedError
        return self.source_field.basis_vectors(r, theta, phi)


class _AnalyticFieldBackend:
    """Backend that delegates to an analytic/provider field source."""

    def __init__(self, provider: SupportsFieldEvaluation):
        self.provider = provider

    def evaluate(self, r, theta, phi):
        return self.provider.evaluate(r, theta, phi)

    def basis_vectors(self, r, theta, phi):
        return self.provider.basis_vectors(r, theta, phi)


class _ComponentFieldView:
    """Lazy single-component view over another evaluable field."""

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


class Field(_EvaluableMixin):
    """Unified field abstraction for discrete and coefficient-backed data.

    Coefficient-backed fields carry a ``FieldSpec`` describing basis family,
    field type, and whether SH storage is mean-free. Solve and projection
    options such as regularization remain separate from that structural spec.
    """

    def __init__(self, backend: _FieldBackend):
        if not callable(getattr(backend, "evaluate", None)):
            raise TypeError(
                "Field expects an internal field backend. Use Field.from_grid_values(...), "
                "Field.from_coefficients(...), or Field.from_provider(...)."
            )
        self._backend = backend

    # --- Property Delegation ---
    @cached_property
    def _coefficient_backend(self) -> Optional[_CoefficientFieldBackend]:
        return self._backend if isinstance(self._backend, _CoefficientFieldBackend) else None

    @cached_property
    def _sampled_backend(self) -> Optional[_SampledFieldBackend]:
        return self._backend if isinstance(self._backend, _SampledFieldBackend) else None

    @cached_property
    def _component_backend(self) -> Optional[_ComponentFieldView]:
        return self._backend if isinstance(self._backend, _ComponentFieldView) else None

    @property
    def grid(self):
        if self._sampled_backend is not None:
            return self._sampled_backend.grid
        if self._coefficient_backend is not None:
            return getattr(self._coefficient_backend.basis, "grid", None)
        return None

    @property
    def v1(self):
        return getattr(self._backend, "v1", None)

    @property
    def v2(self):
        return getattr(self._backend, "v2", None)

    @property
    def v3(self):
        return getattr(self._backend, "v3", None)

    @property
    def basis(self):
        return getattr(self._backend, "basis", None)

    @property
    def coeffs(self):
        return getattr(self._backend, "coeffs", None)

    @property
    def field_type(self):
        return getattr(self._backend, "field_type", None)

    @property
    def component_index(self):
        return getattr(self._backend, "component_index", None)

    @property
    def mean_free(self) -> Optional[bool]:
        return getattr(self._backend, "mean_free", None)

    @property
    def spec(self) -> Optional[FieldSpec]:
        """Return the structural descriptor for coefficient-backed fields."""
        return getattr(self._backend, "spec", None)

    @property
    def r_loc(self):
        return getattr(self._backend, "r_loc", None)

    @property
    def source_field(self):
        return getattr(self._backend, "source_field", None)

    @property
    def weights(self):
        return getattr(self._backend, "weights", None)

    @property
    def reg_lambda(self):
        return getattr(self._backend, "reg_lambda", None)

    @property
    def pinv_rtol(self):
        return getattr(self._backend, "pinv_rtol", None)

    @property
    def magnitude(self) -> Optional[np.ndarray]:
        v1 = self.v1
        if v1 is not None and self.v2 is not None and self.v3 is not None:
            return np.linalg.norm(np.vstack([v1, self.v2, self.v3]), axis=0)
        return None

    # --- Factory Methods ---
    @classmethod
    def from_grid_values(
        cls,
        grid: Grid,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        r_loc: float = None,
        source_field: Optional[SupportsFieldEvaluation] = None,
    ) -> "Field":
        """Construct a sampled/grid-backed field."""
        return cls(
            _SampledFieldBackend(
                grid,
                v1,
                v2,
                v3,
                r_loc=r_loc,
                source_field=source_field,
            )
        )

    @classmethod
    def from_coefficients(
        cls, 
        basis: Any = None, 
        coeffs: np.ndarray = None, 
        field_type: str = "scalar",
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        mean_free: Optional[bool] = None,
        spec: Optional[FieldSpec] = None,
    ) -> "Field":
        """Construct a coefficient-backed field."""
        if spec is None and isinstance(basis, FieldSpec):
            spec = basis
            basis = None
        if spec is not None:
            if basis is not None and basis is not spec.basis:
                raise ValueError("Field construction received both 'basis' and inconsistent 'spec.basis'.")
            basis = spec.basis
            field_type = spec.field_type
            mean_free = spec.mean_free
        if basis is None or coeffs is None:
            raise ValueError("Coefficient-backed fields require both 'basis' and 'coeffs'.")
        return cls(
            _CoefficientFieldBackend(
                basis=basis,
                coeffs=coeffs,
                field_type=field_type,
                weights=weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
                mean_free=mean_free,
            )
        )

    @classmethod
    @classmethod
    def from_provider(cls, provider: SupportsFieldEvaluation) -> "Field":
        """Wrap an analytic/provider-style field behind the ``Field`` facade."""
        return cls(_AnalyticFieldBackend(provider))

    # --- Core Methods ---
    def evaluate(self, r: Any, theta: Any, phi: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._backend.evaluate(r, theta, phi)

    def basis_vectors(self, r: Any, theta: Any, phi: Any):
        return self._backend.basis_vectors(r, theta, phi)

    def evaluate_on_grid(self, grid: Grid, r: Optional[Any] = None):
        """Evaluate a coefficient-backed field on a grid.

        Parameters
        ----------
        grid : Grid
            Target grid.
        r : Any, optional
            Reserved for API symmetry with ``evaluate(r, theta, phi)``.
            Coefficient-backed fields are defined on the associated shell basis,
            so this argument is currently ignored in this method.
        """
        coefficient_backend = self._coefficient_backend
        if coefficient_backend is not None and coefficient_backend.basis is not None:
            return coefficient_backend.spec.evaluate(
                coefficient_backend.coeffs,
                grid,
                coefficient_backend.field_type,
            )
        raise NotImplementedError("evaluate_on_grid valid only for coefficient-backed fields.")

    def regularization_term(self, grid: Grid):
        """Compute the regularization penalty term."""
        coefficient_backend = self._coefficient_backend
        if coefficient_backend is not None and coefficient_backend.basis is not None:
            is_scalar = coefficient_backend.field_type == "scalar"
            reg_op = coefficient_backend.spec.get_regularization_matrix(
                scalar=is_scalar,
                reg_lambda=self.reg_lambda,
            )
            if reg_op is None or self.reg_lambda is None or self.reg_lambda == 0:
                return 0.0
            if not is_scalar:
                return np.tensordot(reg_op, coefficient_backend.coeffs, 2)
            return np.dot(coefficient_backend.coeffs, np.dot(reg_op, coefficient_backend.coeffs))
        raise NotImplementedError("regularization_term valid only for coefficient-backed fields.")

    @staticmethod
    def _from_coefficient_result(res_impl: _CoefficientFieldBackend) -> "Field":
        return Field.from_coefficients(
            spec=res_impl.spec,
            coeffs=res_impl.coeffs,
            weights=res_impl.weights,
            reg_lambda=res_impl.reg_lambda,
            pinv_rtol=res_impl.pinv_rtol,
        )

    def _require_coefficient_backend(self, op_name: str) -> _CoefficientFieldBackend:
        coefficient_backend = self._coefficient_backend
        if coefficient_backend is None:
            raise NotImplementedError(f"{op_name}() only supported for coefficient-backed fields for now.")
        return coefficient_backend

    def curl(self) -> "Field":
        """Compute the radial curl of the field."""
        res_impl = self._require_coefficient_backend("curl").curl()
        return self._from_coefficient_result(res_impl)

    def div(self) -> "Field":
        """Compute the divergence of the field."""
        res_impl = self._require_coefficient_backend("div").div()
        return self._from_coefficient_result(res_impl)

    def toroidal_potential(self) -> "Field":
        """Extract the toroidal potential of the field."""
        res_impl = self._require_coefficient_backend("toroidal_potential").toroidal_potential()
        return self._from_coefficient_result(res_impl)

    def poloidal_potential(self) -> "Field":
        """Extract the poloidal potential of the field."""
        res_impl = self._require_coefficient_backend("poloidal_potential").poloidal_potential()
        return self._from_coefficient_result(res_impl)


class VectorAccessor:
    """Helper class for semantic vector component access."""

    def __init__(self, field: SupportsFieldEvaluation):
        self._field = field

    def _get_component(self, idx: int):
        """Retrieve component by index (0=v1, 1=v2, 2=v3)."""
        attr_name = f"v{idx + 1}"

        # 1. Try accessing property (works for Discrete mode directly)
        val = getattr(self._field, attr_name, None)
        if val is not None:
            return val

        # 2. Return Field in component mode for lazy eval
        return Field(_ComponentFieldView(self._field, idx))

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
