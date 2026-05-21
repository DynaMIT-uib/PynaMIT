"""Basis interface utilities."""

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from pynamit.math import as_linear_map
from pynamit.math.backend import get_array_module


def _backend_array(value, *hints):
    """Return ``value`` on the backend implied by ``hints``."""
    xp = get_array_module(*hints, value)
    return xp.asarray(value)


def _backend_stack(values, axis=0):
    """Stack arrays on their active backend."""
    xp = get_array_module(*values)
    return xp.stack([xp.asarray(value) for value in values], axis=axis)


def basis_kind(basis):
    """Return the normalized kind tag for a basis-like object."""
    kind = getattr(basis, "kind", None)
    return None if kind is None else str(kind).upper()


def is_basis_kind(basis, *kinds):
    """Return whether ``basis`` advertises one of ``kinds``."""
    kind = basis_kind(basis)
    return kind is not None and kind in {str(item).upper() for item in kinds}


def is_sh_basis(basis):
    """Return whether ``basis`` is a spherical-harmonic basis."""
    return is_basis_kind(basis, "SH")


def is_cs_basis(basis):
    """Return whether ``basis`` is a cubed-sphere basis."""
    return is_basis_kind(basis, "CS")


def is_grid_basis(basis):
    """Return whether ``basis`` stores values directly on a grid."""
    return isinstance(basis, GridBasis) or is_basis_kind(basis, "CS", "GRID")


def normalize_horizontal_basis_kind(kind):
    """Normalize a user supplied horizontal-basis kind."""
    normalized = str(kind).strip().upper()
    if normalized not in {"SH", "CS"}:
        raise ValueError("horizontal_basis_kind must be one of ['CS', 'SH'].")
    return normalized


class Basis(ABC):
    """Abstract metadata interface for basis representations.

    Concrete basis classes may expose these fields as regular instance
    attributes. Class-level placeholders satisfy the abstract contract,
    while ``validate_metadata`` checks initialized instances.
    """

    supports_surface_potential_operators = False
    supports_radial_potential_operators = False

    required_attributes = (
        "kind",
        "index_names",
        "index_length",
        "index_arrays",
        "minimum_phi_sampling",
        "caching",
    )

    @property
    def signature(self):
        """Return a stable cache signature for this basis instance."""
        parts = [type(self).__module__, type(self).__qualname__, self.kind]
        for name in ("Nmax", "Mmax", "Nmin", "mean_free", "backend", "is_normalized", "N"):
            if hasattr(self, name):
                parts.append((name, getattr(self, name)))
        return tuple(parts)

    @property
    def coefficient_space_signature(self):
        """Return a signature for coefficient-space compatibility.

        This deliberately describes the coefficient layout and scaling,
        not incidental implementation choices such as evaluation caches
        or the Legendre backend used by an SH basis.
        """
        return (
            type(self).__module__,
            type(self).__qualname__,
            basis_kind(self),
            tuple(self.index_names),
            self.index_length,
        )

    def coefficients_are_compatible_with(self, other):
        """Return whether coefficient vectors share operators."""
        return (
            isinstance(other, Basis)
            and self.coefficient_space_signature == other.coefficient_space_signature
        )

    @property
    @abstractmethod
    def kind(self):
        """Short identifier for the basis."""
        pass

    @property
    @abstractmethod
    def index_names(self):
        """Names of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def index_length(self):
        """Total number of basis functions."""
        pass

    @property
    @abstractmethod
    def index_arrays(self):
        """Arrays of indices used in the basis."""
        pass

    @property
    @abstractmethod
    def minimum_phi_sampling(self):
        """Minimum required sampling in phi direction."""
        pass

    @property
    @abstractmethod
    def caching(self):
        """Whether basis evaluations can be cached."""
        pass

    def validate_metadata(self) -> None:
        """Validate that required basis metadata is initialized."""
        missing = [name for name in self.required_attributes if getattr(self, name, None) is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{type(self).__name__} is missing basis metadata: {joined}.")


class GridBasis(Basis):
    """Basis whose coefficients are values on a native grid."""

    def __init__(self):
        """Initialize default grid-basis metadata."""
        self._kind = "GRID"
        self._index_names = None
        self._index_length = None
        self._index_arrays = None
        self._minimum_phi_sampling = 1
        self._caching = False

    @property
    def kind(self):
        """Short identifier for the grid basis."""
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value

    @property
    def index_names(self):
        """Names of indices used in the basis."""
        return self._index_names

    @index_names.setter
    def index_names(self, value):
        self._index_names = value

    @property
    def index_length(self):
        """Total number of grid coefficients."""
        return self._index_length

    @index_length.setter
    def index_length(self, value):
        self._index_length = value

    @property
    def index_arrays(self):
        """Arrays of grid-coordinate indices used in the basis."""
        return self._index_arrays

    @index_arrays.setter
    def index_arrays(self, value):
        self._index_arrays = value

    @property
    def minimum_phi_sampling(self):
        """Minimum required sampling in phi direction."""
        return self._minimum_phi_sampling

    @minimum_phi_sampling.setter
    def minimum_phi_sampling(self, value):
        self._minimum_phi_sampling = value

    @property
    def caching(self):
        """Whether basis evaluations can be cached."""
        return self._caching

    @caching.setter
    def caching(self, value):
        self._caching = bool(value)


class SurfaceOperators(Basis):
    """Basis with scalar and vector operators on a spherical surface."""

    supports_surface_potential_operators = True

    @abstractmethod
    def evaluate_on_grid(self, grid, derivative=None, cache_in=None, cache_out=False):
        """Evaluate basis functions or derivatives on ``grid``."""
        pass

    @abstractmethod
    def laplacian(self, r=1.0):
        """Return the scalar surface Laplacian operator."""
        pass

    def get_scalar_evaluation_matrix(self, grid):
        """Return the scalar coefficient-to-grid matrix."""
        return _backend_array(
            self.evaluate_on_grid(grid),
            getattr(grid, "theta", None),
            getattr(grid, "phi", None),
        )

    def get_scalar_evaluation_operator(self, grid):
        """Return the scalar coefficient-to-grid operator."""
        matrix = self.get_scalar_evaluation_matrix(grid)
        return as_linear_map(matrix, input_shape=(self.index_length,))

    def get_surface_gradient_matrix(self, grid):
        """Return ``[d_theta, sin(theta)^-1 d_phi]`` on a surface."""
        return _backend_stack(
            [
                self.evaluate_on_grid(grid, derivative="theta"),
                self.evaluate_on_grid(grid, derivative="phi"),
            ]
        )

    def get_surface_gradient_operator(self, grid):
        """Return the scalar-to-vector surface-gradient operator."""
        matrix = self.get_surface_gradient_matrix(grid)
        return as_linear_map(
            matrix,
            input_shape=(self.index_length,),
            output_shape=matrix.shape[:-1],
        )

    def get_rhat_cross_gradient_matrix(self, grid):
        """Return the tangential ``rhat x grad`` operator."""
        grad_theta, grad_phi = self.get_surface_gradient_matrix(grid)
        return _backend_stack([-grad_phi, grad_theta])

    def get_rhat_cross_gradient_operator(self, grid):
        """Return the scalar-to-vector ``rhat x grad`` operator."""
        matrix = self.get_rhat_cross_gradient_matrix(grid)
        return as_linear_map(
            matrix,
            input_shape=(self.index_length,),
            output_shape=matrix.shape[:-1],
        )

    def get_helmholtz_synthesis_matrix(self, grid):
        """Return the canonical tangential Helmholtz synthesis tensor.

        Coefficients are ordered as curl-free then divergence-free
        potentials. Components are ordered as theta then phi.
        """
        return _backend_stack(
            [
                -self.get_surface_gradient_matrix(grid),
                self.get_rhat_cross_gradient_matrix(grid),
            ],
            axis=2,
        )

    def get_helmholtz_synthesis_operator(self, grid):
        """Return the Helmholtz-potential-to-vector operator."""
        matrix = self.get_helmholtz_synthesis_matrix(grid)
        return as_linear_map(
            matrix,
            input_shape=(2, self.index_length),
            output_shape=matrix.shape[:2],
        )

    def get_surface_laplacian_operator(self, r=1.0):
        """Return the surface scalar Laplacian operator."""
        return as_linear_map(self.laplacian(r))


class RadialLaplaceContinuation(ABC):
    """Interface for bases with a radial Laplace continuation."""

    supports_radial_potential_operators = True

    def get_external_potential_continuation_operator(self, start, end):
        """Return the external-potential continuation operator."""
        return as_linear_map(self.external_potential_continuation(start, end))

    def get_internal_potential_continuation_operator(self, start, end):
        """Return the internal-potential continuation operator."""
        return as_linear_map(self.internal_potential_continuation(start, end))

    def get_boundary_potential_discontinuity_operator(self):
        """Return the boundary-potential discontinuity operator."""
        return as_linear_map(self.boundary_potential_discontinuity)

    @abstractmethod
    def external_potential_continuation(self, start, end):
        """Return external-potential continuation."""
        pass

    @abstractmethod
    def internal_potential_continuation(self, start, end):
        """Return internal-potential continuation."""
        pass

    @property
    @abstractmethod
    def boundary_potential_discontinuity(self):
        """Return the regular/irregular potential discontinuity."""
        pass


class BasisView(SurfaceOperators):
    """Coefficient-space view of another evaluable basis."""

    def __init__(
        self,
        parent_basis,
        coefficient_indices=None,
        *,
        metadata=None,
        coefficient_space_signature=None,
        view_name="view",
    ):
        """Initialize a coefficient-space view."""
        if not isinstance(parent_basis, SurfaceOperators):
            raise TypeError("BasisView parent_basis must implement SurfaceOperators.")

        parent_basis.validate_metadata()
        self.parent_basis = parent_basis
        self._parent_coefficient_indices = self._normalize_coefficient_indices(
            parent_basis, coefficient_indices
        )
        self._view_name = str(view_name)
        self._coefficient_space_signature = coefficient_space_signature
        self._related_basis_cache = {}

        self.kind = parent_basis.kind
        self.index_names = list(parent_basis.index_names)
        self.index_length = int(self._parent_coefficient_indices.size)
        self.index_arrays = self._slice_index_arrays(
            parent_basis, self._parent_coefficient_indices
        )
        self.minimum_phi_sampling = parent_basis.minimum_phi_sampling
        self.caching = parent_basis.caching
        self.supports_surface_potential_operators = bool(
            parent_basis.supports_surface_potential_operators
        )
        self.supports_radial_potential_operators = bool(
            parent_basis.supports_radial_potential_operators
        )

        for name, values in zip(self.index_names, self.index_arrays):
            if isinstance(name, str) and name.isidentifier() and not hasattr(self, name):
                setattr(self, name, values)

        for name, value in (metadata or {}).items():
            setattr(self, name, value)

        self.validate_metadata()

    @staticmethod
    def _normalize_coefficient_indices(parent_basis, coefficient_indices):
        """Return validated parent coefficient indices for a view."""
        parent_length = int(parent_basis.index_length)
        if coefficient_indices is None:
            return np.arange(parent_length, dtype=int)

        raw_indices = np.asarray(coefficient_indices)
        if raw_indices.ndim != 1:
            raise ValueError("BasisView coefficient_indices must be one-dimensional.")
        if raw_indices.dtype == bool:
            if raw_indices.size != parent_length:
                raise ValueError(
                    "BasisView boolean coefficient_indices must match parent index_length."
                )
            indices = np.flatnonzero(raw_indices)
        else:
            if not np.issubdtype(raw_indices.dtype, np.integer):
                raise TypeError(
                    "BasisView coefficient_indices must be integers or a boolean mask."
                )
            indices = raw_indices.astype(int, copy=False)

        if np.any(indices < 0) or np.any(indices >= parent_length):
            raise IndexError("BasisView coefficient_indices are outside the parent basis.")
        if np.unique(indices).size != indices.size:
            raise ValueError("BasisView coefficient_indices must not contain duplicates.")
        return indices.copy()

    @staticmethod
    def _slice_index_arrays(parent_basis, coefficient_indices):
        """Slice per-coefficient metadata arrays from the parent."""
        arrays = []
        for values in parent_basis.index_arrays:
            array = np.asarray(values)
            if array.shape == (parent_basis.index_length,):
                arrays.append(array[coefficient_indices])
            elif array.size == parent_basis.index_length:
                arrays.append(array.reshape(parent_basis.index_length)[coefficient_indices])
            else:
                raise ValueError(
                    "BasisView can only slice index_arrays with one value per coefficient."
                )
        return arrays

    @property
    def signature(self):
        """Return a stable cache signature for this basis view."""
        return self.parent_basis.signature + (
            "view",
            self._view_name,
            tuple(int(index) for index in self._parent_coefficient_indices),
            self.coefficient_space_signature,
        )

    @property
    def coefficient_space_signature(self):
        """Return a signature for coefficient-space compatibility."""
        if self._coefficient_space_signature is not None:
            return self._coefficient_space_signature
        parent_indices = np.arange(self.parent_basis.index_length, dtype=int)
        if np.array_equal(self._parent_coefficient_indices, parent_indices):
            return self.parent_basis.coefficient_space_signature
        return (
            "VIEW",
            self.parent_basis.coefficient_space_signature,
            tuple(int(index) for index in self._parent_coefficient_indices),
        )

    @property
    def root_basis(self):
        """Return the non-view ancestor for this basis view."""
        basis = self.parent_basis
        while isinstance(basis, BasisView):
            basis = basis.parent_basis
        return basis

    @property
    def kind(self):
        """Short identifier for the basis."""
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value

    @property
    def index_names(self):
        """Names of indices used in the basis."""
        return self._index_names

    @index_names.setter
    def index_names(self, value):
        self._index_names = value

    @property
    def index_length(self):
        """Total number of basis functions."""
        return self._index_length

    @index_length.setter
    def index_length(self, value):
        self._index_length = value

    @property
    def index_arrays(self):
        """Arrays of indices used in the basis."""
        return self._index_arrays

    @index_arrays.setter
    def index_arrays(self, value):
        self._index_arrays = value

    @property
    def minimum_phi_sampling(self):
        """Minimum required sampling in phi direction."""
        return self._minimum_phi_sampling

    @minimum_phi_sampling.setter
    def minimum_phi_sampling(self, value):
        self._minimum_phi_sampling = value

    @property
    def caching(self):
        """Whether basis evaluations can be cached."""
        return self._caching

    @caching.setter
    def caching(self, value):
        self._caching = bool(value)

    def _slice_coefficient_operator(self, values, operator_name):
        """Slice a parent coefficient-space operator to this view."""
        indices = self._parent_coefficient_indices
        if sp.issparse(values):
            expected_shape = (self.parent_basis.index_length, self.parent_basis.index_length)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{operator_name} has shape {values.shape}, expected {expected_shape}."
                )
            return values.tocsr()[indices, :][:, indices]

        xp = get_array_module(values)
        array = xp.asarray(values)
        if array.ndim == 1:
            if array.size != self.parent_basis.index_length:
                raise ValueError(
                    f"{operator_name} has length {array.size}, expected "
                    f"{self.parent_basis.index_length}."
                )
            return array[indices]
        if array.ndim == 2:
            expected_shape = (self.parent_basis.index_length, self.parent_basis.index_length)
            if array.shape != expected_shape:
                raise ValueError(
                    f"{operator_name} has shape {array.shape}, expected {expected_shape}."
                )
            return array[indices][:, indices]
        raise ValueError(f"{operator_name} must be a 1-D or square 2-D coefficient operator.")

    def _require_radial_support(self):
        """Raise if the parent basis has no radial continuation."""
        if not self.supports_radial_potential_operators:
            raise NotImplementedError(
                f"{type(self.parent_basis).__name__} does not support radial potential operators."
            )

    def evaluate_on_grid(self, grid, derivative=None, cache_in=None, cache_out=False):
        """Evaluate the viewed basis functions on ``grid``."""
        result = self.parent_basis.evaluate_on_grid(
            grid,
            derivative=derivative,
            cache_in=cache_in,
            cache_out=cache_out,
        )
        if cache_out:
            matrix, cache = result
            return matrix[:, self._parent_coefficient_indices], cache
        return result[:, self._parent_coefficient_indices]

    def laplacian(self, r=1.0):
        """Return the viewed scalar surface Laplacian operator."""
        return self._slice_coefficient_operator(self.parent_basis.laplacian(r), "laplacian")

    @property
    def boundary_potential_discontinuity(self):
        """Return the viewed boundary-potential discontinuity."""
        self._require_radial_support()
        return self._slice_coefficient_operator(
            self.parent_basis.boundary_potential_discontinuity,
            "boundary_potential_discontinuity",
        )

    def external_potential_continuation(self, start, end):
        """Return the viewed external-potential continuation."""
        self._require_radial_support()
        return self._slice_coefficient_operator(
            self.parent_basis.external_potential_continuation(start, end),
            "external_potential_continuation",
        )

    def internal_potential_continuation(self, start, end):
        """Return the viewed internal-potential continuation."""
        self._require_radial_support()
        return self._slice_coefficient_operator(
            self.parent_basis.internal_potential_continuation(start, end),
            "internal_potential_continuation",
        )

    def get_external_potential_continuation_operator(self, start, end):
        """Return viewed external-potential continuation operator."""
        return as_linear_map(self.external_potential_continuation(start, end))

    def get_internal_potential_continuation_operator(self, start, end):
        """Return viewed internal-potential continuation operator."""
        return as_linear_map(self.internal_potential_continuation(start, end))

    def get_boundary_potential_discontinuity_operator(self):
        """Return viewed boundary-potential discontinuity operator."""
        self._require_radial_support()
        return as_linear_map(self.boundary_potential_discontinuity)

    def scalar_fields_are_mean_free_by_construction(self):
        """Return whether scalar coefficients omit the mean term."""
        return bool(getattr(self, "mean_free", False))

    def scalar_index_length(self, mean_free=None):
        """Return scalar coefficient count."""
        return int(self.scalar_degrees(mean_free=mean_free).size)

    def scalar_degrees(self, mean_free=None):
        """Return harmonic degrees for the requested scalar space."""
        basis = self.with_mean_free(
            self.mean_free if mean_free is None else bool(mean_free)
        )
        return basis.n

    def scalar_orders(self, mean_free=None):
        """Return harmonic orders for the requested scalar space."""
        basis = self.with_mean_free(
            self.mean_free if mean_free is None else bool(mean_free)
        )
        return basis.m

    def scalar_index_arrays(self, mean_free=None):
        """Return scalar index arrays for the requested scalar space."""
        basis = self.with_mean_free(
            self.mean_free if mean_free is None else bool(mean_free)
        )
        return basis.n, basis.m

    def get_extended_basis(self):
        """Return the parent basis extended to the full scalar space."""
        if hasattr(self.parent_basis, "get_extended_basis"):
            return self.parent_basis.get_extended_basis()
        return self.parent_basis

    def with_mean_free(self, mean_free):
        """Return a compatible mean-free/full basis when available."""
        target_mean_free = bool(mean_free)
        if bool(getattr(self, "mean_free", False)) == target_mean_free:
            return self
        if hasattr(self.parent_basis, "with_mean_free"):
            return self.parent_basis.with_mean_free(target_mean_free)
        raise NotImplementedError(f"{type(self).__name__} does not define mean-free variants.")
